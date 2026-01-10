from __future__ import annotations

from collections import deque
import logging
import os
import time
from typing import Any, Optional, List

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    cache_length,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from .prompt_cache import LRUPromptCache

def build_kmp_lps(pattern: List[int]) -> List[int]:
    """Build KMP LPS table for token stop-sequence matching."""
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps

def generation_loop(dist_state, model, tokenizer, args, prompt_cache_store=None):
    """Main generation loop running on ALL ranks.

    All ranks must execute this together for pipeline parallelism to work.
    """
    rank = dist_state.rank
    logging.info(f"Generation loop started on rank {rank}")
    
    # Initialize prompt cache store if not provided
    if prompt_cache_store is None:
        prompt_cache_store = LRUPromptCache(max_size=max(0, int(args.prompt_cache_size)))

    request_n = 0

    while True:
        # All ranks sync and check for requests
        (
            prompt_tokens,
            max_tokens,
            seed,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            repetition_context_size,
            stop_token_sequences,
            response_queue,
            request,
        ) = dist_state.broadcast_request()

        if prompt_tokens is None:
            # No request - brief sleep to avoid busy-waiting
            time.sleep(0.005)
            continue

        request_id = None
        if rank == 0 and isinstance(request, dict):
            request_id = request.get("request_id")

        # If the client disconnected before generation started, skip work.
        if dist_state.sync_should_cancel(request_id):
            if rank == 0:
                logging.info("Request canceled before start (id=%s)", request_id)
                if response_queue is not None:
                    response_queue.put(None)
                dist_state.clear_request_canceled(request_id)
            continue

        mx.random.seed(int(seed))
        request_n += 1

        if rank == 0:
            logging.info(
                "Request params: req=%d seed=%d max_tokens=%d temperature=%.3f top_p=%.3f top_k=%d repetition_penalty=%.3f repetition_context_size=%d stop_sequences=%d",
                request_n,
                int(seed),
                int(max_tokens) if max_tokens is not None else 0,
                float(temperature),
                float(top_p),
                int(top_k),
                float(repetition_penalty),
                int(repetition_context_size),
                len(stop_token_sequences or []),
            )

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        logits_processors = None
        if repetition_penalty and repetition_penalty != 0.0:
            logits_processors = make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            )
            if rank == 0:
                logging.info(
                    "Using repetition_penalty=%.3f repetition_context_size=%d",
                    repetition_penalty,
                    repetition_context_size,
                )

        # All ranks generate together
        # Try to fetch from prompt cache
        cached_prompt_cache, tokens_to_process = prompt_cache_store.fetch_nearest_cache(
            args.model, prompt_tokens
        )
        full_prompt_len = len(prompt_tokens)

        if rank == 0:
            cache_hit = cached_prompt_cache is not None
            reused_len = max(0, full_prompt_len - len(tokens_to_process))
            cached_len = 0
            if cached_prompt_cache is not None:
                try:
                    cached_len = int(cache_length(cached_prompt_cache))
                except Exception:
                    cached_len = 0
            logging.info(
                "Starting generation: prompt_len=%d cache_hit=%s reused_len=%d to_process_len=%d cached_len=%d max_tokens=%s",
                full_prompt_len,
                cache_hit,
                reused_len,
                len(tokens_to_process),
                cached_len,
                str(max_tokens),
            )
            gen_start_t = time.perf_counter()
            first_token_dt = None
        else:
            gen_start_t = None
            first_token_dt = None
        
        # Create prompt cache if not found
        if cached_prompt_cache is None:
            prompt_cache = make_prompt_cache(model)
        else:
            prompt_cache = cached_prompt_cache

        # If we have an exact cache match, tokens_to_process can be empty. MLX
        # generation requires a non-empty prompt (unless using input embeddings),
        # so ensure we always process at least one token.
        if not tokens_to_process:
            if not prompt_tokens:
                if rank == 0 and response_queue is not None:
                    response_queue.put(
                        {
                            "text": "Error: empty prompt_tokens (cannot generate).",
                            "finish_reason": "error",
                        }
                    )
                    response_queue.put(None)
                continue

            if can_trim_prompt_cache(prompt_cache):
                # Trim one token off the cache and re-process it to seed generation.
                try:
                    trim_prompt_cache(prompt_cache, 1)
                    tokens_to_process = [prompt_tokens[-1]]
                    if rank == 0:
                        logging.debug(
                            "Exact prompt cache hit; trimmed 1 token to avoid empty prompt."
                        )
                except Exception:
                    if rank == 0:
                        logging.debug(
                            "Failed to trim prompt cache for exact match; rebuilding cache.",
                            exc_info=True,
                        )
                    prompt_cache = make_prompt_cache(model)
                    tokens_to_process = prompt_tokens
            else:
                # Fallback: rebuild prompt cache and re-process the full prompt.
                prompt_cache = make_prompt_cache(model)
                tokens_to_process = prompt_tokens

        prompt = mx.array(tokens_to_process, dtype=mx.int32)
        cache_key = prompt_tokens[:]

        # Stop-sequence buffering (rank 0 only) to avoid emitting partial stop strings.
        pending_items = (
            deque() if rank == 0 and response_queue is not None else None
        )

        stop_sequences = [s for s in (stop_token_sequences or []) if s]
        stop_lps = [build_kmp_lps(s) for s in stop_sequences]
        stop_match = [0] * len(stop_sequences)

        last_response = None
        try:
            cancel_check_every = max(1, int(os.environ.get("DISTRIBUTED_CANCEL_CHECK_EVERY", "1")))
            cancel_step = 0
            for response in stream_generate(
                model, tokenizer, prompt, 
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                last_response = response
                cache_key.append(response.token)
                if rank == 0 and response_queue is not None and first_token_dt is None:
                    first_token_dt = time.perf_counter() - gen_start_t
                    try:
                        prompt_tps = float(getattr(response, "prompt_tps", 0.0) or 0.0)
                    except Exception:
                        prompt_tps = 0.0
                    logging.info(
                        "First token: dt=%.3fs prompt_suffix=%d full_prompt_len=%d prompt_tps=%.3f",
                        first_token_dt,
                        int(getattr(response, "prompt_tokens", 0) or 0),
                        full_prompt_len,
                        prompt_tps,
                    )
                holdback = 0
                stop_trim = 0
                if stop_sequences:
                    tok = int(response.token)
                    for i, seq in enumerate(stop_sequences):
                        l = stop_match[i]
                        while l > 0 and seq[l] != tok:
                            l = stop_lps[i][l - 1]
                        if l < len(seq) and seq[l] == tok:
                            l += 1
                        if l > holdback:
                            holdback = l
                        if l == len(seq) and len(seq) > stop_trim:
                            stop_trim = len(seq)
                        stop_match[i] = l

                item = None
                if rank == 0 and response_queue is not None:
                    item = {
                        "text": response.text,
                        "finish_reason": response.finish_reason,
                        # stream_generate reports prompt_tokens as the length of the prompt
                        # it was called with (which may be only the non-cached suffix).
                        # For OpenAI usage, report the full prompt length.
                        "prompt_tokens": full_prompt_len,
                        "generation_tokens": response.generation_tokens,
                        "token": response.token,
                    }
                    pending_items.append(item)

                # Stop early if we hit a stop sequence (discard the stop sequence tokens).
                if stop_trim > 0:
                    if pending_items is not None:
                        for _ in range(min(stop_trim, len(pending_items))):
                            pending_items.pop()
                    break

                # Flush buffered items except the current stop-prefix holdback.
                if pending_items is not None and holdback >= 0:
                    flush_count = len(pending_items) - holdback
                    for _ in range(max(0, flush_count)):
                        response_queue.put(pending_items.popleft())

                cancel_step += 1
                if cancel_step % cancel_check_every == 0 and dist_state.sync_should_cancel(request_id):
                    if rank == 0:
                        logging.info(
                            "Cancel requested (id=%s) at gen_tokens=%d",
                            request_id,
                            int(getattr(response, "generation_tokens", 0) or 0),
                        )
                    break

            # Flush anything left (e.g. overlap prefixes when generation ended).
            if pending_items is not None:
                while pending_items:
                    response_queue.put(pending_items.popleft())

            if rank == 0 and response_queue is not None:
                response_queue.put(None)

            if rank == 0:
                total_dt = time.perf_counter() - gen_start_t
                gen_tokens = int(getattr(last_response, "generation_tokens", 0) or 0) if last_response is not None else 0
                logging.info(
                    "Generation finished: gen_tokens=%d total_dt=%.3fs first_token_dt=%.3fs",
                    gen_tokens,
                    total_dt,
                    first_token_dt if first_token_dt is not None else 0.0,
                )

            # Save full cache (prompt + generated tokens). This maximizes prompt-cache
            # reuse for multi-turn chats where the client includes the previous
            # assistant output in the next request.
            store_tokens = cache_key
            prompt_cache_store.insert_cache(args.model, store_tokens, prompt_cache)
            if rank == 0:
                try:
                    stored_len = int(cache_length(prompt_cache))
                except Exception:
                    stored_len = 0
                logging.info(
                    "Saved prompt cache: key_len=%d cache_len=%d",
                    len(store_tokens),
                    stored_len,
                )

            if rank == 0:
                dist_state.clear_request_canceled(request_id)

        except Exception as e:
            logging.exception(f"Generation error on rank {rank}")
            if rank == 0 and response_queue is not None:
                response_queue.put({"text": f"Error: {e}", "finish_reason": "error"})
                response_queue.put(None)
            if rank == 0:
                dist_state.clear_request_canceled(request_id)
        finally:
            # Ensure no leftover async work from `stream_generate` can interleave
            # with the next request's broadcast/cancel collectives.
            mx.synchronize()

__all__ = ["generation_loop"]
