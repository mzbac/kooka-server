from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
from queue import Queue
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import (
    ArraysCache,
    can_trim_prompt_cache,
    KVCache,
    make_prompt_cache,
    MambaCache,
    RotatingKVCache,
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


@dataclass(frozen=True)
class _PendingRequest:
    prompt_tokens: List[int]
    max_tokens: int
    seed: int
    seed_is_user: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    repetition_context_size: int
    stop_token_sequences: List[List[int]]
    request_id: Optional[str]
    response_queue: Optional[Queue]

    @property
    def batchable(self) -> bool:
        return not self.seed_is_user


@dataclass
class _ActiveRequest:
    prompt_len: int
    cache_key: List[int]
    detokenizer: Any
    stop_sequences: List[List[int]]
    stop_lps: List[List[int]]
    stop_match: List[int]
    pending_items: Optional[Deque[dict]]
    generation_tokens: int
    request_id: Optional[str]
    response_queue: Optional[Queue]


def _is_model_batchable_for_distributed(model: Any) -> bool:
    try:
        cache_types = {type(c) for c in make_prompt_cache(model)}
    except Exception:
        return False
    return cache_types.issubset({KVCache, RotatingKVCache, ArraysCache, MambaCache})


def _prepare_prompt_cache_and_suffix(
    *,
    model: Any,
    prompt_cache_store: LRUPromptCache,
    model_key: str,
    prompt_tokens: List[int],
    rank: int,
) -> Tuple[Optional[List[Any]], List[int]]:
    cached_prompt_cache, tokens_to_process = prompt_cache_store.fetch_nearest_cache(
        model_key, prompt_tokens
    )

    if not tokens_to_process:
        if not prompt_tokens:
            return None, []

        prompt_cache = cached_prompt_cache
        if prompt_cache is not None and can_trim_prompt_cache(prompt_cache):
            try:
                trim_prompt_cache(prompt_cache, 1)
                tokens_to_process = [prompt_tokens[-1]]
                return prompt_cache, tokens_to_process
            except Exception:
                if rank == 0:
                    logging.debug(
                        "Failed to trim prompt cache for exact match; rebuilding cache.",
                        exc_info=True,
                    )
                return make_prompt_cache(model), prompt_tokens

        return make_prompt_cache(model), prompt_tokens

    return cached_prompt_cache, tokens_to_process


def _prime_empty_prompt_cache(
    *,
    model: Any,
    prompt_cache: List[Any],
    tokens_to_process: List[int],
) -> List[int]:
    if not tokens_to_process:
        return tokens_to_process

    needs_priming = False
    for cache_obj in prompt_cache:
        if hasattr(cache_obj, "keys") and getattr(cache_obj, "keys", None) is None:
            needs_priming = True
            break

    if not needs_priming:
        return tokens_to_process

    first_token = int(tokens_to_process[0])
    inp = mx.array([first_token], dtype=mx.int32)[None]
    out = model(inp, cache=prompt_cache)
    try:
        mx.eval(out)
    except Exception:
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            raise

    mx.synchronize()
    return tokens_to_process[1:]


def _sync_canceled_uids(dist_state: Any, active: Dict[int, _ActiveRequest], *, rank: int) -> List[int]:
    if rank == 0:
        local = [
            uid
            for uid, state in active.items()
            if state.request_id and dist_state.is_request_canceled(state.request_id)
        ]
    else:
        local = []

    if getattr(dist_state, "world_size", 1) == 1:
        return [int(x) for x in local]

    if rank == 0:
        meta = mx.array([len(local)], dtype=mx.int32)
    else:
        meta = mx.zeros((1,), dtype=mx.int32)
    meta = mx.distributed.all_sum(meta, stream=mx.cpu)
    mx.eval(meta)

    count = int(meta[0].item())
    if count <= 0:
        return []

    if rank == 0:
        payload = mx.array(local, dtype=mx.int32)
    else:
        payload = mx.zeros((count,), dtype=mx.int32)
    payload = mx.distributed.all_sum(payload, stream=mx.cpu)
    mx.eval(payload)

    return [int(x) for x in payload.tolist()[:count]]


def _finalize_active_request(
    *,
    dist_state: Any,
    state: _ActiveRequest,
    rank: int,
    model_key: str,
    prompt_cache_store: LRUPromptCache,
    prompt_cache: Optional[List[Any]],
) -> None:
    if prompt_cache is not None:
        prompt_cache_store.insert_cache(model_key, state.cache_key, prompt_cache)

    if rank == 0 and state.response_queue is not None:
        if state.pending_items is not None:
            while state.pending_items:
                state.response_queue.put(state.pending_items.popleft())

        state.response_queue.put(None)

        if state.request_id:
            dist_state.clear_request_canceled(state.request_id)


def _serve_one_request_sequential(
    *,
    dist_state: Any,
    model: Any,
    tokenizer: Any,
    args: Any,
    prompt_cache_store: LRUPromptCache,
    prompt_tokens: List[int],
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_context_size: int,
    stop_token_sequences: List[List[int]],
    response_queue: Optional[Queue],
    request_id: Optional[str],
) -> None:
    rank = dist_state.rank

    # If the client disconnected before generation started, skip work.
    if dist_state.sync_should_cancel(request_id):
        if rank == 0:
            logging.info("Request canceled before start (id=%s)", request_id)
            if response_queue is not None:
                response_queue.put(None)
            dist_state.clear_request_canceled(request_id)
        return

    mx.random.seed(int(seed))

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

    full_prompt_len = len(prompt_tokens)

    cached_prompt_cache, tokens_to_process = _prepare_prompt_cache_and_suffix(
        model=model,
        prompt_cache_store=prompt_cache_store,
        model_key=args.model,
        prompt_tokens=prompt_tokens,
        rank=rank,
    )
    prompt_cache = cached_prompt_cache
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(model)

    if not tokens_to_process:
        if rank == 0 and response_queue is not None:
            response_queue.put(
                {
                    "text": "Error: empty prompt_tokens (cannot generate).",
                    "finish_reason": "error",
                }
            )
            response_queue.put(None)
        return

    if rank == 0:
        cache_hit = cached_prompt_cache is not None
        reused_len = max(0, full_prompt_len - len(tokens_to_process))
        logging.info(
            "Starting generation: prompt_len=%d cache_hit=%s reused_len=%d to_process_len=%d max_tokens=%s",
            full_prompt_len,
            cache_hit,
            reused_len,
            len(tokens_to_process),
            str(max_tokens),
        )
        gen_start_t = time.perf_counter()
        first_token_dt = None
    else:
        gen_start_t = None
        first_token_dt = None

    prompt = mx.array(tokens_to_process, dtype=mx.int32)
    cache_key = prompt_tokens[:]

    # Stop-sequence buffering (rank 0 only) to avoid emitting partial stop strings.
    pending_items = deque() if rank == 0 and response_queue is not None else None

    stop_sequences = [s for s in (stop_token_sequences or []) if s]
    stop_lps = [build_kmp_lps(s) for s in stop_sequences]
    stop_match = [0] * len(stop_sequences)

    last_response = None
    try:
        cancel_check_every = max(
            1, int(os.environ.get("DISTRIBUTED_CANCEL_CHECK_EVERY", "8"))
        )
        cancel_step = 0
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            last_response = response
            cache_key.append(int(response.token))
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
            if (
                cancel_step % cancel_check_every == 0
                and dist_state.sync_should_cancel(request_id)
            ):
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
            gen_tokens = (
                int(getattr(last_response, "generation_tokens", 0) or 0)
                if last_response is not None
                else 0
            )
            logging.info(
                "Generation finished: gen_tokens=%d total_dt=%.3fs first_token_dt=%.3fs",
                gen_tokens,
                total_dt,
                first_token_dt if first_token_dt is not None else 0.0,
            )

        # Save full cache (prompt + generated tokens).
        prompt_cache_store.insert_cache(args.model, cache_key, prompt_cache)
        if rank == 0:
            logging.info(
                "Saved prompt cache: key_len=%d cache_items=%d",
                len(cache_key),
                len(prompt_cache) if prompt_cache is not None else 0,
            )

        if rank == 0:
            dist_state.clear_request_canceled(request_id)

    except Exception as e:
        logging.exception("Generation error on rank %d", rank)
        if rank == 0 and response_queue is not None:
            response_queue.put({"text": f"Error: {e}", "finish_reason": "error"})
            response_queue.put(None)
        if rank == 0:
            dist_state.clear_request_canceled(request_id)
    finally:
        # Ensure no leftover async work from `stream_generate` can interleave
        # with the next request's broadcast/cancel collectives.
        mx.synchronize()


def _generation_loop_batched(
    dist_state: Any,
    model: Any,
    tokenizer: Any,
    args: Any,
    prompt_cache_store: LRUPromptCache,
) -> None:
    rank = dist_state.rank
    mx.synchronize()
    if rank == 0:
        init_seed = int(time.time_ns() & 0x7FFFFFFF)
        seed_arr = mx.array([init_seed], dtype=mx.int32)
    else:
        init_seed = 0
        seed_arr = mx.zeros((1,), dtype=mx.int32)
    seed_arr = mx.distributed.all_sum(seed_arr, stream=mx.cpu)
    mx.eval(seed_arr)
    mx.synchronize()
    init_seed = int(seed_arr[0].item())
    mx.random.seed(init_seed)

    max_inflight = max(2, int(getattr(args, "batch_max_inflight", 4)))
    prefill_batch_size = max(1, int(getattr(args, "batch_prefill_batch_size", 2)))
    prefill_batch_size = min(prefill_batch_size, max_inflight)
    prefill_step_size = max(1, int(getattr(args, "batch_prefill_step_size", 2048)))
    steps_per_tick = max(1, int(getattr(args, "batch_steps_per_tick", 1)))
    batch_wait_ms = max(0, int(getattr(args, "batch_wait_ms", 0)))
    wait_steps = max(0, (batch_wait_ms + 4) // 5)

    batch_generator = BatchGenerator(
        model,
        stop_tokens=getattr(tokenizer, "eos_token_ids", set()),
        completion_batch_size=max_inflight,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
    )

    if rank == 0:
        logging.info(
            "Distributed batching enabled: max_inflight=%d prefill_batch_size=%d prefill_step_size=%d steps_per_tick=%d init_seed=%d",
            max_inflight,
            prefill_batch_size,
            prefill_step_size,
            steps_per_tick,
            init_seed,
        )

    pending: Deque[_PendingRequest] = deque()
    active: Dict[int, _ActiveRequest] = {}

    cancel_check_every = max(
        1, int(os.environ.get("DISTRIBUTED_CANCEL_CHECK_EVERY", "8"))
    )
    tick = 0

    while True:
        # Ensure pipeline comms from the previous decode step finish before
        # running any control collectives (broadcast/cancel).
        mx.synchronize()

        if active and tick % cancel_check_every == 0:
            canceled_uids = _sync_canceled_uids(dist_state, active, rank=rank)
            if canceled_uids:
                if rank == 0:
                    logging.info("Canceling %d active requests", len(canceled_uids))
                batch_generator.remove(canceled_uids)
                for uid in canceled_uids:
                    state = active.pop(uid, None)
                    if state is None:
                        continue
                    _finalize_active_request(
                        dist_state=dist_state,
                        state=state,
                        rank=rank,
                        model_key=args.model,
                        prompt_cache_store=prompt_cache_store,
                        prompt_cache=None,
                    )

        # Ingest new requests (bounded) so we don't grow detokenizers/queues without bound.
        while len(active) + len(pending) < max_inflight:
            (
                prompt_tokens,
                max_tokens,
                seed,
                temperature,
                top_p,
                top_k,
                seed_is_user,
                repetition_penalty,
                repetition_context_size,
                stop_token_sequences,
                response_queue,
                request,
            ) = dist_state.broadcast_request()

            if prompt_tokens is None:
                break

            request_id = None
            if rank == 0 and isinstance(request, dict):
                request_id = request.get("request_id")

            pending.append(
                _PendingRequest(
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    seed=seed,
                    seed_is_user=bool(seed_is_user),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                    stop_token_sequences=stop_token_sequences or [],
                    request_id=request_id,
                    response_queue=response_queue,
                )
            )

            # Optional "gather" window when starting from an empty batch.
            if not active and wait_steps > 0 and len(active) + len(pending) < max_inflight:
                for _ in range(wait_steps):
                    if len(active) + len(pending) >= max_inflight:
                        break
                    (
                        prompt_tokens2,
                        max_tokens2,
                        seed2,
                        temperature2,
                        top_p2,
                        top_k2,
                        seed_is_user2,
                        repetition_penalty2,
                        repetition_context_size2,
                        stop_token_sequences2,
                        response_queue2,
                        request2,
                    ) = dist_state.broadcast_request()
                    if prompt_tokens2 is None:
                        time.sleep(0.005)
                        continue
                    request_id2 = None
                    if rank == 0 and isinstance(request2, dict):
                        request_id2 = request2.get("request_id")
                    pending.append(
                        _PendingRequest(
                            prompt_tokens=prompt_tokens2,
                            max_tokens=max_tokens2,
                            seed=seed2,
                            seed_is_user=bool(seed_is_user2),
                            temperature=temperature2,
                            top_p=top_p2,
                            top_k=top_k2,
                            repetition_penalty=repetition_penalty2,
                            repetition_context_size=repetition_context_size2,
                            stop_token_sequences=stop_token_sequences2 or [],
                            request_id=request_id2,
                            response_queue=response_queue2,
                        )
                    )

        drain_batch = bool(active) and bool(pending) and not pending[0].batchable

        # If the next request is not batchable, serve it sequentially once the
        # active batch drains.
        if not active and pending and not pending[0].batchable:
            req = pending.popleft()
            if rank == 0:
                logging.info(
                    "Serving request sequentially (seed specified): max_tokens=%d temperature=%.3f top_p=%.3f top_k=%d",
                    int(req.max_tokens),
                    float(req.temperature),
                    float(req.top_p),
                    int(req.top_k),
                )
            _serve_one_request_sequential(
                dist_state=dist_state,
                model=model,
                tokenizer=tokenizer,
                args=args,
                prompt_cache_store=prompt_cache_store,
                prompt_tokens=req.prompt_tokens,
                max_tokens=req.max_tokens,
                seed=req.seed,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                repetition_context_size=req.repetition_context_size,
                stop_token_sequences=req.stop_token_sequences,
                response_queue=req.response_queue,
                request_id=req.request_id,
            )
            tick += 1
            continue

        if not drain_batch:
            while pending and pending[0].batchable and len(active) < max_inflight:
                req = pending.popleft()

                if dist_state.sync_should_cancel(req.request_id):
                    if rank == 0 and req.request_id:
                        logging.info("Request canceled before start (id=%s)", req.request_id)
                        if req.response_queue is not None:
                            req.response_queue.put(None)
                        dist_state.clear_request_canceled(req.request_id)
                    continue

                prompt_cache, tokens_to_process = _prepare_prompt_cache_and_suffix(
                    model=model,
                    prompt_cache_store=prompt_cache_store,
                    model_key=args.model,
                    prompt_tokens=req.prompt_tokens,
                    rank=rank,
                )

                if not tokens_to_process:
                    if rank == 0 and req.response_queue is not None:
                        req.response_queue.put(
                            {
                                "text": "Error: empty prompt_tokens (cannot generate).",
                                "finish_reason": "error",
                            }
                        )
                        req.response_queue.put(None)
                    continue

                if prompt_cache is None:
                    prompt_cache = make_prompt_cache(model)

                # BatchGenerator's prompt prefill path cannot merge a mix of empty
                # and non-empty KV caches. Prime empty caches with a single token
                # so all inserted prompts have mergeable history caches.
                tokens_to_process = _prime_empty_prompt_cache(
                    model=model,
                    prompt_cache=prompt_cache,
                    tokens_to_process=tokens_to_process,
                )

                sampler = make_sampler(
                    temp=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )
                processors: List[Any] = []
                if req.repetition_penalty and req.repetition_penalty != 0.0:
                    processors = make_logits_processors(
                        repetition_penalty=req.repetition_penalty,
                        repetition_context_size=req.repetition_context_size,
                    )

                (uid,) = batch_generator.insert(
                    [tokens_to_process],
                    req.max_tokens,
                    caches=[prompt_cache],
                    samplers=[sampler],
                    logits_processors=[processors],
                )

                detokenizer = getattr(tokenizer, "detokenizer", None)
                try:
                    reset = getattr(detokenizer, "reset", None)
                    if callable(reset):
                        reset()
                except Exception:
                    pass

                stop_sequences = [s for s in (req.stop_token_sequences or []) if s]
                stop_lps = [build_kmp_lps(s) for s in stop_sequences]
                stop_match = [0] * len(stop_sequences)

                pending_items = (
                    deque() if rank == 0 and req.response_queue is not None else None
                )

                active[uid] = _ActiveRequest(
                    prompt_len=len(req.prompt_tokens),
                    cache_key=req.prompt_tokens[:],
                    detokenizer=detokenizer,
                    stop_sequences=stop_sequences,
                    stop_lps=stop_lps,
                    stop_match=stop_match,
                    pending_items=pending_items,
                    generation_tokens=0,
                    request_id=req.request_id,
                    response_queue=req.response_queue,
                )

                if rank == 0:
                    logging.info(
                        "Batched request inserted: uid=%d prompt_len=%d max_tokens=%d temperature=%.3f top_p=%.3f top_k=%d",
                        int(uid),
                        len(req.prompt_tokens),
                        int(req.max_tokens),
                        float(req.temperature),
                        float(req.top_p),
                        int(req.top_k),
                    )

        if not active:
            time.sleep(0.005)
            tick += 1
            continue

        for _ in range(steps_per_tick):
            if not active:
                break

            responses = batch_generator.next()
            if not responses:
                break

            stop_uids: List[int] = []
            finished: List[Tuple[int, Optional[List[Any]]]] = []

            for r in responses:
                state = active.get(int(r.uid))
                if state is None:
                    continue

                token = int(r.token)
                state.cache_key.append(token)
                state.generation_tokens += 1

                if r.finish_reason != "stop":
                    try:
                        add_token = getattr(state.detokenizer, "add_token", None)
                        if callable(add_token):
                            add_token(token)
                    except Exception:
                        pass
                if r.finish_reason is not None:
                    try:
                        finalize = getattr(state.detokenizer, "finalize", None)
                        if callable(finalize):
                            finalize()
                    except Exception:
                        pass

                try:
                    segment = getattr(state.detokenizer, "last_segment", "")
                except Exception:
                    segment = ""
                if not isinstance(segment, str):
                    segment = ""

                holdback = 0
                stop_trim = 0
                if state.stop_sequences:
                    for i, seq in enumerate(state.stop_sequences):
                        l = state.stop_match[i]
                        while l > 0 and seq[l] != token:
                            l = state.stop_lps[i][l - 1]
                        if l < len(seq) and seq[l] == token:
                            l += 1
                        if l > holdback:
                            holdback = l
                        if l == len(seq) and len(seq) > stop_trim:
                            stop_trim = len(seq)
                        state.stop_match[i] = l

                if state.pending_items is not None and state.response_queue is not None:
                    state.pending_items.append(
                        {
                            "text": segment,
                            "finish_reason": r.finish_reason,
                            "prompt_tokens": state.prompt_len,
                            "generation_tokens": state.generation_tokens,
                            "token": token,
                        }
                    )

                    if stop_trim > 0:
                        for _ in range(min(stop_trim, len(state.pending_items))):
                            state.pending_items.pop()
                        stop_uids.append(int(r.uid))
                    else:
                        flush_count = len(state.pending_items) - holdback
                        for _ in range(max(0, flush_count)):
                            state.response_queue.put(state.pending_items.popleft())

                if r.finish_reason is not None:
                    finished.append((int(r.uid), getattr(r, "prompt_cache", None)))

            if stop_uids:
                batch = getattr(batch_generator, "active_batch", None)
                idx_by_uid: Dict[int, int] = {}
                if batch is not None:
                    idx_by_uid = {
                        int(uid): idx for idx, uid in enumerate(getattr(batch, "uids", []))
                    }
                stop_caches: Dict[int, Optional[List[Any]]] = {}
                for uid in stop_uids:
                    cache = None
                    if batch is not None:
                        idx = idx_by_uid.get(uid)
                        if idx is not None:
                            try:
                                cache = batch.extract_cache(idx)
                            except Exception:
                                cache = None
                    stop_caches[uid] = cache

                batch_generator.remove(stop_uids)
                for uid in stop_uids:
                    state = active.pop(uid, None)
                    if state is None:
                        continue
                    _finalize_active_request(
                        dist_state=dist_state,
                        state=state,
                        rank=rank,
                        model_key=args.model,
                        prompt_cache_store=prompt_cache_store,
                        prompt_cache=stop_caches.get(uid),
                    )

            for uid, prompt_cache in finished:
                state = active.pop(uid, None)
                if state is None:
                    continue
                _finalize_active_request(
                    dist_state=dist_state,
                    state=state,
                    rank=rank,
                    model_key=args.model,
                    prompt_cache_store=prompt_cache_store,
                    prompt_cache=prompt_cache,
                )

        tick += 1

def generation_loop(dist_state, model, tokenizer, args, prompt_cache_store=None):
    """Main generation loop running on ALL ranks.

    All ranks must execute this together for pipeline parallelism to work.
    """
    rank = dist_state.rank
    logging.info("Generation loop started on rank %d", rank)

    # Initialize prompt cache store if not provided.
    if prompt_cache_store is None:
        prompt_cache_store = LRUPromptCache(max_size=max(0, int(args.prompt_cache_size)))

    batch_enabled = bool(getattr(args, "batch", False)) and int(
        getattr(args, "batch_max_inflight", 0)
    ) > 1
    if batch_enabled:
        if _is_model_batchable_for_distributed(model):
            _generation_loop_batched(dist_state, model, tokenizer, args, prompt_cache_store)
            return
        if rank == 0:
            logging.warning(
                "Batching requested but model caches are not batchable; running sequential."
            )

    request_n = 0
    while True:
        (
            prompt_tokens,
            max_tokens,
            seed,
            temperature,
            top_p,
            top_k,
            seed_is_user,
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

        request_n += 1

        if rank == 0:
            logging.info(
                "Request params: req=%d seed=%d seed_is_user=%d max_tokens=%d temperature=%.3f top_p=%.3f top_k=%d repetition_penalty=%.3f repetition_context_size=%d stop_sequences=%d",
                request_n,
                int(seed),
                int(seed_is_user),
                int(max_tokens) if max_tokens is not None else 0,
                float(temperature),
                float(top_p),
                int(top_k),
                float(repetition_penalty),
                int(repetition_context_size),
                len(stop_token_sequences or []),
            )

        _serve_one_request_sequential(
            dist_state=dist_state,
            model=model,
            tokenizer=tokenizer,
            args=args,
            prompt_cache_store=prompt_cache_store,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            stop_token_sequences=stop_token_sequences or [],
            response_queue=response_queue,
            request_id=request_id,
        )

__all__ = ["generation_loop"]
