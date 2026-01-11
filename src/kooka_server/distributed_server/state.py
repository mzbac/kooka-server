from __future__ import annotations

import logging
import time
from queue import Empty, Queue
from threading import Lock
from typing import Any, Optional

import mlx.core as mx

from .constants import (
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_REPETITION_PENALTY,
    MAX_PROMPT_LENGTH,
    MAX_STOP_SEQUENCES,
    MAX_STOP_SEQUENCE_LENGTH,
)


class DistributedState:
    """Coordinates generation requests across distributed ranks."""

    def __init__(self, group: Any):
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.request_queue: Queue[dict] = Queue()  # Only used by rank 0
        self.lock = Lock()
        self.canceled_requests: set[str] = set()  # request_id strings (rank 0)

    def cancel_request(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        with self.lock:
            self.canceled_requests.add(str(request_id))

    def is_request_canceled(self, request_id: Optional[str]) -> bool:
        if not request_id:
            return False
        with self.lock:
            return str(request_id) in self.canceled_requests

    def clear_request_canceled(self, request_id: Optional[str]) -> None:
        if not request_id:
            return
        with self.lock:
            self.canceled_requests.discard(str(request_id))

    def sync_should_cancel(self, request_id: Optional[str]) -> bool:
        """
        Collective cancel check (all ranks must call).

        Rank 0 checks the local canceled set for request_id. The result is then
        synchronized to all ranks using an all_sum collective so that pipeline
        parallelism can stop cleanly without deadlocking other ranks.
        """
        local = 0
        if self.rank == 0 and request_id and self.is_request_canceled(request_id):
            local = 1

        if self.world_size == 1:
            return local > 0

        # Avoid interleaving control collectives (cancel/stop checks) with async
        # pipeline prefetch from `stream_generate`, which can otherwise deadlock
        # multi-rank runs.
        mx.synchronize()
        flag = mx.array([local], dtype=mx.int32)
        flag = mx.distributed.all_sum(flag, stream=mx.cpu)
        mx.eval(flag)
        return int(flag[0].item()) > 0

    def broadcast_request(self):
        """Broadcast request from rank 0 to all ranks.

        Returns (prompt_tokens, max_tokens, seed, temperature, top_p, top_k,
        seed_is_user, repetition_penalty, repetition_context_size,
        stop_token_sequences, response_queue, request) or (None, 0, 0, 0.0,
        0.0, 0, 0, 0.0, ...).
        """
        prompt_tokens = None
        max_tokens = 256
        seed = 0
        seed_is_user = 0
        temperature = 0.0
        top_p = 0.0
        top_k = 0
        repetition_penalty = DEFAULT_REPETITION_PENALTY
        repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE
        stop_token_sequences = []
        response_queue = None
        request = None

        # Rank 0 checks for new request
        if self.rank == 0:
            try:
                request = self.request_queue.get_nowait()
                prompt_tokens = request["prompt_tokens"]
                max_tokens = request["max_tokens"]
                seed_raw = request.get("seed")
                seed_is_user = 1 if request.get("seed_is_user") else 0
                if seed_raw is None:
                    seed = int(time.time_ns() & 0x7FFFFFFF)
                else:
                    seed = int(seed_raw)
                temperature = float(request.get("temperature") or 0.0)
                top_p = float(request.get("top_p") or 0.0)
                top_k = int(request.get("top_k") or 0)
                rp = request.get("repetition_penalty", repetition_penalty)
                repetition_penalty = float(repetition_penalty if rp is None else rp)
                rcs = request.get("repetition_context_size", repetition_context_size)
                repetition_context_size = int(repetition_context_size if rcs is None else rcs)
                stop_token_sequences = request.get("stop_token_sequences") or []
                response_queue = request["response_queue"]
                logging.info(
                    "Broadcasting request: prompt_len=%d, max_tokens=%d, stop_sequences=%d",
                    len(prompt_tokens) if prompt_tokens else 0,
                    int(max_tokens) if max_tokens is not None else 0,
                    len(stop_token_sequences or []),
                )
            except Empty:
                prompt_tokens = None

        # Broadcast metadata first so idle polling only does one collective.
        # Metadata: [length, max_tokens, seed, top_k, stop_count, repetition_context_size, seed_is_user]
        if self.rank == 0:
            length = len(prompt_tokens) if prompt_tokens else 0
            if length > MAX_PROMPT_LENGTH:
                length = MAX_PROMPT_LENGTH
            stop_count = min(len(stop_token_sequences or []), MAX_STOP_SEQUENCES)
            meta = mx.array(
                [length, max_tokens, seed, top_k, stop_count, repetition_context_size, seed_is_user],
                dtype=mx.int32,
            )
        else:
            meta = mx.zeros((7,), dtype=mx.int32)

        t0 = time.perf_counter()
        meta = mx.distributed.all_sum(meta, stream=mx.cpu)
        mx.eval(meta)
        meta_dt = time.perf_counter() - t0

        length = int(meta[0].item())
        max_tokens = int(meta[1].item())
        seed = int(meta[2].item())
        top_k = int(meta[3].item())
        stop_count = int(meta[4].item())
        repetition_context_size = int(meta[5].item())
        seed_is_user = int(meta[6].item())

        # Broadcast floats: [temperature, top_p, repetition_penalty]
        if length == 0:
            return (
                None,
                0,
                0,
                0.0,
                0.0,
                0,
                0,
                0.0,
                DEFAULT_REPETITION_CONTEXT_SIZE,
                [],
                None,
                None,
            )

        if self.rank == 0:
            meta_f = mx.array([temperature, top_p, repetition_penalty], dtype=mx.float32)
        else:
            meta_f = mx.zeros((3,), dtype=mx.float32)

        t1 = time.perf_counter()
        meta_f = mx.distributed.all_sum(meta_f, stream=mx.cpu)
        mx.eval(meta_f)
        meta_f_dt = time.perf_counter() - t1
        temperature = float(meta_f[0].item())
        top_p = float(meta_f[1].item())
        repetition_penalty = float(meta_f[2].item())

        # Broadcast actual prompt tokens
        if self.rank == 0:
            tokens = mx.array(prompt_tokens[:length], dtype=mx.int32)
        else:
            tokens = mx.zeros((length,), dtype=mx.int32)

        t2 = time.perf_counter()
        tokens = mx.distributed.all_sum(tokens, stream=mx.cpu)
        mx.eval(tokens)
        tokens_dt = time.perf_counter() - t2

        prompt = tokens.tolist()

        stop_token_sequences_out = []
        if stop_count > 0:
            # Broadcast stop token sequences (padded per-sequence)
            if self.rank == 0:
                lens = [0] * stop_count
                flat = [0] * (stop_count * MAX_STOP_SEQUENCE_LENGTH)
                for i, seq in enumerate((stop_token_sequences or [])[:stop_count]):
                    if not seq:
                        continue
                    seq = [int(x) for x in seq]
                    l = min(len(seq), MAX_STOP_SEQUENCE_LENGTH)
                    lens[i] = l
                    start = i * MAX_STOP_SEQUENCE_LENGTH
                    flat[start : start + l] = seq[:l]
                stop_lens = mx.array(lens, dtype=mx.int32)
                stop_tokens = mx.array(flat, dtype=mx.int32)
            else:
                stop_lens = mx.zeros((stop_count,), dtype=mx.int32)
                stop_tokens = mx.zeros((stop_count * MAX_STOP_SEQUENCE_LENGTH,), dtype=mx.int32)

            t3 = time.perf_counter()
            stop_lens = mx.distributed.all_sum(stop_lens, stream=mx.cpu)
            stop_tokens = mx.distributed.all_sum(stop_tokens, stream=mx.cpu)
            mx.eval(stop_lens, stop_tokens)
            stop_dt = time.perf_counter() - t3

            for i in range(stop_count):
                l = int(stop_lens[i].item())
                if l <= 0:
                    continue
                start = i * MAX_STOP_SEQUENCE_LENGTH
                stop_token_sequences_out.append(stop_tokens[start : start + l].tolist())

        if self.rank == 0:
            logging.info(
                "Broadcast timings: meta=%.3fs floats=%.3fs tokens=%.3fs stop=%.3fs (length=%d stop_count=%d)",
                meta_dt,
                meta_f_dt,
                tokens_dt,
                stop_dt if stop_count > 0 else 0.0,
                length,
                stop_count,
            )

        return (
            prompt,
            max_tokens,
            seed,
            temperature,
            top_p,
            top_k,
            seed_is_user,
            repetition_penalty,
            repetition_context_size,
            stop_token_sequences_out,
            response_queue,
            request,
        )


__all__ = ["DistributedState"]
