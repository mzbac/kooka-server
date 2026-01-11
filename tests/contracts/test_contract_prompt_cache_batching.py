from __future__ import annotations

from threading import Thread
from typing import Dict, List

import pytest

from tests.contracts._client import http_json, join_url, stream_openai_sse


pytestmark = pytest.mark.contract


def _warm_prompt_cache(
    *, base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str, prompt: str
) -> None:
    body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": prompt}],
    }
    status, payload = http_json(
        "POST",
        join_url(base_url, "/chat/completions"),
        body=body,
        headers=headers,
        timeout_s=timeout_s,
    )
    assert status == 200, f"warmup chat completion failed: {status} {payload}"


def test_batch_prefill_mixed_prompt_cache_does_not_crash(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    hit_prompt = "Say OK in one word."
    miss_prompt = "Say OK. (Different prompt to force a cache miss.)"

    _warm_prompt_cache(
        base_url=base_url,
        headers=headers,
        timeout_s=timeout_s,
        model_id=model_id,
        prompt=hit_prompt,
    )

    base_body = {
        "model": model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 16,
    }
    bodies = [
        {**base_body, "messages": [{"role": "user", "content": hit_prompt}]},
        {**base_body, "messages": [{"role": "user", "content": miss_prompt}]},
    ]

    errors: List[BaseException] = []

    def _worker(body: dict) -> None:
        try:
            chunks, saw_done = stream_openai_sse(
                join_url(base_url, "/chat/completions"),
                body,
                headers=headers,
                timeout_s=timeout_s,
                max_seconds=180.0,
            )
            assert saw_done
            assert chunks
        except BaseException as e:
            errors.append(e)

    threads = [Thread(target=_worker, args=(b,), daemon=True) for b in bodies]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=240.0)

    assert not any(t.is_alive() for t in threads), "Timed out waiting for concurrent streams"
    if errors:
        raise AssertionError(f"Concurrent batch requests failed: {errors!r}")

    status, payload = http_json(
        "GET",
        join_url(base_url, "/health"),
        headers=headers,
        timeout_s=timeout_s,
    )
    assert status == 200, f"Server unhealthy after concurrent batch requests: {status} {payload}"
    assert payload.get("status") == "ok"

