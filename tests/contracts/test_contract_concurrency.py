from __future__ import annotations

from threading import Thread
from typing import Dict, List

import pytest

from tests.contracts._client import join_url, stream_openai_sse


pytestmark = pytest.mark.contract


def test_concurrent_openai_chat_stream_does_not_hang(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    body = {
        "model": model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
    }

    errors: List[BaseException] = []
    results: List[bool] = []

    def _worker() -> None:
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
            results.append(True)
        except BaseException as e:
            errors.append(e)

    threads = [Thread(target=_worker, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=240.0)

    assert not any(t.is_alive() for t in threads), "Timed out waiting for concurrent streams"
    if errors:
        raise AssertionError(f"Concurrent stream failures: {errors!r}")
    assert len(results) == len(threads)

