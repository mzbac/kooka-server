from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def http_json(
    method: str,
    url: str,
    body: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 60.0,
) -> Tuple[int, Dict[str, Any]]:
    data = None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"_raw": raw}
            return resp.status, payload
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        return e.code, payload


def get_default_model(base_url: str, headers: Dict[str, str], timeout_s: float) -> str:
    status, payload = http_json("GET", join_url(base_url, "/v1/models"), headers=headers, timeout_s=timeout_s)
    if status != 200:
        raise AssertionError(f"GET /v1/models expected 200, got {status}: {payload}")
    if payload.get("object") != "list":
        raise AssertionError(f"/v1/models: object != list: {payload}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise AssertionError(f"/v1/models: data not a non-empty list: {payload}")
    mid = (data[0] or {}).get("id")
    if not isinstance(mid, str) or not mid:
        raise AssertionError(f"/v1/models: first model missing id: {payload}")
    return mid


def _iter_sse_data_payloads(resp, max_seconds: float) -> Iterator[str]:
    start = time.time()
    buf = ""

    while True:
        if (time.time() - start) > max_seconds:
            raise TimeoutError(f"Timed out streaming after {max_seconds:.0f}s")

        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")

        # SSE event delimiter: blank line
        while True:
            idx = buf.find("\n\n")
            if idx < 0:
                break
            event = buf[:idx]
            buf = buf[idx + 2 :]
            if not event:
                continue
            lines = event.rstrip("\n").split("\n")
            data_line = next((l for l in lines if l.startswith("data: ")), None)
            if not data_line:
                continue
            payload = data_line[6:].strip()
            if not payload:
                continue
            if payload.startswith(":"):
                continue
            yield payload


def stream_openai_sse(
    url: str,
    body: dict,
    headers: Dict[str, str],
    timeout_s: float,
    max_seconds: float = 120.0,
) -> Tuple[List[dict], bool]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    chunks: List[dict] = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for payload in _iter_sse_data_payloads(resp, max_seconds=max_seconds):
            if payload == "[DONE]":
                saw_done = True
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                chunks.append(obj)
    return chunks, saw_done


def stream_anthropic_sse(
    url: str,
    body: dict,
    headers: Dict[str, str],
    timeout_s: float,
    max_seconds: float = 120.0,
) -> List[dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    events: List[dict] = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for payload in _iter_sse_data_payloads(resp, max_seconds=max_seconds):
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            events.append(obj)
            if obj.get("type") == "message_stop":
                break
    return events

