from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from tests.contracts._client import http_json, join_url, stream_anthropic_sse, stream_openai_sse


pytestmark = pytest.mark.contract


def _is_minimax_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "minimax" in mid or "m2.1" in mid


def _get_tool_schema() -> Tuple[str, Dict[str, Any]]:
    tool_name = "echo"
    tool_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    return tool_name, tool_schema


def _opencode_check_openai_stream(chunks: List[dict], saw_done: bool) -> Tuple[bool, bool]:
    assert saw_done, "missing [DONE]"
    assert chunks, "no json chunks received"

    saw_delta = False
    saw_usage = False
    saw_tool_call = False
    saw_finish_tool_calls = False

    for obj in chunks:
        if obj.get("usage"):
            saw_usage = True

        choices = obj.get("choices") or []
        if not choices:
            continue

        choice0 = choices[0] or {}
        if choice0.get("finish_reason") == "tool_calls":
            saw_finish_tool_calls = True

        delta = choice0.get("delta") or {}
        if delta.get("content") is not None or delta.get("tool_calls") is not None:
            saw_delta = True

        tool_calls = delta.get("tool_calls") or []
        if tool_calls:
            assert isinstance(tool_calls, list), "delta.tool_calls not list"
            for tc in tool_calls:
                fn = (tc or {}).get("function") or {}
                assert isinstance(fn.get("name"), str) and fn["name"], "tool_call missing function.name"
                assert isinstance(fn.get("arguments"), str), "tool_call.function.arguments not string"
            saw_tool_call = True

    assert saw_delta, "never observed delta.content or delta.tool_calls"
    assert saw_usage, "never observed a usage chunk (expected include_usage=true)"

    return saw_tool_call, saw_finish_tool_calls


def _opencode_check_anthropic_stream(events: List[dict]) -> Tuple[bool, bool]:
    assert events, "anthropic stream: no json events received"

    saw_start = any((e.get("type") == "message_start") for e in events if isinstance(e, dict))
    saw_stop = any((e.get("type") == "message_stop") for e in events if isinstance(e, dict))
    assert saw_start, "anthropic stream: missing message_start"
    assert saw_stop, "anthropic stream: missing message_stop"

    saw_text_delta_or_tool = any(
        (e.get("type") == "content_block_delta" and (e.get("delta") or {}).get("type") == "text_delta")
        for e in events
        if isinstance(e, dict)
    )
    saw_tool_use = any(
        (e.get("type") == "content_block_start" and (e.get("content_block") or {}).get("type") == "tool_use")
        for e in events
        if isinstance(e, dict)
    )
    assert saw_text_delta_or_tool or saw_tool_use, "anthropic stream: never observed text_delta or tool_use"

    saw_usage = False
    for e in events:
        if not isinstance(e, dict):
            continue
        if isinstance(e.get("usage"), dict):
            saw_usage = True
        msg = e.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("usage"), dict):
            saw_usage = True
    assert saw_usage, "anthropic stream: never observed usage"

    saw_stop_tool_use = any(
        (e.get("type") == "message_delta" and (e.get("delta") or {}).get("stop_reason") == "tool_use")
        for e in events
        if isinstance(e, dict)
    )

    return saw_tool_use, saw_stop_tool_use


def test_opencode_openai_stream_shape(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    stream_body = {
        "model": model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
    }
    chunks, saw_done = stream_openai_sse(
        join_url(base_url, "/chat/completions"),
        stream_body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=120.0,
    )
    _opencode_check_openai_stream(chunks, saw_done)


def test_opencode_openai_tool_call_stream(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    tool_name, tool_schema = _get_tool_schema()
    tool_body = {
        "model": model_id,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "You are a tool calling assistant. You MUST call the provided tool."},
            {"role": "user", "content": 'Call the tool echo with arguments {"text":"OK"}. Return only the tool call.'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": tool_name, "description": "Echo the provided text.", "parameters": tool_schema},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
    }
    chunks, saw_done = stream_openai_sse(
        join_url(base_url, "/chat/completions"),
        tool_body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=180.0,
    )
    saw_tool_call, saw_finish_tool_calls = _opencode_check_openai_stream(chunks, saw_done)
    assert saw_tool_call, "tool call stream: never observed delta.tool_calls"
    assert saw_finish_tool_calls, "tool call stream: never observed finish_reason=tool_calls"


def test_opencode_anthropic_messages_stream(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    body = {
        "model": model_id,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say OK."}]}],
    }
    events = stream_anthropic_sse(
        join_url(base_url, "/v1/messages?beta=true"),
        body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=120.0,
    )
    _opencode_check_anthropic_stream(events)


@pytest.mark.minimax
def test_opencode_anthropic_tool_use_stream_minimax_contract(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_minimax_model(model_id):
        pytest.skip(f"MiniMax contract test skipped for model={model_id!r}")

    tool_name, tool_schema = _get_tool_schema()
    body = {
        "model": model_id,
        "stream": True,
        "max_tokens": 128,
        "system": [{"type": "text", "text": "You are a tool calling assistant. You MUST call the provided tool."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Output exactly this and nothing else: <invoke name="echo"><parameter name="text">OK</parameter></invoke>',
                    }
                ],
            }
        ],
        "tools": [{"name": tool_name, "description": "Echo the provided text.", "input_schema": tool_schema}],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    events = stream_anthropic_sse(
        join_url(base_url, "/v1/messages?beta=true"),
        body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=180.0,
    )
    saw_tool_use, saw_stop_tool_use = _opencode_check_anthropic_stream(events)
    assert saw_tool_use, "anthropic tool use stream: missing tool_use content_block_start"
    assert saw_stop_tool_use, "anthropic tool use stream: missing message_delta stop_reason=tool_use"


@pytest.mark.minimax
def test_opencode_anthropic_tool_use_non_stream_minimax_contract(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_minimax_model(model_id):
        pytest.skip(f"MiniMax contract test skipped for model={model_id!r}")

    tool_name, tool_schema = _get_tool_schema()
    body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 128,
        "system": [{"type": "text", "text": "You are a tool calling assistant. You MUST call the provided tool."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Output exactly this and nothing else: <invoke name="echo"><parameter name="text">OK</parameter></invoke>',
                    }
                ],
            }
        ],
        "tools": [{"name": tool_name, "description": "Echo the provided text.", "input_schema": tool_schema}],
        "tool_choice": {"type": "tool", "name": tool_name},
    }
    status, payload = http_json(
        "POST", join_url(base_url, "/v1/messages?beta=true"), body=body, headers=headers, timeout_s=timeout_s
    )
    assert status == 200, f"POST /v1/messages (tool use) expected 200, got {status}: {payload}"
    content = payload.get("content") or []
    assert isinstance(content, list), f"messages: content not list: {payload}"
    tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    assert tool_blocks, f"messages: missing tool_use blocks: {payload}"
    assert payload.get("stop_reason") == "tool_use", f"messages: stop_reason != tool_use: {payload}"
    for b in tool_blocks:
        assert b.get("name") == tool_name, f"messages: unexpected tool name: {payload}"
        inp = b.get("input")
        assert isinstance(inp, dict), f"messages: tool_use.input not object: {payload}"
        assert isinstance(inp.get("text"), str) and inp["text"], f"messages: tool_use.input missing text: {payload}"
