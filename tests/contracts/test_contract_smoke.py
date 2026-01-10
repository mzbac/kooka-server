from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from tests.contracts._client import http_json, join_url, stream_anthropic_sse, stream_openai_sse


pytestmark = pytest.mark.contract


def _is_minimax_model(model_id: str) -> bool:
    mid = model_id.lower()
    return "minimax" in mid or "m2.1" in mid


def _check_openai_chat_response(payload: dict) -> None:
    assert isinstance(payload.get("id"), str), "chat: missing id"
    assert isinstance(payload.get("choices"), list) and payload["choices"], "chat: missing choices[0]"
    choice0 = payload["choices"][0]
    assert "finish_reason" in choice0, "chat: missing finish_reason"
    msg = choice0.get("message") or {}
    assert msg.get("role") == "assistant", "chat: message.role != assistant"
    # content may be empty/None when tool_calls happen; allow string/None.
    if "content" in msg:
        assert isinstance(msg["content"], (str, type(None))), "chat: message.content not str|null"
    if "tool_calls" in msg:
        assert isinstance(msg["tool_calls"], list), "chat: message.tool_calls not a list"
        for tc in msg["tool_calls"]:
            fn = (tc or {}).get("function") or {}
            assert isinstance(fn.get("name"), str) and fn["name"], "chat: tool_call missing function.name"
            assert isinstance(fn.get("arguments"), str), "chat: tool_call.function.arguments not string"


def _check_openai_tool_call_response(payload: dict, expected_tool_name: str) -> None:
    _check_openai_chat_response(payload)
    choice0 = payload["choices"][0]
    assert choice0.get("finish_reason") == "tool_calls", f"chat: finish_reason != tool_calls: {payload}"
    msg = choice0.get("message") or {}
    tool_calls = msg.get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls, f"chat: missing tool_calls: {payload}"
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        assert fn.get("name") == expected_tool_name, f"chat: unexpected tool name: {payload}"
        args_s = fn.get("arguments")
        assert isinstance(args_s, str), f"chat: tool_call.function.arguments not string: {payload}"
        try:
            args = json.loads(args_s) if args_s else {}
        except json.JSONDecodeError as e:
            raise AssertionError(f"chat: tool_call.function.arguments not valid JSON: {e}: {payload}") from e
        assert isinstance(args, dict), f"chat: tool_call.function.arguments not object: {payload}"
        assert isinstance(args.get("text"), str) and args["text"], f"chat: tool args missing text: {payload}"


def _check_openai_chat_stream(chunks: List[dict], saw_done: bool) -> None:
    assert saw_done, "stream: missing [DONE]"
    assert chunks, "stream: no json chunks received"
    saw_delta = False
    saw_usage = False
    for obj in chunks:
        if obj.get("usage"):
            saw_usage = True
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        if delta.get("content") is not None or delta.get("tool_calls") is not None:
            saw_delta = True
        tool_calls = delta.get("tool_calls") or []
        if tool_calls:
            assert isinstance(tool_calls, list), "stream: delta.tool_calls not list"
            for tc in tool_calls:
                fn = (tc or {}).get("function") or {}
                assert isinstance(fn.get("name"), str) and fn["name"], "stream: tool_call missing function.name"
                assert isinstance(fn.get("arguments"), str), "stream: tool_call.function.arguments not string"
    assert saw_delta, "stream: never observed delta.content or delta.tool_calls"
    assert saw_usage, "stream: never observed a usage chunk (expected include_usage=true)"


def _check_openai_tool_call_stream(chunks: List[dict], saw_done: bool, expected_tool_name: str) -> None:
    assert saw_done, "stream: missing [DONE]"
    assert chunks, "stream: no json chunks received"
    saw_tool_call = False
    saw_finish_tool_calls = False
    for obj in chunks:
        choices = obj.get("choices") or []
        if not choices:
            continue
        choice0 = choices[0] or {}
        if choice0.get("finish_reason") == "tool_calls":
            saw_finish_tool_calls = True
        delta = choice0.get("delta") or {}
        tool_calls = delta.get("tool_calls") or []
        if not tool_calls:
            continue
        assert isinstance(tool_calls, list), f"stream: delta.tool_calls not list: {obj}"
        saw_tool_call = True
        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            assert fn.get("name") == expected_tool_name, f"stream: unexpected tool name: {obj}"
            args_s = fn.get("arguments")
            assert isinstance(args_s, str), f"stream: tool_call.function.arguments not string: {obj}"
            # Arguments may arrive as a full JSON string; ensure it's parseable.
            try:
                json.loads(args_s) if args_s else {}
            except json.JSONDecodeError as e:
                raise AssertionError(f"stream: tool_call.function.arguments not valid JSON: {e}: {obj}") from e

    assert saw_tool_call, "stream: never observed delta.tool_calls (expected tool call)"
    assert saw_finish_tool_calls, "stream: never observed finish_reason=tool_calls"


def _check_anthropic_message_response(payload: dict) -> None:
    assert isinstance(payload.get("id"), str) and payload["id"], "messages: missing id"
    assert payload.get("type") == "message", f"messages: type != message: {payload}"
    assert payload.get("role") == "assistant", f"messages: role != assistant: {payload}"

    content = payload.get("content")
    assert isinstance(content, list), f"messages: content not list: {payload}"
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            assert isinstance(block.get("text"), str), f"messages: text block missing text: {payload}"
        if btype == "tool_use":
            assert isinstance(block.get("id"), str), f"messages: tool_use missing id: {payload}"
            assert isinstance(block.get("name"), str), f"messages: tool_use missing name: {payload}"

    stop_reason = payload.get("stop_reason")
    assert isinstance(stop_reason, (str, type(None))), f"messages: stop_reason not str|null: {payload}"

    usage = payload.get("usage")
    if usage is None:
        return
    assert isinstance(usage, dict), f"messages: usage not object: {payload}"
    assert isinstance(usage.get("input_tokens"), int), f"messages: usage.input_tokens not int: {payload}"
    assert isinstance(usage.get("output_tokens"), int), f"messages: usage.output_tokens not int: {payload}"


def _check_anthropic_tool_use_response(payload: dict, expected_tool_name: str) -> None:
    _check_anthropic_message_response(payload)
    content = payload.get("content") or []
    tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
    assert tool_blocks, f"messages: missing tool_use blocks: {payload}"
    for b in tool_blocks:
        assert b.get("name") == expected_tool_name, f"messages: unexpected tool name: {payload}"
        inp = b.get("input")
        assert isinstance(inp, dict), f"messages: tool_use.input not object: {payload}"
        assert isinstance(inp.get("text"), str) and inp["text"], f"messages: tool_use.input missing text: {payload}"


def _check_anthropic_stream(events: List[dict]) -> None:
    assert events, "anthropic stream: no json events received"

    saw_start = any((e.get("type") == "message_start") for e in events if isinstance(e, dict))
    saw_stop = any((e.get("type") == "message_stop") for e in events if isinstance(e, dict))
    assert saw_start, "anthropic stream: missing message_start"
    assert saw_stop, "anthropic stream: missing message_stop"

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

    saw_text_delta = any(
        (e.get("type") == "content_block_delta" and (e.get("delta") or {}).get("type") == "text_delta")
        for e in events
        if isinstance(e, dict)
    )
    saw_tool_use = any(
        (e.get("type") == "content_block_start" and (e.get("content_block") or {}).get("type") == "tool_use")
        for e in events
        if isinstance(e, dict)
    )
    assert saw_text_delta or saw_tool_use, "anthropic stream: never observed text_delta or tool_use"


def _check_anthropic_tool_use_stream(events: List[dict], expected_tool_name: str) -> None:
    _check_anthropic_stream(events)
    saw_tool_use = any(
        (
            e.get("type") == "content_block_start"
            and (e.get("content_block") or {}).get("type") == "tool_use"
            and (e.get("content_block") or {}).get("name") == expected_tool_name
        )
        for e in events
        if isinstance(e, dict)
    )
    assert saw_tool_use, "anthropic stream: missing tool_use content_block_start"


def _get_tool_schema() -> Tuple[str, Dict[str, Any]]:
    tool_name = "echo"
    tool_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    return tool_name, tool_schema


def test_v1_models_shape(ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float) -> None:
    status, payload = http_json("GET", join_url(base_url, "/v1/models"), headers=headers, timeout_s=timeout_s)
    assert status == 200, f"GET /v1/models expected 200, got {status}: {payload}"
    assert payload.get("object") == "list", f"/v1/models: object != list: {payload}"
    data = payload.get("data")
    assert isinstance(data, list) and data, f"/v1/models: data not a non-empty list: {payload}"
    mid = (data[0] or {}).get("id")
    assert isinstance(mid, str) and mid, f"/v1/models: first model missing id: {payload}"


def test_openai_chat_completions_non_stream(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    chat_body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Say OK."}],
    }
    status, payload = http_json(
        "POST", join_url(base_url, "/chat/completions"), body=chat_body, headers=headers, timeout_s=timeout_s
    )
    assert status == 200, f"POST /chat/completions expected 200, got {status}: {payload}"
    _check_openai_chat_response(payload)


def test_openai_chat_completions_stream(
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
    _check_openai_chat_stream(chunks, saw_done)


def test_openai_tool_call_non_stream(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    tool_name, tool_schema = _get_tool_schema()
    tool_body = {
        "model": model_id,
        "stream": False,
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
    status, payload = http_json(
        "POST", join_url(base_url, "/chat/completions"), body=tool_body, headers=headers, timeout_s=timeout_s
    )
    assert status == 200, f"POST /chat/completions (tool call) expected 200, got {status}: {payload}"
    _check_openai_tool_call_response(payload, tool_name)


def test_openai_tool_call_stream(
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
    _check_openai_tool_call_stream(chunks, saw_done, tool_name)


def test_anthropic_messages_non_stream_beta_query_string(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    msg_body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say OK."}]}],
    }
    status, payload = http_json(
        "POST", join_url(base_url, "/v1/messages?beta=true"), body=msg_body, headers=headers, timeout_s=timeout_s
    )
    assert status == 200, f"POST /v1/messages expected 200, got {status}: {payload}"
    _check_anthropic_message_response(payload)


def test_anthropic_messages_stream_beta_query_string(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    msg_stream_body = {
        "model": model_id,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say OK."}]}],
    }
    events = stream_anthropic_sse(
        join_url(base_url, "/v1/messages?beta=true"),
        msg_stream_body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=120.0,
    )
    _check_anthropic_stream(events)


@pytest.mark.minimax
def test_anthropic_tool_use_non_stream_minimax_contract(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_minimax_model(model_id):
        pytest.skip(f"MiniMax contract test skipped for model={model_id!r}")
    tool_name, tool_schema = _get_tool_schema()
    msg_tool_body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 256,
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
        "POST", join_url(base_url, "/v1/messages?beta=true"), body=msg_tool_body, headers=headers, timeout_s=timeout_s
    )
    assert status == 200, f"POST /v1/messages (tool use) expected 200, got {status}: {payload}"
    _check_anthropic_tool_use_response(payload, tool_name)


@pytest.mark.minimax
def test_anthropic_tool_use_stream_minimax_contract(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_minimax_model(model_id):
        pytest.skip(f"MiniMax contract test skipped for model={model_id!r}")
    tool_name, tool_schema = _get_tool_schema()
    msg_tool_stream_body = {
        "model": model_id,
        "stream": True,
        "max_tokens": 256,
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
        msg_tool_stream_body,
        headers=headers,
        timeout_s=timeout_s,
        max_seconds=180.0,
    )
    _check_anthropic_tool_use_stream(events, tool_name)
