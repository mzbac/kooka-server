from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.contracts._client import http_json, join_url, stream_anthropic_sse


pytestmark = [pytest.mark.contract, pytest.mark.nemotron]


def _is_nemotron_model(model_id: str) -> bool:
    return "nemotron" in model_id.lower()


def _get_tool_schema() -> tuple[str, Dict[str, Any]]:
    tool_name = "echo"
    tool_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    return tool_name, tool_schema


def _check_anthropic_tool_use_stream(events: list[dict], expected_tool_name: str) -> None:
    saw_tool_use = any(
        (
            e.get("type") == "content_block_start"
            and (e.get("content_block") or {}).get("type") == "tool_use"
            and (e.get("content_block") or {}).get("name") == expected_tool_name
        )
        for e in events
        if isinstance(e, dict)
    )
    assert saw_tool_use, "anthropic tool use stream: missing tool_use content_block_start"

    saw_stop_tool_use = any(
        (e.get("type") == "message_delta" and (e.get("delta") or {}).get("stop_reason") == "tool_use")
        for e in events
        if isinstance(e, dict)
    )
    assert saw_stop_tool_use, "anthropic tool use stream: missing message_delta stop_reason=tool_use"


def test_anthropic_tool_use_non_stream_nemotron_prompt(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_nemotron_model(model_id):
        pytest.skip(f"Nemotron contract test skipped for model={model_id!r}")

    tool_name, tool_schema = _get_tool_schema()
    body = {
        "model": model_id,
        "stream": False,
        "max_tokens": 512,
        "system": [{"type": "text", "text": "You are a tool calling assistant. You MUST call the provided tool."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": 'Call the tool echo with input {"text":"OK"}. Return only the tool call.'}],
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


def test_anthropic_tool_use_stream_nemotron_prompt(
    ensure_server: Dict[str, str], base_url: str, headers: Dict[str, str], timeout_s: float, model_id: str
) -> None:
    if not _is_nemotron_model(model_id):
        pytest.skip(f"Nemotron contract test skipped for model={model_id!r}")

    tool_name, tool_schema = _get_tool_schema()
    body = {
        "model": model_id,
        "stream": True,
        "max_tokens": 512,
        "system": [{"type": "text", "text": "You are a tool calling assistant. You MUST call the provided tool."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": 'Call the tool echo with input {"text":"OK"}. Return only the tool call.'}],
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
    _check_anthropic_tool_use_stream(events, tool_name)
