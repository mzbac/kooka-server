from __future__ import annotations

import json

import pytest

from kooka_server.api.anthropic.messages import convert_anthropic_to_openai_messages, process_message_content


@pytest.mark.unit
def test_anthropic_tool_use_arguments_are_json_string() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "echo",
                        "input": {"text": "hi"},
                    }
                ],
            }
        ]
    }

    messages = convert_anthropic_to_openai_messages(body)
    process_message_content(messages)

    tool_calls = messages[0].get("tool_calls")
    assert isinstance(tool_calls, list) and tool_calls
    args = tool_calls[0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"text": "hi"}


@pytest.mark.unit
def test_process_message_content_normalizes_tool_call_arguments() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "echo", "arguments": {"text": "ok"}},
                }
            ],
        }
    ]

    process_message_content(messages)

    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"text": "ok"}
