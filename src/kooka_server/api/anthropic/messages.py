from __future__ import annotations

import json
from typing import Any, Optional


def convert_anthropic_to_openai_messages(body: dict) -> list[dict]:
    messages: list[dict] = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_id = block.get("tool_use_id", "")
                result = block.get("content", "")
                if isinstance(result, list):
                    result = "".join(b.get("text", "") for b in result if b.get("type") == "text")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result,
                    }
                )

        if role == "assistant":
            msg_dict: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            messages.append(msg_dict)
        elif tool_results:
            messages.extend(tool_results)
        else:
            messages.append({"role": role, "content": "".join(text_parts)})

    return messages


def convert_anthropic_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", "") or "",
                    "parameters": tool.get("input_schema", {}) or {},
                },
            }
        )
    return converted or None


def process_message_content(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_fragments = [fragment.get("text", "") for fragment in content if fragment.get("type") == "text"]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                func = tool_call.get("function")
                if not isinstance(func, dict):
                    continue
                args = func.get("arguments")
                if isinstance(args, str):
                    if args:
                        try:
                            json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            func["arguments"] = "{}"
                elif args is not None:
                    func["arguments"] = json.dumps(args, ensure_ascii=False)
