from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from ...tool_fixes import ToolFixContext, apply as apply_tool_fixes


def tool_names_from_openai_tools(tools: Any) -> set[str]:
    if not isinstance(tools, list):
        return set()

    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name") or tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) < 2:
        return s
    if not lines[-1].startswith("```"):
        return s
    return "\n".join(lines[1:-1]).strip()


def parse_json_tool_calls(text: str, allowed_names: set[str]) -> Optional[list[dict]]:
    if not allowed_names:
        return None

    s = _strip_json_fence(text)
    if not s:
        return None

    def make_calls(payload: Any) -> Optional[list[dict]]:
        if isinstance(payload, dict):
            calls = [payload]
        elif isinstance(payload, list):
            calls = payload
        else:
            return None

        out: list[dict] = []
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not name:
                continue
            if name not in allowed_names:
                continue
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                continue

            out.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                    "index": idx,
                }
            )

        return out or None

    try:
        if out := make_calls(json.loads(s)):
            return out
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(s):
        if ch not in "{[":
            continue
        try:
            payload, _end = decoder.raw_decode(s[idx:])
        except json.JSONDecodeError:
            continue
        if out := make_calls(payload):
            return out

    return None


def normalize_finish_reason_for_tool_calls(finish_reason: Any, *, saw_tool_calls: bool) -> Any:
    """
    Normalize OpenAI `finish_reason` when tool calls are present.

    Some backends report the final chunk as "stop" even if tool calls were emitted earlier.
    """
    if not saw_tool_calls or finish_reason is None:
        return finish_reason
    if finish_reason in {"stop", "tool_call"}:
        return "tool_calls"
    return finish_reason


def make_openai_tool_call(
    *,
    name: str,
    arguments: Any,
    tool_call_id: Optional[str] = None,
    index: Optional[int] = None,
) -> dict:
    if tool_call_id is None:
        tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

    if arguments is None:
        arguments_s = "{}"
    elif isinstance(arguments, str):
        arguments_s = arguments
    else:
        arguments_s = json.dumps(arguments, ensure_ascii=False)

    out: dict[str, Any] = {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments_s},
    }
    if index is not None:
        out["index"] = index
    return out


def apply_tool_fixes_to_openai_tool_calls(
    tool_calls: list[Any],
    *,
    tool_parser_type: Optional[str],
    tools: Optional[list[dict]],
) -> list[Any]:
    """
    Apply model-aware tool-call fixes to OpenAI tool call payloads.

    OpenAI tool calls carry `function.arguments` as a JSON string; fixes operate on
    a parsed Python object and are re-serialized back to a string.
    """
    if not tool_calls:
        return tool_calls

    ctx = ToolFixContext(tool_parser_type=tool_parser_type, tools=tools)

    fixed: list[Any] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            fixed.append(tc)
            continue
        func = tc.get("function")
        if not isinstance(func, dict):
            fixed.append(tc)
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            fixed.append(tc)
            continue
        args_s = func.get("arguments")
        if not isinstance(args_s, str):
            fixed.append(tc)
            continue
        try:
            args_obj = json.loads(args_s) if args_s else {}
        except json.JSONDecodeError:
            fixed.append(tc)
            continue

        fixed_call = apply_tool_fixes({"name": name, "arguments": args_obj}, ctx)
        if not isinstance(fixed_call, dict):
            fixed.append(tc)
            continue

        tc2 = dict(tc)
        func2 = dict(func)
        func2["arguments"] = json.dumps(fixed_call.get("arguments", args_obj), ensure_ascii=False)
        tc2["function"] = func2
        fixed.append(tc2)

    return fixed
