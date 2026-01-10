from __future__ import annotations

from typing import Any


def _summarize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return {"type": type(messages).__name__}

    summarized: list[dict[str, Any]] = []
    for msg in messages[:50]:
        if not isinstance(msg, dict):
            summarized.append({"type": type(msg).__name__})
            continue

        role = msg.get("role")
        content = msg.get("content")

        item: dict[str, Any] = {"role": role}
        if isinstance(content, str):
            item["content_len"] = len(content)
        elif isinstance(content, list):
            item["content_items"] = len(content)
        elif content is None:
            item["content_len"] = 0
        else:
            item["content_type"] = type(content).__name__

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            item["tool_calls"] = len(tool_calls)

        summarized.append(item)

    extra = len(messages) - len(summarized)
    if extra > 0:
        summarized.append({"truncated_messages": extra})
    return summarized


def _summarize_tools(tools: Any) -> Any:
    if tools is None:
        return None
    if not isinstance(tools, list):
        return {"type": type(tools).__name__}

    names: list[str] = []
    for tool in tools[:200]:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name") or tool.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    out: dict[str, Any] = {"count": len(tools), "names": names[:50]}
    if len(names) > 50:
        out["truncated_names"] = len(names) - 50
    return out


def redact_request_body(body: Any) -> Any:
    """
    Return a safe-to-log summary of a request body.

    Never include user text; only emit metadata (lengths, counts, and tool names).
    """
    if not isinstance(body, dict):
        return {"type": type(body).__name__}

    out: dict[str, Any] = {}

    model = body.get("model")
    if isinstance(model, str):
        out["model"] = model

    for key in ("stream",):
        if key in body:
            out[key] = bool(body.get(key))

    for key in (
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
    ):
        if key in body:
            out[key] = body.get(key)

    if "messages" in body:
        out["messages"] = _summarize_messages(body.get("messages"))

    if "tools" in body:
        out["tools"] = _summarize_tools(body.get("tools"))

    for key in ("stop", "stop_sequences"):
        if key in body:
            stop = body.get(key)
            if isinstance(stop, str):
                out[key] = {"count": 1, "lens": [len(stop)]}
            elif isinstance(stop, list):
                lens = [len(s) for s in stop if isinstance(s, str)]
                out[key] = {"count": len(lens), "lens": lens[:20]}
            else:
                out[key] = {"type": type(stop).__name__}

    extras = sorted(set(body.keys()) - set(out.keys()))
    if extras:
        out["extra_keys"] = extras

    return out


__all__ = ["redact_request_body"]

