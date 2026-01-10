from __future__ import annotations

from typing import Any


def maybe_patch_tool_parser(tokenizer: Any) -> None:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        return

    tool_parser = getattr(tokenizer, "tool_parser", None)
    tool_parser_module = getattr(tool_parser, "__module__", None)
    if tool_parser_module != "mlx_lm.tool_parsers.json_tools":
        return

    if "<function=" not in chat_template or "<parameter=" not in chat_template:
        return

    try:
        from mlx_lm.tool_parsers import qwen3_coder
    except ModuleNotFoundError:
        return

    setattr(tokenizer, "_tool_parser", qwen3_coder.parse_tool_call)
    setattr(tokenizer, "_tool_call_start", qwen3_coder.tool_call_start)
    setattr(tokenizer, "_tool_call_end", qwen3_coder.tool_call_end)

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["tool_parser_type"] = "qwen3_coder"


__all__ = ["maybe_patch_tool_parser"]

