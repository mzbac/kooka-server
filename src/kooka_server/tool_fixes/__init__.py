from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from .common import ToolCall, ToolFix, ToolFixContext, infer_tool_parser_type


@lru_cache(maxsize=32)
def get_profile(tool_parser_type: Optional[str]) -> Tuple[ToolFix, ...]:
    if not tool_parser_type:
        return ()
    if tool_parser_type == "minimax_m2":
        from . import minimax_m2

        return minimax_m2.PROFILE
    return ()


def apply(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
    out = tool_call
    for fix in get_profile(ctx.tool_parser_type):
        out = fix(out, ctx)
    return out


__all__ = ["ToolFixContext", "apply", "get_profile", "infer_tool_parser_type"]
