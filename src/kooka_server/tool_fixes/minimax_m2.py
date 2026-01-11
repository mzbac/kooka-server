from __future__ import annotations

import re
from typing import Any

from .common import (
    ToolCall,
    ToolFixContext,
    get_tool_parameters_schema,
    normalize_dot_ext_spacing_strict,
    normalize_pathlike_strings_strict,
)

_HYPHEN_SPACE_RE = re.compile(r"-(?:[ \t]+)(?=\S)")
_DOT_SPACE_RE = re.compile(r"\.(?:[ \t]+)(?=\S)")


def fix_dot_ext_spacing(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
    if not isinstance(tool_call, dict):
        return tool_call

    name = tool_call.get("name")
    if not isinstance(name, str) or not name:
        return tool_call

    schema = get_tool_parameters_schema(ctx.tools, name)
    if not isinstance(schema, dict):
        return tool_call

    arguments: Any = tool_call.get("arguments")
    fixed = normalize_dot_ext_spacing_strict(arguments, schema)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


def fix_hyphen_spacing_in_paths(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
    if not isinstance(tool_call, dict):
        return tool_call

    name = tool_call.get("name")
    if not isinstance(name, str) or not name:
        return tool_call

    schema = get_tool_parameters_schema(ctx.tools, name)
    if not isinstance(schema, dict):
        return tool_call

    arguments: Any = tool_call.get("arguments")

    def transform(value: str) -> str:
        if "- " not in value and "-\t" not in value:
            return value
        return _HYPHEN_SPACE_RE.sub("-", value)

    fixed = normalize_pathlike_strings_strict(arguments, schema, transform)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


def fix_dot_spacing_in_paths(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
    if not isinstance(tool_call, dict):
        return tool_call

    name = tool_call.get("name")
    if not isinstance(name, str) or not name:
        return tool_call

    schema = get_tool_parameters_schema(ctx.tools, name)
    if not isinstance(schema, dict):
        return tool_call

    arguments: Any = tool_call.get("arguments")

    def transform(value: str) -> str:
        if ". " not in value and ".\t" not in value:
            return value
        return _DOT_SPACE_RE.sub(".", value)

    fixed = normalize_pathlike_strings_strict(arguments, schema, transform)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


PROFILE = (fix_hyphen_spacing_in_paths, fix_dot_spacing_in_paths, fix_dot_ext_spacing)
