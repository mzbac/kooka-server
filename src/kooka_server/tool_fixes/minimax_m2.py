from __future__ import annotations

import re
import re
from typing import Any

from .common import (
    ToolCall,
    ToolFixContext,
    filter_by_schema,
    get_tool_parameters_schema,
    normalize_dot_ext_spacing_strict,
    normalize_identifier_strings_strict,
    normalize_pathlike_strings_strict,
)

_TIGHTEN_HYPHEN_RE = re.compile(r"(?<=\S)[ \t]*-[ \t]*(?=\S)")
_TIGHTEN_DOT_RE = re.compile(r"(?<=\S)[ \t]*\.[ \t]*(?=\S)")


def fix_schema_normalization(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
    if not isinstance(tool_call, dict):
        return tool_call

    name = tool_call.get("name")
    if not isinstance(name, str) or not name:
        return tool_call

    schema = get_tool_parameters_schema(ctx.tools, name)
    if not isinstance(schema, dict):
        return tool_call

    arguments: Any = tool_call.get("arguments")
    fixed = filter_by_schema(arguments, schema)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


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
        if "- " not in value and "-\t" not in value and " -" not in value and "\t-" not in value:
            return value
        return _TIGHTEN_HYPHEN_RE.sub("-", value)

    fixed = normalize_pathlike_strings_strict(arguments, schema, transform)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


def fix_hyphen_spacing_in_ids(tool_call: ToolCall, ctx: ToolFixContext) -> ToolCall:
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
        if "- " not in value and "-\t" not in value and " -" not in value and "\t-" not in value:
            return value
        return _TIGHTEN_HYPHEN_RE.sub("-", value)

    fixed = normalize_identifier_strings_strict(arguments, schema, transform)
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
        if ". " not in value and ".\t" not in value and " ." not in value and "\t." not in value:
            return value
        return _TIGHTEN_DOT_RE.sub(".", value)

    fixed = normalize_pathlike_strings_strict(arguments, schema, transform)
    if fixed == arguments:
        return tool_call

    out = dict(tool_call)
    out["arguments"] = fixed
    return out


PROFILE = (
    fix_schema_normalization,
    fix_hyphen_spacing_in_ids,
    fix_hyphen_spacing_in_paths,
    fix_dot_spacing_in_paths,
    fix_dot_ext_spacing,
)
