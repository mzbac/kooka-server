from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


ToolCall = dict[str, Any]
ToolFix = Callable[[ToolCall, "ToolFixContext"], ToolCall]


@dataclass(frozen=True)
class ToolFixContext:
    """Context for model/tool-specific tool-call fixups."""

    tool_parser_type: Optional[str]
    tools: Optional[list[dict]]


def infer_tool_parser_type(tokenizer: Any) -> Optional[str]:
    tool_parser = getattr(tokenizer, "tool_parser", None)
    if callable(tool_parser):
        module = getattr(tool_parser, "__module__", None)
        if isinstance(module, str) and module.startswith("mlx_lm.tool_parsers."):
            return module.rsplit(".", 1)[-1]

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        tool_parser_type = init_kwargs.get("tool_parser_type")
        if isinstance(tool_parser_type, str) and tool_parser_type:
            return tool_parser_type

    return None


_DOT_EXTS = (
    "js",
    "ts",
    "jsx",
    "tsx",
    "mjs",
    "cjs",
    "json",
    "md",
    "html",
    "css",
    "yml",
    "yaml",
    "toml",
    "py",
    "sh",
    "go",
    "rs",
)
_DOTSPACE_EXT_RE = re.compile(rf"\.(?:[ \t]+)({'|'.join(_DOT_EXTS)})\b")


_PATHLIKE_ARG_KEYS = frozenset(
    {
        "path",
        "paths",
        "file",
        "files",
        "file_path",
        "file_paths",
        "filepath",
        "filepaths",
        "filename",
        "filenames",
        "source",
        "destination",
        "src",
        "dst",
    }
)


def is_pathlike_key(key: str) -> bool:
    key_norm = key.lower()
    if key_norm in _PATHLIKE_ARG_KEYS:
        return True
    return key_norm.endswith(("path", "paths", "filepath", "filepaths", "filename", "filenames"))


def get_tool_parameters_schema(tools: Optional[list[dict]], tool_name: str) -> Optional[dict]:
    """Return the JSON schema for a tool's parameters, if present."""
    if not tools:
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        func: Any = None
        if tool.get("type") == "function":
            func = tool.get("function")
        else:
            func = tool

        if not isinstance(func, dict):
            continue
        name = func.get("name") or tool.get("name")
        if name != tool_name:
            continue

        params = func.get("parameters") or tool.get("parameters")
        if isinstance(params, dict):
            return params
        return None
    return None


def normalize_dot_ext_spacing_strict(arguments: Any, schema: Optional[dict]) -> Any:
    """Normalize '. js' -> '.js' for schema-defined path/file-like fields."""
    if not isinstance(schema, dict):
        return arguments

    def walk(value: Any, current_schema: Optional[dict], key: Optional[str]) -> Any:
        if not isinstance(current_schema, dict):
            return value

        # Merge object properties across unions.
        schema_type = current_schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((t for t in schema_type if t != "null"), schema_type[0] if schema_type else None)

        if isinstance(value, dict):
            properties: dict[str, Any] = {}
            if schema_type in (None, "object") and isinstance(current_schema.get("properties"), dict):
                properties.update(current_schema["properties"])
            for union_key in ("anyOf", "oneOf", "allOf"):
                union_val = current_schema.get(union_key)
                if isinstance(union_val, list):
                    for branch in union_val:
                        if isinstance(branch, dict) and isinstance(branch.get("properties"), dict):
                            properties.update(branch["properties"])

            if not properties:
                return value

            out: dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(k, str) and k in properties:
                    out[k] = walk(v, properties[k], k)
                else:
                    out[k] = v
            return out

        if isinstance(value, list):
            items_schema = current_schema.get("items")
            if isinstance(items_schema, dict):
                return [walk(v, items_schema, key) for v in value]
            return value

        if isinstance(value, str):
            if key is not None and is_pathlike_key(key) and _DOTSPACE_EXT_RE.search(value):
                return _DOTSPACE_EXT_RE.sub(r".\1", value)
            return value

        return value

    return walk(arguments, schema, None)
