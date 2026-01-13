from __future__ import annotations

import json
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


_COMPOUND_KEY_PARTS: dict[str, tuple[str, ...]] = {
    "filepath": ("file", "path"),
    "filepaths": ("file", "paths"),
    "filename": ("file", "name"),
    "filenames": ("file", "names"),
    "dirpath": ("dir", "path"),
    "dirpaths": ("dir", "paths"),
    "dirname": ("dir", "name"),
    "dirnames": ("dir", "names"),
}


def _split_key_parts(key: str) -> list[str]:
    key_s = str(key)
    key_s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key_s)
    key_s = key_s.replace("-", "_").lower()
    key_s = re.sub(r"[^a-z0-9]+", "_", key_s).strip("_")
    if not key_s:
        return []

    parts: list[str] = []
    for part in key_s.split("_"):
        if not part:
            continue
        if part in _COMPOUND_KEY_PARTS:
            parts.extend(_COMPOUND_KEY_PARTS[part])
        else:
            parts.append(part)
    return parts


def is_pathlike_key(key: str) -> bool:
    parts = _split_key_parts(key)
    if not parts:
        return False

    last = parts[-1]
    if last in {"path", "paths"}:
        return True
    if last in {"file", "files"}:
        return True
    if last in {"dir", "dirs", "directory", "directories", "folder", "folders"}:
        return True
    if last in {"src", "dst", "source", "destination"}:
        return True
    if last in {"name", "names"} and any(part in parts for part in {"file", "dir", "directory", "folder"}):
        return True
    return False


def is_identifier_key(key: str) -> bool:
    parts = _split_key_parts(key)
    if not parts:
        return False
    return parts[-1] in {"id", "ids"}


def is_pathlike_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False

    fmt = schema.get("format")
    if isinstance(fmt, str) and fmt:
        parts = _split_key_parts(fmt)
        if parts and parts[-1] in {"path", "paths"}:
            # Treat bare "path" as a generic filesystem path signal; otherwise require a file/dir hint
            # to avoid matching unrelated formats like "json-path".
            if len(parts) == 1:
                return True
            if any(part in {"file", "dir", "directory", "folder"} for part in parts):
                return True

    for union_key in ("anyOf", "oneOf", "allOf"):
        union_val = schema.get(union_key)
        if isinstance(union_val, list):
            for branch in union_val:
                if is_pathlike_schema(branch):
                    return True

    return False


def is_identifier_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False

    fmt = schema.get("format")
    if isinstance(fmt, str) and fmt.lower() == "uuid":
        return True

    for union_key in ("anyOf", "oneOf", "allOf"):
        union_val = schema.get(union_key)
        if isinstance(union_val, list):
            for branch in union_val:
                if is_identifier_schema(branch):
                    return True

    return False


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


def _normalize_strings_strict(
    arguments: Any,
    schema: Optional[dict],
    *,
    should_transform_key: Callable[[str], bool],
    should_transform_schema: Optional[Callable[[Any], bool]] = None,
    transform: Callable[[str], str],
) -> Any:
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

            additional_props_schema: Optional[dict] = None
            additional_props = current_schema.get("additionalProperties", True)
            if isinstance(additional_props, dict):
                additional_props_schema = additional_props
            for union_key in ("anyOf", "oneOf", "allOf"):
                union_val = current_schema.get(union_key)
                if isinstance(union_val, list):
                    for branch in union_val:
                        if isinstance(branch, dict) and isinstance(branch.get("properties"), dict):
                            properties.update(branch["properties"])
                        if additional_props_schema is None and isinstance(branch, dict):
                            branch_additional_props = branch.get("additionalProperties")
                            if isinstance(branch_additional_props, dict):
                                additional_props_schema = branch_additional_props

            if not properties and additional_props_schema is None:
                return value

            out: dict[str, Any] = {}
            for k, v in value.items():
                if isinstance(k, str) and k in properties:
                    out[k] = walk(v, properties[k], k)
                elif additional_props_schema is not None:
                    out[k] = walk(v, additional_props_schema, k if isinstance(k, str) else None)
                else:
                    out[k] = v
            return out

        if isinstance(value, list):
            items_schema = current_schema.get("items")
            if isinstance(items_schema, dict):
                return [walk(v, items_schema, key) for v in value]
            return value

        if isinstance(value, str):
            schema_matches = should_transform_schema(current_schema) if should_transform_schema is not None else False
            key_matches = key is not None and should_transform_key(key)
            if schema_matches or key_matches:
                return transform(value)
            return value

        return value

    return walk(arguments, schema, None)


def normalize_pathlike_strings_strict(arguments: Any, schema: Optional[dict], transform: Callable[[str], str]) -> Any:
    """Apply a normalization function to schema-defined path/file-like string fields."""
    return _normalize_strings_strict(
        arguments,
        schema,
        should_transform_key=is_pathlike_key,
        should_transform_schema=is_pathlike_schema,
        transform=transform,
    )


def normalize_identifier_strings_strict(arguments: Any, schema: Optional[dict], transform: Callable[[str], str]) -> Any:
    """Apply a normalization function to schema-defined identifier/id-like string fields."""
    return _normalize_strings_strict(
        arguments,
        schema,
        should_transform_key=is_identifier_key,
        should_transform_schema=is_identifier_schema,
        transform=transform,
    )


def normalize_dot_ext_spacing_strict(arguments: Any, schema: Optional[dict]) -> Any:
    """Normalize '. js' -> '.js' for schema-defined path/file-like fields."""
    def transform(value: str) -> str:
        if not _DOTSPACE_EXT_RE.search(value):
            return value
        return _DOTSPACE_EXT_RE.sub(r".\1", value)

    return normalize_pathlike_strings_strict(arguments, schema, transform)


def filter_by_schema(value: Any, schema: Any) -> Any:
    """
    Best-effort schema normalization (not full validation).

    - When a schema expects a string, coerce non-string values to JSON strings.
    - When `additionalProperties` is false, drop unknown object keys.
    - Recurse through objects and arrays.
    """
    if not isinstance(schema, dict):
        return value

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if value is None and "null" in schema_type:
            return None
        schema_type = next((t for t in schema_type if t != "null"), schema_type[0] if schema_type else None)

    if schema_type == "string":
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    if schema_type == "array":
        if not isinstance(value, list):
            return value
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            return [filter_by_schema(item, items_schema) for item in value]
        return value

    properties: dict[str, Any] = {}
    if schema_type in (None, "object") and isinstance(schema.get("properties"), dict):
        properties.update(schema["properties"])

    additional_props = schema.get("additionalProperties", True)
    additional_props_schema: Optional[dict] = additional_props if isinstance(additional_props, dict) else None
    for union_key in ("anyOf", "oneOf", "allOf"):
        union_val = schema.get(union_key)
        if isinstance(union_val, list):
            for branch in union_val:
                if isinstance(branch, dict) and isinstance(branch.get("properties"), dict):
                    properties.update(branch["properties"])
                if additional_props_schema is None and isinstance(branch, dict):
                    branch_additional_props = branch.get("additionalProperties")
                    if isinstance(branch_additional_props, dict):
                        additional_props_schema = branch_additional_props

    is_object_schema = schema_type == "object" or (
        schema_type is None and (properties or additional_props is False or additional_props_schema is not None)
    )
    if is_object_schema:
        if not isinstance(value, dict):
            return value

        out: dict[str, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key in properties:
                out[key] = filter_by_schema(val, properties[key])
            elif additional_props is False:
                continue
            elif additional_props_schema is not None:
                out[key] = filter_by_schema(val, additional_props_schema)
            else:
                out[key] = val
        return out

    return value
