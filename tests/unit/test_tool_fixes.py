from __future__ import annotations

import json

import pytest


@pytest.mark.unit
def test_tool_fixes_minimax_path_normalization_is_schema_aware() -> None:
    from kooka_server.tool_fixes import ToolFixContext, apply as apply_tool_fixes

    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "elementId": {"type": "string"},
                        "path": {"type": "string"},
                        "filename": {"type": "string"},
                        "filePath": {"type": "string"},
                        "file_name": {"type": "string"},
                        "target": {"type": "string", "format": "filepath"},
                        "trace": {"type": "string", "format": "uuid"},
                        "mapping": {
                            "type": "object",
                            "additionalProperties": {"type": "string", "format": "filepath"},
                        },
                        "labels": {"type": "object", "additionalProperties": {"type": "string"}},
                        "content": {"type": "string"},
                        "nested": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "other": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]

    tool_call = {
        "name": "write_file",
        "arguments": {
            "id": "high- score",
            "elementId": "high - score",
            "trace": "f81d4fae - 7dec - 11d0 - a765 - 00a0c91e6bf6",
            "path": "main . js",
            "filename": "style . css",
            "filePath": "src/main . js",
            "file_name": "style. css",
            "target": "assets/style . css",
            "mapping": {"a": "assets/style . css"},
            "labels": {"k": {"nested": "x"}},
            "content": {"text": "keep. js"},
            "nested": {"file_path": "src/main . ts", "other": "x. js"},
            "unknown": "foo. js",
        },
    }

    ctx = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=tools,
    )

    fixed = apply_tool_fixes(tool_call, ctx)
    assert fixed["arguments"]["id"] == "high-score"
    assert fixed["arguments"]["elementId"] == "high-score"
    assert fixed["arguments"]["trace"] == "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
    assert fixed["arguments"]["path"] == "main.js"
    assert fixed["arguments"]["filename"] == "style.css"
    assert fixed["arguments"]["filePath"] == "src/main.js"
    assert fixed["arguments"]["file_name"] == "style.css"
    assert fixed["arguments"]["target"] == "assets/style.css"
    assert fixed["arguments"]["mapping"]["a"] == "assets/style.css"
    assert fixed["arguments"]["labels"]["k"] == json.dumps({"nested": "x"}, ensure_ascii=False)
    assert fixed["arguments"]["nested"]["file_path"] == "src/main.ts"
    assert fixed["arguments"]["content"] == json.dumps({"text": "keep. js"}, ensure_ascii=False)
    assert fixed["arguments"]["nested"]["other"] == "x. js"
    assert "unknown" not in fixed["arguments"]

    # Non-MiniMax profiles should be no-ops.
    ctx_other = ToolFixContext(
        tool_parser_type="json_tools",
        tools=tools,
    )
    fixed_other = apply_tool_fixes(tool_call, ctx_other)
    assert fixed_other == tool_call

    ctx_no_schema = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=None,
    )
    fixed_no_schema = apply_tool_fixes(tool_call, ctx_no_schema)
    assert fixed_no_schema == tool_call


@pytest.mark.unit
def test_tool_fixes_minimax_repo_path_spacing_is_normalized() -> None:
    from kooka_server.tool_fixes import ToolFixContext, apply as apply_tool_fixes

    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]

    tool_call = {
        "name": "write_file",
        "arguments": {
            "path": "/repo/kooka--server. public/kooka- server/validate_mermaid. js",
            "content": "OK",
        },
    }

    ctx = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=tools,
    )

    fixed = apply_tool_fixes(tool_call, ctx)
    assert fixed["arguments"]["path"] == "/repo/kooka--server.public/kooka-server/validate_mermaid.js"


@pytest.mark.unit
def test_tool_fixes_minimax_union_additional_properties_schema_is_applied() -> None:
    from kooka_server.tool_fixes import ToolFixContext, apply as apply_tool_fixes

    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_mapping",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mapping_union": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                },
                                {"type": "null"},
                            ]
                        }
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]

    tool_call = {
        "name": "write_mapping",
        "arguments": {
            "mapping_union": {"a": {"nested": "x"}},
        },
    }

    ctx = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=tools,
    )

    fixed = apply_tool_fixes(tool_call, ctx)
    assert fixed["arguments"]["mapping_union"]["a"] == json.dumps({"nested": "x"}, ensure_ascii=False)


@pytest.mark.unit
def test_tool_fixes_minimax_root_string_schema_is_normalized() -> None:
    from kooka_server.tool_fixes import ToolFixContext, apply as apply_tool_fixes

    tools = [
        {
            "type": "function",
            "function": {
                "name": "path_string",
                "parameters": {"type": "string", "format": "filepath"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "uuid_string",
                "parameters": {"type": "string", "format": "uuid"},
            },
        },
    ]

    ctx = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=tools,
    )

    fixed_path = apply_tool_fixes(
        {"name": "path_string", "arguments": "assets/style . css"},
        ctx,
    )
    assert fixed_path["arguments"] == "assets/style.css"

    fixed_uuid = apply_tool_fixes(
        {"name": "uuid_string", "arguments": "f81d4fae - 7dec - 11d0 - a765 - 00a0c91e6bf6"},
        ctx,
    )
    assert fixed_uuid["arguments"] == "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"
