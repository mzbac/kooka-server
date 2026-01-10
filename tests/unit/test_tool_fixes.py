from __future__ import annotations

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
                        "path": {"type": "string"},
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
            "path": "main. js",
            "content": "keep. js",
            "nested": {"file_path": "src/main. ts", "other": "x. js"},
            "unknown": "foo. js",
        },
    }

    ctx = ToolFixContext(
        tool_parser_type="minimax_m2",
        tools=tools,
    )

    fixed = apply_tool_fixes(tool_call, ctx)
    assert fixed["arguments"]["path"] == "main.js"
    assert fixed["arguments"]["nested"]["file_path"] == "src/main.ts"
    assert fixed["arguments"]["content"] == "keep. js"
    assert fixed["arguments"]["nested"]["other"] == "x. js"
    assert fixed["arguments"]["unknown"] == "foo. js"

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

