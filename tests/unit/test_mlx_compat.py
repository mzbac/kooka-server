from __future__ import annotations

import sys
import types

import pytest


class _DummyTokenizer:
    def __init__(self, chat_template: str, tool_parser, init_kwargs: dict):
        self.chat_template = chat_template
        self._tool_parser = tool_parser
        self._tool_call_start = "<tool_call>"
        self._tool_call_end = "</tool_call>"
        self.init_kwargs = init_kwargs

    @property
    def tool_parser(self):
        return self._tool_parser


@pytest.mark.unit
def test_maybe_patch_tool_parser_patches_json_tools_tokenizer() -> None:
    from kooka_server.mlx_utils.tokenizer_compat import maybe_patch_tool_parser

    # Build a minimal fake `mlx_lm.tool_parsers` so this test doesn't depend on
    # the installed mlx_lm distribution including those modules.
    import mlx_lm  # noqa: F401

    tool_parsers_mod = types.ModuleType("mlx_lm.tool_parsers")
    json_tools_mod = types.ModuleType("mlx_lm.tool_parsers.json_tools")
    qwen3_coder_mod = types.ModuleType("mlx_lm.tool_parsers.qwen3_coder")

    def _json_parse_tool_call(text, tools=None):
        raise ValueError("json_tools should not be used after patch")

    _json_parse_tool_call.__module__ = json_tools_mod.__name__
    json_tools_mod.parse_tool_call = _json_parse_tool_call
    json_tools_mod.tool_call_start = "<tool_call>"
    json_tools_mod.tool_call_end = "</tool_call>"

    def _qwen3_parse_tool_call(text, tools=None):
        return {"name": "echo", "arguments": {"text": "OK"}}

    _qwen3_parse_tool_call.__module__ = qwen3_coder_mod.__name__
    qwen3_coder_mod.parse_tool_call = _qwen3_parse_tool_call
    qwen3_coder_mod.tool_call_start = "<tool_call>"
    qwen3_coder_mod.tool_call_end = "</tool_call>"

    tool_parsers_mod.json_tools = json_tools_mod
    tool_parsers_mod.qwen3_coder = qwen3_coder_mod

    sys.modules[tool_parsers_mod.__name__] = tool_parsers_mod
    sys.modules[json_tools_mod.__name__] = json_tools_mod
    sys.modules[qwen3_coder_mod.__name__] = qwen3_coder_mod

    tok = _DummyTokenizer(
        chat_template="... <tool_call>\\n<function=echo>\\n<parameter=text>OK</parameter>\\n</function>\\n</tool_call> ...",
        tool_parser=json_tools_mod.parse_tool_call,
        init_kwargs={},
    )
    maybe_patch_tool_parser(tok)

    assert tok._tool_parser is qwen3_coder_mod.parse_tool_call
    assert tok._tool_call_start == qwen3_coder_mod.tool_call_start
    assert tok._tool_call_end == qwen3_coder_mod.tool_call_end
    assert tok.init_kwargs.get("tool_parser_type") == "qwen3_coder"
