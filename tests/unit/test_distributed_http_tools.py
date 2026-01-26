from __future__ import annotations

import pytest


@pytest.mark.unit
def test_apply_chat_template_safe_normalizes_empty_tools_to_none() -> None:
    from kooka_server.distributed_server.http import apply_chat_template_safe

    seen = {}

    class _DummyTokenizer:
        def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
            seen["tools"] = tools
            seen["add_generation_prompt"] = add_generation_prompt
            seen["tokenize"] = tokenize
            return "OK"

    tok = _DummyTokenizer()
    out = apply_chat_template_safe(
        tok,
        [{"role": "user", "content": "hi"}],
        tools=[],
        add_generation_prompt=True,
        tokenize=False,
    )

    assert out == "OK"
    assert seen["tools"] is None
    assert seen["add_generation_prompt"] is True
    assert seen["tokenize"] is False

