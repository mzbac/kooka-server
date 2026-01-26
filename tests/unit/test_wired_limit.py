from __future__ import annotations

import sys
import types

import pytest


@pytest.mark.unit
def test_set_default_wired_limit_uses_recommended_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    mlx_mod = types.ModuleType("mlx")
    mlx_mod.__path__ = []

    core_mod = types.ModuleType("mlx.core")

    class _DummyMetal:
        def is_available(self) -> bool:
            return True

        def device_info(self) -> dict:
            return {"max_recommended_working_set_size": 123}

    def _set_wired_limit(value: int) -> None:
        called["value"] = value

    core_mod.metal = _DummyMetal()
    core_mod.set_wired_limit = _set_wired_limit

    monkeypatch.setitem(sys.modules, "mlx", mlx_mod)
    monkeypatch.setitem(sys.modules, "mlx.core", core_mod)

    from kooka_server.mlx_utils.wired_limit import set_default_wired_limit

    set_default_wired_limit()

    assert called["value"] == 123

