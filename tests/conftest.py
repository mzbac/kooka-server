from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(_REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

_SRC = _REPO_ROOT / "src"
src_str = str(_SRC)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

from tests.contracts._client import get_default_model, http_json, join_url  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-url",
        default=os.environ.get("BASE_URL", "http://127.0.0.1:8080"),
        help="Base URL for contract tests (default: $BASE_URL or http://127.0.0.1:8080).",
    )
    parser.addoption(
        "--api-key",
        default=os.environ.get("API_KEY", None),
        help="Optional API key for Authorization header (default: $API_KEY).",
    )
    parser.addoption(
        "--timeout-s",
        type=float,
        default=float(os.environ.get("TIMEOUT_S", "60.0")),
        help="HTTP timeout seconds (default: $TIMEOUT_S or 60).",
    )
    parser.addoption(
        "--model",
        default=os.environ.get("MODEL", None),
        help="Model override for contract tests (default: $MODEL or first entry in /v1/models).",
    )


@pytest.fixture(scope="session")
def base_url(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--base-url"))


@pytest.fixture(scope="session")
def timeout_s(pytestconfig: pytest.Config) -> float:
    return float(pytestconfig.getoption("--timeout-s"))


@pytest.fixture(scope="session")
def headers(pytestconfig: pytest.Config) -> Dict[str, str]:
    api_key = pytestconfig.getoption("--api-key")
    if not api_key:
        return {}
    return {"authorization": f"Bearer {api_key}"}


@pytest.fixture(scope="session")
def ensure_server(base_url: str, headers: Dict[str, str], timeout_s: float) -> Dict[str, Any]:
    try:
        status, payload = http_json("GET", join_url(base_url, "/health"), headers=headers, timeout_s=timeout_s)
    except Exception as e:
        pytest.fail(f"Failed to reach server at {base_url}: {e}")
        raise

    assert status == 200, f"GET /health expected 200, got {status}: {payload}"
    assert payload.get("status") == "ok", f"/health status != ok: {payload}"
    return payload


@pytest.fixture(scope="session")
def model_id(
    pytestconfig: pytest.Config,
    ensure_server: Dict[str, Any],
    base_url: str,
    headers: Dict[str, str],
    timeout_s: float,
) -> str:
    override = pytestconfig.getoption("--model")
    if override:
        return str(override)
    return get_default_model(base_url, headers=headers, timeout_s=timeout_s)
