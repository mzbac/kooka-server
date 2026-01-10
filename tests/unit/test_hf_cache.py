from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.mark.unit
def test_list_mlx_lm_models_from_hf_cache_filters_and_handles_binary_refs() -> None:
    from kooka_server.hf_utils.hf_cache import list_mlx_lm_models_from_hf_cache

    repo_id = "mlx-community/Test-Model"
    encoded_repo_dir = "models--mlx-community--Test-Model"
    commit = "abc123"

    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(tmp)
        repo = hub / encoded_repo_dir

        # Create a valid refs/main and a binary refs/._main (AppleDouble artifact).
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text(commit, encoding="utf-8")
        (repo / "refs" / "._main").write_bytes(b"\xb0\x00\x00\x00")

        # Minimal snapshot contents required by /v1/models listing.
        snapshot = repo / "snapshots" / commit
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
            (snapshot / name).write_text("{}\n", encoding="utf-8")

        models = list_mlx_lm_models_from_hf_cache(hub_cache_dir=hub)
        assert repo_id in models

        models_filtered = list_mlx_lm_models_from_hf_cache(hub_cache_dir=hub, filter_repo_id=repo_id)
        assert models_filtered == [repo_id]

        models_filtered_out = list_mlx_lm_models_from_hf_cache(hub_cache_dir=hub, filter_repo_id="other/repo")
        assert models_filtered_out == []

        # Missing required files should exclude the repo.
        (snapshot / "tokenizer_config.json").unlink()
        models_missing = list_mlx_lm_models_from_hf_cache(hub_cache_dir=hub)
        assert repo_id not in models_missing

