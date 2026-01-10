from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mlx_lm.utils import sharded_load as _sharded_load

from .minimax_pipeline import patch_minimax_for_pipeline


def _should_patch_minimax(repo: str) -> bool:
    repo_s = str(repo)
    if "minimax" in repo_s.lower():
        return True

    repo_path = Path(repo_s)
    if repo_path.is_dir():
        config = repo_path / "config.json"
        if config.is_file():
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            model_type = data.get("model_type")
            return isinstance(model_type, str) and model_type.lower() == "minimax"

    return False


def sharded_load(
    repo: str,
    pipeline_group: Optional[Any] = None,
    tensor_group: Optional[Any] = None,
    return_config: bool = False,
):
    if _should_patch_minimax(repo):
        patch_minimax_for_pipeline()
    return _sharded_load(
        repo,
        pipeline_group=pipeline_group,
        tensor_group=tensor_group,
        return_config=return_config,
    )
