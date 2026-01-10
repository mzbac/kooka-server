from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

from ..hf_utils.hf_cache import list_mlx_lm_models_from_hf_cache

_REQUIRED_HF_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)


def _filter_repo_id_from_models_path(path: str) -> Optional[str]:
    parts = urlparse(path).path.split("/")
    if len(parts) > 3:
        return "/".join(parts[3:])
    return None


def _active_model_id(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    model_path = Path(model)
    if model_path.exists():
        return str(model_path.resolve())
    return model


def list_models(
    *,
    created: int,
    active_model: Optional[str],
    request_path: str,
    required_hf_files: Sequence[str] = _REQUIRED_HF_FILES,
) -> list[dict]:
    filter_repo_id = _filter_repo_id_from_models_path(request_path)

    model_ids: list[str] = []
    if active_id := _active_model_id(active_model):
        if filter_repo_id is None or filter_repo_id == active_id:
            model_ids.append(active_id)

    model_ids.extend(
        list_mlx_lm_models_from_hf_cache(
            filter_repo_id=filter_repo_id,
            required_files=required_hf_files,
        )
    )

    seen: set[str] = set()
    model_ids = [model_id for model_id in model_ids if not (model_id in seen or seen.add(model_id))]

    return [{"id": model_id, "object": "model", "created": created} for model_id in model_ids]


def json_response(
    *,
    created: int,
    active_model: Optional[str],
    request_path: str,
    required_hf_files: Sequence[str] = _REQUIRED_HF_FILES,
) -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": list_models(
                created=created,
                active_model=active_model,
                request_path=request_path,
                required_hf_files=required_hf_files,
            ),
        },
        ensure_ascii=False,
    ).encode("utf-8")
