from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

_DEFAULT_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)


def _resolve_hf_hub_cache_dir() -> Path:
    """
    Resolve the Hugging Face hub cache directory.

    We intentionally avoid depending on huggingface_hub internals here so this works even
    when `scan_cache_dir()` is unavailable or fails due to non-text filesystem artifacts
    (e.g. macOS AppleDouble `._*` files on external drives).
    """
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser() / "hub")

    return (Path.home() / ".cache" / "huggingface" / "hub")


def _decode_repo_id(models_dir_name: str) -> str:
    # huggingface_hub encodes repo ids as `models--org--name` (slashes replaced by `--`).
    suffix = models_dir_name[len("models--") :]
    return suffix.replace("--", "/")


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _iter_hf_cached_mlx_models(
    hub_cache_dir: Path,
    filter_repo_id: Optional[str],
    required_files: Sequence[str],
) -> Iterable[str]:
    if not hub_cache_dir.is_dir():
        return

    for repo_dir in sorted(hub_cache_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        name = repo_dir.name
        if not name.startswith("models--"):
            continue
        if name.startswith("models--._"):
            continue

        repo_id = _decode_repo_id(name)
        if filter_repo_id is not None and repo_id != filter_repo_id:
            continue

        ref_main = repo_dir / "refs" / "main"
        if not ref_main.is_file():
            continue

        commit = _read_text_file(ref_main)
        if not commit:
            continue

        snapshot = repo_dir / "snapshots" / commit
        if not snapshot.is_dir():
            continue

        present = set()
        try:
            for child in snapshot.iterdir():
                if child.name.startswith("._"):
                    continue
                if child.is_file():
                    present.add(child.name)
        except OSError:
            continue

        if all(req in present for req in required_files):
            yield repo_id


def list_mlx_lm_models_from_hf_cache(
    *,
    filter_repo_id: Optional[str] = None,
    hub_cache_dir: Optional[Path] = None,
    required_files: Sequence[str] = _DEFAULT_REQUIRED_FILES,
) -> list[str]:
    """
    Return repo ids for cached models that look compatible with MLX-LM.

    This is used to populate `/v1/models`. It is resilient to non-UTF8 artifacts
    in the cache directory (common on macOS external drives).
    """
    hub_dir = hub_cache_dir or _resolve_hf_hub_cache_dir()
    return list(_iter_hf_cached_mlx_models(hub_dir, filter_repo_id, required_files))

