from __future__ import annotations

import logging


def set_default_wired_limit() -> None:
    """Set MLX Metal wired limit to the recommended working set size.

    This reduces the probability of slowdowns during the first token (TTFT) on
    Apple Silicon by keeping model buffers resident.
    """
    try:
        import mlx.core as mx
    except Exception:
        return

    try:
        metal = getattr(mx, "metal", None)
        if metal is None or not metal.is_available():
            return

        if not hasattr(mx, "set_wired_limit"):
            return

        info = metal.device_info()
        if not isinstance(info, dict):
            return

        wired_limit = info.get("max_recommended_working_set_size")
        if wired_limit is None:
            return

        mx.set_wired_limit(wired_limit)
    except Exception:
        logging.debug("Failed to set MLX wired limit", exc_info=True)


__all__ = ["set_default_wired_limit"]

