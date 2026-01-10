from __future__ import annotations

import os
from typing import Any, Callable, Optional

import mlx.core as mx


def patch_minimax_for_pipeline() -> None:
    from mlx_lm.models import minimax as minimax_mod
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.pipeline import PipelineMixin

    minimax_model_cls = minimax_mod.MiniMaxModel
    if getattr(minimax_model_cls, "_kooka_pipeline_patched", False):
        return

    def pipeline_layers(self) -> list[Any]:
        start_idx = getattr(self, "start_idx", 0)
        end_idx = getattr(self, "end_idx", None)
        return self.layers[start_idx:end_idx]

    def pipeline(self, group: Any) -> None:
        split = os.environ.get("MINIMAX_PIPELINE_SPLIT")
        if not split:
            PipelineMixin.pipeline(self, group)
            return

        counts = [int(x.strip()) for x in split.split(",") if x.strip()]
        if len(counts) != group.size():
            raise ValueError(f"MINIMAX_PIPELINE_SPLIT expects {group.size()} entries but got {len(counts)}")

        total_layers = len(self.layers)
        if sum(counts) != total_layers:
            raise ValueError(f"MINIMAX_PIPELINE_SPLIT must sum to {total_layers} layers but got {sum(counts)}")

        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()

        prefix = sum(counts[: self.pipeline_rank])
        self.end_idx = total_layers - prefix
        self.start_idx = self.end_idx - counts[self.pipeline_rank]

        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx

    original_call: Callable[..., Any] = minimax_model_cls.__call__

    def pipeline_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        pipeline_size = getattr(self, "pipeline_size", 1)
        if pipeline_size <= 1 and not hasattr(self, "start_idx"):
            return original_call(self, inputs, mask=mask, cache=cache)

        h = self.embed_tokens(inputs)

        layers = pipeline_layers(self)
        if cache is None:
            cache = [None] * len(layers)

        if mask is None:
            mask = minimax_mod.create_attention_mask(h, cache[0] if cache else None)

        pipeline_rank = getattr(self, "pipeline_rank", 0)
        pipeline_size = getattr(self, "pipeline_size", 1)

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(layers, cache):
            if layer is None:
                continue
            h = layer(h, mask, c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache and cache[-1] is not None and hasattr(cache[-1], "keys"):
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)

    def make_cache(self) -> list[Any]:
        inner = getattr(self, "model", None)
        if inner is None:
            return []
        layers = getattr(inner, "pipeline_layers", None)
        if callable(layers):
            layers = layers()
        if not isinstance(layers, list):
            layers = getattr(inner, "layers", [])
        return [KVCache() for _ in layers]

    setattr(minimax_model_cls, "pipeline_layers", property(pipeline_layers))
    setattr(minimax_model_cls, "pipeline", pipeline)
    setattr(minimax_model_cls, "__call__", pipeline_call)

    setattr(minimax_mod.Model, "make_cache", make_cache)

    setattr(minimax_model_cls, "_kooka_pipeline_patched", True)

