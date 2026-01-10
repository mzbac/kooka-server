from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

from .tokenizer_compat import maybe_patch_tool_parser


class KookaModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model_key: Optional[tuple[Any, ...]] = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.draft_model: Any = None
        self.cache_types: set[type] = set()

        self.default_model_map: dict[str, str] = {}
        if self.cli_args.model is not None:
            self.default_model_map[self.cli_args.model] = "default_model"
            self.load(self.cli_args.model, draft_model_path="default_model")

    def load(self, model_path: str, adapter_path: Optional[str] = None, draft_model_path: Optional[str] = None):
        model_path = self.default_model_map.get(model_path, model_path)
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.model_key = None
        self.cache_types = set()

        tokenizer_config: dict[str, Any] = {"trust_remote_code": True if self.cli_args.trust_remote_code else None}
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError("A model path has to be given via --model or in the HTTP request")
            adapter_path = adapter_path or self.cli_args.adapter_path
            model, tokenizer = load(self.cli_args.model, adapter_path=adapter_path, tokenizer_config=tokenizer_config)
        else:
            model, tokenizer = load(model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config)

        if self.cli_args.use_default_chat_template and tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
        maybe_patch_tool_parser(tokenizer)

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        def validate_draft_tokenizer(draft_tokenizer: Any) -> None:
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer; speculative decoding may misbehave."
                )

        if draft_model_path == "default_model" and self.cli_args.draft_model is not None:
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)
        elif draft_model_path is not None and draft_model_path != "default_model":
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)

        for c in make_prompt_cache(self.model):
            self.cache_types.add(type(c))
        if self.draft_model is not None:
            for c in make_prompt_cache(self.draft_model):
                self.cache_types.add(type(c))

        return self.model, self.tokenizer
