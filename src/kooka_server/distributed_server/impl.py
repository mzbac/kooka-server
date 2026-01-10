# Copyright © 2025 Apple Inc.

"""
Distributed HTTP server for MLX LM.

Run with mlx.launch for distributed inference across multiple machines:

```bash
# Local multi-process test (2 processes on same machine)
mlx.launch -n 2 \
    --env MLX_METAL_FAST_SYNCH=1 \
    -- \
    python mlx_lm/examples/distributed_server.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --port 8080

# Two machines with ring backend
mlx.launch --backend ring --hostfile hosts.json \
    --env MLX_METAL_FAST_SYNCH=1 \
    -- \
    python mlx_lm/examples/distributed_server.py \
    --model mlx-community/Llama-3.3-70B-Instruct-4bit \
    --host 0.0.0.0 --port 8080
```

Architecture:
- All ranks run the generation loop simultaneously (required for pipeline parallelism)
- Rank 0 receives HTTP requests and broadcasts prompts to all ranks
- All ranks call model forward together for each token
- Only rank 0 returns HTTP responses

Supports all server.py functionality:
- OpenAI-compatible /v1/chat/completions and /v1/completions
- Anthropic-compatible /v1/messages
- Tool calling (via tokenizer tool parsers)
- Streaming responses

For more information on running distributed programs with MLX see:
https://ml-explore.github.io/mlx/build/html/usage/distributed.html
"""

import argparse
import logging
import warnings
from threading import Thread
from typing import List, Optional

import mlx.core as mx

from ..mlx_utils.mlx_lm_compat import sharded_load
from ..mlx_utils.tokenizer_compat import maybe_patch_tool_parser

from .generation import generation_loop
from .http import run_http_server
from .state import DistributedState

def _run(args: argparse.Namespace) -> None:
    # Initialize distributed
    group = mx.distributed.init()
    rank = group.rank()
    world_size = group.size()

    logging.basicConfig(
        level=getattr(logging, str(getattr(args, "log_level", "INFO")).upper(), logging.INFO),
        format=f"%(asctime)s - [Rank {rank}] %(message)s",
        force=True,
    )

    logging.info(f"Distributed: rank {rank}/{world_size}")

    # Load model with distributed sharding. Prefer pipeline-parallel (our main
    # multi-machine use-case), but fall back to tensor parallelism when a model
    # doesn't support pipelining (useful for quick local validation).
    logging.info(f"Loading {args.model} (distributed sharding)")
    try:
        model, tokenizer = sharded_load(args.model, pipeline_group=group, tensor_group=None)
    except ValueError as e:
        msg = str(e)
        if "does not support pipelining" not in msg:
            raise
        if world_size != 1:
            raise ValueError(
                "Model does not support pipelining (required for serve-distributed with world_size>1). "
                "Use a pipeline-capable MLX model (e.g. MiniMax) or run single-machine `serve`."
            ) from e
        logging.info("Model does not support pipelining; falling back to tensor sharding (world_size=1)")
        model, tokenizer = sharded_load(args.model, pipeline_group=None, tensor_group=group)
    logging.info("Model loaded")

    if args.use_default_chat_template and tokenizer.chat_template is None:
        tokenizer.chat_template = tokenizer.default_chat_template
    if args.chat_template:
        tokenizer.chat_template = args.chat_template
    maybe_patch_tool_parser(tokenizer)

    dist_state = DistributedState(group)

    if rank == 0:
        # Rank 0: HTTP server in background, generation loop in foreground
        http_thread = Thread(target=run_http_server, args=(dist_state, tokenizer, args), daemon=True)
        http_thread.start()
        warnings.warn("kooka-server serve-distributed: early development; contract enforced by pytest contract tests (tests/).")

    # All ranks run generation loop
    generation_loop(dist_state, model, tokenizer, args)


def serve_distributed(args: argparse.Namespace) -> None:
    _run(args)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="kooka-server distributed server")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace repo")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--chat-template", type=str, default="")
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=2,
        help="Maximum number of prompt-cache entries to keep (LRU). Set to 0 to disable.",
    )

    args = parser.parse_args(argv)
    _run(args)


if __name__ == "__main__":
    main()
