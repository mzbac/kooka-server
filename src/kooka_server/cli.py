import argparse
import json
import logging

from .distributed import serve_distributed
from .mlx_utils.wired_limit import set_default_wired_limit
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="kooka-server")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve_p = sub.add_parser("serve", help="Run single-machine server")
    serve_p.add_argument("--model", required=False, default=None, help="Model path or HF repo id")
    serve_p.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    serve_p.add_argument("--trust-remote-code", action="store_true")
    serve_p.add_argument("--draft-model", default=None, help="Optional model for speculative decoding")
    serve_p.add_argument("--num-draft-tokens", type=int, default=3)
    serve_p.add_argument("--use-default-chat-template", action="store_true")
    serve_p.add_argument("--chat-template", default="", help="Override tokenizer chat template")
    serve_p.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help='JSON args for apply_chat_template, e.g. \'{"enable_thinking": false}\'',
    )
    serve_p.add_argument("--temp", type=float, default=0.0)
    serve_p.add_argument("--top-p", type=float, default=1.0)
    serve_p.add_argument("--top-k", type=int, default=0)
    serve_p.add_argument("--min-p", type=float, default=0.0)
    serve_p.add_argument("--max-tokens", type=int, default=512)
    serve_p.add_argument("--kv-bits", type=int, default=None, help="KV cache quantization bits (None disables)")
    serve_p.add_argument("--kv-group-size", type=int, default=64, help="KV cache quantization group size")
    serve_p.add_argument("--quantized-kv-start", type=int, default=0, help="Token position to begin KV quantization")

    dist_p = sub.add_parser("serve-distributed", help="Run distributed server (launch via mlx.launch)")
    dist_p.add_argument("--model", required=True, help="Model path or HF repo id")
    dist_p.add_argument("--host", default="0.0.0.0")
    dist_p.add_argument("--port", type=int, default=8080)
    dist_p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    dist_p.add_argument("--max-tokens", type=int, default=32000)
    dist_p.add_argument("--chat-template", default="", help="Override tokenizer chat template")
    dist_p.add_argument("--use-default-chat-template", action="store_true")
    dist_p.add_argument("--temperature", type=float, default=0.0)
    dist_p.add_argument("--top-p", type=float, default=1.0)
    dist_p.add_argument("--top-k", type=int, default=0)
    dist_p.add_argument(
        "--prompt-cache-size",
        type=int,
        default=2,
        help="Maximum number of prompt-cache entries to keep (LRU). Set to 0 to disable.",
    )
    dist_p.add_argument(
        "--batch",
        action="store_true",
        help="Enable server-side micro-batching across concurrent clients.",
    )
    dist_p.add_argument(
        "--batch-max-inflight",
        type=int,
        default=4,
        help="Maximum number of concurrent sequences in the active batch.",
    )
    dist_p.add_argument(
        "--batch-prefill-batch-size",
        type=int,
        default=2,
        help="Number of prompts to prefill together (lower uses less memory).",
    )
    dist_p.add_argument(
        "--batch-prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill steps (tokens).",
    )
    dist_p.add_argument(
        "--batch-steps-per-tick",
        type=int,
        default=1,
        help="Number of batch decode steps to run between request polls.",
    )
    dist_p.add_argument(
        "--batch-wait-ms",
        type=int,
        default=10,
        help="When starting a new batch, wait up to this long to gather more requests.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_default_wired_limit()

    if args.cmd == "serve":
        serve(args)
        return
    if args.cmd == "serve-distributed":
        serve_distributed(args)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")
