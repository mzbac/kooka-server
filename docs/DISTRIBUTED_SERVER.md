# Distributed HTTP Server (Pipeline Parallel, MiniMax)

`kooka-server` ships a distributed, pipeline-parallel HTTP server intended for MiniMax models.

- Command (via `uvx`): `UV_PRERELEASE=allow uvx kooka-server serve-distributed ...`
- Implementation: `src/kooka_server/distributed.py`

Important:
- Launch via `mlx.launch` so all ranks participate in the generation loop.
- For multi-machine setups, prefer `env ...` as the launched command so `uvx` is resolved on each node via `PATH`.
- When running via `uvx`, you currently need to allow pre-releases because `mlx-lm==0.30.2` depends on `transformers==5.0.0rc1`:
  - use `uvx --prerelease=allow ...` or set `UV_PRERELEASE=allow` (recommended for `mlx.launch`).

## Requirements

- macOS Apple Silicon (arm64) recommended (MLX wheels are platform-specific).
- `uv` installed and on `PATH` on every machine you plan to run ranks on.
- `mlx.launch` available (installed alongside `mlx` / `mlx-lm` dependencies).

## Multi-Machine (ring backend)

### hosts.json

Create a hostfile (example):

```json
[
  {"ssh": "localhost", "ips": ["<HOST1_IP>"]},
  {"ssh": "user@<HOST2_IP>", "ips": ["<HOST2_IP>"]}
]
```

### Launch

For MiniMax pipeline parallelism, set `MINIMAX_PIPELINE_SPLIT` so it sums to the model's `num_hidden_layers` (MiniMax-M2.1 uses 62).

This example runs `kooka-server` via `uvx` on every rank; make sure `uv` is available on every node.

```bash
mlx.launch --backend ring --hostfile hosts.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env UV_PRERELEASE=allow \
  --env MINIMAX_PIPELINE_SPLIT=25,37 \
  -- uvx kooka-server serve-distributed \
    --model mlx-community/MiniMax-M2.1-4bit \
    --host 0.0.0.0 --port 8080 \
    --temperature 1.0 --top-p 0.95 --top-k 40
```

## API Endpoints

- `POST /v1/chat/completions` (OpenAI-compatible)
- `POST /chat/completions` (OpenAI-compatible alias)
- `POST /v1/completions`
- `POST /v1/messages` (Anthropic-compatible)
- `GET /v1/models`
- `GET /health`
