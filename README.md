# kooka-server

Production-focused local inference web server for MLX / `mlx-lm`.

## Quickstart (single machine)
```bash
# in a Python env with mlx + mlx-lm installed
kooka-server serve --model mlx-community/MiniMax-M2.1-4bit --host 127.0.0.1 --port 8080

# contract tests
pytest -m contract --base-url http://127.0.0.1:8080
```

## Quickstart (distributed)
```bash
# single-node (world_size=1) via mx.distributed
kooka-server serve-distributed --model mlx-community/MiniMax-M2.1-4bit --host 0.0.0.0 --port 8080

# local multi-process (world_size=2)
mlx.launch -n 2 --env MLX_METAL_FAST_SYNCH=1 -- \
  kooka-server serve-distributed --model mlx-community/MiniMax-M2.1-4bit --host 0.0.0.0 --port 8080

# contract tests (run against rank0)
pytest -m contract --base-url http://127.0.0.1:8080
```
