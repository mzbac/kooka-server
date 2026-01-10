# kooka-server

Local inference web server for MLX / `mlx-lm`.

Note: `mlx-lm==0.30.2` depends on `transformers==5.0.0rc1`, so `uvx` needs pre-releases enabled:

```bash
export UV_PRERELEASE=allow
```

## Quickstart (single machine)
```bash
uvx kooka-server serve --model mlx-community/MiniMax-M2.1-4bit --host 127.0.0.1 --port 8080
```

## Quickstart (distributed)
See `docs/DISTRIBUTED_SERVER.md`.
