# Repository Guidelines

## Project Structure & Module Organization
- `src/kooka_server/`: Python package (server + CLI).
  - `cli.py`: `kooka-server` entrypoint (arg parsing + subcommands).
  - `server.py`: single-machine HTTP server and request handling.
  - `distributed.py`: `mlx.launch`-based distributed server.
  - `api/`: request/response shaping helpers (keep `server.py` thin).
  - `hf_utils/`: Hugging Face cache + hub helpers (used by `/v1/models`).
  - `tool_fixes/`: model-aware tool-call fixes (pure, testable transforms).
  - `mlx_utils/`: MLX-LM integration seams (loaders, distributed-only patches).
- `docs/`: design notes and operational docs.
- `tests/`: pytest unit + contract tests.
- `mem/`: scratchpad for agent planning files (ignored by git).
- `scripts/`: helper scripts for local/dev ops.

## Build, Test, and Development Commands
- `python -m pip install -e ".[dev]"`: editable install for local development.
- `python -m kooka_server serve --model <hf_repo_or_path> --host 127.0.0.1 --port 8080`: run single-machine server.
- `mlx.launch -n 2 --env MLX_METAL_FAST_SYNCH=1 -- kooka-server serve-distributed --model <hf_repo_or_path> --port 8080`: run local multi-process distributed server.



## Coding Style & Naming Conventions
- Python: 4-space indentation; prefer type hints and small, focused functions.
- Names: `snake_case` for modules/functions, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants.
- Logging: use `logging` (not `print`); avoid `DEBUG` in production since request bodies may include sensitive text.
- Compatibility: keep OpenAI-compatible routes stable (e.g., `/v1/chat/completions`, `/v1/completions`, `/v1/models`).
- Comments/docstrings: keep them short and purpose-driven; avoid speculative or redundant commentary.

## MLX-LM Integration Guidelines
- Single-machine server should use upstream `mlx_lm.utils.load` (no model overrides).
- Distributed-only model quirks should be implemented as **small runtime patches** under `src/kooka_server/mlx_utils/`
  and invoked only by distributed loading code (e.g., before calling `mlx_lm.utils.sharded_load`).
- Avoid copying full upstream model implementations into `kooka-server`; prefer minimal patch layers.

## Testing Guidelines
- Contract tests (run against a running server):
  - `pytest -m contract --base-url http://127.0.0.1:8080`
  - Unit-only (no server required): `pytest -m unit`
  - Nemotron-focused tool-use contract (do not weaken the MiniMax contract):
    - `pytest -m nemotron --base-url http://127.0.0.1:8080 --model <nemotron-model-id>`
- When fixing API/serialization issues, add/extend checks under `tests/` to prevent regressions.
- Treat existing contract tests as “frozen”: if a specific model needs a different prompt/expectation, add a new
  model-specific smoke test rather than changing the baseline contract.

## Planning / Scratch Files
- If you use the planning-with-files (create plan) workflow, keep all generated planning/progress/notes files under `mem/`
  so they don’t pollute the repo (since `mem/` is git-ignored).

## Commit & Pull Request Guidelines
- Use clear, imperative commit summaries going forward (e.g., `Fix JSON validation for chat requests`).
- PRs should include: a short problem statement, repro steps (or `curl` examples), and doc updates in `docs/` when APIs/flags change.
