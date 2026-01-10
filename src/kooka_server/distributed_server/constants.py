from __future__ import annotations

# Default decoding controls (aligned with single-machine `serve` defaults).
DEFAULT_REPETITION_PENALTY = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20

# Maximum prompt length for broadcasting (in tokens).
MAX_PROMPT_LENGTH = 131072

# Stop sequence broadcast limits.
MAX_STOP_SEQUENCES = 8
MAX_STOP_SEQUENCE_LENGTH = 256

__all__ = [
    "DEFAULT_REPETITION_PENALTY",
    "DEFAULT_REPETITION_CONTEXT_SIZE",
    "MAX_PROMPT_LENGTH",
    "MAX_STOP_SEQUENCES",
    "MAX_STOP_SEQUENCE_LENGTH",
]
