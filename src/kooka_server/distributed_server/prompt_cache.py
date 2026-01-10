from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, List

from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache


class LRUPromptCache:
    """LRU cache for prompt prefixes to speed up generation."""

    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        count: int

    @dataclass
    class SearchResult:
        model: Any
        exact: List[int]
        shorter: List[int]
        longer: List[int]
        common_prefix: int

    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache = {}
        self._lru = deque()

    def _search(self, model, tokens):
        """Search the cache for a prompt cache. Return exact or close match."""
        if model not in self._cache:
            return self.SearchResult(model, None, None, None, 0)

        current = self._cache[model]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return self.SearchResult(model, tokens, None, None, 0)

        shorter = None
        if last_cache_index > 0:
            shorter = tokens[: last_cache_index + 1]

        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "cache" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return self.SearchResult(model, None, shorter, longer, common_prefix)

    def _get(self, model, tokens):
        current = self._cache[model]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model, tokens):
        path = [self._cache[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for i in reversed(range(len(tokens))):
            d_prev, d, t = path[i], path[i + 1], tokens[i]
            if len(d) > 0:
                break
            del d_prev[t]

    def _extract(self, model, tokens):
        cache_entry = self._get(model, tokens)
        if cache_entry.count == 1:
            self._delete(model, tokens)
            self._lru.remove((model, tokens))
            return cache_entry

        cache_entry.count -= 1
        return self.CacheEntry(
            copy.deepcopy(cache_entry.prompt_cache),
            1,
        )

    def fetch_nearest_cache(self, model, tokens):
        result = self._search(model, tokens)
        if result.exact is not None:
            cache_entry = self._extract(result.model, result.exact)
            return cache_entry.prompt_cache, []

        if result.shorter is not None:
            cache_entry = self._extract(result.model, result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens[prefix_len:]

        if result.longer is not None:
            cache_entry = self._get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache_entry = self.CacheEntry(
                    copy.deepcopy(cache_entry.prompt_cache),
                    1,
                )
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache_entry.prompt_cache, num_to_trim)
                return cache_entry.prompt_cache, tokens[prefix:]

        return None, tokens

    def insert_cache(self, model, tokens, prompt_cache):
        if model not in self._cache:
            self._cache[model] = {}
        current = self._cache[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        if "cache" in current:
            current["cache"].count += 1
            self._lru.remove((model, tokens))
        else:
            current["cache"] = self.CacheEntry(prompt_cache, 1)

        self._lru.append((model, tokens))
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.popleft()
            self._delete(model, tokens)


__all__ = ["LRUPromptCache"]

