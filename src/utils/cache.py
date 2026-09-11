"""Caching layer for search results and generated answers."""

import hashlib
import json
import os
from typing import Any, Optional


class QueryCache:
    """Disk-backed query and result cache."""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialize query cache directory."""
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_key(self, namespace: str, query: str, **kwargs: Any) -> str:
        """Generate deterministic cache key."""
        payload = f"{namespace}:{query}:{sorted(kwargs.items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, namespace: str, query: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve cached result if available."""
        key = self._get_key(namespace, query, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(
        self, namespace: str, query: str, data: Any, **kwargs: Any
    ) -> None:
        """Save result to cache."""
        key = self._get_key(namespace, query, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
