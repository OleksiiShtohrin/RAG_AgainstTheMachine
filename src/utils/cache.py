"""Caching layer for search results and generated answers."""

import hashlib
import json
import os
from typing import Any, Optional


class QueryCache:
    """Disk-backed query and result cache."""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialize query cache with a storage directory.

        Args:
            cache_dir: Directory path where cached responses are stored.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_key(self, namespace: str, query: str, **kwargs: Any) -> str:
        """Generate deterministic cache key from query parameters.

        Args:
            namespace: Context category for the key (e.g. 'search', 'answer').
            query: The input query string.
            **kwargs: Extra parameters influencing query results.

        Returns:
            Hexadecimal SHA-256 hash string serving as filename.
        """
        payload = f"{namespace}:{query}:{sorted(kwargs.items())}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, namespace: str, query: str, **kwargs: Any) -> Optional[Any]:
        """Retrieve cached result if available on disk.

        Args:
            namespace: Cache category identifier.
            query: The query text to look up.
            **kwargs: Additional parameters matching the cached state.

        Returns:
            Deserialized cached data if valid file exists, None otherwise.
        """
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
        """Save result to cache on disk.

        Args:
            namespace: Cache category identifier.
            query: The query text being saved.
            data: JSON-serializable payload to store.
            **kwargs: Associated parameters to form the unique key.
        """
        key = self._get_key(namespace, query, **kwargs)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
