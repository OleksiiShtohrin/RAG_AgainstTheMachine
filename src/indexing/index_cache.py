"""In-memory cache for indexing artifacts to avoid redundant disk I/O."""

from typing import Any


class IndexCache:
    """In-memory cache for loaded indexes."""

    def __init__(self) -> None:
        """Initialize an empty index cache."""
        self._cache: dict[str, Any] = {}

    def get(self, index_dir: str) -> Any | None:
        """Return cached index for a directory, if available.

        Args:
            index_dir: Target directory path used as the cache key.

        Returns:
            The cached index instance or None if not found.
        """
        return self._cache.get(index_dir)

    def set(self, index_dir: str, index: Any) -> None:
        """Store an index instance associated with a directory.

        Args:
            index_dir: Target directory path.
            index: Loaded index object to store in memory.
        """
        self._cache[index_dir] = index
