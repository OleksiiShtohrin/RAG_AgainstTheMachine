from typing import Any


class IndexCache:
    """In-memory cache for loaded indexes."""

    def __init__(self) -> None:
        """Initialize an empty index cache."""
        self._cache: dict[str, Any] = {}

    def get(self, index_dir: str) -> Any | None:
        """Return cached index for a directory, if available."""
        return self._cache.get(index_dir)

    def set(self, index_dir: str, index: Any) -> None:
        """Store an index for a directory."""
        self._cache[index_dir] = index
