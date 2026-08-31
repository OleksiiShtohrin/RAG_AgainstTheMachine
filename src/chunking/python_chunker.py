"""Python source code chunking strategy."""

from typing import List
from src.chunking.base import BaseChunker, Chunk
from src.chunking.markdown_chunker import MarkdownChunker


class PythonChunker(BaseChunker):
    """Chunks Python files with windowing fallback."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: int = 800,
        overlap: int = 150,
    ) -> None:
        """Initialize python chunker."""
        super().__init__(max_chunk_size=max_chunk_size)
        self.delegate = MarkdownChunker(
            max_chunk_size=max_chunk_size,
            target_chunk_size=target_chunk_size,
            overlap=overlap,
        )

    def chunk(self, file_path: str, content: str) -> List[Chunk]:
        """Chunk Python files using overlapping structural blocks."""
        return self.delegate.chunk(file_path, content)
