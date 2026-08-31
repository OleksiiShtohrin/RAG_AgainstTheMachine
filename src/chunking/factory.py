"""Factory to resolve appropriate chunkers based on file extensions."""

import os
from typing import Dict, Type
from src.chunking.base import BaseChunker
from src.chunking.markdown_chunker import MarkdownChunker
from src.chunking.python_chunker import PythonChunker


class ChunkerFactory:
    """Selects and instantiates the correct chunker strategy for a given file."""

    _EXTENSION_MAP: Dict[str, Type[BaseChunker]] = {
        ".py": PythonChunker,
        ".md": MarkdownChunker,
        ".markdown": MarkdownChunker,
        ".txt": MarkdownChunker,
        ".rst": MarkdownChunker,
    }

    @classmethod
    def get_chunker(
        cls, file_path: str, max_chunk_size: int = 2000
    ) -> BaseChunker:
        """Get the matching chunker instance for a file.

        Args:
            file_path: Path to the target file.
            max_chunk_size: Maximum character limit per chunk.

        Returns:
            An instantiated BaseChunker strategy.
        """
        _, ext = os.path.splitext(file_path.lower())
        chunker_class = cls._EXTENSION_MAP.get(ext, MarkdownChunker)
        return chunker_class(max_chunk_size=max_chunk_size)
