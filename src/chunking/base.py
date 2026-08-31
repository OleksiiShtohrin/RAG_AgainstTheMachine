"""Base interfaces and data structures for document chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """Represents a text chunk with its exact location in the source file."""

    content: str
    file_path: str
    first_character_index: int
    last_character_index: int

    def __post_init__(self) -> None:
        """Validate character indices and length."""
        if self.first_character_index < 0:
            raise ValueError("first_character_index cannot be negative.")
        if self.last_character_index < self.first_character_index:
            raise ValueError("last_character_index cannot be less than first.")


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the chunker with a maximum character size limit.

        Args:
            max_chunk_size: Maximum allowed characters per chunk (<= 2000).
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive.")
        if max_chunk_size > 2000:
            raise ValueError("max_chunk_size cannot exceed 2000 characters.")
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, file_path: str, content: str) -> List[Chunk]:
        """Split file content into a list of Chunks.

        Args:
            file_path: Relative path to the source file.
            content: Raw text content of the file.

        Returns:
            List of Chunk objects with accurate character indices.
        """
        pass
