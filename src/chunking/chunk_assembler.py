"""Chunk assembler module for slicing text ranges into bounded chunks."""

from src.chunking.base import Chunk


class ChunkAssembler:
    """Assembles text segments into overlapping chunks within size limits."""

    def __init__(
        self,
        max_chunk_size: int,
        target_chunk_size: int,
        overlap: int,
    ) -> None:
        """Initialize ChunkAssembler with size boundaries.

        Args:
            max_chunk_size: Absolute maximum character limit per chunk.
            target_chunk_size: Target slice size.
            overlap: Number of characters to overlap between slices.
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

        if target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        chunk_size = min(
            target_chunk_size,
            max_chunk_size,
        )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk size"
            )

        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def assemble(
        self,
        file_path: str,
        content: str,
        start: int,
        end: int,
    ) -> list[Chunk]:
        """Slice a text span into a list of Chunk objects.

        Args:
            file_path: Origin relative file path.
            content: Complete file content string.
            start: Start character offset of the segment.
            end: End character offset of the segment.

        Returns:
            List of generated Chunk instances.
        """
        chunk_size = min(
            self.target_chunk_size,
            self.max_chunk_size,
        )

        if self.overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk size"
            )

        if end - start <= chunk_size:
            return [
                Chunk(
                    content=content[start:end],
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                )
            ]

        chunks = []

        step = chunk_size - self.overlap
        current_start = start

        while current_start < end:
            current_end = min(
                current_start + chunk_size,
                end,
            )

            chunks.append(
                Chunk(
                    content=content[current_start:current_end],
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=current_end,
                )
            )

            if current_end == end:
                break

            current_start += step

        return chunks
