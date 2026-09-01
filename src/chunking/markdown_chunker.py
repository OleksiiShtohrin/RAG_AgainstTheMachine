"""Markdown and generic text chunking strategy with semantic sliding window."""

from typing import List

from src.chunking.base import BaseChunker, Chunk


class MarkdownChunker(BaseChunker):
    """Chunks Markdown and text files with overlap for high retrieval recall."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: int = 800,
        overlap: int = 150,
    ) -> None:
        """Initialize with max limit and target window."""
        super().__init__(
            max_chunk_size=max_chunk_size,
        )

        if target_chunk_size <= 0:
            raise ValueError(
                "target_chunk_size must be positive"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        self.target_chunk_size = min(
            target_chunk_size,
            self.max_chunk_size,
        )

        self.overlap = min(
            overlap,
            self.target_chunk_size - 1,
        )

    def chunk(
        self,
        file_path: str,
        content: str,
    ) -> List[Chunk]:
        """Split text content into overlapping chunks within max_chunk_size."""
        if not content.strip():
            return []

        chunks: List[Chunk] = []

        text_len = len(content)
        start = 0

        while start < text_len:
            end = min(
                start + self.target_chunk_size,
                text_len,
            )

            # Try to break at paragraph or newline boundary.
            if end < text_len:
                last_p = content.rfind(
                    "\n\n",
                    start,
                    end,
                )

                if last_p > start + (
                    self.target_chunk_size // 2
                ):
                    end = last_p + 2
                else:
                    last_nl = content.rfind(
                        "\n",
                        start,
                        end,
                    )

                    if last_nl > start + (
                        self.target_chunk_size // 2
                    ):
                        end = last_nl + 1

            chunk_len = end - start

            if chunk_len > self.max_chunk_size:
                end = start + self.max_chunk_size

            chunk_text = content[start:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end,
                    )
                )

            if end >= text_len:
                break

            step = max(
                1,
                (end - start) - self.overlap,
            )

            start += step

        return chunks
