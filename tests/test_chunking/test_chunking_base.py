import pytest

from src.chunking.markdown_chunker import MarkdownChunker


def test_max_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        MarkdownChunker(max_chunk_size=0)


def test_max_chunk_size_cannot_exceed_2000():
    with pytest.raises(ValueError):
        MarkdownChunker(max_chunk_size=2001)


def test_max_chunk_size_2000_is_valid():
    chunker = MarkdownChunker(max_chunk_size=2000)

    assert chunker.max_chunk_size == 2000
