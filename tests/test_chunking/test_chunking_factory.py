from src.chunking.factory import ChunkerFactory
from src.chunking.markdown_chunker import MarkdownChunker
from src.chunking.python_chunker import PythonChunker


def test_python_extension_returns_python_chunker():
    chunker = ChunkerFactory.get_chunker(
        "example.py",
        max_chunk_size=100,
    )

    assert isinstance(chunker, PythonChunker)


def test_markdown_extension_returns_markdown_chunker():
    chunker = ChunkerFactory.get_chunker(
        "example.md",
        max_chunk_size=100,
    )

    assert isinstance(chunker, MarkdownChunker)


def test_markdown_extension_is_case_insensitive():
    chunker = ChunkerFactory.get_chunker(
        "README.MD",
        max_chunk_size=100,
    )

    assert isinstance(chunker, MarkdownChunker)


def test_text_extension_returns_markdown_chunker():
    chunker = ChunkerFactory.get_chunker(
        "example.txt",
        max_chunk_size=100,
    )

    assert isinstance(chunker, MarkdownChunker)


def test_unknown_extension_uses_markdown_chunker():
    chunker = ChunkerFactory.get_chunker(
        "example.xyz",
        max_chunk_size=100,
    )

    assert isinstance(chunker, MarkdownChunker)
