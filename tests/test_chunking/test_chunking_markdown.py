from src.chunking.markdown_chunker import MarkdownChunker


def test_markdown_chunker_creates_chunks():
    content = "# Title\n\nThis is some text."

    chunker = MarkdownChunker(max_chunk_size=100)

    chunks = list(
        chunker.chunk(
            content=content,
            file_path="test.md",
        )
    )

    assert chunks
    assert all(chunk.file_path == "test.md" for chunk in chunks)
    assert all(chunk.content for chunk in chunks)


def test_markdown_chunker_preserves_offsets():
    content = "# Title\n\nHello world."

    chunker = MarkdownChunker(max_chunk_size=100)

    chunks = list(
        chunker.chunk(
            content=content,
            file_path="test.md",
        )
    )

    for chunk in chunks:
        assert content[
            chunk.first_character_index:chunk.last_character_index
        ] == chunk.content


def test_markdown_chunker_respects_max_chunk_size():
    content = (
        "# Title\n\n"
        "This is a long piece of text that should "
        "be split into several chunks."
    )

    chunker = MarkdownChunker(max_chunk_size=30)

    chunks = list(
        chunker.chunk(
            content=content,
            file_path="test.md",
        )
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 30 for chunk in chunks)


def test_markdown_chunker_applies_overlap():
    content = (
        "This is the first part of the text. "
        "This is the second part of the text. "
        "This is the third part of the text."
    )

    chunker = MarkdownChunker(
        max_chunk_size=50,
        overlap=10,
    )

    chunks = list(
        chunker.chunk(
            content=content,
            file_path="test.md",
        )
    )

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.content[-10:] == current.content[:10]
