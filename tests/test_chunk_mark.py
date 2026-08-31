import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.chunking import MarkdownChunker

text = """
# Hello

This is a test document.

## Section

Some more text here.
"""

chunker = MarkdownChunker(
    max_chunk_size=50,
    target_chunk_size=30,
    overlap=5,
)

chunks = chunker.chunk(
    "data/raw/example.md",
    text,
)

for chunk in chunks:
    print(chunk)
    print("length:", len(chunk.content))
    print("---")