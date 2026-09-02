import os
import sys
import shutil
import tempfile

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.indexing.indexer import CorpusIndexer
from src.ingestion.corpus_reader import CorpusReader


TEST_CONTENT = """# Example

def hello():
    return "hello"

"""


with tempfile.TemporaryDirectory() as temp_dir:
    raw_dir = os.path.join(temp_dir, "raw")
    output_dir = os.path.join(temp_dir, "processed")

    os.makedirs(raw_dir)

    file_path = os.path.join(
        raw_dir,
        "example.py",
    )

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(TEST_CONTENT)

    reader = CorpusReader(raw_dir)

    documents = reader.read()

    print("DOCUMENTS:", len(documents))

    assert len(documents) == 1
    assert documents[0].content == TEST_CONTENT

    indexer = CorpusIndexer(
        max_chunk_size=100,
    )

    index = indexer.index_directory(
        raw_dir=raw_dir,
        output_dir=output_dir,
    )

    print("CHUNKS:", len(index.chunks))

    assert len(index.chunks) > 0

    for chunk in index.chunks:
        print(
            "CHUNK:",
            chunk.file_path,
            chunk.first_character_index,
            chunk.last_character_index,
        )

        assert chunk.content == TEST_CONTENT[
            chunk.first_character_index:
            chunk.last_character_index
        ]

        assert (
            chunk.last_character_index
            - chunk.first_character_index
            <= 100
        )

    index_file = os.path.join(
        output_dir,
        "bm25_index.pkl",
    )

    assert os.path.exists(index_file)

    loaded_index = CorpusIndexer.load_index(
        output_dir,
    )

    assert len(loaded_index.chunks) == len(index.chunks)

    print("INDEX PERSISTENCE: PASS")
    print("CORPUS READER → INDEXER: PASS")
