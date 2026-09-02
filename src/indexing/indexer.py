"""Corpus indexer for chunking documents and building a BM25 index."""

import os
import pickle

from tqdm import tqdm

from src.chunking.base import Chunk
from src.chunking.factory import ChunkerFactory
from src.indexing.bm25_index import BM25Index
from src.ingestion.corpus_reader import CorpusReader


class CorpusIndexer:
    """Manage the end-to-end indexing lifecycle of a codebase."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize the indexer with a maximum chunk size."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive.")

        if max_chunk_size > 2000:
            raise ValueError(
                "max_chunk_size cannot exceed 2000 characters."
            )

        self.max_chunk_size = max_chunk_size

    def index_directory(
        self,
        raw_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ) -> BM25Index:
        """Read, chunk, index, and persist the corpus."""
        reader = CorpusReader(raw_dir)

        all_chunks: list[Chunk] = []

        documents = list(reader.iter_documents())

        for document in tqdm(
            documents,
            desc="Chunking",
            unit="file",
        ):
            chunker = ChunkerFactory.get_chunker(
                document.file_path,
                max_chunk_size=self.max_chunk_size,
            )

            chunks = chunker.chunk(
                document.file_path,
                document.content,
            )

            all_chunks.extend(chunks)

        index = BM25Index.build(all_chunks)

        os.makedirs(output_dir, exist_ok=True)

        index_file = os.path.join(
            output_dir,
            "bm25_index.pkl",
        )

        with open(index_file, "wb") as file:
            pickle.dump(index, file)

        return index

    @classmethod
    def load_index(
        cls,
        output_dir: str = "data/processed",
    ) -> BM25Index:
        """Load a persisted BM25 index from disk."""
        index_file = os.path.join(
            output_dir,
            "bm25_index.pkl",
        )

        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"Index file not found at: {index_file}"
            )

        with open(index_file, "rb") as file:
            index: BM25Index = pickle.load(file)

        return index
