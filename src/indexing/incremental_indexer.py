"""Incremental indexer supporting delta updates without full rebuilds."""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Set

from src.chunking.base import Chunk
from src.chunking.factory import ChunkerFactory
from src.indexing.bm25_index import BM25Index
from src.indexing.indexer import CorpusIndexer
from src.ingestion.corpus_reader import CorpusReader


class IncrementalIndexer:
    """Manages incremental delta indexing for modified files."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialize incremental indexer with maximum chunk size.

        Args:
            max_chunk_size: Maximum allowed characters per chunk.
        """
        self.max_chunk_size = max_chunk_size
        self.base_indexer = CorpusIndexer(max_chunk_size=max_chunk_size)

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        """Compute SHA-256 hash of a file on disk.

        Args:
            filepath: Path to the target file.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _collect_files(self, raw_dir: str) -> list[str]:
        """Collect supported corpus files from the raw directory.

        Args:
            raw_dir: Root path of raw corpus files.

        Returns:
            List of resolved string file paths.
        """
        reader = CorpusReader(raw_dir)
        return [str(path) for path in reader._collect_files()]

    def _process_file(self, filepath: str) -> list[Chunk]:
        """Read and chunk one individual corpus file.

        Args:
            filepath: Absolute or relative file path to read.

        Returns:
            List of generated Chunk objects for this file.
        """
        path = Path(filepath)
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        reader = CorpusReader(str(path.parent))
        file_path = reader._relative_path(path)

        chunker = ChunkerFactory.get_chunker(
            file_path,
            max_chunk_size=self.max_chunk_size,
        )

        return chunker.chunk(file_path, content)

    def update_index(
        self, raw_dir: str = "data/raw", output_dir: str = "data/processed"
    ) -> BM25Index:
        """Incrementally update BM25 index by detecting file modifications.

        Args:
            raw_dir: Path to directory of source files.
            output_dir: Directory where index and manifest are stored.

        Returns:
            Updated BM25Index instance.
        """
        manifest_path = os.path.join(output_dir, "manifest.json")
        index_path = os.path.join(output_dir, "bm25_index.pkl")

        current_files = self._collect_files(raw_dir)
        old_manifest: Dict[str, str] = {}

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)

        new_manifest: Dict[str, str] = {}
        changed_files: Set[str] = set()

        for fpath in current_files:
            fhash = self._compute_file_hash(fpath)
            new_manifest[fpath] = fhash
            if fpath not in old_manifest or old_manifest[fpath] != fhash:
                changed_files.add(fpath)

        deleted_files = set(old_manifest.keys()) - set(new_manifest.keys())

        if (
            not changed_files
            and not deleted_files
            and os.path.exists(index_path)
        ):
            print("Incremental: No files changed. Using existing index.")
            return CorpusIndexer.load_index(output_dir)

        print(
            f"Incremental: {len(changed_files)} changed, "
            f"{len(deleted_files)} deleted."
        )

        existing_chunks: List[Chunk] = []
        if os.path.exists(index_path):
            old_index = CorpusIndexer.load_index(output_dir)
            existing_chunks = [
                c
                for c in old_index.chunks
                if c.file_path not in changed_files
                and c.file_path not in deleted_files
            ]

        new_chunks: List[Chunk] = []
        for fpath in changed_files:
            new_chunks.extend(self._process_file(fpath))

        all_chunks = existing_chunks + new_chunks
        updated_index = BM25Index.build(all_chunks)

        os.makedirs(output_dir, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(updated_index, f)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(new_manifest, f, indent=2)

        return updated_index
