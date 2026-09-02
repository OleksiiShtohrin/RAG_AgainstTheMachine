"""Lexical retriever implementation using BM25."""

from typing import List
from src.indexing.bm25_index import BM25Index
from src.models.source import MinimalSource
from src.retrieval.base import BaseRetriever


class LexicalRetriever(BaseRetriever):
    """Retriever based on BM25 lexical ranking."""

    def __init__(self, index: BM25Index) -> None:
        """Initialize with an in-memory BM25 index.

        Args:
            index: Loaded BM25Index instance.
        """
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k source locations for the query.

        Args:
            query: Question text.
            k: Number of results to return.

        Returns:
            List of MinimalSource items.
        """
        if k <= 0 or not query.strip():
            return []

        ranked_docs = self.index.score_query(query)
        top_k_indices = ranked_docs[:k]

        results: List[MinimalSource] = []
        for doc_id, _score in top_k_indices:
            chunk = self.index.chunks[doc_id]
            results.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        return results
