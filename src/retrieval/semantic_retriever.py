"""Semantic vector retriever implementation."""

from typing import List
from src.indexing.semantic_index import SemanticIndex
from src.models.source import MinimalSource
from src.retrieval.base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    """Retrieves document chunks using vector embedding similarity."""

    def __init__(self, semantic_index: SemanticIndex) -> None:
        """Initialize semantic retriever with a loaded SemanticIndex.

        Args:
            semantic_index: Vector index containing embeddings and chunks.
        """
        self.semantic_index = semantic_index

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k source locations based on semantic similarity.

        Args:
            query: The user query string.
            k: Number of most relevant candidates to return.

        Returns:
            List of top-k MinimalSource objects covering matching spans.
        """
        if k <= 0 or not query.strip():
            return []

        ranked_docs = self.semantic_index.score_query(query, top_k=k)
        results: List[MinimalSource] = []
        for doc_id, _score in ranked_docs:
            chunk = self.semantic_index.chunks[doc_id]
            results.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        return results
