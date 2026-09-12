"""Hybrid retriever fusing lexical and semantic rankings via RRF."""

from typing import Dict, List
from src.indexing.bm25_index import BM25Index
from src.indexing.semantic_index import SemanticIndex
from src.models.source import MinimalSource
from src.retrieval.base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Combines BM25 and vector semantic search via Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_index: BM25Index,
        semantic_index: SemanticIndex,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever with both index instances.

        Args:
            bm25_index: Pre-built lexical BM25 index.
            semantic_index: Pre-built dense vector semantic index.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (default: 60).
        """
        self.bm25_index = bm25_index
        self.semantic_index = semantic_index
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k sources using fused rankings.

        Args:
            query: Search query text.
            k: Number of top fused candidates to return.

        Returns:
            List of top-k MinimalSource instances ranked by RRF score.
        """
        if k <= 0 or not query.strip():
            return []

        # 1. Lexical ranking (top-50)
        bm25_hits = self.bm25_index.score_query(query)[:50]
        # 2. Semantic ranking (top-50)
        semantic_hits = self.semantic_index.score_query(query, top_k=50)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_id, _score) in enumerate(bm25_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, (doc_id, _score) in enumerate(semantic_hits, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        sorted_docs = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        results: List[MinimalSource] = []
        for doc_id, _ in sorted_docs:
            chunk = self.bm25_index.chunks[doc_id]
            results.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        return results
