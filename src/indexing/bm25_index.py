"""BM25 index wrapper using rank_bm25."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm
from rank_bm25 import BM25Okapi

from src.chunking.base import Chunk
from src.indexing.tokenizer import CodeTokenizer


@dataclass
class BM25Index:
    """BM25 index wrapper holding chunks and model."""

    chunks: List[Chunk]
    bm25: Optional[BM25Okapi]

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "BM25Index":
        """Build a BM25 index from chunks."""
        if not chunks:
            return cls(
                chunks=[],
                bm25=None,
            )

        corpus_tokens: List[List[str]] = []

        for chunk in tqdm(
            chunks,
            desc="Tokenizing",
            unit="chunk",
        ):
            text = (
                f"{chunk.file_path} "
                f"{chunk.file_path} "
                f"{chunk.content}"
            )
            tokens = CodeTokenizer.tokenize(text)
            corpus_tokens.append(tokens)

        bm25 = BM25Okapi(corpus_tokens)

        return cls(
            chunks=chunks,
            bm25=bm25,
        )

    def score_query(
        self,
        query: str,
    ) -> List[Tuple[int, float]]:
        """Score all chunks against query and return sorted results."""
        query_tokens = CodeTokenizer.tokenize(query)

        if not query_tokens or not self.chunks or self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query_tokens)

        doc_scores = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score > 0
        ]

        return sorted(
            doc_scores,
            key=lambda x: x[1],
            reverse=True,
        )
