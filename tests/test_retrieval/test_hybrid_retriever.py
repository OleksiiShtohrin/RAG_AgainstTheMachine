from src.chunking.base import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever


class FakeBM25Index:
    """Fake BM25 index for hybrid retriever tests."""

    def __init__(self) -> None:
        self.chunks = [
            Chunk(
                content="first chunk",
                file_path="test.py",
                first_character_index=0,
                last_character_index=11,
            ),
            Chunk(
                content="second chunk",
                file_path="test.py",
                first_character_index=12,
                last_character_index=24,
            ),
        ]

    def score_query(
        self,
        query: str,
    ) -> list[tuple[int, float]]:
        return [
            (0, 10.0),
            (1, 5.0),
        ]


class FakeSemanticIndex:
    """Fake semantic index for hybrid retriever tests."""

    def score_query(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        return [
            (0, 0.9),
            (1, 0.8),
        ]


def test_retrieve_fuses_both_rankings() -> None:
    retriever = HybridRetriever(
        FakeBM25Index(),  # type: ignore[arg-type]
        FakeSemanticIndex(),  # type: ignore[arg-type]
    )

    results = retriever.retrieve("test query", k=2)

    assert len(results) == 2
    assert results[0].file_path == "test.py"
    assert results[0].first_character_index == 0
    assert results[0].last_character_index == 11


def test_retrieve_uses_ranking_not_raw_scores() -> None:
    class BM25Index(FakeBM25Index):
        def score_query(
            self,
            query: str,
        ) -> list[tuple[int, float]]:
            return [
                (0, 1.0),
                (1, 1000.0),
            ]

    class SemanticIndex(FakeSemanticIndex):
        def score_query(
            self,
            query: str,
            top_k: int,
        ) -> list[tuple[int, float]]:
            return [
                (0, 0.9),
                (1, 0.1),
            ]

    retriever = HybridRetriever(
        BM25Index(),  # type: ignore[arg-type]
        SemanticIndex(),  # type: ignore[arg-type]
    )

    results = retriever.retrieve("test query", k=1)

    assert len(results) == 1
    assert results[0].first_character_index == 0


def test_retrieve_empty_query() -> None:
    retriever = HybridRetriever(
        FakeBM25Index(),  # type: ignore[arg-type]
        FakeSemanticIndex(),  # type: ignore[arg-type]
    )

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retrieve_non_positive_k() -> None:
    retriever = HybridRetriever(
        FakeBM25Index(),  # type: ignore[arg-type]
        FakeSemanticIndex(),  # type: ignore[arg-type]
    )

    assert retriever.retrieve("test query", k=0) == []
    assert retriever.retrieve("test query", k=-1) == []


def test_retrieve_includes_results_from_one_ranking() -> None:
    class BM25Index(FakeBM25Index):
        def score_query(
            self,
            query: str,
        ) -> list[tuple[int, float]]:
            return [(0, 10.0)]

    class SemanticIndex(FakeSemanticIndex):
        def score_query(
            self,
            query: str,
            top_k: int,
        ) -> list[tuple[int, float]]:
            return [(1, 0.9)]

    retriever = HybridRetriever(
        BM25Index(),  # type: ignore[arg-type]
        SemanticIndex(),  # type: ignore[arg-type]
    )

    results = retriever.retrieve("test query", k=2)

    assert len(results) == 2
    assert results[0].first_character_index == 0
    assert results[1].first_character_index == 12
