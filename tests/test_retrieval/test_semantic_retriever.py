from src.chunking.base import Chunk
from src.retrieval.semantic_retriever import SemanticRetriever


class FakeSemanticIndex:
    """Fake semantic index for retriever tests."""

    def __init__(self) -> None:
        self.chunks = [
            Chunk(
                content="first chunk",
                file_path="test.py",
                first_character_index=10,
                last_character_index=21,
            ),
        ]

    def score_query(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        return [(0, 0.95)]


class TrackingSemanticIndex(FakeSemanticIndex):
    """Fake index that records the requested top_k."""

    def __init__(self) -> None:
        super().__init__()
        self.received_top_k = None

    def score_query(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        self.received_top_k = top_k
        return [(0, 0.95)]


def test_retrieve_returns_sources() -> None:
    index = FakeSemanticIndex()
    retriever = SemanticRetriever(index)  # type: ignore[arg-type]

    results = retriever.retrieve("test query", k=1)

    assert len(results) == 1
    assert results[0].file_path == "test.py"
    assert results[0].first_character_index == 10
    assert results[0].last_character_index == 21


def test_retrieve_empty_query() -> None:
    retriever = SemanticRetriever(FakeSemanticIndex())  # type: ignore[arg-type]

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retrieve_non_positive_k() -> None:
    retriever = SemanticRetriever(FakeSemanticIndex())  # type: ignore[arg-type]

    assert retriever.retrieve("test query", k=0) == []
    assert retriever.retrieve("test query", k=-1) == []


def test_retrieve_passes_k_to_index() -> None:
    index = TrackingSemanticIndex()
    retriever = SemanticRetriever(index)  # type: ignore[arg-type]

    retriever.retrieve("test query", k=7)

    assert index.received_top_k == 7
