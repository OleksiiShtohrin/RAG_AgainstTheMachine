from src.chunking.base import Chunk
from src.indexing.bm25_index import BM25Index
from src.retrieval.lexical_retriever import LexicalRetriever


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            content=(
                "def check_wheel_size(directory):\n"
                "    Check the size of wheel files.\n"
                "    return wheel_size\n"
            ),
            file_path="data/raw/example.py",
            first_character_index=0,
            last_character_index=100,
        ),
        Chunk(
            content=(
                "class Database:\n"
                "    def connect(self):\n"
                "        return connection\n"
            ),
            file_path="data/raw/database.py",
            first_character_index=0,
            last_character_index=90,
        ),
        Chunk(
            content=(
                "The documentation explains how users "
                "configure the application.\n"
            ),
            file_path="data/raw/README.md",
            first_character_index=0,
            last_character_index=80,
        ),
    ]


def test_build_index() -> None:
    index = BM25Index.build(make_chunks())

    assert len(index.chunks) == 3


def test_exact_query_returns_matching_chunk() -> None:
    index = BM25Index.build(make_chunks())

    results = index.score_query("check_wheel_size")

    assert results
    assert results[0][0] == 0


def test_related_query_returns_matching_chunk() -> None:
    index = BM25Index.build(make_chunks())

    results = index.score_query("wheel size")

    assert results
    assert results[0][0] == 0


def test_unrelated_query_returns_no_results() -> None:
    index = BM25Index.build(make_chunks())

    results = index.score_query(
        "quantum banana spaceship"
    )

    assert results == []


def test_empty_query_returns_no_results() -> None:
    index = BM25Index.build(make_chunks())

    assert index.score_query("") == []
    assert index.score_query("   ") == []


def test_scores_are_sorted_descending() -> None:
    index = BM25Index.build(make_chunks())

    results = index.score_query("wheel")

    scores = [score for _, score in results]

    assert scores == sorted(scores, reverse=True)


def test_empty_index() -> None:
    index = BM25Index.build([])

    assert index.chunks == []
    assert index.score_query("wheel size") == []


def test_lexical_retriever_returns_top_k() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    results = retriever.retrieve(
        query="wheel",
        k=1,
    )

    assert len(results) == 1


def test_lexical_retriever_returns_metadata() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    results = retriever.retrieve(
        query="wheel size",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.file_path == "data/raw/example.py"
    assert result.first_character_index == 0
    assert result.last_character_index == 100


def test_lexical_retriever_respects_k() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    results = retriever.retrieve(
        query="wheel",
        k=5,
    )

    assert len(results) <= 5


def test_lexical_retriever_zero_k() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    assert retriever.retrieve("wheel", k=0) == []


def test_lexical_retriever_negative_k() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    assert retriever.retrieve("wheel", k=-1) == []


def test_lexical_retriever_empty_query() -> None:
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []
