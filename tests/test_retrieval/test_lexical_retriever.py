from src.indexing.bm25_index import BM25Index
from src.retrieval.lexical_retriever import LexicalRetriever

from tests.test_indexing.test_bm25_retrieval import make_chunks


def test_lexical_retriever_empty_query_returns_empty():
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_lexical_retriever_non_positive_k_returns_empty():
    index = BM25Index.build(make_chunks())
    retriever = LexicalRetriever(index)

    assert retriever.retrieve("wheel", k=0) == []
    assert retriever.retrieve("wheel", k=-1) == []
