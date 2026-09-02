import pytest

from src.retrieval.base import BaseRetriever


def test_base_retriever_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseRetriever()
