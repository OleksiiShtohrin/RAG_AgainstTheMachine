"""Abstract base class for retrieval mechanisms."""

from abc import ABC, abstractmethod
from typing import List
from src.models.source import MinimalSource


class BaseRetriever(ABC):
    """Abstract interface for all document retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Retrieve top-k relevant MinimalSource snippets for a given query.

        Args:
            query: The user or dataset question text.
            k: Number of top results to return.

        Returns:
            List of top-k MinimalSource citations.
        """
        pass
