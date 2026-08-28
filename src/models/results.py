"""Data models for search and answer results used in CLI and evaluation."""

from typing import List
from pydantic import BaseModel
from src.models.source import MinimalSource


class MinimalSearchResults(BaseModel):
    """Search result for a single question containing retrieved sources."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Answer result for a single question including LLM generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Batch output format for the search_dataset command."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch output format for the answer_dataset command."""

    search_results: List[MinimalAnswer]
    k: int
