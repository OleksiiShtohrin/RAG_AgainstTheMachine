"""Data models for unanswered and answered questions and datasets."""

from typing import List, Union
from pydantic import BaseModel
from src.models.source import MinimalSource


class UnansweredQuestion(BaseModel):
    """Represents a query without an answer."""

    question_id: str
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with ground-truth sources and an answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a collection of RAG questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]
