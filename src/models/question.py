"""Data models for unanswered and answered questions and datasets."""

import uuid
from typing import List, Union
from pydantic import BaseModel, Field
from src.models.source import MinimalSource


class UnansweredQuestion(BaseModel):
    """Represents a query without an answer.

    Attributes:
        question_id: Unique string identifier for the question.
        question: Query text asked by the user or dataset.
    """

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a question with ground-truth sources and an answer.

    Attributes:
        sources: Ground-truth reference source citations.
        answer: Expected reference answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a collection of RAG questions.

    Attributes:
        rag_questions: List of answered or unanswered question objects.
    """

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]
