import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    UnansweredQuestion,
    RagDataset,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def test_minimal_source_creation():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=50,
    )

    assert source.file_path == "src/main.py"
    assert source.first_character_index == 10
    assert source.last_character_index == 50


def test_minimal_source_requires_all_fields():
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="src/main.py",
            first_character_index=10,
        )


def test_unanswered_question_generates_question_id():
    question = UnansweredQuestion(
        question="What is Python?"
    )

    assert question.question
    assert question.question_id
    assert isinstance(question.question_id, str)


def test_unanswered_questions_get_different_ids():
    first = UnansweredQuestion(question="Question 1")
    second = UnansweredQuestion(question="Question 2")

    assert first.question_id != second.question_id


def test_answered_question_inherits_unanswered_question():
    question = AnsweredQuestion(
        question="What is Python?",
        answer="A programming language.",
        sources=[],
    )

    assert question.question == "What is Python?"
    assert question.answer == "A programming language."
    assert question.question_id
    assert isinstance(question, UnansweredQuestion)


def test_answered_question_accepts_sources():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=20,
    )

    question = AnsweredQuestion(
        question="What is Python?",
        answer="A programming language.",
        sources=[source],
    )

    assert len(question.sources) == 1
    assert question.sources[0].file_path == "src/main.py"
    assert question.sources[0].first_character_index == 0
    assert question.sources[0].last_character_index == 20


def test_minimal_source_json_serialization():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
    )

    data = source.model_dump()

    assert data == {
        "file_path": "src/main.py",
        "first_character_index": 10,
        "last_character_index": 30,
    }


def test_rag_dataset_accepts_questions():
    unanswered = UnansweredQuestion(
        question="What is Python?"
    )

    answered = AnsweredQuestion(
        question="What is RAG?",
        answer="Retrieval-Augmented Generation.",
        sources=[],
    )

    dataset = RagDataset(
        rag_questions=[unanswered, answered]
    )

    assert len(dataset.rag_questions) == 2
    assert dataset.rag_questions[0].question == "What is Python?"
    assert dataset.rag_questions[1].question == "What is RAG?"


def test_minimal_search_results_accepts_sources():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    result = MinimalSearchResults(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
    )

    assert result.question_id == "test-123"
    assert result.question == "What is Python?"
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0].file_path == "src/main.py"
    assert result.retrieved_sources[0].first_character_index == 0
    assert result.retrieved_sources[0].last_character_index == 50


def test_minimal_answer_creation():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    result = MinimalAnswer(
        question_id="test-123",
        question="What is Python?",
        answer="Python is a programming language.",
        retrieved_sources=[source],
    )

    assert result.question_id == "test-123"
    assert result.question == "What is Python?"
    assert result.answer == "Python is a programming language."
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0].file_path == "src/main.py"


def test_minimal_answer_inherits_search_results():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    result = MinimalAnswer(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
        answer="Python is a programming language.",
    )

    assert isinstance(result, MinimalSearchResults)
    assert result.question_id == "test-123"
    assert result.question == "What is Python?"
    assert result.answer == "Python is a programming language."
    assert result.retrieved_sources == [source]


def test_student_search_results_accepts_search_results():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    search_result = MinimalSearchResults(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
    )

    result = StudentSearchResults(
        search_results=[search_result],
        k=5,
    )

    assert result.k == 5
    assert len(result.search_results) == 1
    assert result.search_results[0].question_id == "test-123"
    assert result.search_results[0].question == "What is Python?"
    assert result.search_results[0].retrieved_sources == [source]


def test_student_search_results_and_answer_accepts_answers():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    answer = MinimalAnswer(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
        answer="Python is a programming language.",
    )

    result = StudentSearchResultsAndAnswer(
        search_results=[answer],
        k=5,
    )

    assert result.k == 5
    assert len(result.search_results) == 1
    assert result.search_results[0].question_id == "test-123"
    assert result.search_results[0].question == "What is Python?"
    assert result.search_results[0].answer == (
        "Python is a programming language."
    )
    assert result.search_results[0].retrieved_sources == [source]


def test_minimal_search_results_requires_question():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    with pytest.raises(ValueError):
        MinimalSearchResults(
            question_id="test-123",
            retrieved_sources=[source],
        )


def test_minimal_source_requires_file_path():
    with pytest.raises(ValueError):
        MinimalSource(
            first_character_index=0,
            last_character_index=10,
        )


def test_student_search_results_requires_k():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=0,
        last_character_index=50,
    )

    search_result = MinimalSearchResults(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
    )

    with pytest.raises(ValueError):
        StudentSearchResults(
            search_results=[search_result],
        )
