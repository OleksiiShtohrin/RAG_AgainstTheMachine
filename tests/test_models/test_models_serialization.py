from src.models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def test_minimal_source_model_dump():
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


def test_minimal_source_json_round_trip():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
    )

    json_data = source.model_dump_json()
    restored = MinimalSource.model_validate_json(json_data)

    assert restored == source


def test_minimal_search_results_json_round_trip():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
    )

    result = MinimalSearchResults(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
    )

    json_data = result.model_dump_json()
    restored = MinimalSearchResults.model_validate_json(json_data)

    assert restored == result
    assert restored.retrieved_sources[0] == source


def test_minimal_answer_json_round_trip():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
    )

    result = MinimalAnswer(
        question_id="test-123",
        question="What is Python?",
        retrieved_sources=[source],
        answer="Python is a programming language.",
    )

    json_data = result.model_dump_json()
    restored = MinimalAnswer.model_validate_json(json_data)

    assert restored == result
    assert restored.answer == "Python is a programming language."
    assert restored.retrieved_sources[0] == source


def test_student_search_results_json_round_trip():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
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

    json_data = result.model_dump_json()
    restored = StudentSearchResults.model_validate_json(json_data)

    assert restored == result
    assert restored.k == 5
    assert restored.search_results[0] == search_result
    assert restored.search_results[0].retrieved_sources[0] == source


def test_student_search_results_and_answer_json_round_trip():
    source = MinimalSource(
        file_path="src/main.py",
        first_character_index=10,
        last_character_index=30,
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

    json_data = result.model_dump_json()
    restored = StudentSearchResultsAndAnswer.model_validate_json(json_data)

    assert restored == result
    assert restored.k == 5
    assert restored.search_results[0] == answer
    assert restored.search_results[0].answer == (
        "Python is a programming language."
    )
    assert restored.search_results[0].retrieved_sources[0] == source
