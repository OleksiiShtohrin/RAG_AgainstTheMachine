from fastapi.testclient import TestClient
from src.models.source import MinimalSource

from src.server import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-api",
    }


def test_search(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, k: int):
            return [
                MinimalSource(
                    file_path="test.py",
                    first_character_index=0,
                    last_character_index=10,
                )
            ]

    monkeypatch.setattr(
        "src.server.get_retriever",
        lambda: FakeRetriever(),
    )

    response = client.post(
        "/search",
        json={
            "query": "What is Python?",
            "k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is Python?",
        "k": 5,
        "results": [
            {
                "file_path": "test.py",
                "first_character_index": 0,
                "last_character_index": 10,
            }
        ],
    }


def test_answer(monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, k: int):
            return [
                MinimalSource(
                    file_path="test.py",
                    first_character_index=0,
                    last_character_index=10,
                )
            ]

    class FakeGenerator:
        def generate_answer(self, query: str, sources):
            return "Python is a programming language."

    monkeypatch.setattr(
        "src.server.get_retriever",
        lambda: FakeRetriever(),
    )
    monkeypatch.setattr(
        "src.server.get_generator",
        lambda: FakeGenerator(),
    )

    response = client.post(
        "/answer",
        json={
            "query": "What is Python?",
            "k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is Python?",
        "answer": "Python is a programming language.",
        "sources": [
            {
                "file_path": "test.py",
                "first_character_index": 0,
                "last_character_index": 10,
            }
        ],
    }


def test_search_rejects_empty_query() -> None:
    response = client.post(
        "/search",
        json={
            "query": "",
            "k": 5,
        },
    )

    assert response.status_code == 400


def test_answer_rejects_empty_query() -> None:
    response = client.post(
        "/answer",
        json={
            "query": "",
            "k": 5,
        },
    )

    assert response.status_code == 400
