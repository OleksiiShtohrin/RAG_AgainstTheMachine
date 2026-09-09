from src.cli import CLI
from src.models.source import MinimalSource
import json


def test_search_calls_retriever_and_prints_sources(
    monkeypatch,
    capsys,
):
    source = MinimalSource(
        file_path="src/example.py",
        first_character_index=10,
        last_character_index=30,
    )

    class FakeIndex:
        pass

    class FakeRetriever:
        def __init__(self, index):
            assert index is fake_index

        def retrieve(self, query, k):
            assert query == "How does this work?"
            assert k == 5
            return [source]

    fake_index = FakeIndex()

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        lambda index_dir: fake_index,
    )
    monkeypatch.setattr(
        "src.cli.LexicalRetriever",
        FakeRetriever,
    )

    CLI().search(
        query="How does this work?",
        k=5,
        index_dir="data/processed",
    )

    captured = capsys.readouterr()

    assert "src/example.py [10:30]" in captured.out


def test_search_rejects_empty_query(capsys):
    CLI().search(query="   ")

    captured = capsys.readouterr()

    assert "Error: Search query cannot be empty." in captured.err


def test_search_rejects_non_positive_k(capsys):
    CLI().search(
        query="How does this work?",
        k=0,
    )

    captured = capsys.readouterr()

    assert "Warning: k <= 0 requested, 0 results returned." in captured.out


def test_search_handles_missing_index(monkeypatch, capsys):
    def fake_load_index(index_dir):
        raise FileNotFoundError("Index file not found.")

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        fake_load_index,
    )

    CLI().search(
        query="How does this work?",
        k=5,
    )

    captured = capsys.readouterr()

    assert "An error occurred during search:" in captured.err
    assert "Index file not found." in captured.err


def test_search_dataset_processes_questions(
    monkeypatch,
    tmp_path,
):
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        """
        {
            "rag_questions": [
                {
                    "question_id": "q1",
                    "question": "What is Python?"
                },
                {
                    "question_id": "q2",
                    "question": "What is BM25?"
                }
            ]
        }
        """,
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path="src/example.py",
        first_character_index=10,
        last_character_index=30,
    )

    class FakeIndex:
        pass

    class FakeRetriever:
        def __init__(self, index):
            assert index is fake_index

        def retrieve(self, query, k):
            assert k == 5

            if query == "What is Python?":
                return [source]

            if query == "What is BM25?":
                return [source]

            raise AssertionError(f"Unexpected query: {query}")

    fake_index = FakeIndex()

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        lambda index_dir: fake_index,
    )
    monkeypatch.setattr(
        "src.cli.LexicalRetriever",
        FakeRetriever,
    )

    output_dir = tmp_path / "results"

    CLI().search_dataset(
        dataset_path=str(dataset_path),
        k=5,
        save_directory=str(output_dir),
    )

    result_file = output_dir / "questions.json"

    assert result_file.exists()

    data = json.loads(result_file.read_text(encoding="utf-8"))

    assert data["k"] == 5
    assert len(data["search_results"]) == 2

    assert data["search_results"][0]["question_id"] == "q1"
    assert data["search_results"][0]["question"] == "What is Python?"

    assert len(data["search_results"][0]["retrieved_sources"]) == 1
    assert (
        data["search_results"][0]["retrieved_sources"][0]["file_path"]
        == "src/example.py"
    )


def test_search_dataset_handles_missing_dataset(tmp_path, capsys):
    dataset_path = tmp_path / "missing.json"
    output_dir = tmp_path / "results"

    CLI().search_dataset(
        dataset_path=str(dataset_path),
        k=5,
        save_directory=str(output_dir),
    )

    captured = capsys.readouterr()

    assert "Error: Could not read dataset file" in captured.err


def test_answer_calls_generator_and_prints_answer(
    monkeypatch,
    capsys,
):
    source = MinimalSource(
        file_path="src/example.py",
        first_character_index=10,
        last_character_index=30,
    )

    class FakeIndex:
        pass

    class FakeRetriever:
        def __init__(self, index):
            assert index is fake_index

        def retrieve(self, query, k):
            assert query == "How does this work?"
            assert k == 5
            return [source]

    class FakeGenerator:
        def __init__(self, model_name):
            assert model_name == "Qwen/Qwen3-0.6B"

        def generate_answer(self, question, sources):
            assert question == "How does this work?"
            assert sources == [source]
            return "It works like this."

    fake_index = FakeIndex()

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        lambda index_dir: fake_index,
    )
    monkeypatch.setattr(
        "src.cli.LexicalRetriever",
        FakeRetriever,
    )
    monkeypatch.setattr(
        "src.cli.AnswerGenerator",
        FakeGenerator,
    )

    CLI().answer(
        query="How does this work?",
        k=5,
        model_name="Qwen/Qwen3-0.6B",
    )

    captured = capsys.readouterr()

    # assert "It works like this." in captured.out
    assert "--- Retrieved Sources ---" in captured.out
    assert "src/example.py [10:30]" in captured.out
    assert "--- Answer ---" in captured.out
    assert "It works like this." in captured.out


def test_answer_rejects_empty_query(capsys):
    CLI().answer(query="   ")

    captured = capsys.readouterr()

    assert "Error: Query cannot be empty." in captured.err


def test_answer_rejects_non_positive_k(capsys):
    CLI().answer(
        query="How does this work?",
        k=0,
    )

    captured = capsys.readouterr()

    assert "Warning: k <= 0 requested, 0 results returned." in captured.out


def test_answer_handles_missing_index(monkeypatch, capsys):
    def fake_load_index(index_dir):
        raise FileNotFoundError("Index file not found.")

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        fake_load_index,
    )

    CLI().answer(
        query="How does this work?",
        k=5,
    )

    captured = capsys.readouterr()

    assert "An error occurred during answer generation:" in captured.err
    assert "Index file not found." in captured.err


def test_evaluate_calculates_and_prints_recall(tmp_path, capsys):
    student_results_path = tmp_path / "student_results.json"
    dataset_path = tmp_path / "dataset.json"

    student_results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": "Where is hello?",
                        "retrieved_sources": [
                            {
                                "file_path": "src/example.py",
                                "first_character_index": 10,
                                "last_character_index": 30,
                            }
                        ],
                    }
                ],
                "k": 5,
            }
        ),
        encoding="utf-8",
    )

    dataset_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "q1",
                        "question": "Where is hello?",
                        "answer": "It is in example.py.",
                        "sources": [
                            {
                                "file_path": "src/example.py",
                                "first_character_index": 10,
                                "last_character_index": 30,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cli = CLI()
    cli.evaluate(
        student_search_results_path=str(student_results_path),
        dataset_path=str(dataset_path),
    )

    captured = capsys.readouterr()

    assert "Recall@1: 1.000" in captured.out
    assert "Recall@3: 1.000" in captured.out
    assert "Recall@5: 1.000" in captured.out
    assert "Recall@10: 1.000" in captured.out


def test_evaluate_handles_missing_student_results(tmp_path, capsys):
    dataset_path = tmp_path / "dataset.json"

    dataset_path.write_text(
        json.dumps({"rag_questions": []}),
        encoding="utf-8",
    )

    cli = CLI()
    cli.evaluate(
        student_search_results_path=str(
            tmp_path / "missing.json"
        ),
        dataset_path=str(dataset_path),
    )

    captured = capsys.readouterr()

    assert "Error: Could not load files." in captured.err


def test_evaluate_handles_malformed_student_results(
    tmp_path,
    capsys,
):
    student_results_path = tmp_path / "student_results.json"
    dataset_path = tmp_path / "dataset.json"

    student_results_path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    dataset_path.write_text(
        json.dumps({"rag_questions": []}),
        encoding="utf-8",
    )

    cli = CLI()
    cli.evaluate(
        student_search_results_path=str(student_results_path),
        dataset_path=str(dataset_path),
    )

    captured = capsys.readouterr()

    assert "Error: Could not load files." in captured.err


def test_answer_dataset_processes_questions(
    monkeypatch,
    tmp_path,
):
    student_results_path = tmp_path / "student_results.json"
    save_directory = tmp_path / "output"

    student_results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": "What does hello do?",
                        "retrieved_sources": [
                            {
                                "file_path": "src/example.py",
                                "first_character_index": 10,
                                "last_character_index": 30,
                            }
                        ],
                    },
                    {
                        "question_id": "q2",
                        "question": "What does goodbye do?",
                        "retrieved_sources": [],
                    },
                ],
                "k": 5,
            }
        ),
        encoding="utf-8",
    )

    class FakeGenerator:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_answer(self, question, sources):
            return f"Answer for: {question}"

    monkeypatch.setattr(
        "src.cli.AnswerGenerator",
        FakeGenerator,
    )

    cli = CLI()
    cli.answer_dataset(
        student_search_results_path=str(student_results_path),
        save_directory=str(save_directory),
    )

    output_path = save_directory / "student_results.json"

    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["k"] == 5
    assert len(data["search_results"]) == 2

    assert data["search_results"][0]["question_id"] == "q1"
    assert (
        data["search_results"][0]["answer"]
        == "Answer for: What does hello do?"
    )

    assert data["search_results"][1]["question_id"] == "q2"
    assert (
        data["search_results"][1]["answer"]
        == "Answer for: What does goodbye do?"
    )


def test_answer_dataset_handles_missing_input(
    tmp_path,
    capsys,
):
    cli = CLI()

    cli.answer_dataset(
        student_search_results_path=str(
            tmp_path / "missing.json"
        ),
        save_directory=str(tmp_path / "output"),
    )

    captured = capsys.readouterr()

    assert "Error: Invalid file at" in captured.err


def test_answer_dataset_handles_malformed_input(
    tmp_path,
    capsys,
):
    student_results_path = tmp_path / "student_results.json"

    student_results_path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    cli = CLI()

    cli.answer_dataset(
        student_search_results_path=str(student_results_path),
        save_directory=str(tmp_path / "output"),
    )

    captured = capsys.readouterr()

    assert "Error: Invalid file at" in captured.err
