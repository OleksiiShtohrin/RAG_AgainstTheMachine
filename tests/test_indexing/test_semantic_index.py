import torch
from src.indexing.semantic_index import SemanticIndex
from src.chunking.base import Chunk


def test_build_empty_chunks() -> None:
    index = SemanticIndex.build([])

    assert index.chunks == []
    assert index.embeddings.shape == (0, 384)


def test_build_creates_embeddings() -> None:
    chunks = [
        Chunk(
            content="hello world",
            file_path="test.py",
            first_character_index=0,
            last_character_index=11,
        ),
        Chunk(
            content="another chunk",
            file_path="test.py",
            first_character_index=12,
            last_character_index=25,
        ),
    ]

    index = SemanticIndex.build(
        chunks,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert len(index.chunks) == 2
    assert index.embeddings.shape == (2, 384)


def test_embeddings_are_normalized() -> None:
    chunks = [
        Chunk(
            content="hello world",
            file_path="test.py",
            first_character_index=0,
            last_character_index=11,
        ),
        Chunk(
            content="another chunk",
            file_path="test.py",
            first_character_index=12,
            last_character_index=25,
        ),
    ]

    index = SemanticIndex.build(chunks)

    norms = torch.linalg.vector_norm(
        index.embeddings,
        dim=1,
    )

    assert torch.allclose(
        norms,
        torch.ones(2),
        atol=1e-5,
    )


def test_score_query_returns_ranked_results() -> None:
    chunks = [
        Chunk(
            content="Python programming language",
            file_path="python.py",
            first_character_index=0,
            last_character_index=29,
        ),
        Chunk(
            content="database connection settings",
            file_path="database.py",
            first_character_index=0,
            last_character_index=31,
        ),
    ]

    index = SemanticIndex.build(chunks)

    results = index.score_query(
        "Python programming",
        top_k=1,
    )

    assert len(results) == 1

    chunk_id, score = results[0]

    assert chunk_id == 0
    assert isinstance(score, float)


def test_score_query_respects_top_k_and_order() -> None:
    chunks = [
        Chunk(
            content="Python programming language",
            file_path="python.py",
            first_character_index=0,
            last_character_index=29,
        ),
        Chunk(
            content="database connection settings",
            file_path="database.py",
            first_character_index=0,
            last_character_index=31,
        ),
    ]

    index = SemanticIndex.build(chunks)

    results = index.score_query(
        "Python programming",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == 0
    assert results[0][1] >= results[1][1]


def test_score_query_empty_query() -> None:
    chunks = [
        Chunk(
            content="Python programming language",
            file_path="python.py",
            first_character_index=0,
            last_character_index=29,
        ),
    ]

    index = SemanticIndex.build(chunks)

    assert index.score_query("") == []
    assert index.score_query("   ") == []


def test_score_query_top_k_larger_than_chunks() -> None:
    chunks = [
        Chunk(
            content="Python programming language",
            file_path="python.py",
            first_character_index=0,
            last_character_index=29,
        ),
        Chunk(
            content="database connection settings",
            file_path="database.py",
            first_character_index=0,
            last_character_index=31,
        ),
    ]

    index = SemanticIndex.build(chunks)

    results = index.score_query(
        "Python programming",
        top_k=10,
    )

    assert len(results) == 2


def test_save_and_load(tmp_path) -> None:
    chunks = [
        Chunk(
            content="Python programming language",
            file_path="python.py",
            first_character_index=0,
            last_character_index=29,
        ),
        Chunk(
            content="database connection settings",
            file_path="database.py",
            first_character_index=0,
            last_character_index=31,
        ),
    ]

    index = SemanticIndex.build(chunks)

    index_path = tmp_path / "semantic_index.pkl"
    index.save(str(index_path))

    loaded = SemanticIndex.load(str(index_path))

    assert loaded.chunks == index.chunks
    assert loaded.model_name == index.model_name
    assert torch.equal(
        loaded.embeddings,
        index.embeddings,
    )


def test_score_query_loads_model_only_once(monkeypatch) -> None:

    index = SemanticIndex(
        chunks=[
            Chunk(
                content="hello world",
                file_path="test.py",
                first_character_index=0,
                last_character_index=11,
            )
        ],
        embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    tokenizer_calls = 0
    model_calls = 0

    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return (
                torch.tensor(
                    [[[1.0, 0.0], [1.0, 0.0]]]
                ),
            )

    def fake_tokenizer(*args, **kwargs):
        nonlocal tokenizer_calls
        tokenizer_calls += 1
        return FakeTokenizer()

    def fake_model(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return FakeModel()

    monkeypatch.setattr(
        "src.indexing.semantic_index.AutoTokenizer.from_pretrained",
        fake_tokenizer,
    )
    monkeypatch.setattr(
        "src.indexing.semantic_index.AutoModel.from_pretrained",
        fake_model,
    )

    index.score_query("hello")
    index.score_query("world")

    assert tokenizer_calls == 1
    assert model_calls == 1
