from src.cli import CLI
from src.models.source import MinimalSource
from src.utils.cache import QueryCache


def test_search_uses_cache_on_second_call(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = MinimalSource(
        file_path="test.py",
        first_character_index=0,
        last_character_index=10,
    )

    retrieve_calls = 0

    def fake_retrieve(self, query: str, k: int):
        nonlocal retrieve_calls
        retrieve_calls += 1
        return [source]

    monkeypatch.setattr(
        "src.cli.LexicalRetriever.retrieve",
        fake_retrieve,
    )

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        lambda index_dir: object(),
    )

    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(
        "src.cli.QueryCache",
        lambda: QueryCache(cache_dir=str(cache_dir)),
        raising=False,
    )

    cli = CLI()

    cli.search("What is Python?", k=5)
    cli.search("What is Python?", k=5)

    assert retrieve_calls == 1

    output = capsys.readouterr().out
    assert output.count("test.py [0:10]") == 2


def test_search_hybrid_uses_cache_on_second_call(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = MinimalSource(
        file_path="test.py",
        first_character_index=0,
        last_character_index=10,
    )

    retrieve_calls = 0

    def fake_retrieve(self, query: str, k: int):
        nonlocal retrieve_calls
        retrieve_calls += 1
        return [source]

    monkeypatch.setattr(
        "src.retrieval.hybrid_retriever.HybridRetriever.retrieve",
        fake_retrieve,
    )

    monkeypatch.setattr(
        "src.cli.CorpusIndexer.load_index",
        lambda index_dir: object(),
    )

    monkeypatch.setattr(
        "src.indexing.semantic_index.SemanticIndex.load",
        lambda index_dir: object(),
    )

    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(
        "src.cli.QueryCache",
        lambda: QueryCache(cache_dir=str(cache_dir)),
        raising=False,
    )

    cli = CLI()

    cli.search_hybrid("What is Python?", k=5)
    cli.search_hybrid("What is Python?", k=5)

    assert retrieve_calls == 1

    output = capsys.readouterr().out
    assert output.count("test.py [0:10]") == 2
