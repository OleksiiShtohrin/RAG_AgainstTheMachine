from src.indexing.index_cache import IndexCache


def test_index_cache_returns_same_index_for_second_load(tmp_path):
    cache = IndexCache()

    index = object()

    first = cache.get(str(tmp_path))
    assert first is None

    cache.set(str(tmp_path), index)

    second = cache.get(str(tmp_path))

    assert second is index


def test_corpus_indexer_loads_index_from_cache_on_second_call(
    tmp_path,
    monkeypatch,
) -> None:
    from src.indexing.indexer import CorpusIndexer

    index = object()
    load_calls = 0

    def fake_load(path):
        nonlocal load_calls
        load_calls += 1
        return index

    monkeypatch.setattr(
        "src.indexing.indexer.pickle.load",
        fake_load,
    )

    index_file = tmp_path / "bm25_index.pkl"
    index_file.write_bytes(b"fake")

    first = CorpusIndexer.load_index(str(tmp_path))
    second = CorpusIndexer.load_index(str(tmp_path))

    assert first is second
    assert first is index
    assert load_calls == 1


def test_index_cache_stores_and_returns_index() -> None:
    cache = IndexCache()
    index = object()

    cache.set("data/processed", index)

    assert cache.get("data/processed") is index


def test_index_cache_returns_none_for_missing_index() -> None:
    cache = IndexCache()

    assert cache.get("data/processed") is None
