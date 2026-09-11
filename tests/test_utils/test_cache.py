from src.utils.cache import QueryCache


def test_cache_stores_and_returns_query_result(tmp_path) -> None:
    cache = QueryCache(cache_dir=str(tmp_path))

    query = "What is Python?"
    result = [
        {
            "file_path": "test.py",
            "first_character_index": 0,
            "last_character_index": 100,
        }
    ]

    cache.set(
        "search",
        query,
        result,
    )

    cached = cache.get(
        "search",
        query,
    )

    assert cached == result


def test_cache_returns_none_for_missing_query(tmp_path) -> None:
    cache = QueryCache(cache_dir=str(tmp_path))

    cached = cache.get(
        "search",
        "query that is not cached",
    )

    assert cached is None
