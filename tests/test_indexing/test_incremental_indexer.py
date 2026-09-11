import json
from pathlib import Path

from src.indexing.incremental_indexer import IncrementalIndexer


def test_second_update_skips_unchanged_files(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    raw_dir.mkdir()
    (raw_dir / "test.py").write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    indexer = IncrementalIndexer()

    indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    processed_files: list[Path] = []

    original_process_file = indexer._process_file

    def tracking_process_file(path: Path):
        processed_files.append(path)
        return original_process_file(path)

    monkeypatch.setattr(
        indexer,
        "_process_file",
        tracking_process_file,
    )

    indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    assert processed_files == []


def test_update_reprocesses_changed_file(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    raw_dir.mkdir()
    test_file = raw_dir / "test.py"
    test_file.write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    indexer = IncrementalIndexer()

    indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    processed_files: list[Path] = []
    original_process_file = indexer._process_file

    def tracking_process_file(path: str):
        processed_files.append(Path(path))
        return original_process_file(path)

    monkeypatch.setattr(
        indexer,
        "_process_file",
        tracking_process_file,
    )

    test_file.write_text(
        "print('changed')\n",
        encoding="utf-8",
    )

    indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    assert processed_files == [test_file]


def test_update_removes_deleted_file(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    raw_dir.mkdir()

    first_file = raw_dir / "first.py"
    second_file = raw_dir / "second.py"

    first_file.write_text(
        "print('first')\n",
        encoding="utf-8",
    )
    second_file.write_text(
        "print('second')\n",
        encoding="utf-8",
    )

    indexer = IncrementalIndexer()

    first_index = indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    assert len(first_index.chunks) > 0

    second_file.unlink()

    second_index = indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    assert all(
        chunk.file_path != str(second_file)
        for chunk in second_index.chunks
    )


def test_update_saves_manifest(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    raw_dir.mkdir()

    test_file = raw_dir / "test.py"
    test_file.write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    indexer = IncrementalIndexer()

    indexer.update_index(
        raw_dir=str(raw_dir),
        output_dir=str(output_dir),
    )

    manifest_path = output_dir / "manifest.json"

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert str(test_file) in manifest
    assert isinstance(manifest[str(test_file)], str)
    assert len(manifest[str(test_file)]) == 64
