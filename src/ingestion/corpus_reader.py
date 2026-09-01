"""Read supported text files from the corpus."""

from pathlib import Path
from typing import Iterator

from src.ingestion.document import Document


class CorpusReader:
    """Discover and read text files from a corpus directory."""

    IGNORED_DIRS = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "venv",
        ".venv",
        "build",
        "dist",
        "egg-info",
    }

    SUPPORTED_EXTENSIONS = {
        ".py",
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cu",
    }

    def __init__(self, corpus_path: str = "data/raw") -> None:
        """Initialize the reader with the corpus root directory."""
        self.corpus_path = Path(corpus_path)

    def read(self) -> list[Document]:
        """Read all supported text files and return them as Documents."""
        return list(self.iter_documents())

    def iter_documents(self) -> Iterator[Document]:
        """Yield supported corpus files one at a time."""
        if not self.corpus_path.is_dir():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_path}"
            )

        for file_path in self._collect_files():
            try:
                content = file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, UnicodeError):
                continue

            yield Document(
                file_path=self._relative_path(file_path),
                content=content,
            )

    def _collect_files(self) -> list[Path]:
        """Find supported files while skipping generated/cache directories."""
        files: list[Path] = []

        for path in self.corpus_path.rglob("*"):
            if not path.is_file():
                continue

            if any(part in self.IGNORED_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            files.append(path)

        return sorted(files)

    def _relative_path(self, file_path: Path) -> str:
        """Return a project-relative path when possible."""
        resolved_file = file_path.resolve()
        project_root = Path.cwd().resolve()

        try:
            relative = resolved_file.relative_to(project_root)
            return relative.as_posix()
        except ValueError:
            return resolved_file.as_posix()
