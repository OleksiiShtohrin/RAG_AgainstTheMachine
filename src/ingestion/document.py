"""Domain model for a source document read from the corpus."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Represents the complete text of one corpus file."""

    file_path: str
    content: str
