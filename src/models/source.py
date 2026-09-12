"""Data models representing retrieved source snippets."""

from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Represents a minimal source citation with exact character offsets.

    Attributes:
        file_path: Relative path to the origin file within the corpus.
        first_character_index: Zero-based starting character position.
        last_character_index: Ending character position in the file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
