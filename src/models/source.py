"""Data models representing retrieved source snippets."""

from pydantic import BaseModel


class MinimalSource(BaseModel):
    """Represents a minimal source citation with exact character offsets."""

    file_path: str
    first_character_index: int
    last_character_index: int
