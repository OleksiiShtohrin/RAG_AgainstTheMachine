"""Robust tokenizer for technical documentation and source code."""

import re
from typing import List


class CodeTokenizer:
    """Tokenizes text and code into normalized searchable tokens."""

    _SPLIT_REGEX = re.compile(r"[^a-zA-Z0-9_]+")
    _CAMEL_REGEX = re.compile(r"([a-z0-9])([A-Z])")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Split text into lowercase tokens and subwords."""
        if not text:
            return []

        # Expand camelCase: "OpenAIServer" -> "Open AI Server"
        expanded = cls._CAMEL_REGEX.sub(r"\1 \2", text)
        raw_tokens = cls._SPLIT_REGEX.split(expanded)

        tokens: List[str] = []
        for token in raw_tokens:
            token = token.strip("_").lower()
            if len(token) > 1:  # ignore single noise characters
                tokens.append(token)
                # Split snake_case sub-parts
                if "_" in token:
                    parts = token.split("_")
                    for p in parts:
                        if len(p) > 1 and p != token:
                            tokens.append(p)

        return tokens
