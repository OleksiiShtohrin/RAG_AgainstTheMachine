"""Prompt engineering and context builder for grounded question answering."""

import re
from typing import Dict, List
from src.models.source import MinimalSource


class ContextPrompter:
    """Builds prompts with retrieved context chunks for LLM inference."""

    SYSTEM_PROMPT = (
        "You are an expert technical assistant for the vLLM codebase. "
        "Answer the question directly, accurately, and concisely based ONLY "
        "on the provided context. Do not output internal thought processes. "
        "If the answer is not present, state that it is unavailable."
    )

    @classmethod
    def load_source_texts(
        cls, sources: List[MinimalSource], max_total_chars: int = 3000
    ) -> str:
        """Read text snippets from disk based on MinimalSource coordinates."""
        context_parts: List[str] = []
        current_chars = 0

        for idx, source in enumerate(sources, 1):
            try:
                with open(
                    source.file_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    file_content = f.read()
                snippet = file_content[
                    source.first_character_index:source.last_character_index
                ].strip()
                if not snippet:
                    continue

                part = f"[Source {idx}: {source.file_path}]\n{snippet}\n"
                if current_chars + len(part) > max_total_chars:
                    break

                context_parts.append(part)
                current_chars += len(part)
            except Exception:
                continue

        return "\n".join(context_parts)

    @classmethod
    def build_chat_messages(
        cls, question: str, context: str
    ) -> List[Dict[str, str]]:
        """Construct chat messages adhering to conversational format."""
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Direct Answer:"
        )
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def clean_output(cls, text: str) -> str:
        """Remove any <think>...</think> tags if model produces them."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()
