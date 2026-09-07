from src.generation.prompter import ContextPrompter
from src.models.source import MinimalSource


def test_load_source_texts_reads_requested_range(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "AAAA BBBB CCCC",
        encoding="utf-8",
    )

    source = MinimalSource(
        file_path=str(source_file),
        first_character_index=5,
        last_character_index=9,
    )

    context = ContextPrompter.load_source_texts([source])

    assert "BBBB" in context


def test_load_source_texts_skips_missing_file():
    source = MinimalSource(
        file_path="does/not/exist.py",
        first_character_index=0,
        last_character_index=10,
    )

    context = ContextPrompter.load_source_texts([source])

    assert context == ""


def test_build_chat_messages():
    messages = ContextPrompter.build_chat_messages(
        "How does this work?",
        "def hello():\n    pass",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "How does this work?" in messages[1]["content"]
    assert "def hello()" in messages[1]["content"]


def test_clean_output_removes_think_tags():
    result = ContextPrompter.clean_output(
        "<think>internal reasoning</think>The answer."
    )

    assert result == "The answer."
