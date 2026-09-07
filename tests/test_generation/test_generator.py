import torch

from src.generation import generator


class FakeInputs(dict):
    """Fake tokenizer output."""

    def __init__(self):
        input_ids = torch.tensor([[1, 2]])
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class FakeTokenizer:
    """Fake tokenizer for unit tests."""

    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        return "PROMPT"

    def __call__(self, prompts, return_tensors="pt"):
        return FakeInputs()

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return ["<think>internal</think>Final answer."]


class FakeModel:
    """Fake language model for unit tests."""

    def __init__(self):
        self.generate_called = False

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_called = True
        return torch.tensor([[1, 2, 3, 4]])


def make_generator(monkeypatch):
    """Create AnswerGenerator with fake model and tokenizer."""
    tokenizer = FakeTokenizer()
    model = FakeModel()

    monkeypatch.setattr(
        generator.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        generator.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    answer_generator = generator.AnswerGenerator()

    return answer_generator, tokenizer, model


def test_default_model():
    assert generator.AnswerGenerator.DEFAULT_MODEL == "Qwen/Qwen3-0.6B"


def test_cpu_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    answer_generator, _, _ = make_generator(monkeypatch)

    assert answer_generator.device == "cpu"


def test_empty_question(monkeypatch):
    answer_generator, _, _ = make_generator(monkeypatch)

    result = answer_generator.generate_answer("", [])

    assert result == "No question provided."


def test_generate_answer(monkeypatch):
    answer_generator, _, model = make_generator(monkeypatch)

    result = answer_generator.generate_answer(
        "How does this work?",
        [],
    )

    assert result == "Final answer."
    assert model.generate_called is True


def test_sources_are_added_to_prompt(monkeypatch, tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "def important_function():\n"
        "    return 'important information'\n",
        encoding="utf-8",
    )

    source = generator.MinimalSource(
        file_path=str(source_file),
        first_character_index=0,
        last_character_index=72,
    )

    answer_generator, tokenizer, _ = make_generator(monkeypatch)

    captured_messages = {}

    def fake_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ):
        captured_messages["messages"] = messages
        return "PROMPT"

    tokenizer.apply_chat_template = fake_apply_chat_template

    answer_generator.generate_answer(
        "What does important_function do?",
        [source],
    )

    user_message = captured_messages["messages"][1]["content"]

    assert "important_function" in user_message
    assert "important information" in user_message
