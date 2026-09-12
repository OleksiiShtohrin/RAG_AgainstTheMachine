"""Fast LLM response generation engine using HuggingFace Transformers."""

import torch
from typing import Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.generation.prompter import ContextPrompter
from src.models.source import MinimalSource


class AnswerGenerator:
    """Handles local LLM inference for grounded answer generation."""

    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """Initialize tokenizer and model for causal language modeling.

        Args:
            model_name: HuggingFace model identifier or local directory path.
            device: Computing target device ('cuda', 'mps', 'cpu').
                If None, detects available accelerators automatically.
        """
        self.model_name = model_name
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        model: Any = self.model
        model.eval()

    def generate_answer(
        self,
        question: str,
        sources: List[MinimalSource],
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a grounded answer based on retrieved context sources.

        Args:
            question: The user query string.
            sources:
                List of retrieved citations containing file paths and spans.
            max_new_tokens: Maximum number of tokens the model can generate.

        Returns:
            The decoded and cleaned text response from the model.
        """
        if not question.strip():
            return "No question provided."

        top_sources = sources[:3]
        context = ContextPrompter.load_source_texts(
            top_sources, max_total_chars=3000
        )
        messages = ContextPrompter.build_chat_messages(question, context)

        # Apply chat template with non-thinking mode for Qwen3
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_text = str(raw_prompt)

        # Force close thinking block so Qwen3 writes the final answer directly
        if "<think>" not in prompt_text:
            prompt_text = f"{prompt_text}<think>\n\n</think>\n\n"

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(
            self.device
        )

        model: Any = self.model
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return ContextPrompter.clean_output(response)
