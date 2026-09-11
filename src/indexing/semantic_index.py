"""Semantic vector index using lightweight all-MiniLM-L6-v2 embeddings."""

import pickle
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from src.chunking.base import Chunk


@dataclass
class SemanticIndex:
    """Vector index storing normalized chunk embeddings for cosine search."""

    chunks: List[Chunk]
    embeddings: torch.Tensor  # Shape: (N, D) normalized vectors
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _tokenizer: Any = None
    _model: Any = None
    _device: str = "cpu"

    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> "SemanticIndex":
        """Build vector embeddings index from chunks."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        all_embeddings: List[torch.Tensor] = []

        texts = [f"{c.file_path}: {c.content}" for c in chunks]

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Generating Embeddings",
            unit="batch",
        ):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                outputs = model(**encoded)
                # Mean Pooling taking attention mask into account
                token_embeddings = outputs[0]
                input_mask_expanded = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(
                    token_embeddings * input_mask_expanded, dim=1
                )
                sum_mask = torch.clamp(
                    input_mask_expanded.sum(dim=1),
                    min=1e-9
                )
                mean_pooled = sum_embeddings / sum_mask
                normalized = F.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu())

        full_embeddings = (
            torch.cat(all_embeddings, dim=0)
            if all_embeddings
            else torch.empty((0, 384))
        )

        return cls(
            chunks=chunks,
            embeddings=full_embeddings,
            model_name=model_name,
        )

    def save(self, filepath: str) -> None:
        """Persist vector index to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "SemanticIndex":
        """Load vector index from disk."""
        with open(filepath, "rb") as f:
            index: SemanticIndex = pickle.load(f)
        return index

    def score_query(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Calculate cosine similarity against all chunk embeddings."""
        if not query.strip() or self.embeddings.numel() == 0:
            return []

        self._load_model()

        assert self._tokenizer is not None
        assert self._model is not None

        tokenizer = self._tokenizer
        model = self._model

        encoded = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = model(**encoded)
            token_embeddings = outputs[0]
            input_mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_emb = torch.sum(token_embeddings * input_mask, dim=1)
            sum_mask = torch.clamp(input_mask.sum(dim=1), min=1e-9)
            query_emb = F.normalize(sum_emb / sum_mask, p=2, dim=1)

        # Dot product with normalized vectors equals cosine similarity
        scores = torch.matmul(self.embeddings, query_emb.squeeze(0))
        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(self.chunks))
        )

        return [
            (idx.item(), score.item())
            for idx, score in zip(top_indices, top_scores)
        ]

    def _load_model(self) -> None:
        """Load tokenizer and model once for runtime queries."""
        if self._model is not None and self._tokenizer is not None:
            return

        self._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self._model = AutoModel.from_pretrained(
            self.model_name
        ).to(self._device)
        self._model.eval()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude runtime model objects from persisted index."""
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore index with unloaded runtime model."""
        self.__dict__.update(state)
