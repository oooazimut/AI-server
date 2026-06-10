from __future__ import annotations

import math
import os
from typing import Protocol

SparseVector = dict[int, float]


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> SparseVector:
        """Return a normalized sparse vector for cosine similarity."""


class FastEmbedEmbeddingProvider:
    """Local embedding provider backed by the optional fastembed package."""

    name = "fastembed"

    def __init__(self, *, model_name: str | None = None, cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed. Install retrieval extras: uv sync --extra retrieval"
            ) from exc

        kwargs: dict[str, str] = {}
        if model_name:
            kwargs["model_name"] = model_name
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = TextEmbedding(**kwargs)

    def embed(self, text: str) -> SparseVector:
        vector = next(self.model.embed([text]))
        return dense_to_sparse_vector(vector)


def create_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    selected = (provider or os.getenv("AI_SERVER_EMBEDDINGS_PROVIDER") or "fastembed").strip().lower()
    if selected == "fastembed":
        return FastEmbedEmbeddingProvider(
            model_name=os.getenv("AI_SERVER_FASTEMBED_MODEL") or None,
            cache_dir=os.getenv("AI_SERVER_FASTEMBED_CACHE_DIR") or None,
        )
    raise ValueError(f"Unknown embedding provider: {selected}")


def dense_to_sparse_vector(values: object) -> SparseVector:
    vector: SparseVector = {}
    for index, value in enumerate(values):
        number = float(value)
        if number:
            vector[index] = number
    return normalize_sparse_vector(vector)


def normalize_sparse_vector(vector: SparseVector) -> SparseVector:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items()}
