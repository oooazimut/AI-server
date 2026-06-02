from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
import re
from typing import Protocol


SparseVector = dict[int, float]


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> SparseVector:
        """Return a normalized sparse vector for cosine similarity."""


class LocalHashingEmbeddingProvider:
    """Deterministic local fallback used for development and tests.

    It is not a semantic embedding model. Production can switch to FastEmbed
    with AI_SERVER_EMBEDDINGS_PROVIDER=fastembed without changing retriever code.
    """

    name = "local_hashing"

    def __init__(self, *, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> SparseVector:
        features = _tokenize(text)
        normalized = _normalize(text)
        features.extend(_char_ngrams(normalized, size=4, limit=1200))
        if not features:
            return {}

        counts = Counter(features)
        vector: SparseVector = {}
        for feature, count in counts.items():
            index = _stable_index(feature, self.dimensions)
            vector[index] = vector.get(index, 0.0) + (1.0 + math.log(count))
        return normalize_sparse_vector(vector)


class FastEmbedEmbeddingProvider:
    """Real local embedding provider backed by the optional fastembed package."""

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
    selected = (provider or os.getenv("AI_SERVER_EMBEDDINGS_PROVIDER") or "hashing").strip().lower()
    if selected in {"hashing", "local", "local_hashing"}:
        dimensions = int(os.getenv("AI_SERVER_HASHING_EMBEDDING_DIMENSIONS") or "512")
        return LocalHashingEmbeddingProvider(dimensions=dimensions)
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


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[0-9a-zа-яё_\.]{2,}", _normalize(value), flags=re.IGNORECASE)


def _char_ngrams(value: str, *, size: int, limit: int) -> list[str]:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) < size:
        return []
    return [compact[index : index + size] for index in range(min(len(compact) - size + 1, limit))]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def _stable_index(value: str, dimensions: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions
