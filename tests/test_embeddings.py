import pytest

from ai_server.embeddings import create_embedding_provider, dense_to_sparse_vector
from tests.fakes import FakeEmbeddingProvider


def test_create_embedding_provider_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_embedding_provider("unknown")


def test_dense_to_sparse_vector_normalizes_values():
    vector = dense_to_sparse_vector([3.0, 0.0, 4.0])

    assert vector == {0: 0.6, 2: 0.8}


def test_test_embedding_provider_is_deterministic():
    provider = FakeEmbeddingProvider(dimensions=128)

    first = provider.embed("создать задачу в Битрикс24")
    second = provider.embed("создать задачу в Битрикс24")

    assert first
    assert first == second
