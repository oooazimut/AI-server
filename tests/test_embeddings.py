from ai_server.embeddings import LocalHashingEmbeddingProvider, create_embedding_provider


def test_create_default_embedding_provider():
    provider = create_embedding_provider("hashing")

    assert isinstance(provider, LocalHashingEmbeddingProvider)
    assert provider.name == "local_hashing"


def test_hashing_embedding_provider_is_deterministic():
    provider = LocalHashingEmbeddingProvider(dimensions=128)

    first = provider.embed("создать задачу в Битрикс24")
    second = provider.embed("создать задачу в Битрикс24")

    assert first
    assert first == second
