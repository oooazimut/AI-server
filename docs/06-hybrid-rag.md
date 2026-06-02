# Hybrid RAG

## Что считаем RAG

RAG/knowledge - это слой фактов и источников, на которые специалист опирается при ответе или выборе действия.

Для Битрикс24-специалиста это сейчас:

```text
agents/bitrix24/knowledge/topics/*.md
```

Здесь лежат перенесенные знания старого `BitrixAIAgent`: задачи, документы, REST, проекты и CRM.

## Что считаем Skills

Skills - это процедурные сценарии, то есть инструкции как действовать в типовом кейсе.

```text
agents/bitrix24/skills/*.md
```

Например `tasks_create_edit.md` говорит, какие поля проверить перед созданием задачи и почему write-действие должно идти через подтверждение.

## Почему гибрид

В production одного vector search обычно мало. Нужна комбинация:

```text
keyword/BM25-like search
+ vector search
+ metadata filters
+ access policies
+ optional rerank
```

Точный поиск важен для ID, REST-методов, названий полей, дат и имен файлов. Vector search полезен для смысловых запросов, когда пользователь формулирует не теми словами, что в документе.

## Текущий MVP

Сейчас реализован `HybridKnowledgeRetriever`:

- keyword-score по токенам и секциям markdown;
- vector-score через pluggable `EmbeddingProvider`;
- default provider: `LocalHashingEmbeddingProvider` для разработки и тестов;
- production-ready provider: `FastEmbedEmbeddingProvider` для реальных локальных embeddings;
- общий итоговый score;
- endpoint ручной проверки: `GET /agents/{agent_id}/knowledge/search?q=...`.

Hashing provider - fallback, а не semantic model. Он нужен, чтобы проект всегда стартовал локально и чтобы контракт retrieval был покрыт тестами.

## Как включить реальные локальные embeddings

Установить optional retrieval-зависимости:

```powershell
uv sync --extra dev --extra retrieval
```

Включить fastembed-провайдер:

```env
AI_SERVER_EMBEDDINGS_PROVIDER=fastembed
AI_SERVER_FASTEMBED_MODEL=
AI_SERVER_FASTEMBED_CACHE_DIR=var/embedding_models
```

`AI_SERVER_FASTEMBED_MODEL` можно оставить пустым, тогда `fastembed` возьмет модель по умолчанию. Для production на русском контенте нужно отдельно выбрать и закрепить multilingual embedding model, затем зафиксировать ее в env/config.

## Будущий production-вариант

```text
knowledge sources
  ↓
chunking
  ↓
embeddings
  ↓
pgvector / Qdrant
  ↓
hybrid search: BM25 + vector
  ↓
access filters + rerank
  ↓
context for specialist
```

