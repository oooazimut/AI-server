# Периодическая индексация портала

Источник переноса: `BitrixAIAgent/app/agent/search_indexer.py` и
`BitrixAIAgent/app/agent/portal_search.py`.

## Роль

Data pipeline. По расписанию синхронизирует метаданные задач, проектов, диска и
содержимое файлов в локальный поисковый индекс.

## Входы

- Bitrix REST read-only методы.
- Настройки лимитов и интервалов индексации.

## Выходы

- Локальный индекс для `portal_search`.
- Контентный cache для документов.

## State

- `var/search_index.sqlite`.
- `var/search_content`.
- `var/search_indexer_state.json`.
- `var/search_indexer.lock`.

## Правило переноса

Это не RAG сам по себе, а подготовка данных для поиска/RAG. Bitrix24-специалист
пользуется результатом индекса, но не запускает полный обход портала из своего
LLM-loop.

