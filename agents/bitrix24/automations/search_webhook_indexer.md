# Индексация файлов по webhook

Источник переноса: `BitrixAIAgent/app/agent/search_webhook_indexer.py`.

## Роль

Event worker для быстрых изменений в индексе портала. Обрабатывает события диска:
создание, изменение, перемещение, переименование, удаление файла.

## Входы

- Disk/file webhook-события из `bitrix_webhook_event_queue`.

## Выходы

- Upsert/delete элемента в `portal_search`.
- Обновление metadata файла в локальном индексе.
- Обновление локального текста файла, если расширение поддерживается и включён
  `SEARCH_WEBHOOK_CONTENT_ENABLED`.

## State

- `var/search_index.sqlite`.
- `var/search_content`.

## Правило переноса

Webhook-индексатор должен быть отдельным worker рядом с периодическим индексатором.
Периодическая индексация отвечает за полноту, webhook-индексация - за свежесть.
