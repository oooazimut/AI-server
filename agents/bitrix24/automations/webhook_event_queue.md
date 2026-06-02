# Очередь webhook-событий Битрикс24

Источник переноса: `BitrixAIAgent/app/agent/webhook_event_queue.py` и
`BitrixAIAgent/app/agent/webhook_event_processor.py`.

## Роль

Транспортный worker. Принимает события Битрикс24, дедуплицирует их, кладёт в
локальную очередь и обрабатывает с retry/backoff. Это не агент и не место для
доменной логики.

## Входы

- HTTP webhook от Битрикс24.
- Синтетические события от `bitrix_reconciler`.

## Выходы

- Сообщения пользователей передаются в channel/orchestrator flow.
- События задач передаются в `bitrix_task_quality_control`.
- События диска передаются в `bitrix_search_webhook_indexer`.

## State

- `var/webhook_event_queue.sqlite`.

## Правило переноса

Очередь должна остаться самостоятельной инфраструктурой Bitrix-контура. Оркестратор
не должен вручную управлять её retry-логикой.

