# Bitrix24 Workers / Automations

## Зачем отдельный слой

В старом `BitrixAIAgent` фоновые процессы запускались из `app/main.py` рядом с
чатовым runtime. В новой архитектуре они остаются в Bitrix-домене, но отделяются
от `Bitrix24Specialist`.

`Bitrix24Specialist` отвечает за экспертную обработку запросов людей.
`Bitrix Workers / Automations` отвечают за события, расписания, очереди, индексы
и автономные бизнес-процессы.

## Переносимые процессы

| Automation | Старый модуль | Тип | Назначение |
| --- | --- | --- | --- |
| `bitrix_webhook_event_queue` | `app.agent.webhook_event_queue`, `app.agent.webhook_event_processor` | event worker | Очередь, retry, dedupe и маршрутизация webhook-событий |
| `bitrix_portal_search_indexer` | `app.agent.search_indexer`, `app.agent.portal_search` | data pipeline | Периодическая индексация задач, проектов, диска и контента |
| `bitrix_search_webhook_indexer` | `app.agent.search_webhook_indexer` | event worker | Быстрое обновление индекса по disk/file событиям |
| `bitrix_reconciler` | `app.agent.reconciler` | scheduled worker | Восстановление потерянных task/disk событий |
| `bitrix_task_supervisor` | `app.agent.supervisor` | business workflow | Контроль просроченных задач и уведомления |
| `bitrix_task_quality_control` | `app.agent.quality_control`, `app.agent.task_closure` | business workflow | Проверка качества закрытия задач |
| `bitrix_vehicle_usage` | `app.agent.vehicle_usage` | business workflow | Учёт использования служебных машин |

## Граница ответственности

- Оркестратор не знает деталей Bitrix REST и не управляет циклом worker-ов.
- Bitrix24-специалист знает, что такие возможности существуют, и может объяснять
  их состояние или использовать результат индекса.
- Tool Gateway выполняет Bitrix REST вызовы и применяет policy layer.
- Worker runtime запускает процессы, хранит state и отдаёт status endpoints.

## Runtime `var`

Runtime state из старого `var/` является частью боевого cutover. Он не хранится
в Git, но должен быть перенесён в новый `AI-server/var` после остановки старого
сервиса:

- `var/search_index.sqlite`;
- `var/search_content`;
- `var/webhook_event_queue.sqlite`;
- `var/dialog_state.sqlite`;
- `var/bitrix_oauth.sqlite`;
- `var/bitrix_write_audit.jsonl`;
- `var/quality_control_state.json`;
- `var/supervisor_state.json`;
- `var/vehicle_usage.sqlite`;
- `var/search_indexer_state.json`;
- `var/attachments`;
- `var/document_drafts`.

Не переносим как состояние сервиса:

- stale lock-файлы, например `var/search_indexer.lock`;
- dev/ngrok логи;
- временные `tmp*` каталоги.

OAuth state, старые диалоги, audit, вложения и индексы чувствительны, но для
боевого переезда их нужно переносить. Ограничение только в том, что они не
коммитятся и копируются не из живого пишущего процесса, а в момент cutover.

План миграции:

```powershell
uv run python scripts/import_bitrix_var.py --profile cutover
```

Фактическое копирование после остановки старого сервиса:

```powershell
uv run python scripts/import_bitrix_var.py --profile cutover --execute
```

Перед заменой существующих файлов скрипт переносит старые target-файлы в
`var/legacy/backups/<timestamp>/`, если не указан `--no-backup`.

## Порядок технического переноса

1. Зарегистрировать automation manifests и API чтения каталога.
2. Встроить `var/` как runtime-контур и подготовить cutover-миграцию.
3. Вынести общий Bitrix client/OAuth/event parser в `integrations/bitrix`.
4. Перенести очередь webhook-событий и processor как первый реальный worker.
5. Подключить portal search indexer, потому что он кормит RAG и поиск.
6. Перенести `quality_control` и `supervisor` через policy layer и dry-run.
7. Отдельно решить судьбу `vehicle_usage`: оставить в Bitrix-домене или выделить
   самостоятельного специалиста.

Устаревший `event_poller` из старого проекта не переносится. Для Bitrix24
целевой входной канал - webhook-режим через очередь событий.
