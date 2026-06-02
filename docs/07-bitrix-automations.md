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
| `bitrix_event_poller` | `app.agent.event_poller` | channel adapter | Polling входящих сообщений, если webhook недоступен |

## Граница ответственности

- Оркестратор не знает деталей Bitrix REST и не управляет циклом worker-ов.
- Bitrix24-специалист знает, что такие возможности существуют, и может объяснять
  их состояние или использовать результат индекса.
- Tool Gateway выполняет Bitrix REST вызовы и применяет policy layer.
- Worker runtime запускает процессы, хранит state и отдаёт status endpoints.

## State

Runtime state из старого `var/` не переносится как исходный код, но учитывается
как контракт данных:

- `var/search_index.sqlite`;
- `var/search_content`;
- `var/webhook_event_queue.sqlite`;
- `var/quality_control_state.json`;
- `var/supervisor_state.json`;
- `var/vehicle_usage.sqlite`;
- `var/search_indexer_state.json`;
- `var/search_indexer.lock`.

OAuth state и вложения считаются чувствительными данными и не копируются без
отдельного решения.

## Порядок технического переноса

1. Зарегистрировать automation manifests и API чтения каталога.
2. Вынести общий Bitrix client/OAuth/event parser в `integrations/bitrix`.
3. Перенести очередь webhook-событий и processor как первый реальный worker.
4. Подключить portal search indexer, потому что он кормит RAG и поиск.
5. Перенести `quality_control` и `supervisor` через policy layer и dry-run.
6. Отдельно решить судьбу `vehicle_usage`: оставить в Bitrix-домене или выделить
   самостоятельного специалиста.

