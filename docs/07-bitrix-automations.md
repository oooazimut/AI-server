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
| `bitrix_chat_adapter` | `app.agent.runtime_v2`, `app.agent.dialog_state_store` | channel adapter | Входящие сообщения Bitrix-чата, pending write confirmations, аудит действий |
| `bitrix_webhook_event_queue` | `app.agent.webhook_event_queue`, `app.agent.webhook_event_processor` | event worker | Очередь, retry, dedupe и маршрутизация webhook-событий |
| `bitrix_portal_search_indexer` | `app.agent.search_indexer`, `app.agent.portal_search` | data pipeline | Периодическая metadata/delta-индексация задач, проектов и диска |
| `bitrix_search_webhook_indexer` | `app.agent.search_webhook_indexer` | event worker | Быстрое обновление индекса по disk/file событиям |
| `bitrix_reconciler` | `app.agent.reconciler` | scheduled worker | Восстановление потерянных task/disk событий |
| `bitrix_task_supervisor` | `app.agent.supervisor` | business workflow | Контроль просроченных задач и уведомления |
| `bitrix_task_quality_control` | `app.agent.quality_control` | business workflow | Проверка качества закрытия задач по webhook |
| `bitrix_task_closure` | `app.agent.task_closure` | chat tool | Закрытие задачи по запросу человека через LLM Битрикс-субагента и pending confirmation |
| `logistics_vehicle_usage` | `app.agent.vehicle_usage` | business workflow | Утренний учёт использования служебных машин через будущего субагента `Логист` |

## Граница ответственности

- Оркестратор не знает деталей Bitrix REST и не управляет циклом worker-ов.
- Bitrix24-специалист знает, что такие возможности существуют, и может объяснять
  их состояние или использовать результат индекса.
- Tool Gateway выполняет Bitrix REST вызовы и применяет policy layer.
- Worker runtime запускает процессы, хранит state и отдаёт status endpoints.
- Bitrix message adapter хранит ожидающие подтверждения write-действий в
  `var/dialog_state.sqlite` и выполняет их только после прямого ответа
  пользователя `да`; отмена делается ответом `отмена`.
- Подтверждённые write-действия дополнительно проверяют
  `AGENT_WRITE_ALLOWED_USER_IDS` или ограниченное правило
  `AGENT_LIMITED_TASK_CREATE_USER_IDS` + `AGENT_LIMITED_TASK_CREATE_PROJECT_ID`.
  Если `BITRIX_OAUTH_REQUIRED_FOR_WRITES=true`, выполнение идёт только через
  OAuth-токен пользователя.
- Фоновый quality-control в боевом режиме требует
  `QUALITY_CONTROL_ACTOR_USER_ID`; без служебного OAuth actor он не выполняет
  write-действия через общий webhook-токен.

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
- `var/learning_events.jsonl`;
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
4. Перенести очередь webhook-событий и message processor как первый реальный worker.
5. Подключить portal search indexer, потому что он кормит RAG и поиск.
6. Перенести `quality_control` и `supervisor` через policy layer и dry-run.
7. `vehicle_usage` переносить не в Bitrix-специалиста, а в отдельного
   субагента `Логист`: scheduler утром инициирует сбор данных, а Логист
   разбирает ответы и формирует отчёт.

Устаревший `event_poller` из старого проекта не переносится. Для Bitrix24
целевой входной канал - webhook-режим через очередь событий.

## Текущий webhook contour

В `AI-server` уже есть:

- `backend/ai_server/integrations/bitrix` - REST client, OAuth state reader и
  parser входящих Bitrix-событий;
- `backend/ai_server/workers/bitrix/webhook_event_queue.py` - совместимая
  SQLite-очередь `webhook_events`;
- `backend/ai_server/channels/bitrix.py` - message adapter, который превращает
  Bitrix message event в `AgentTask` для Оркестратора, а также перехватывает
  подтверждение/отмену ожидающих Bitrix-действий;
- `backend/ai_server/integrations/bitrix/dialog_state.py` - совместимое
  хранилище `dialog_states`, pending Bitrix write actions и JSONL-аудит
  подтверждённых/отменённых действий;
- `backend/ai_server/integrations/bitrix/portal_search.py` - reader/writer
  совместимой SQLite-таблицы `portal_search_items`;
- `backend/ai_server/document_text.py` - извлечение текста из `.txt`, `.csv`,
  `.doc`, `.docx`, `.xls`, `.xlsx`, `.pdf`;
- `backend/ai_server/workers/bitrix/search_indexer.py` - фоновый и ручной
  metadata/delta/content-indexer задач, проектов, диска и содержимого файлов;
- `backend/ai_server/workers/bitrix/search_webhook_indexer.py` - обработчик
  disk/file webhook-событий с обновлением metadata и, если включено, текста файла;
- `backend/ai_server/workers/bitrix/quality_control.py` - LLM-driven обработчик
  `onTaskUpdate`: worker передаёт модели событие и ID задачи, модель сама
  вызывает read-tools `bitrix_task_get`/`bitrix_task_results_list`, затем
  вызывает `quality_control_action`; backend применяет dedupe,
  dry-run/policy/OAuth-actor перед approve/disapprove/renew/comment/notify;
- `backend/ai_server/agents/bitrix_task_closure.py` - чатовый tool закрытия
  задачи: LLM Битрикс-субагент готовит `task_id`/`task_query` и `result_text`,
  пользователь подтверждает, затем Битрикс-специалист в режиме `task_closure`
  сам выбирает read/write tools; backend исполняет tools и применяет guardrails;
- `POST /bitrix/events` - endpoint приёма webhook-событий;
- `GET /bitrix/status`, `GET /bitrix/webhook-events/status`,
  `GET /bitrix/search/status` и `GET /bitrix/search` - runtime status/search;
- `GET /bitrix/search/indexer/status`, `POST /bitrix/search/reindex`,
  `POST /bitrix/search/reindex-delta`, `POST /bitrix/search/reindex-content` -
  статус и ручной запуск индексатора.
- `GET /bitrix/quality-control/status` - статус webhook-контроля качества.

Worker очереди включается отдельно:

```env
AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED=true
```

Для безопасной разработки можно держать:

```env
AGENT_DRY_RUN=true
```

Фоновая индексация портала по умолчанию не стартует сама, чтобы разработочный
сервер случайно не начал обходить боевой Bitrix:

```env
SEARCH_BACKGROUND_INDEXER_ENABLED=true
```

Webhook-обновление файлового индекса тоже включается явно:

```env
SEARCH_WEBHOOK_INDEXER_ENABLED=true
SEARCH_WEBHOOK_CONTENT_ENABLED=true
```

`portal_search` уже умеет читать и обновлять старую SQLite-таблицу
`portal_search_items`. Перенесён metadata/delta/content-контур: задачи, проекты,
диск, содержимое документов и disk/file webhook-и.

`dialog_state` читает старую форму `pending_action` из `dialog_state.sqlite`.
Формат ключа сохранён: `chat:{chat_id}:user:{user_id}`,
`dialog:{dialog_id}:user:{user_id}` или `user:{user_id}`. Это позволяет при
cutover перенести незавершённые подтверждения из старого агента без ручной
конвертации.
