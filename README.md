# AI Server

Корпоративный сервер ИИ-агентов для офисных и клиентских сценариев.

Первый MVP: разделить старый автономный `BitrixAIAgent` на две роли:

- `internal_orchestrator` - Переговорщик: входная точка, маршрутизатор и голос
  системы для сотрудников;
- `bitrix24` - исполнитель точных структурированных команд Битрикс24 без
  собственных skills, knowledge topics и смыслового разбора.

## Архитектура

```text
Bitrix24 chat / local test
  ↓
Internal Orchestrator / Переговорщик
  ↓
Agent Registry
  ↓
Bitrix24 Specialist
  ↓
Tool Gateway + Policy Layer
  ↓
Bitrix REST / Portal Search / Documents

Bitrix24 webhooks / schedules
  ↓
Bitrix Workers / Automations
  ↓
Tool Gateway + State Stores
```

## Структура

```text
agents/
  internal_orchestrator/
    manifest.yaml
    instructions.md
    skills/
  bitrix24/
    manifest.yaml
    instructions.md
    automations/
  logistics/
    manifest.yaml
    instructions.md
    automations/
backend/ai_server/
  agents/
  orchestrators/
  tools/
  workers/
  knowledge.py
  skills.py
  registry.py
  models.py
var/
  README.md
  ...
```

## Быстрый запуск прототипа

```powershell
cd C:\Users\office3pc\PyProjects\AI-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:AI_SERVER_ENV_FILE="C:\Users\office3pc\PyProjects\BitrixAIAgent\.env,C:\Users\office3pc\PyProjects\BitrixAIAgent\.env.webhook.local,.env.local"
uvicorn --app-dir backend ai_server.main:app --reload
```

Проверка:

```text
GET  http://127.0.0.1:8000/health
GET  http://127.0.0.1:8000/agents
GET  http://127.0.0.1:8000/agents/bitrix24/automations
GET  http://127.0.0.1:8000/automations
GET  http://127.0.0.1:8000/bitrix/status
GET  http://127.0.0.1:8000/bitrix/search/status
GET  http://127.0.0.1:8000/bitrix/search/indexer/status
GET  http://127.0.0.1:8000/bitrix/oauth/status
GET  http://127.0.0.1:8000/logistics/vehicle-usage/status
GET  http://127.0.0.1:8000/admin/bitrix/search?q=...  (X-Admin-Secret)
POST http://127.0.0.1:8000/admin/bitrix/search/reindex  (X-Admin-Secret)
POST http://127.0.0.1:8000/admin/bitrix/search/reindex-delta  (X-Admin-Secret)
POST http://127.0.0.1:8000/admin/bitrix/search/reindex-content  (X-Admin-Secret)
POST http://127.0.0.1:8000/bitrix/events
```


## Hybrid RAG

Knowledge retrieval использует реальные embeddings через `fastembed`. Для локального запуска retrieval нужно установить extra:

```powershell
uv sync --extra dev --extra retrieval
```

```env
AI_SERVER_EMBEDDINGS_PROVIDER=fastembed
AI_SERVER_FASTEMBED_CACHE_DIR=var/embedding_models
```

## LLM model

Единственный автономный агент — `internal_orchestrator`. Он всегда использует
Pro-модель, определяет смысл запроса и передаёт Bitrix-исполнителю точную
структурированную команду. Bitrix не имеет собственного LLM и выполняет только
контракт, ACL/OAuth и конкретные REST-действия.

```env
AI_SERVER_ENV_FILE=.env,.env.local
AI_SERVER_LLM_PROVIDER=deepseek
AI_SERVER_LLM_MODEL=deepseek-v4-pro
AI_SERVER_ORCHESTRATOR_LLM_MODEL=deepseek-v4-pro
AI_SERVER_LLM_BASE_URL=
AI_SERVER_LLM_API_KEY=
AI_SERVER_LLM_MAX_TOKENS=3000
```

`GET /health` показывает провайдера, фактическую модель оркестратора, политику
`pro_only_fail_closed` и готовность LLM, но не показывает ключи.

Если нужно наложить env-файлы старого `BitrixAIAgent` и локальные секреты нового
сервера, укажите их через запятую или точку с запятой:

```env
AI_SERVER_ENV_FILE=C:\Users\office3pc\PyProjects\BitrixAIAgent\.env,C:\Users\office3pc\PyProjects\BitrixAIAgent\.env.webhook.local,.env.local
```

## Technical footer

Для внутренних каналов можно включить короткий технический footer только для
админов и директора. Footer строится не от "агента вообще", а от фактических
`model_usage` текущего ответа: какие агенты, провайдеры и модели участвовали.
Footer отражает фактически использованную оркестратором модель.
Клиентские каналы не должны подключать этот footer.

```env
AI_SERVER_TECH_FOOTER_ENABLED=true
AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS=1,9
AI_SERVER_TECH_FOOTER_BALANCE_ENABLED=true
AI_SERVER_TECH_FOOTER_BALANCE_CACHE_SECONDS=300
AI_SERVER_DEEPSEEK_BALANCE_BASE_URL=https://api.deepseek.com
```

## Runtime var и cutover

`var/` - локальный runtime-контур сервера. В Git хранится только каркас каталога,
а реальные данные остаются локальными: индексы, очереди, OAuth, старые диалоги,
audit, learning-события, вложения и drafts.

Журнал накопления примеров пишется в `var/learning_events.jsonl`. Это append-only
контур для будущих evals, разборов качества и датасетов дообучения: запрос,
ответ агента, handoff, действия, model usage и ручной feedback.

```env
LEARNING_EVENTS_ENABLED=true
LEARNING_EVENTS_CAPTURE_TEXT=true
LEARNING_EVENTS_MAX_TEXT_CHARS=8000
```

План переноса из старого `BitrixAIAgent/var`:

```powershell
uv run python scripts/import_bitrix_var.py --profile cutover
```

Фактический перенос выполнять после остановки старого сервиса:

```powershell
uv run python scripts/import_bitrix_var.py --profile cutover --execute
```

## Bitrix webhook contour

Новый сервер уже умеет принимать webhook-события Битрикс24 в совместимую SQLite
очередь:

```env
PUBLIC_BASE_URL=https://your-public-host.example
WEBHOOK_SECRET=change-me
BITRIX_REST_WEBHOOK_URL=https://example.bitrix24.ru/rest/...
BITRIX_BOT_ID=...
BITRIX_BOT_TOKEN=...
AGENT_DRY_RUN=true
```

По умолчанию worker очереди не запускается автоматически. Для обработки очереди
и отправки ответов в Bitrix-чат:

```env
AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED=true
AGENT_DRY_RUN=false
```

Тестовый чат для dev-контура создаётся отдельно, чтобы не смешивать старый
моноагентный сервис и новую связку `internal_orchestrator -> bitrix24`:

```powershell
uv run python scripts/create_bitrix_dev_chat.py --title "AI dev" --users 1,9
uv run python scripts/create_bitrix_dev_chat.py --title "AI dev" --users 1,9 --execute
```

Первая команда печатает dry-run план и замаскированный payload, вторая реально
создаёт групповой чат Bitrix24 через `imbot.v2.Chat.add`.

Если используем старый OAuth-бот из `BitrixAIAgent`, env-файлы нужно накладывать
так же, как в старом сервисе:

```powershell
uv run python scripts/create_bitrix_dev_chat.py `
  --env-file C:\Users\office3pc\PyProjects\BitrixAIAgent\.env `
  --env-file C:\Users\office3pc\PyProjects\BitrixAIAgent\.env.webhook.local `
  --oauth-db-path C:\Users\office3pc\PyProjects\BitrixAIAgent\var\bitrix_oauth.sqlite `
  --title "AI dev" --users 1,9 --execute
```

Контролируемая smoke-проверка сценария `tasks.task.add` через тот же Bitrix
channel processor:

```powershell
uv run python scripts/smoke_bitrix_task_create_flow.py `
  --env-file C:\Users\office3pc\PyProjects\BitrixAIAgent\.env `
  --env-file C:\Users\office3pc\PyProjects\BitrixAIAgent\.env.webhook.local `
  --oauth-db-path C:\Users\office3pc\PyProjects\BitrixAIAgent\var\bitrix_oauth.sqlite `
  --chat-id 3955 --dialog-id chat3955 --user-id 9 --confirm
```

Проверка состояния:

```text
GET http://127.0.0.1:8000/bitrix/status
GET http://127.0.0.1:8000/bitrix/webhook-events/status
```

## Bitrix OAuth local app

OAuth нужен не для общения с Оркестратором, а для полномочий write-действий:
создать задачу, закрыть задачу, записать результат или выполнить фоновое
действие от имени служебного actor. Локальное приложение Bitrix открывает
`/bitrix/app`, callback приходит на `/bitrix/oauth/callback`, install payload
принимается на `/bitrix/install`, а сохранённые токены лежат в
`var/bitrix_oauth.sqlite`.

```env
PUBLIC_BASE_URL=https://your-public-host.example
BITRIX_DOMAIN=example.bitrix24.ru
BITRIX_OAUTH_CLIENT_ID=
BITRIX_OAUTH_CLIENT_SECRET=
BITRIX_OAUTH_REQUIRED_FOR_WRITES=true
```

```text
GET  http://127.0.0.1:8000/bitrix/oauth/status
GET  http://127.0.0.1:8000/bitrix/oauth/start
GET  http://127.0.0.1:8000/bitrix/app
POST http://127.0.0.1:8000/bitrix/install
```

Кнопки в Bitrix-чате не являются отдельным бизнес-контуром. Если они нужны,
оставляем только кнопку-ссылку для первичной OAuth-регистрации пользователя.

## Bitrix portal search

Новый сервер умеет читать и обновлять PostgreSQL-индекс портала. После
индексации доступны:

```text
GET http://127.0.0.1:8000/bitrix/search/status
GET http://127.0.0.1:8000/bitrix/search/indexer/status
GET http://127.0.0.1:8000/admin/bitrix/search?q=договор&scope=documents
POST http://127.0.0.1:8000/admin/bitrix/search/reindex
POST http://127.0.0.1:8000/admin/bitrix/search/reindex-delta
POST http://127.0.0.1:8000/admin/bitrix/search/reindex-content
```

Эти административные запросы требуют заголовок `X-Admin-Secret`.
Поддерживаемые scope: `all`, `documents`, `files`, `tasks`, `projects`.

Фоновый индексатор metadata/delta/content включается явно:

```env
SEARCH_BACKGROUND_INDEXER_ENABLED=true
SEARCH_WEBHOOK_INDEXER_ENABLED=true
SEARCH_WEBHOOK_CONTENT_ENABLED=true
SEARCH_CONTENT_MAX_FILES=80
SEARCH_CONTENT_MAX_BYTES=20971520
SEARCH_CONTENT_ALLOWED_EXTENSIONS=.txt,.csv,.doc,.docx,.xlsx,.xls,.pdf
```

`portal_search` подключён как структурированный Bitrix tool для документов и
файлов. Смысл запроса, точные параметры и формат ответа определяет только
оркестратор; Bitrix проверяет доступ текущего пользователя и выполняет команду.

События закрытия задач также передаются оркестратору. Старые самостоятельные
Bitrix quality-control и supervisor удалены.

## Logistics vehicle usage

Утренний учёт служебных автомобилей по умолчанию выключен. При явном включении
его worker выполняет только детерминированное расписание, dedupe, отправку
заранее заданных сообщений и сохранение данных; самостоятельный LLM-специалист
`logistics` в runtime не создаётся.

```env
VEHICLE_USAGE_ENABLED=false
VEHICLE_USAGE_DRY_RUN=true
VEHICLE_USAGE_MANAGER_USER_ID=9
VEHICLE_USAGE_DIALOG_ID=chat9
VEHICLE_USAGE_REQUEST_TIME=08:30
VEHICLE_USAGE_REMINDER_INTERVAL_MINUTES=30
VEHICLE_USAGE_REMINDER_DELAYS_MINUTES=30,60
VEHICLE_USAGE_MAX_REMINDERS=2
VEHICLE_USAGE_WORKDAY_MODE=weekday
VEHICLE_USAGE_UNKNOWN_FILL_TIME=12:00
VEHICLE_USAGE_AUTO_DAY_OFF_TIME=18:00
VEHICLE_USAGE_ADMIN_NOTIFY_USER_IDS=1,9
VEHICLE_USAGE_ADMIN_USER_IDS=1
VEHICLE_USAGE_ALLOWED_USER_IDS=9
VEHICLE_USAGE_STAFF_ROSTER=1|15|Иван Петров;2|16|Пётр Иванов
```

```text
GET  http://127.0.0.1:8000/logistics/vehicle-usage/status
```

## Bitrix task closure from chat

Закрытие задачи по сообщению человека идёт через общий контур:
`internal_orchestrator -> structured Bitrix command -> numbered draft ->
confirmation`. Оркестратор определяет задачу, формирует четыре блока результата
и выбирает точные операции. Bitrix исполняет их после подтверждения и применяет
guardrails: OAuth/write-policy, проверку черновика и идемпотентность.

## Document contour

Поиск документов Bitrix выполняется структурированными инструментами
`bitrix24`. Смысл запроса, параметры и формат ответа определяет Pro-оркестратор.
Старые ПТО/Картотека LLM-прототипы и совместимые `/agent/...` маршруты удалены.

## Admin skills/RAG editor

Старый `project_shell` был админским tool для выполнения команд из Bitrix-чата.
В новом проекте его не переносим как произвольную консоль. Правильная замена -
отдельный админский LLM-специалист для редактирования `agents/*/skills` и
`agents/*/knowledge/topics`: read/list, подготовка diff, подтверждение человеком
и применение ограниченного patch без доступа к секретам и произвольным shell
командам.

## Документы

- `docs/00-vision.md` - цель и границы проекта.
- `docs/01-architecture.md` - слои и основные компоненты.
- `docs/02-agents.md` - роли агентов и контракт взаимодействия.
- `docs/03-client-support-flow.md` - клиентский сценарий техподдержки.
- `docs/04-security-and-policies.md` - безопасность, доступы, подтверждения.
- docs/05-mvp-roadmap.md - первая дорожная карта.
- docs/06-hybrid-rag.md - hybrid retrieval, RAG и skills.
- docs/07-bitrix-automations.md - фоновые Bitrix workers и порядок переноса.




