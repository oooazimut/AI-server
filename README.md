# AI Server

Корпоративный сервер ИИ-агентов для офисных и клиентских сценариев.

Первый MVP: разделить старый автономный `BitrixAIAgent` на две роли:

- `internal_orchestrator` - входная точка и маршрутизатор для сотрудников;
- `bitrix24` - узкий специалист по Битрикс24 со своими instructions, skills, knowledge topics и tools.

## Архитектура

```text
Bitrix24 chat / local test
  ↓
Internal Orchestrator
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
    skills/
    knowledge/topics/
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
GET  http://127.0.0.1:8000/agents/bitrix24/skills
GET  http://127.0.0.1:8000/agents/bitrix24/automations
GET  http://127.0.0.1:8000/automations
GET  http://127.0.0.1:8000/bitrix/status
GET  http://127.0.0.1:8000/bitrix/search/status
GET  http://127.0.0.1:8000/bitrix/search/indexer/status
GET  http://127.0.0.1:8000/bitrix/search?q=...
POST http://127.0.0.1:8000/bitrix/search/reindex
POST http://127.0.0.1:8000/bitrix/search/reindex-delta
POST http://127.0.0.1:8000/bitrix/search/reindex-content
POST http://127.0.0.1:8000/bitrix/events
POST http://127.0.0.1:8000/orchestrator/test
```

Пример `POST /orchestrator/test`:

```json
{
  "text": "Найди просроченные задачи в Битриксе",
  "user_id": "9"
}
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

Модель верхнего уровня зафиксирована в конфиге как `deepseek-v4-flash`.
Bitrix24-специалист работает как LLM-субагент: сначала модель выбирает tool calls
и передаёт структурированные аргументы, затем backend-tools выполняют только
валидацию, policy/OAuth и конкретные Bitrix REST действия. Backend не должен
выбирать бизнес-сценарий вместо LLM-субагента.

```env
AI_SERVER_ENV_FILE=.env,.env.local
AI_SERVER_LLM_PROVIDER=deepseek
AI_SERVER_LLM_MODEL=deepseek-v4-flash
AI_SERVER_LLM_BASE_URL=
AI_SERVER_LLM_API_KEY=
AI_SERVER_LLM_MAX_TOKENS=3000
```

`GET /health` показывает `llm_provider`, `llm_model` и `llm_configured`, но не
показывает ключи.

Если нужно наложить env-файлы старого `BitrixAIAgent` и локальные секреты нового
сервера, укажите их через запятую или точку с запятой:

```env
AI_SERVER_ENV_FILE=C:\Users\office3pc\PyProjects\BitrixAIAgent\.env,C:\Users\office3pc\PyProjects\BitrixAIAgent\.env.webhook.local,.env.local
```

## Technical footer

Для внутренних каналов можно включить короткий технический footer только для
админов и директора. Footer строится не от "агента вообще", а от фактических
`model_usage` текущего ответа: какие агенты, провайдеры и модели участвовали.
Если в ответе работали только deterministic skills/API, footer так и пишет, что
LLM не использовалась. Клиентские каналы не должны подключать этот footer.

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
audit, вложения и drafts.

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

## Bitrix portal search

Новый сервер умеет читать и обновлять локальный индекс портала
`var/search_index.sqlite`. После cutover-миграции старого `BitrixAIAgent/var`
доступны:

```text
GET http://127.0.0.1:8000/bitrix/search/status
GET http://127.0.0.1:8000/bitrix/search/indexer/status
GET http://127.0.0.1:8000/bitrix/search?q=договор&scope=documents
POST http://127.0.0.1:8000/bitrix/search/reindex
POST http://127.0.0.1:8000/bitrix/search/reindex-delta
POST http://127.0.0.1:8000/bitrix/search/reindex-content
```

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

`portal_search` также подключён как Bitrix tool и используется Bitrix24-специалистом
для запросов про документы, файлы, договоры и портал.
## Документы

- `docs/00-vision.md` - цель и границы проекта.
- `docs/01-architecture.md` - слои и основные компоненты.
- `docs/02-agents.md` - роли агентов и контракт взаимодействия.
- `docs/03-client-support-flow.md` - клиентский сценарий техподдержки.
- `docs/04-security-and-policies.md` - безопасность, доступы, подтверждения.
- docs/05-mvp-roadmap.md - первая дорожная карта.
- docs/06-hybrid-rag.md - hybrid retrieval, RAG и skills.
- docs/07-bitrix-automations.md - фоновые Bitrix workers и порядок переноса.




