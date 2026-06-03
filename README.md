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
uvicorn backend.ai_server.main:app --reload
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
GET  http://127.0.0.1:8000/bitrix/search?q=...
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

Проверка состояния:

```text
GET http://127.0.0.1:8000/bitrix/status
GET http://127.0.0.1:8000/bitrix/webhook-events/status
```

## Bitrix portal search

Новый сервер умеет читать локальный индекс портала из `var/search_index.sqlite`.
После cutover-миграции старого `BitrixAIAgent/var` доступны:

```text
GET http://127.0.0.1:8000/bitrix/search/status
GET http://127.0.0.1:8000/bitrix/search?q=договор&scope=documents
```

Поддерживаемые scope: `all`, `documents`, `files`, `tasks`, `projects`.

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




