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
backend/ai_server/
  agents/
  orchestrators/
  tools/
  knowledge.py
  skills.py
  registry.py
  models.py
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
POST http://127.0.0.1:8000/orchestrator/test
```

Пример `POST /orchestrator/test`:

```json
{
  "text": "Найди просроченные задачи в Битриксе",
  "user_id": "9"
}
```

## Документы

- `docs/00-vision.md` - цель и границы проекта.
- `docs/01-architecture.md` - слои и основные компоненты.
- `docs/02-agents.md` - роли агентов и контракт взаимодействия.
- `docs/03-client-support-flow.md` - клиентский сценарий техподдержки.
- `docs/04-security-and-policies.md` - безопасность, доступы, подтверждения.
- `docs/05-mvp-roadmap.md` - первая дорожная карта.
