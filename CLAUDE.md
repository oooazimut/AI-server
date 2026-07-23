# Проект: AI-server

Главное архитектурное правило находится в `ORCHESTRATOR_AUTHORITY.md`:
`internal_orchestrator` — единственный текущий автономный агент, а все
подключённые специалисты являются исполнителями структурированных команд.

## Архитектурные принципы и стандарты кода
При проведении `/code-review ultra` строго оценивайте код на соответствие следующим концепциям:

### 1. Чистая Архитектура (Clean Architecture)
- **Направление зависимостей:** Внутренние слои (Domain, Use Cases) не должны зависеть от внешних слоев (Инфраструктура, UI, БД). Зависимости должны быть направлены строго внутрь.
- **Изоляция:** Проверяйте, чтобы детали реализации (например, ORM-модели, HTTP-библиотеки) не проникали в бизнес-логику.
- **Порты и Адаптеры:** Интерфейсы (порты) должны определяться на уровне бизнес-логики, а их реализация (адаптеры) — на уровне инфраструктуры.

### 2. Принципы SOLID
- **SRP:** Каждый класс/модуль должен иметь только одну причину для изменения. Ищите перегруженные контроллеры и сервисы-"боги".
- **OCP:** Код должен быть открыт для расширения, но закрыт для модификации (использование полиморфизма вместо бесконечных `if-else` или `switch`).
- **LSP:** Наследники или реализации интерфейсов не должны ломать поведение, ожидаемое от базового типа.
- **ISP:** Клиенты не должны зависеть от методов, которые они не используют. Разделяйте "толстые" интерфейсы.
- **DIP:** Модули верхних уровней не должны зависеть от модулей нижних уровней. Оба должны зависеть от абстракций.

### 3. Агентно-Ориентированное Программирование

### 4. Запахи кода (Code Smells)
- **Длинные методы и гигантские классы:** Сигнализируйте, если метод превышает 20-30 строк или класс разросся.
- **Завистливые функции (Feature Envy):** Метод класса обращается к данным другого класса чаще, чем к своим собственным.
- **Одержимость примитивами (Primitive Obsession):** Использование строк/чисел вместо Value Objects (например, `string email` вместо класса `Email`).
- **Длинные списки параметров:** Передача более 3-4 аргументов в метод (требуйте группировки в DTO).
- **Дублирование кода (DRY):** Выделяйте повторяющуюся логику.


## ⚠️ ЗАКОН ПРОЕКТА: ЧИСТАЯ АРХИТЕКТУРА

**Любая работа над кодом — добавление фичи, рефакторинг, исправление бага, написание теста — ОБЯЗАНА соблюдать принципы Чистой Архитектуры. Это не рекомендация. Это условие приёмки любого изменения.**

Нарушение архитектурных принципов, перечисленных ниже, недопустимо ни при каких обстоятельствах, ни ради скорости, ни ради удобства, ни «временно».

---

## Архитектурные слои и направление зависимостей

```
channels/          ←  точки входа (HTTP webhooks, Bitrix events)
orchestrators/     ←  маршрутизация запросов между специалистами
agents/            ←  бизнес-логика специалистов (Use Cases)
tools/             ←  инструменты агентов (порты для внешних операций)
integrations/      ←  инфраструктура (Bitrix API, OAuth, поисковый индекс)
workers/           ←  фоновые задачи (воркеры, адаптеры)
```

**Правило зависимостей (АБСОЛЮТНОЕ):**
- Зависимости направлены СТРОГО внутрь: `channels → orchestrators → agents → tools → integrations`
- `agents/` не импортирует из `channels/` или `workers/` — никогда
- `integrations/` не знает об `agents/` или `tools/` — никогда
- `channels/` не импортирует из `workers/` напрямую — только через порты (`channels/ports.py`)
- `workers/` может зависеть от `agents/` (внешнее → внутреннее — допустимо)

---

## Обязательные правила при написании кода

### 1. Порты и адаптеры (DIP)

Каждая внешняя зависимость описывается через `Protocol` (порт). Конкретные реализации (адаптеры) — в инфраструктурном слое.

Существующие порты:
- `agents/ports.py` — очереди, каналы, хранилище оркестратора, scheduler и публикация результатов
- `agents/bitrix24/ports.py` — `TaskDraftStorePort`
- `tools/bitrix_ports.py` — `BitrixTaskPort`, `BitrixUserPort`, `BitrixBotPort`, `BitrixDiskPort`, `BitrixRestPort`, `BitrixToolClientPort`, `BitrixFileDownloadPort`, `BitrixWritePort`
- `channels/ports.py` — технические channel-протоколы

**При добавлении новой внешней зависимости** — сначала порт в нужном `ports.py`, потом конкретный класс.

### 2. Инъекция зависимостей (не Service Locator)

`get_settings()` запрещено вызывать внутри методов бизнес-логики. Допустимо только:
- в factory-методах (`build()`, `lifespan`)
- в точках входа (`__init__` с `settings or get_settings()` — только для backward compat)

Новые классы должны получать `Settings` через параметр конструктора.

### 3. Расширяемость специалистов (OCP)

Добавление специалиста N не требует изменения существующего кода кроме:
1. Добавить поле в `SpecialistDeps` (`specialists.py`)
2. Заполнить поле в `startup.py` и `agent_worker.py`
3. Написать executor-класс с `execute_structured_command()`; свободный текст и собственный LLM запрещены
4. Создать `manifest.yaml` с `entrypoint`

`BitrixWebhookProcessor`, `build_specialist_registry()` — не трогать.

### 4. Агентная изоляция

- Текущий специалист не принимает смысловых решений: он исполняет точную структурированную команду оркестратора
- Взаимодействие между агентами — только через `PlanAuthoritativeOrchestrator` (hub-and-spoke)
- Прямые вызовы agent→agent запрещены
- Каждый агент владеет собственным `AgentStore` с namespace по `agent_id`
- Toolset инжектируется снаружи (через `SpecialistDeps`), агент не создаёт инструменты сам
- Per-request данные (user_id, dialog_key) передаются через `task.context`, не через поля агента

### 5. Request-scoped инструменты

`BitrixToolset` с `dialog_key` и `user_id` создаётся per-request в `_build_request_toolsets()` и передаётся через `task.context["_bitrix_tools"]`. Синглтоны с пользовательскими данными недопустимы.

### 6. Запахи кода — немедленное исправление

При обнаружении следующего — исправить до слияния:
- **Feature Envy** — метод A обращается к данным объекта B чаще, чем к своим
- **Service Locator** — `get_settings()` внутри метода бизнес-логики
- **Длинный if/elif** для диспетчеризации инструментов — заменить dispatch-таблицей (dict)
- **Обратная зависимость** — внутренний слой импортирует внешний

### 7. Доменная изоляция агентов (АБСОЛЮТНО)

Каждый агент-специалист знает ТОЛЬКО о том, что входит в его предметную зону.
Инструменты специалиста соответствуют ИСКЛЮЧИТЕЛЬНО его области ответственности.

**Запрещено:**
- любому executor/operator принимать свободный текст и повторно определять намерение;
- `Bitrix24Specialist` читать данные логистики или самостоятельно менять указанный инструмент;
- логистической и диагностической автоматизации иметь LLM;
- любому исполнителю импортировать инфраструктуру чужой предметной зоны.

**Действующие контуры:**
- `PlanAuthoritativeOrchestrator`: единственный Pro-смысловой контур;
- `Bitrix24Specialist`: структурированный executor команд Bitrix;
- `logistics`: детерминированная автоматизация учёта машин;
- `diagnost`: технический сборщик событий, трасс и инцидентов.

**Коммуникация:**
- Вся коммуникация между агентами — ТОЛЬКО через `PlanAuthoritativeOrchestrator`
- Оркестратор — единственный посредник между людьми и специалистами
- Специалист не знает о существовании других специалистов
- Исполнитель возвращает структурированный результат → оркестратор собирает ответ → канал доставляет адресату

---

## О проекте

Корпоративный мультиагентный AI-сервер. FastAPI-приложение, принимающее события Bitrix24 (чат-бот, вебхуки) и маршрутизирующее запросы через оркестратор к агентам-специалистам. Используется сотрудниками компании.

**Оркестратор** (`internal_orchestrator`) — старший ИИ-агент (Переговорщик). Получает запросы от людей и специалистов, принимает решения о маршрутизации, координирует специалистов, синтезирует ответы. Имеет собственные RAG/skills/knowledge/историю диалогов.

**Действующие компоненты:**
| ID | Класс | Область |
|---|---|---|
| `bitrix24` | `Bitrix24Specialist` | Подключённый executor: задачи, календарь, диск и поиск |
| `logistics` | `LogisticsSpecialist` | Структурированный executor учёта машин без собственной LLM |
| `diagnost` | operator manifest | Технический сборщик, не пользовательский агент |

Старые LLM-прототипы `pto`, `kartoteka`, `logistics` и `diagnost` удалены. Физический Logistics executor сохранён без собственной LLM.

---

## Структура проекта

```
backend/ai_server/
├── main.py                         # FastAPI app, подключение роутеров
├── startup.py                      # lifespan: инициализация HTTP-инфраструктуры (без consumer loops)
├── agent_worker.py                 # Standalone-процесс: все consumer loops (orchestrator, specialists, workers)
├── settings.py                     # Settings (pydantic), get_settings(), все env-переменные
├── models.py                       # Pydantic-модели: AgentTask, AgentResult, ActionRecord, ...
├── specialists.py                  # Specialist Protocol, SpecialistDeps, build_specialist_registry
├── registry.py                     # Загрузка AgentManifest из manifest.yaml
│
├── agents/                         # Бизнес-логика специалистов
│   ├── base.py                     # Общий lifecycle исполнителей и tool-вызовов
│   ├── ports.py                    # Queue, Channel, Store, Scheduler и ResultPublisher Protocol
│   ├── bitrix24/                   # Специалист Bitrix24
│   │   ├── __init__.py             # Ре-экспорт публичного API Bitrix24 executor
│   │   ├── specialist.py           # Bitrix24Specialist
│   │   ├── ports.py                # TaskDraftStorePort
│   │   └── tools/                  # AgentTool-классы специалиста
│   │       ├── task_create.py      # TaskCreateDraftTool, TaskCreateConfirmTool, TaskDraftDiscardTool, BitrixTaskCreateDraft
│   │       ├── bitrix_api.py       # BitrixApiTool
│   │       └── portal_search.py    # PortalSearchTool
│   └── logistics/tools/            # Детерминированные инструменты vehicle_usage
│
├── orchestrators/
│   ├── internal.py                 # Общий lifecycle/transport единственного оркестратора
│   ├── plan_authoritative.py       # Единственный живой Pro plan-authoritative runtime
│   ├── bitrix_semantics.py         # Смысл, шаблоны и параметры Bitrix-команд
│   └── bitrix_formatter.py         # Единственный форматтер пользовательских Bitrix-ответов
│
├── channels/
│   ├── bitrix.py                   # BitrixWebhookProcessor (точка входа Bitrix-событий)
│   └── ports.py                    # Технические channel-протоколы
│
├── tools/
│   ├── bitrix_ports.py             # Все Bitrix-порты (Task, User, Bot, Disk, ...) — Protocol-интерфейсы
│   ├── bitrix_policy.py            # Политики write-операций (allow/confirm/deny)
│   ├── document_access/            # DocumentToolset (PTO): поиск, чтение, сравнение документов
│   └── vehicle_usage.py            # VehicleUsageStorePort и детерминированные helpers
│
├── integrations/
│   └── bitrix/
│       ├── client.py               # BitrixClient (HTTP к Bitrix24 REST API)
│       ├── oauth.py                # BitrixOAuthService
│       ├── dialog_state.py         # BitrixPendingActionService, DialogStateStore
│       ├── events.py               # Парсинг входящих событий
│       └── portal_search/          # PortalSearchIndex: SQLite-индекс задач/диска/проектов
│
├── workers/
│   ├── bitrix/
│   │   ├── webhook_event_queue.py  # WebhookEventQueue: SQLite-очередь с dedupe/retry
│   │   ├── search_indexer.py       # PortalSearchIndexerWorker: фоновая индексация
│   │   ├── search_webhook_adapter.py   # SearchWebhookHandlerAdapter
│   │   ├── search_webhook_indexer.py   # Обработка disk-webhook для индекса
│   │   ├── reconciler.py           # Сверка потерянных событий
│   │   └── task_close_direct_dispatcher.py # Передача закрытия задачи оркестратору
│   └── logistics/
│       └── staff_sync.py           # Синхронизация сотрудников, run_staff_sync
│
├── routes/
│   ├── admin.py                    # GET /health, GET /agents, GET /automations
│   ├── bitrix.py                   # Bitrix webhook endpoints
│   └── logistics.py                # GET /logistics/vehicle-usage/status
│
├── agent_store.py                  # AgentStore: SQLite per-agent namespace
├── agent_scheduler.py              # AgentScheduler (APScheduler), SchedulerPort re-export
├── llm.py                          # LLMClient Protocol, OpenAICompatibleLLMClient
├── retrieval.py                    # HybridKnowledgeRetriever (RAG)
├── knowledge.py                    # MarkdownKnowledgeBase
├── skills.py                       # SkillStore
├── transcription.py                # STT: OpenAI / Yandex SpeechKit
├── attachments.py                  # AttachmentService
├── learning.py                     # LearningEventRecorder
├── technical_footer.py             # TechnicalFooterService (служебная подпись ответа)
└── utils.py                        # MOSCOW_TZ, optional_int, confidence, ...

agents/                             # Манифесты агентов (вне пакета)
├── bitrix24/
│   ├── manifest.yaml
│   ├── instructions.md
│   ├── skills/
│   └── knowledge/
├── internal_orchestrator/
│   ├── manifest.yaml
│   ├── instructions.md
│   ├── skills/
│   └── knowledge/
├── logistics/
│   └── manifest.yaml
└── diagnost/
    └── manifest.yaml

tests/                              # pytest, без сети — все внешние вызовы мокируются
var/                                # Рантайм: SQLite, вложения, индексы — не коммитится
```

---

## Манифесты агентов

Манифест (`manifest.yaml`) описывает агента декларативно. Загружается через `registry.py`. Поля:
- `id` — уникальный идентификатор
- `kind` — `specialist` | `orchestrator` | `operator`
- `audience` — `employee` | (будущее: `customer`)
- `entrypoint` — `ai_server.agents.module.ClassName` — динамически загружается через `importlib`
- `capabilities` — список строк, описывает возможности агента
- `tools` — список инструментов агента
- `automations` — список фоновых автоматизаций (воркеры, расписания)

Инструкции агента — в `instructions.md` рядом с манифестом. Знания — в `knowledge/`. Навыки — в `skills/`.

---

## Добавление нового специалиста

1. Создать `agents/<id>/manifest.yaml` с `kind: specialist`, `reasoning_mode: executor`, `entrypoint: ai_server.agents.<module>.<Class>`
2. Реализовать `execute_structured_command()` с версионированной схемой команды; собственный LLM, свободный текст и смена выбранного инструмента запрещены
3. Добавить поля зависимостей в `SpecialistDeps` (`specialists.py`)
4. Заполнить поля в `startup.py`

Исполнитель без `execute_structured_command()` реестром не создаётся. Новый
автономный тип агента требует отдельного явного решения пользователя.

---

## Хранилище агента (AgentStore)

Каждый агент-специалист и оркестратор владеет **собственным изолированным хранилищем**. Хранилище — это порт (`AgentDialogStorePort`, `agents/ports.py`): агент зависит от абстракции, конкретный адаптер инжектируется снаружи через `SpecialistDeps`.

### Обязанности хранилища

- **История диалогов** — обязательно: `load_turns(dialog_key)`, `append_turn(dialog_key, user_text, response)`
- **Домен-специфичные данные** — опционально: черновики, индексы файлов, кэши — каждый агент добавляет нужные таблицы/коллекции в своём адаптере
- **Изоляция по namespace** — агенты не пересекаются; хранилище одного агента не видно другому

### DIP

Агент знает только о порте (`AgentDialogStorePort`). Конкретная реализация передаётся через `SpecialistDeps` → `build()`. Замена хранилища (PostgreSQL → что угодно) требует только нового адаптера — агент не меняется.

### Текущая реализация: PostgreSQL

Все агенты используют адаптер `PostgresAgentSchema` (`integrations/postgres/agent_schema.py`). Каждый агент — отдельная PostgreSQL-схема:

```python
# integrations/postgres/<agent_id>_agent.py
from .agent_schema import PostgresAgentSchema

class PostgresMyAgentStore(PostgresAgentSchema):
    _SCHEMA = "my_agent"  # своя схема в PostgreSQL

    async def ensure_schema(self) -> None:
        await super().ensure_schema()  # создаёт my_agent.dialog_history
        async with await self._connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS my_agent.my_table (
                    id BIGSERIAL PRIMARY KEY,
                    search_text TEXT NOT NULL,
                    data JSONB NOT NULL
                )
            """)
```

Базовый класс даёт: `ensure_schema()`, `load_turns()`, `append_turn()`, `_connect()` (async), `_sync_connect()` (sync).

Поиск по домен-таблицам — PostgreSQL `ILIKE` по колонке `search_text`, не FTS.

**Существующие адаптеры:**

| Файл | Класс | Схема |
|---|---|---|
| `integrations/postgres/bitrix_agent.py` | `PostgresBitrixAgentStore` | `bitrix24` |
| `integrations/postgres/orchestrator_agent.py` | `PostgresOrchestratorStore` | `orchestrator` |
| `integrations/postgres/vehicle_usage.py` | `PostgresVehicleUsageStore` | `vehicle_usage` |

Хранилище нового исполнителя добавляется только вместе с действующим структурированным контрактом.

---

## Процессная архитектура (Sprint 27)

Сервис разделён на два независимых системных процесса:

| Юнит | Процесс | Назначение |
|---|---|---|
| `ai-server` | `uvicorn --workers 1` | HTTP: приём вебхуков → Redis-очередь и статус-эндпоинты |
| `ai-server-worker` | `python -m ai_server.agent_worker` | Consumer loops: оркестратор, специалисты, фоновые воркеры — **1 экземпляр** |

**Правило:** Consumer loops (`orchestrator.run`, `specialist.run`, `run_webhook_event_worker`) запускаются **только** в `agent_worker.py`. В `startup.py` (uvicorn) их запускать запрещено — это CA-нарушение и причина ghost turns.

---

## Фоновые воркеры

Все запускаются в `agent_worker.py`:

| Воркер | Файл | Назначение |
|---|---|---|
| WebhookEventQueue | `workers/bitrix/webhook_event_queue.py` | Redis-очередь Bitrix webhook-событий: dedupe, retry с exponential backoff |
| PortalSearchIndexerWorker | `workers/bitrix/search_indexer.py` | Фоновая синхронизация задач/диска/проектов в поисковый индекс |
| SearchWebhookHandlerAdapter | `workers/bitrix/search_webhook_adapter.py` | Обновление индекса по disk-webhook в реальном времени |
| BitrixReconciler | `workers/bitrix/reconciler.py` | Компенсация потерянных webhook-событий (сверка с Bitrix) |
| TaskCloseDirectDispatcher | `workers/bitrix/task_close_direct_dispatcher.py` | Передача события закрытия задачи оркестратору |
| run_staff_sync | `workers/logistics/staff_sync.py` | Синхронизация списка сотрудников из Bitrix для vehicle usage |

---

## Конфигурация

Все параметры — через переменные окружения. Локально: `.env.local` (не коммитится). Прод: `/opt/ai-server/.env.prod` (не в git).

Полный список с типами и дефолтами — `backend/ai_server/settings.py::Settings`.

**Ключевые переменные:**

| Группа | Переменные |
|---|---|
| Bitrix24 REST | `BITRIX_REST_WEBHOOK_URL`, `BITRIX_BOT_TOKEN`, `BITRIX_BOT_ID`, `BITRIX_DOMAIN` |
| Bitrix24 OAuth | `BITRIX_OAUTH_CLIENT_ID`, `BITRIX_OAUTH_CLIENT_SECRET`, `BITRIX_OAUTH_ENABLED` |
| LLM | `AI_SERVER_LLM_PROVIDER`, `AI_SERVER_LLM_MODEL`, `AI_SERVER_LLM_BASE_URL`, `AI_SERVER_LLM_API_KEY` |
| STT | `STT_PROVIDER` (`openai` / `yandex_speechkit`), `OPENAI_API_KEY`, `YANDEX_API_KEY` |
| Воркеры | `WEBHOOK_EVENT_QUEUE_ENABLED`, `AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED`, `RECONCILE_ENABLED` |
| Логистика | `VEHICLE_USAGE_ENABLED`, `VEHICLE_USAGE_MANAGER_USER_ID`, `VEHICLE_USAGE_DIALOG_ID` |
| Безопасность | `WEBHOOK_SECRET`, `AGENT_DRY_RUN` |

Новые env-переменные: добавить в `Settings` с дефолтом и при необходимости `@property *_configured`.

---

## Транскрипция голоса

Настраивается через `STT_PROVIDER`:

- **`openai`** — `gpt-4o-transcribe`, лимит 25 МБ, любой аудиоформат. Требует `OPENAI_API_KEY`.
- **`yandex_speechkit`** — синхронный API, лимит 1 МБ, требует OggOpus. При `YANDEX_SPEECHKIT_CONVERT_TO_OGG=true` конвертирует через `ffmpeg`. Требует `YANDEX_API_KEY` или `YANDEX_IAM_TOKEN` + `YANDEX_FOLDER_ID`.

---

## Команды

```bash
uv sync --extra dev --extra retrieval   # установка зависимостей
uv run pytest -v                        # все тесты
uv run pytest -v -q                     # тесты (краткий вывод)
uv run ruff check . && uv run ruff format --check .   # линт — обязателен перед коммитом
uv run ruff check --fix . && uv run ruff format .     # авто-исправление форматирования
uv run uvicorn ai_server.main:app --reload            # локальный запуск
```

---

## Тесты

- Размещаются в `tests/`, запускаются через pytest, `pythonpath = ["backend"]`
- Сеть не используется: все внешние вызовы мокируются
- Тесты, зависящие от Settings, изолируют окружение:
  ```python
  monkeypatch.setenv("AI_SERVER_ENV_FILE", "")  # отключить .env.local
  monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://example.bitrix24.ru/...")
  ```
  без `AI_SERVER_ENV_FILE=""` подтянется `.env.local` и тест получит боевые значения

---

## Ветки и деплой

- `dev` — основная ветка разработки. Прямые пуши в `dev`/`main` запрещены всем кроме владельца. Остальные: feature-ветка → PR в `dev`.
- `main` — продакшен. Путь: PR `dev` → `main` с зелёным CI (lint + test).
- **Пуш в `main` автоматически деплоится** (GitHub Actions job `deploy`): SSH → `git pull` → `uv sync` → `sudo systemctl restart ai-server` → `sudo systemctl restart ai-server-worker`.
- Прод живёт в `/opt/ai-server`, конфиг — `/opt/ai-server/.env.prod`.
- Два systemd-юнита: `ai-server` и `ai-server-worker`. Планировщик работает внутри единственного worker-процесса.

> **ЗАКОН ДЕПЛОЯ:** Любые изменения на проде — только через PR `dev` → `main` на GitHub. Прямое редактирование файлов на сервере (`/opt/ai-server`) строго запрещено: хотфиксы теряются при следующем деплое, а `git pull` блокируется из-за локальных изменений. Срочный хотфикс = ветка → PR → merge → автодеплой.

---

## Операционные уроки из staging-работ

Перед любыми действиями с сервером, staging/prod, Redis/Postgres или GitHub перечитай этот раздел. Он фиксирует повторяющиеся ошибки, которые уже приводили к лишним попыткам.

- **PowerShell + SSH quoting:** не передавай сложные SQL/Python/shell-фрагменты через многоуровневые `"`/`'`. Для многострочного Python используй безопасный паттерн: локально собрать код в base64 и на сервере выполнить `python -c 'import base64,sys; exec(base64.b64decode(sys.argv[1]))' <base64>`. Для SQL предпочитай простые `psql -At -F '|' -c 'select ...'` без вложенных запятых в PowerShell-строке или выноси сложную логику в Python/base64.
- **sudo и stdin:** не совмещай `printf password | sudo -S ...` с here-doc (`<<'PY'`) или командами, которым тоже нужен stdin. Here-doc перехватывает stdin, sudo получает не пароль и падает. Если нужен sudo + Python, передавай код через `-c`/base64/argv, а не через stdin.
- **Не используй удалённые `$(...)` внутри PowerShell double quotes:** PowerShell может попытаться интерпретировать `$()` локально. Если нужен timestamp или динамическое значение, вычисли его отдельно локально либо используй простой серверный вызов без вложенной подстановки.
- **Git в `/opt/ai-server-staging`:** каталог принадлежит пользователю `azimut`. Не выполняй `sudo git` в staging repo: у root нет GitHub SSH-ключа и можно создать root-owned refs/logs. Git-команды выполнять так: `sudo -u azimut git -c safe.directory=/opt/ai-server-staging ...`. Если уже появились root-owned `.git/refs` или `.git/logs`, исправлять только конкретный staging `.git`-путь, не трогая production.
- **`safe.directory`:** для серверных git-команд в `/opt/ai-server-staging` и `/opt/ai-server` используй `git -c safe.directory=<path> ...`, а не глобальное изменение конфигурации, если это не требуется явно.
- **Staging reset/switch:** перед `reset --hard` или переключением ветки всегда сначала проверить `git status --short --branch`. Делать это только в `/opt/ai-server-staging`; production-папку между ветками не переключать.
- **Production vs staging:** production `/opt/ai-server` не трогать при тестах. Для staging проверять Redis DB 1, staging DB `ai_server_staging`, staging services `ai-server-staging.service` и `ai-server-staging-worker.service`. Production остаётся на Redis DB 0 и своих service units.
- **Секреты:** при проверке `.env`, webhook, API keys и токенов не печатать значения. Печатать только `configured`, имя переменной, модель, host/db/prefix без секретной части.
- **Локальные тесты:** в Codex sandbox может не быть `python`, `py`, `pytest` или `ruff`. Сначала проверь `.venv`; если её нет, для синтаксиса можно использовать bundled Python, а полноценные `pytest/ruff` запускать в серверном/staging venv или после явной установки зависимостей.
- **Логи и диагностика:** для “проверь сообщение” сначала собрать: service state, branch+commit, `/health`, Redis pending/processing, `journalctl` за последние минуты, последние `diagnost.events`. Не делать вывод по одному источнику.

---

## Конвенции кода

- Python 3.11+, ruff: длина строки 120, double quotes, target py311
- Настройки линта — в `pyproject.toml`
- Секреты (токены, ключи API, пароли) **никогда не коммитятся**: репозиторий публичный
- Комментарии только там, где WHY неочевидно — не описывать WHAT делает код
- Без backwards-compatibility заглушек: если что-то удалено — удалено полностью

---

## Тестовое покрытие (ОБЯЗАТЕЛЬНО)

**Любой новый код обязан сопровождаться тестами.** Это не рекомендация — это условие приёмки.

### Правила:

1. **Новый модуль** → новый `tests/test_<module>.py` с тестами всех публичных методов.
2. **Новый инструмент агента** (`AgentTool`) → прямые тесты через `.execute(args, user_id=...)`, без поднятия специалиста.
3. **Новый Postgres-адаптер** → тесты с мокированием `_sync_connect()` / `_connect()` через `monkeypatch` или `patch.object`.
4. **Новая бизнес-логика** (чистые функции, builders, helpers) → тесты без моков.
5. **Новый HTTP-эндпоинт** → тест через `TestClient(app)`.

### Что мокировать:

- Внешние сервисы (Bitrix API, Redis, PostgreSQL) — **всегда мокировать**, сеть в тестах запрещена.
- `_sync_connect()` / `_connect()` в Postgres-сторах — через `monkeypatch.setattr(store, "_sync_connect", ...)`.
- `AsyncMock` для async-методов клиентов.

### Пример минимального теста инструмента:

```python
def test_my_tool_ok():
    client = AsyncMock()
    client.some_method = AsyncMock(return_value={"id": 1})
    tool = MyTool(client=client)
    result = anyio_run(tool.execute({"arg": "value"}, user_id=9))
    assert result.status == ToolStatus.OK
```

### Проверка покрытия перед коммитом:

```bash
uv run pytest -q   # все тесты должны быть зелёными
```
