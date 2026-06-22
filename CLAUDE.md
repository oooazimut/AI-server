# Проект: AI-server

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
- `agents/ports.py` — `SchedulerPort`, `BitrixToolsetPort`, `PtoToolsetPort`, `VehicleUsageToolsetPort`, `AgentDialogStorePort`, `SpecialistOutputPort`
- `integrations/bitrix/ports.py` — `BitrixTaskPort`, `BitrixUserPort`, `BitrixBotPort`, `BitrixDiskPort`, `BitrixRestPort`, `BitrixToolClientPort`, `BitrixFileDownloadPort`, `BitrixWritePort`, `BitrixSupervisorPort`
- `channels/ports.py` — `SearchWebhookHandlerPort`, `QualityControlHandlerPort`

**При добавлении новой внешней зависимости** — сначала порт в нужном `ports.py`, потом конкретный класс.

### 2. Инъекция зависимостей (не Service Locator)

`get_settings()` запрещено вызывать внутри методов бизнес-логики. Допустимо только:
- в factory-методах (`build()`, `lifespan`)
- в точках входа (`__init__` с `settings or get_settings()` — только для backward compat)

Новые классы должны получать `Settings` через параметр конструктора.

### 3. Расширяемость специалистов (OCP)

Добавление специалиста N не требует изменения существующего кода кроме:
1. Добавить поле в `SpecialistDeps` (`specialists.py`)
2. Заполнить поле в `startup.py`
3. Написать класс специалиста с методом `build()`
4. Создать `manifest.yaml` с `entrypoint`

`BitrixWebhookProcessor`, `build_specialist_registry()` — не трогать.

### 4. Агентная изоляция

- Специалист принимает решения автономно в рамках своих инструментов
- Взаимодействие между агентами — только через `InternalOrchestrator` (hub-and-spoke)
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
- `LogisticsSpecialist` с инструментами отправки сообщений в Bitrix или прямым вызовом Bitrix API
- `Bitrix24Specialist` с инструментами чтения `vehicle_usage` БД или данных других специалистов
- `PtoSpecialist` с инструментами работы с задачами Bitrix или vehicle_usage
- Любой специалист, импортирующий или вызывающий инфраструктуру чужой предметной зоны

**Допустимые инструменты по зонам:**
- `LogisticsSpecialist`: `vehicle_usage_context`, `vehicle_usage_save_draft`, `vehicle_usage_save_report`
- `Bitrix24Specialist`: `bitrix_task_*`, `bitrix_disk_*`, `bitrix_send_message`, `bitrix_notify_users`
- `PtoSpecialist`: `document_search`, `document_read`, `document_compare`
- `InternalOrchestrator`: `call_specialist_{id}`, `schedule_reminder` (координация, не бизнес-логика)

**Коммуникация:**
- Вся коммуникация между агентами — ТОЛЬКО через `InternalOrchestrator`
- Оркестратор — единственный посредник между людьми и специалистами
- Специалист не знает о существовании других специалистов
- Специалист инициирует исходящие задачи через `SpecialistOutputPort` → оркестратор → адресат

---

## О проекте

Корпоративный мультиагентный AI-сервер. FastAPI-приложение, принимающее события Bitrix24 (чат-бот, вебхуки) и маршрутизирующее запросы через оркестратор к агентам-специалистам. Используется сотрудниками компании.

**Оркестратор** (`internal_orchestrator`) — старший ИИ-агент (Переговорщик). Получает запросы от людей и специалистов, принимает решения о маршрутизации, координирует специалистов, синтезирует ответы. Имеет собственные RAG/skills/knowledge/историю диалогов.

**Специалисты:**
| ID | Класс | Область |
|---|---|---|
| `bitrix24` | `Bitrix24Specialist` | Задачи, диск, поиск по порталу, документы, доставка сообщений |
| `pto` | `PtoSpecialist` | Технические документы, регламенты, сравнение таблиц |
| `logistics` | `LogisticsSpecialist` | Учёт служебных автомобилей, утренние отчёты |

---

## Структура проекта

```
backend/ai_server/
├── main.py                         # FastAPI app, подключение роутеров
├── startup.py                      # lifespan: создание всех сервисов и воркеров
├── settings.py                     # Settings (pydantic), get_settings(), все env-переменные
├── models.py                       # Pydantic-модели: AgentTask, AgentResult, ActionRecord, ...
├── specialists.py                  # Specialist Protocol, SpecialistDeps, build_specialist_registry
├── registry.py                     # Загрузка AgentManifest из manifest.yaml
│
├── agents/                         # Бизнес-логика специалистов
│   ├── base.py                     # BaseSpecialist: цикл decide→execute→compose
│   ├── ports.py                    # SchedulerPort, BitrixToolsetPort, PtoToolsetPort, VehicleUsageToolsetPort, AgentDialogStorePort, SpecialistOutputPort
│   ├── specialist_llm_shared.py    # Утилиты общие для LLM-сервисов
│   ├── bitrix24/                   # Специалист Bitrix24
│   │   ├── __init__.py             # Ре-экспорт публичного API (Bitrix24Specialist, LLM-классы, task_create)
│   │   ├── specialist.py           # Bitrix24Specialist, BitrixProposalService, IncompleteProposal
│   │   ├── llm.py                  # BitrixLLMService, BitrixAgentLLM Protocol, dataclasses
│   │   └── task_create.py          # BitrixTaskCreateDraft, build_task_create_draft_from_args
│   ├── pto/                        # Специалист ПТО
│   │   ├── __init__.py             # Ре-экспорт публичного API (PtoSpecialist, LLM-классы)
│   │   ├── specialist.py           # PtoSpecialist
│   │   └── llm.py                  # PtoLLMService, PtoAgentLLM Protocol, dataclasses
│   └── logistics/                  # Специалист Логистика
│       ├── __init__.py             # Ре-экспорт публичного API (LogisticsSpecialist, LLM-классы)
│       ├── specialist.py           # LogisticsSpecialist
│       └── llm.py                  # LogisticsLLMService, LogisticsAgentLLM Protocol, dataclasses
│
├── orchestrators/
│   ├── internal.py                 # InternalOrchestrator (Переговорщик): агентный цикл decide→execute→compose
│   └── internal_llm.py             # OrchestratorLLMService, InternalOrchestratorLLM Protocol
│
├── channels/
│   ├── bitrix.py                   # BitrixWebhookProcessor (точка входа Bitrix-событий)
│   └── ports.py                    # SearchWebhookHandlerPort, QualityControlHandlerPort
│
├── tools/
│   ├── bitrix.py                   # BitrixToolset
│   ├── bitrix_policy.py            # Политики write-операций
│   ├── document_access/            # DocumentToolset (PTO): поиск, чтение, сравнение документов
│   └── vehicle_usage.py            # VehicleUsageToolset, VehicleUsageStore
│
├── integrations/
│   └── bitrix/
│       ├── client.py               # BitrixClient (HTTP к Bitrix24 REST API)
│       ├── oauth.py                # BitrixOAuthService
│       ├── dialog_state.py         # BitrixPendingActionService, DialogStateStore
│       ├── events.py               # Парсинг входящих событий
│       ├── ports.py                # Все Bitrix-порты (Task, User, Bot, Disk, ...)
│       ├── bitrix_store.py         # BitrixAgentStore (состояние Bitrix24-специалиста)
│       └── portal_search/          # PortalSearchIndex: SQLite-индекс задач/диска/проектов
│
├── workers/
│   ├── bitrix/
│   │   ├── webhook_event_queue.py  # WebhookEventQueue: SQLite-очередь с dedupe/retry
│   │   ├── search_indexer.py       # PortalSearchIndexerWorker: фоновая индексация
│   │   ├── search_webhook_adapter.py   # SearchWebhookHandlerAdapter
│   │   ├── search_webhook_indexer.py   # Обработка disk-webhook для индекса
│   │   ├── reconciler.py           # Сверка потерянных событий
│   │   ├── supervisor.py           # Мониторинг просроченных задач
│   │   ├── quality_control.py      # LLM-оценка качества закрытия задач
│   │   └── quality_control_adapter.py  # QualityControlHandlerAdapter
│   └── logistics/
│       └── staff_sync.py           # Синхронизация сотрудников, run_staff_sync
│
├── routes/
│   ├── admin.py                    # GET /health, GET /agents, GET /automations
│   ├── agents.py                   # POST /orchestrator/test (dev endpoint)
│   ├── bitrix.py                   # Bitrix webhook endpoints
│   ├── learning.py                 # POST /learning/feedback
│   └── logistics.py                # GET/POST /logistics/vehicle-usage
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
└── pto/
    └── manifest.yaml

tests/                              # pytest, без сети — все внешние вызовы мокируются
var/                                # Рантайм: SQLite, вложения, индексы — не коммитится
```

---

## Манифесты агентов

Манифест (`manifest.yaml`) описывает агента декларативно. Загружается через `registry.py`. Поля:
- `id` — уникальный идентификатор
- `kind` — `specialist` | `orchestrator`
- `audience` — `employee` | (будущее: `customer`)
- `entrypoint` — `ai_server.agents.module.ClassName` — динамически загружается через `importlib`
- `capabilities` — список строк, описывает возможности агента
- `tools` — список инструментов агента
- `automations` — список фоновых автоматизаций (воркеры, расписания)

Инструкции агента — в `instructions.md` рядом с манифестом. Знания — в `knowledge/`. Навыки — в `skills/`.

---

## Добавление нового специалиста

1. Создать `agents/<id>/manifest.yaml` с `kind: specialist`, `entrypoint: ai_server.agents.<module>.<Class>`
2. Написать класс специалиста, унаследовав `BaseSpecialist`, реализовать `build()`, `tool_definitions()`, `_execute_tool_call()`, `_llm_failure_result()`, `_logs()`
3. Добавить поля зависимостей в `SpecialistDeps` (`specialists.py`)
4. Заполнить поля в `startup.py`

`BitrixWebhookProcessor`, `build_specialist_registry()` — не трогать.

---

## Фоновые воркеры

| Воркер | Файл | Назначение |
|---|---|---|
| WebhookEventQueue | `workers/bitrix/webhook_event_queue.py` | SQLite-очередь Bitrix webhook-событий: dedupe, retry с exponential backoff |
| PortalSearchIndexerWorker | `workers/bitrix/search_indexer.py` | Фоновая синхронизация задач/диска/проектов в поисковый индекс |
| SearchWebhookHandlerAdapter | `workers/bitrix/search_webhook_adapter.py` | Обновление индекса по disk-webhook в реальном времени |
| BitrixReconciler | `workers/bitrix/reconciler.py` | Компенсация потерянных webhook-событий (сверка с Bitrix) |
| TaskSupervisor | `workers/bitrix/supervisor.py` | Мониторинг просроченных задач, уведомления ответственным |
| QualityControlHandlerAdapter | `workers/bitrix/quality_control_adapter.py` | LLM-оценка качества закрытия задач через webhook |
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
| Воркеры | `WEBHOOK_EVENT_QUEUE_ENABLED`, `AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED`, `SUPERVISOR_ENABLED`, `RECONCILE_ENABLED` |
| Качество | `QUALITY_CONTROL_WEBHOOK_ENABLED`, `QUALITY_CONTROL_ACTOR_USER_ID` |
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
- **Пуш в `main` автоматически деплоится** (GitHub Actions job `deploy`): SSH → `git pull` → `uv sync` → `sudo systemctl restart ai-server`.
- Прод живёт в `/opt/ai-server`, конфиг — `/opt/ai-server/.env.prod`.

---

## Конвенции кода

- Python 3.11+, ruff: длина строки 120, double quotes, target py311
- Настройки линта — в `pyproject.toml`
- Секреты (токены, ключи API, пароли) **никогда не коммитятся**: репозиторий публичный
- Комментарии только там, где WHY неочевидно — не описывать WHAT делает код
- Без backwards-compatibility заглушек: если что-то удалено — удалено полностью
