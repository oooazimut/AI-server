# Agent System Roadmap

Документ фиксирует общий план развития AI-server как multi-agent системы. Его цель - дать общий контекст для отдельных чатов/веток Codex, чтобы каждая ветка решала один пункт и не ломала текущую архитектуру.

## Рабочая папка

Работать в текущем репозитории:

```text
C:\Users\BOSS_PC\Documents\Codex\2026-06-24\oooazimut-ai-server-https-github-com\work\AI-server
```

Не создавать новую копию проекта без отдельной причины.

## Текущая архитектура

AI-server - FastAPI backend с multi-agent схемой.

Основной поток должен сохраняться:

```text
Пользователь -> internal_orchestrator -> специалисты -> internal_orchestrator -> пользователь
```

Главный агент:

- `internal_orchestrator` / Переговорщик - принимает запросы, выбирает специалистов, собирает финальный ответ и контролирует подтверждения.

Текущие специалисты:

- `bitrix24` - задачи, проекты, CRM, диск Bitrix, поиск по порталу, сообщения.
- `pto` - техническая документация, сметы, сравнение документов, комплектность.
- `logistics` - служебные автомобили, статусы сотрудников, утренние отчеты.

## Главные ограничения

- Не ломать текущий поток пользователь -> Оркестратор -> специалисты -> Оркестратор -> пользователь.
- Не смешивать ответственность агентов.
- Специалисты не должны знать друг о друге напрямую.
- Оркестратор не должен выполнять доменную работу, если есть подходящий специалист.
- Diagnostic agent не должен участвовать в обычном пользовательском диалоге.
- Любые write-действия и опасные операции должны проходить через policy/approval слой.
- После изменений запускать тесты.

## Целевое развитие

### 1. Правила как мини-книга

Переструктурировать правила агентов в формат "мини-книги":

- краткое оглавление сверху;
- основные разделы через `##`;
- внутри разделов: когда применять, что делать, чего не делать, когда уточнять, когда нужно подтверждение, примеры;
- сохранить текущий смысл правил;
- сделать структуру удобной для RAG.

Первый кандидат: `agents/internal_orchestrator/`.

Ожидаемая ветка:

```text
feature/rules-minibook
```

### 2. Модульные правила и добавление новых агентов

Новый агент должен добавляться как отдельный модуль:

```text
agents/<agent_id>/
  manifest.yaml
  instructions.md
  skills/
  knowledge/topics/
```

Код специалиста:

```text
backend/ai_server/agents/<agent_id>/
```

Добавление нового агента не должно требовать переписывания старых правил. Оркестратор должен получать только минимальную routing-информацию о новом агенте.

Ожидаемая ветка:

```text
feature/modular-agent-rules
```

### 3. Rule index и загрузка правил по ситуации

Сейчас часть правил может попадать в контекст сразу. Цель - перейти к более управляемой схеме:

- короткое ядро правил всегда доступно Оркестратору;
- подробные правила хранятся как главы;
- `rule_index.yaml` или аналогичный индекс описывает, когда какую главу использовать;
- RAG/read-rule механизм достает релевантные разделы по ситуации.

Пример индекса:

```yaml
rules:
  - id: routing_guidelines
    file: knowledge/routing_guidelines.md
    use_when: "Нужно выбрать специалиста"
  - id: escalation_policy
    file: knowledge/escalation_policy.md
    use_when: "Задача пришла от специалиста или есть _source/_intent"
```

Ожидаемая ветка:

```text
feature/rule-index
```

### 4. TraceRecorder

Добавить инфраструктурный слой полной трассировки. Боевые агенты не должны знать о diagnostic agent; они только пишут trace через общий сервис.

Минимальные события:

- `user_message_received`
- `orchestrator_context_loaded`
- `orchestrator_rules_retrieved`
- `orchestrator_decision`
- `specialist_called`
- `specialist_rules_retrieved`
- `specialist_llm_decision`
- `tool_called`
- `tool_result`
- `specialist_final_answer`
- `orchestrator_compose`
- `message_sent_to_user`
- `human_feedback_received`

Каждое событие должно иметь общий `trace_id`, а отдельные шаги - `span_id` и при необходимости `parent_span_id`.

Ожидаемая ветка:

```text
feature/trace-recorder
```

### 5. Feedback -> incidents

В проекте уже есть:

- `var/learning_events.jsonl`
- `GET /learning/status`
- `GET /learning/events`
- `POST /learning/feedback`

Нужно развить это в incident-механику:

- низкая оценка или статус "не выполнено" создают incident;
- incident связан с исходным событием, trace, ответом, actions, model_usage и feedback;
- incident можно просмотреть и проанализировать.

Ожидаемая ветка:

```text
feature/learning-incidents
```

### 6. Diagnostic agent

Добавить отдельного агента для анализа ошибок.

Он не участвует в обычной цепочке ответа пользователю.

Задачи diagnostic agent:

- читать learning events;
- читать feedback;
- читать trace;
- определять вероятный этап сбоя;
- предлагать, что исправить: правило, skill, tool, код или тест;
- предлагать regression tests.

Он может знать обо всех агентах как об объектах анализа, но специалисты не должны знать о нем.

Ожидаемая ветка:

```text
feature/diagnostic-agent
```

### 7. Batch-анализ ошибок

Ошибки нужно не только разбирать по одной, но и копить.

Цель:

- группировать однотипные incidents;
- искать общий корень;
- предлагать общий patch plan;
- предлагать regression tests;
- помечать incidents как покрытые конкретным исправлением.

Пример:

```text
Много incidents вида "товар не найден, хотя есть" ->
общий patch: fallback-поиск по части названия, артикулу, синонимам, уточнение склада.
```

Ожидаемая ветка:

```text
feature/batch-incident-analysis
```

## Рекомендуемый порядок работ

1. `feature/rules-minibook`
2. `feature/modular-agent-rules`
3. `feature/rule-index`
4. `feature/trace-recorder`
5. `feature/learning-incidents`
6. `feature/diagnostic-agent`
7. `feature/batch-incident-analysis`

## Шаблон для нового чата Codex

```text
Работаем с проектом AI-server:
C:\Users\BOSS_PC\Documents\Codex\2026-06-24\oooazimut-ai-server-https-github-com\work\AI-server

Не создавай новую копию проекта.
Работай от текущего состояния репозитория.

Глобальный roadmap смотри в:
docs/08-agent-system-roadmap.md

Текущая задача:
<название пункта>

Ограничения:
- не ломать поток пользователь -> Оркестратор -> специалисты -> Оркестратор -> пользователь;
- не смешивать ответственность агентов;
- diagnostic_agent не должен участвовать в обычном диалоге;
- правила должны быть модульными;
- после изменений запускать тесты.
```

## Проверки

Базовая команда тестов:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp var\tmp\pytest -p no:cacheprovider
```

Линтер:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```
