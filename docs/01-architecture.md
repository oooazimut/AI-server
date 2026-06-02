# Архитектура

## Стартовое решение

Первый рабочий MVP строится вокруг двух агентов:

```text
Сотрудник
  ↓
Channel Adapter: Bitrix24 chat / local test
  ↓
Внутренний оркестратор
  ↓
Битрикс24-специалист
  ↓
Tool Gateway: Bitrix REST / portal search / documents

Параллельно:

Bitrix24 webhooks / schedules
  ↓
Bitrix Workers / Automations
  ↓
Tool Gateway + локальный runtime state
```

Старый проект `BitrixAIAgent` совмещал канал общения, оркестрацию, LLM tool loop и Bitrix-экспертизу. В новом проекте эти роли разделяются.

## Слои

```text
Каналы общения
  ↓
Channel Adapters
  ↓
Оркестраторы / операторы
  ↓
Agent Registry
  ↓
Специалисты
  ↓
Tool Gateway + Policy Layer
  ↓
Битрикс24 / 1С / Excel / Git / сети / файлы

Фоновые события и расписания
  ↓
Workers / Automations
  ↓
Tool Gateway + State Stores
```

## Agent Package

Каждый специалист должен быть отдельным модулем:

```text
agents/<agent_id>/
  manifest.yaml
  instructions.md
  skills/*.md
  knowledge/topics/*.md
  automations/*.md
  evals.yaml
```

На первом этапе реализованы пакеты:

- `agents/internal_orchestrator`;
- `agents/bitrix24`.

## Контракты

Backend использует собственные тонкие контракты:

- `AgentTask` - входящая задача для агента;
- `AgentResult` - структурированный результат агента;
- `AgentAutomationManifest` - карточка фонового процесса или бизнес-автоматизации;
- `ToolDefinition` - описание инструмента;
- `ToolResult` - результат инструмента;
- `PolicyDecision` - решение policy layer.

Эти контракты оставляют возможность позже подключить LangGraph как runtime процессов и MCP как стандартный транспорт инструментов.

## Workers / Automations

Фоновые процессы не считаются субагентами. Они принадлежат домену специалиста,
но запускаются как отдельные сервисы:

- channel adapters принимают сообщения и события;
- event workers обрабатывают webhook/queue поток;
- scheduled workers выполняют периодические проверки;
- data pipelines поддерживают поисковые индексы и RAG-источники;
- business workflows выполняют доменные процессы с политиками доступа.

Для Bitrix24 это означает: `Bitrix24Specialist` отвечает за экспертную обработку
запросов людей, а индексация, quality-control, supervisor, reconciler и очередь
webhook-событий живут в слое `Workers / Automations`.

## Почему пока без прямой зависимости от LangGraph/MCP

Сейчас важнее правильно выделить доменные границы: кто маршрутизирует, кто является специалистом, где лежат skills/RAG, какие инструменты разрешены и что требует подтверждения. После стабилизации сценариев можно заменить внутренний runtime на LangGraph и вынести Bitrix/tools в MCP-серверы.
