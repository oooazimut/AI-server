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
```

## Agent Package

Каждый специалист должен быть отдельным модулем:

```text
agents/<agent_id>/
  manifest.yaml
  instructions.md
  skills/*.md
  knowledge/topics/*.md
  evals.yaml
```

На первом этапе реализованы пакеты:

- `agents/internal_orchestrator`;
- `agents/bitrix24`.

## Контракты

Backend использует собственные тонкие контракты:

- `AgentTask` - входящая задача для агента;
- `AgentResult` - структурированный результат агента;
- `ToolDefinition` - описание инструмента;
- `ToolResult` - результат инструмента;
- `PolicyDecision` - решение policy layer.

Эти контракты оставляют возможность позже подключить LangGraph как runtime процессов и MCP как стандартный транспорт инструментов.

## Почему пока без прямой зависимости от LangGraph/MCP

Сейчас важнее правильно выделить доменные границы: кто маршрутизирует, кто является специалистом, где лежат skills/RAG, какие инструменты разрешены и что требует подтверждения. После стабилизации сценариев можно заменить внутренний runtime на LangGraph и вынести Bitrix/tools в MCP-серверы.
