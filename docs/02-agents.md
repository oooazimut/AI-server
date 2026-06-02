# Агенты

## Типы агентов

- `orchestrator` - внутренний диспетчер задач сотрудников.
- `operator` - внешний оператор клиентского сценария.
- `specialist` - узкий эксперт с собственными инструкциями, RAG и инструментами.

## Контракт вызова специалиста

```json
{
  "task_id": "...",
  "user": { "id": "...", "role": "..." },
  "request": "...",
  "context": {},
  "files": [],
  "allowed_actions": [],
  "required_output_format": "structured_result"
}
```

## Ответ специалиста

```json
{
  "status": "completed | needs_clarification | needs_human | failed",
  "answer": "...",
  "artifacts": [],
  "actions_taken": [],
  "actions_requiring_approval": [],
  "confidence": 0.8,
  "logs": []
}
```

## Принцип

Агенты думают и принимают решение в своей области. Инструменты выполняют конкретные действия.
