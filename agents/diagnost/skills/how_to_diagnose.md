# Как провести диагностику

Используй этот скил как отправную точку для любого запроса на диагностику системы.

## Быстрый старт

1. **Сводка за период** → сначала всегда вызови `diagnost_error_report(since_hours=24)`.
2. **Конкретная проблема** → ищи событие через `diagnost_search_events(query="ключевые слова")`.
3. **Конкретный инцидент** → получи детали через `diagnost_get_incident(incident_id="...")`.
4. **Все открытые инциденты** → `diagnost_list_incidents(status="open")`.
5. **Ручной инцидент** → `diagnost_create_incident(event_id="...", comment="описание")`.

## Когда что применять

| Ситуация | Инструмент |
|---|---|
| "Покажи ошибки за неделю" | `diagnost_error_report(since_hours=168)` |
| "Что случилось с запросом про камеру" | `diagnost_search_events(query="камера")` |
| "Открой инцидент №X" | `diagnost_get_incident(incident_id="X")` |
| "Сколько открытых проблем?" | `diagnost_list_incidents(status="open", limit=50)` |
| "Пометь это событие как проблему" | `diagnost_create_incident(event_id="...", comment="...")` |

## Как автоматически создаются инциденты

Воркер диагноста создаёт инцидент при каждом событии оркестратора с:
- `status = "failed"` → `reason = "failed"`
- `confidence < 0.5` → `reason = "low_confidence"`

Ручные инциденты создаются через `diagnost_create_incident` с `reason = "manual"`.

## Статусы инцидентов

- `open` — требует внимания
- `resolved` — закрыт

Фильтрация: `diagnost_list_incidents(status="open")` или `diagnost_list_incidents(status="resolved")`.
