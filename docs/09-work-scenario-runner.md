# Work Scenario Runner

Минимальный сценарный стенд для проверки рабочих цепочек AI-server через HTTP API.

## Назначение

Runner имитирует запросы пользователя и проверяет цепочку:

```text
Пользователь -> /orchestrator/test -> Переговорщик -> специалист -> Переговорщик -> ответ
```

Дополнительно runner может:

- найти связанный `learning_event`;
- прочитать trace по `trace_id`;
- отправить тестовый feedback;
- создать incident;
- вызвать `/learning/diagnose`;
- вывести краткий отчет по каждому сценарию.

## Файлы

```text
tests/scenarios/work_scenarios.yaml
scripts/work_scenario_runner.py
```

## Запуск

Сначала запустить AI-server, затем из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts\work_scenario_runner.py --base-url http://127.0.0.1:8000
```

Если `WEBHOOK_SECRET` включен:

```powershell
.\.venv\Scripts\python.exe scripts\work_scenario_runner.py --secret <secret>
```

Сохранить отчет:

```powershell
.\.venv\Scripts\python.exe scripts\work_scenario_runner.py --output outputs\scenario_runs\last_run.json
```

Прогнать только первые два сценария:

```powershell
.\.venv\Scripts\python.exe scripts\work_scenario_runner.py --limit 2
```

## Формат сценария

```yaml
scenarios:
  - id: secure_org_data_open_search
    title: "Поиск открытой инструкции во внутренней базе"
    text: "Найди инструкцию TL-WR820N"
    expected:
      status: completed
      handoff_to_any:
        - secure_org_data
      answer_contains_any:
        - TL-WR820N
      trace_events_any:
        - orchestrator_decision
```

Для проверки feedback/incidents:

```yaml
feedback:
  rating: 3
  rating_scale: 10
  outcome: not_completed
  comment: "Тестовая плохая оценка."
  tags:
    - catalog
  diagnose: true
```

## Важно

- Сценарии должны начинаться с безопасных read-only запросов.
- `outputs/` не коммитить без отдельной команды.
- Write-сценарии и protected/secret данные добавлять только после согласования правил доступа.
