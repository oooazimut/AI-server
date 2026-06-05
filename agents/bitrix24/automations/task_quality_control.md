# Контроль качества закрытия задач

Источник переноса: `BitrixAIAgent/app/agent/quality_control.py`.
Чатовая часть старого `task_closure.py` перенесена отдельно как tool
`task_closure` Битрикс-субагента.

## Роль

Бизнес-автоматизация по событиям задач. Проверяет закрытие, результат, шаблон
ответа и качество описания выполненных работ.

## Входы

- `ONTASKUPDATE` из очереди webhook-событий.
- `task_id` из события.
- Read-tools `bitrix_task_get` и `bitrix_task_results_list`, которые вызывает
  сам LLM quality-control агент.
- Action-tool `quality_control_action`, который модель вызывает после анализа.

## Выходы

- Вернуть задачу в работу.
- Одобрить задачу.
- Уведомить ответственного или директора.
- Записать результат проверки в state.

## State

- `var/quality_control_state.json`.

## Правило переноса

Это автономная Bitrix-автоматизация. Она запускает LLM quality-control агента:
модель сама выбирает read-tools, сама оценивает результат и сама вызывает
action-tool. Backend остаётся исполнителем tools, policy, OAuth actor, dedupe и
state. Автоматические write-действия требуют служебного OAuth actor и явной
политики.

Закрытие задачи по просьбе человека из чата проходит не через этот worker, а
через цепочку `оркестратор -> LLM Bitrix24 -> task_closure -> pending
confirmation -> Bitrix REST`.
