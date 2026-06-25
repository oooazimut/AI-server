# Создание и изменение задач

## Создание задачи

1. Проверь, хватает ли данных: название (`title`), исполнитель, срок.
2. **Исполнитель по имени** — сначала вызови `bitrix_api` → `user.search` с именем/фамилией, получи числовой ID из ответа, затем передай его в `task_create_draft` как `responsible_id`. Не передавай имя напрямую в `task_create_draft`.
3. **Исполнитель — текущий пользователь** — используй `responsible_self: true`.
4. **Проект по названию** — сначала вызови `bitrix_api` → `sonet_group.get`, найди нужную группу по полю `NAME` в ответе, возьми её числовой `ID`, затем передай как `group_id` в `task_create_draft`. Не передавай название напрямую.
5. **Срок** — если указан относительный («через неделю», «в пятницу»), вычисли `deadline_iso` самостоятельно по `current_datetime`. Если срок не указан, проверь правила в `retrieval_context`; если правил нет — задай уточняющий вопрос.
6. Если данных не хватает, верни `needs_clarification` — не вызывай `task_create_draft` без `title`, `responsible_id`/`responsible_self` и `deadline_iso`/`no_deadline`.
7. `task_create_draft` валидирует контракт и готовит pending action; write-действие не выполняется без подтверждения пользователя.

## Изменение задачи

- Для обновления полей задачи используй `bitrix_api` → `tasks.task.update`.
- Для добавления комментария — `bitrix_api` → `tasks.task.commentitem.add`.
- Для делегирования — `bitrix_api` → `tasks.task.delegate`.
- Все write-методы требуют confirm policy (подтверждение пользователя).

## Уведомление пользователя

- Системное уведомление отправляется через `bitrix_api` → `im.notify.system.add` (allow-policy, без подтверждения).
- Поля: `USER_ID` (числовой ID), `MESSAGE` (текст).
