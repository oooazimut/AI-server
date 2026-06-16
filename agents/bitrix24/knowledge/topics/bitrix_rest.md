# Bitrix REST API

Этот раздел читать через `knowledge_base`, когда нужно вспомнить формат вызова Bitrix REST или подобрать метод.

## Общий формат

Вызов REST выполняется через tool `bitrix_api`:

```json
{
  "action": "call",
  "method": "tasks.task.list",
  "params": {
    "filter": {},
    "select": ["ID", "TITLE"]
  },
  "summary": "Короткое описание действия"
}
```

Для подтверждения ожидающего изменения:

```json
{"action": "confirm_pending", "method": "_pending", "params": {}}
```

Для отмены:

```json
{"action": "cancel_pending", "method": "_pending", "params": {}}
```

## Пользователи

`user.search` - поиск сотрудников.

```json
{
  "action": "call",
  "method": "user.search",
  "params": {
    "FILTER": {"ACTIVE": true, "FIND": "Смородин"},
    "SORT": "LAST_NAME",
    "ORDER": "ASC",
    "LIMIT": 10
  }
}
```

`user.get` - чтение пользователя по фильтру.

Запрещено через агента: создание, изменение, увольнение и удаление пользователей.

## Задачи

`tasks.task.list` - список задач.

Частые поля:
- `ID`
- `TITLE`
- `STATUS`
- `RESPONSIBLE_ID`
- `CREATED_BY`
- `DEADLINE`
- `GROUP_ID`
- `DESCRIPTION`
- `CLOSED_DATE`
- `CREATED_DATE`
- `CHANGED_DATE`

```json
{
  "action": "call",
  "method": "tasks.task.list",
  "params": {
    "filter": {"!STATUS": 5},
    "select": ["ID", "TITLE", "STATUS", "RESPONSIBLE_ID", "CREATED_BY", "DEADLINE"],
    "order": {"DEADLINE": "ASC"}
  }
}
```

Статусы задач:
- `1` - новая
- `2` - ждёт выполнения
- `3` - выполняется
- `4` - ждёт контроля
- `5` - завершена
- `6` - отложена
- `7` - отклонена

`tasks.task.get` - карточка задачи.

```json
{
  "action": "call",
  "method": "tasks.task.get",
  "params": {
    "taskId": 7797,
    "select": ["ID", "TITLE", "DESCRIPTION", "STATUS", "RESPONSIBLE_ID", "CREATED_BY", "DEADLINE"]
  }
}
```

`tasks.task.add` - создать задачу. Это изменение, backend остановит до подтверждения.

```json
{
  "action": "call",
  "method": "tasks.task.add",
  "params": {
    "fields": {
      "TITLE": "Название задачи",
      "DESCRIPTION": "Описание",
      "RESPONSIBLE_ID": 9,
      "CREATED_BY": "current_user.id",
      "DEADLINE": "2026-05-14T19:00:00",
      "TASK_CONTROL": "Y"
    }
  },
  "summary": "Создать задачу ..."
}
```

`tasks.task.result.list` - результаты выполнения задачи.

`tasks.task.result.add` - добавить результат выполнения задачи. Для закрытия
через чат использовать не напрямую, а через tool `task_closure`.

`tasks.task.complete` - перевести задачу в выполненные/на контроль. Для
закрытия через чат использовать не напрямую, а через tool `task_closure`.

`task_closure` не выполняет закрытие сразу: он создаёт pending
`ai_server.task_closure`, который исполняется только после подтверждения
пользователем.

Backend автоматически усиливает создание задач:
- принудительно ставит `CREATED_BY = current_user.id`, даже если модель передала другое значение;
- не выбирает дедлайн за модель: если пользователь не указал срок, модель должна сама применить правила knowledge или спросить уточнение;
- если модель явно передала `NO_DEADLINE=true`, backend преобразует это в отсутствие срока;
- backend может включать технические safety/policy-поля только как guardrail, а не как доменное решение.

`task.commentitem.getlist` - комментарии задачи.

`task.commentitem.add` - добавить комментарий. Это изменение, нужно подтверждение.

`tasks.task.disapprove` - вернуть задачу на доработку. Это изменение, нужно подтверждение или отдельная фоновая политика контроля результата.

`tasks.task.renew` - переоткрыть задачу. Это изменение.

## Календарь и напоминания

`calendar.*` разрешён политикой backend:
- методы чтения выполняются сразу;
- методы изменения (`add`, `update`, `delete`, `remove` и похожие) требуют подтверждения пользователя и OAuth-доступ.

Для просьб вроде "напомни мне завтра" использовать календарные методы Bitrix,
а не отвечать, что метод запрещён. Если для создания события не хватает даты,
времени, названия или календаря/секции, уточнить недостающие данные.

## Проекты / рабочие группы

`sonet_workgroup.list` - список всех рабочих групп / проектов.

```json
{
  "action": "call",
  "method": "sonet_workgroup.list",
  "params": {}
}
```

Поиск проекта по имени:

```json
{
  "action": "call",
  "method": "sonet_workgroup.list",
  "params": {"filter": {"%NAME": "Транзит"}}
}
```

`sonet_group.get` - получить конкретную группу/проект по ID или фильтру.

```json
{
  "action": "call",
  "method": "sonet_group.get",
  "params": {"FILTER": {"ID": 42}}
}
```

Изменение проектов (`sonet_group.update`, `sonet_group.create`, удаление) требует подтверждения.

Подтверждённые изменения в Bitrix должны выполняться через OAuth-токен текущего пользователя, если он подключён. Если OAuth обязателен и токена нет, не выполнять действие через общий входящий webhook; попросить пользователя один раз открыть локальное приложение `ИИ Агент-помощник` в Bitrix24.

При `sonet_group.create` по умолчанию создавать открытый проект:
`PROJECT=Y`, `VISIBLE=Y`, `OPENED=Y`, `CLOSED=N`. Если `OWNER_ID` явно не указан,
backend подставляет `OWNER_ID=current_user.id`. Если пользователь явно просит
закрытый/приватный проект, использовать его явное значение.

## Диск и файлы

`disk.file.get` - метаданные файла и ссылка скачивания, если доступна.

```json
{
  "action": "call",
  "method": "disk.file.get",
  "params": {"id": 2111}
}
```

`disk.folder.getchildren` - список файлов в папке.

`disk.attachedObject.get` - информация о вложении задачи.

`disk.storage.getlist` - список хранилищ Диска, включая личные диски пользователей.

`disk.storage.addfolder` - создать папку в корне хранилища. Это изменение, нужно подтверждение.

`disk.folder.addsubfolder` - создать дочернюю папку внутри папки. Это изменение, нужно подтверждение.

`disk.folder.uploadfile` - загрузить новый файл в папку. Это изменение, нужно подтверждение.

Для физического чтения или смыслового сравнения технических документов запрос
должен идти к ПТО-специалисту. Битрикс24-специалист может найти файл или ссылку,
но не должен сам выполнять экспертизу документа.

## CRM

`crm.lead.list`, `crm.lead.get` - чтение лидов.

`crm.deal.list`, `crm.deal.get` - чтение сделок.

`crm.status.list` - статусы, стадии, справочники CRM.

Создание и изменение CRM-сущностей (`crm.lead.add`, `crm.deal.update` и т.п.) требует подтверждения и доступно только разрешённым пользователям.
