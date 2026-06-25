# Политики доступа к Bitrix REST API

Раздел описывает, какие методы разрешены, какие требуют подтверждения, а какие
запрещены совсем. Соблюдение этих правил — зона ответственности агента, а не только backend guardrail.

## Как работает политика

Каждый вызов `bitrix_api` проходит через `decide_bitrix_method_policy`:

1. **allow** — выполняется немедленно без подтверждения пользователем.
2. **confirm** — backend откладывает выполнение до явного подтверждения пользователем в чате.
3. **deny** — отклоняется сразу; агент не должен пытаться обойти ограничение.

## Разрешено без подтверждения (allow)

Чтение:
- `user.get`, `user.search`
- `tasks.task.get`, `tasks.task.list`, `tasks.task.result.list`
- `task.commentitem.get`, `task.commentitem.getlist`
- `disk.file.get`, `disk.folder.get`, `disk.folder.getchildren`
- `disk.attachedobject.get`, `disk.storage.getlist`, `disk.storage.getchildren`
- `sonet_group.get`, `sonet_group.user.get`
- `crm.lead.get`, `crm.lead.list`, `crm.deal.get`, `crm.deal.list`, `crm.status.list`
- `catalog.*` — весь раздел каталога
- `batch`, `app.info`, `profile`, `user.current`

Операции с суффиксом `.get`, `.list`, `.search` — как правило разрешены.

Уведомления:
- `im.notify.system.add` — системное уведомление, разрешено без подтверждения.

## Требует подтверждения (confirm)

Любой write-метод (суффикс `.add`, `.update`, `.delete`, `.create`, `.complete`,
`.approve`, `.disapprove`, `.renew`, `.start`, `.pause`, `.delegate`, `.remove`)
с одним из префиксов:
- `tasks.*` / `task.*`
- `sonet_group.*`
- `crm.*`
- `disk.*`
- `calendar.*`
- `catalog.*`

Дополнительно требуют подтверждения явно:
- `disk.storage.addfolder`, `disk.storage.uploadfile`
- `disk.folder.addsubfolder`, `disk.folder.uploadfile`

## Запрещено (deny)

- `user.add`, `user.update`, `user.delete`, `user.dismiss` — управление учётными записями.
- Методы с префиксом `user.` (кроме `user.get`, `user.search` → allow).
- Методы с префиксом `department.*`, `humanresources.*`.
- Методы с префиксом `imbot.*` (боты), `im.*` — **кроме** `im.notify.system.add`.
- Методы с префиксом `rest.*`.
- Любой метод, не попавший ни в один из разрешённых паттернов.

