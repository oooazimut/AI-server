# Vehicle Usage Production Runbook

Цель: перенести сценарий "Отчет по машинам и людям" из staging/dev в production без ручных правок в базе.

## Текущее различие

- Staging/dev уже работает на расширенной схеме `logistics`:
  - `vehicle_daily_drivers`
  - `vehicle_usage_operators`
  - `vehicle_usage_revisions`
  - `vehicle_payment_cards`
  - дополнительные audit/status колонки в существующих таблицах
- Production сейчас на старой схеме:
  - нет `vehicle_daily_drivers`
  - нет `vehicle_usage_operators`
  - нет `vehicle_usage_revisions`
  - нет части колонок `active`, `assignment_status`, `updated_by_user_id`, `finalized_by_user_id`
- Миграция должна выполняться кодом через `PostgresVehicleUsageStore.ensure_schema()`, а не ручным SQL.

## Production env

Перед переносом проверить и согласовать:

```env
VEHICLE_USAGE_ENABLED=true
VEHICLE_USAGE_DRY_RUN=false
VEHICLE_USAGE_MANAGER_USER_ID=13
VEHICLE_USAGE_DIALOG_ID=13
VEHICLE_USAGE_ALLOWED_USER_IDS=13
VEHICLE_USAGE_ADMIN_USER_IDS=1
VEHICLE_USAGE_ADMIN_NOTIFY_USER_IDS=13,9,1
VEHICLE_USAGE_REQUEST_TIME=08:30
VEHICLE_USAGE_REMINDER_DELAYS_MINUTES=30,60
VEHICLE_USAGE_MAX_REMINDERS=2
VEHICLE_USAGE_WORKDAY_MODE=weekday
VEHICLE_USAGE_STAFF_SYNC_ENABLED=false
```

`VEHICLE_USAGE_ADMIN_USER_IDS` нужно задать явно. Для production согласовано: администратор Bitrix ID 1, оператор отчета Коверга Дмитрий Bitrix ID 13. Если оставить администраторов пустыми, код использует fallback на `VEHICLE_USAGE_MANAGER_USER_ID`, и Коверга Дмитрий сможет менять список операторов.

## Порядок переноса

1. До deploy проверить backup production-кода и production-БД.
2. Убедиться, что staging/dev:
   - тесты `tests/test_logistics_specialist.py tests/test_postgres_stores.py` проходят;
   - ручные проверки через dev bot/runner прошли;
   - список сотрудников и машин соответствует production;
   - операторы в staging: тестовый пользователь и Коверга Дмитрий.
3. Смержить dev-правки в `main` только после согласования.
4. Обновить production-код без ручного переключения production-папки между ветками.
5. Применить production env из раздела выше.
6. Запустить production приложение так, чтобы `ensure_schema()` выполнил миграцию `logistics`.
7. Проверить только health/status.
8. Проверить read-only:
   - список операторов;
   - список сотрудников;
   - список машин;
   - наличие новых таблиц/колонок.
9. Выполнить один контролируемый сценарий:
   - запросить отчет за дату, где данные уже есть;
   - затем один новый тестовый отчет только с согласованным оператором.
10. После успешной проверки оставить штатный режим.

## Rollback

- Если ошибка до записи новых production-отчетов: откатить код к предыдущему production commit и перезапустить production-сервисы.
- Если ошибка после записи новых production-отчетов: не удалять строки вручную; сначала сохранить дамп затронутых `logistics.vehicle_usage_*`, затем согласовать точечную корректировку.
- Новые таблицы можно оставить: старый код их не использует.

## Контрольные проверки

```bash
systemctl is-active ai-server.service ai-server-worker.service ai-server-scheduler.service
```

Команды через Bitrix:

```text
Логист покажи операторов отчета по машинам
Логист покажи отчет по машинам за <контрольная дата>
Логист покажи отчет по Борисову за <месяц>
Логист покажи отчет по Авто 2 за <месяц>
```
