# AI-server project rules for Codex agents

Этот файл нужно читать в начале каждой сессии по проекту AI-server. Он фиксирует практические правила, которые экономят время на повторяющихся ошибках.

## Контур и архитектура

- Работать только в dev-контуре. Не трогать Bitrix prod-бота, prod webhook и prod-процессы.
- Все действия из чата Bitrix, включая отчеты Диагноста, должны идти через оркестратор. Не вызывать Diagnostic Agent напрямую как обычного специалиста.
- Diagnostic Agent отвечает за диагностику, diagnostic_report и отчеты по ошибкам. Bitrix-команды отчетов доступны только dev/admin пользователям.
- Перед работой проверить ветку и состояние: `git status --short --branch`. Ожидаемая ветка для текущих работ: `feature/real-integration-tests`.
- Не печатать содержимое `.env.local`, webhook URL, токены и секреты. Можно проверять наличие переменных, но не выводить значения.

## Где смотреть контекст

- Базовая архитектура: `docs/01-architecture.md`, `docs/02-agents.md`.
- Текущий план агентной системы: `docs/08-agent-system-roadmap.md`.
- Scenario runner: `docs/09-work-scenario-runner.md`.
- Реальные интеграционные тесты и diagnostic_report: `docs/10-real-integration-tests-roadmap.md`.

## Кодировки и кириллица

- На Windows не доверять выводу PowerShell, если русские строки выглядят как `????` или mojibake. Сначала проверить, не проблема ли это консольной кодировки.
- Для HTTP-запросов с кириллицей предпочитать `curl.exe` с `-X POST`, `-H "Content-Type: application/json; charset=utf-8"` и JSON из файла или аккуратно экранированную строку.
- Не делать вывод, что бот или API сломаны, только по битому отображению кириллицы в терминале.
- При чтении UTF-8 документов PowerShell может показывать битый текст. Для поиска использовать `rg`, для точной проверки можно открыть файл другим способом или проверить байты/кодировку.
- В новых и изменяемых файлах сохранять UTF-8. Не смешивать кодировки в одном документе.

## Песочница, pytest и временные файлы

- `pytest` на этой машине может падать из-за прав на стандартные временные каталоги и `.pytest_cache`. Запускать тесты с явными каталогами внутри рабочей области:

```powershell
.\.venv\Scripts\python.exe -m pytest tests --basetemp ..\..\..\ai-server-feature-real-integration-tests\work\pytest-tmp -o cache_dir=..\..\..\ai-server-feature-real-integration-tests\work\pytest-cache
```

- Если тесты, HTTP-запросы или restart dev-сервера упираются в sandbox/network, повторить команду с разрешением на escalation, а не искать обходные хаки.
- Не удалять чужие временные каталоги и артефакты без явной причины. Сначала понять, кто их создал и нужны ли они текущей проверке.

## Dev server и живые проверки

- После изменений в backend перезапустить dev uvicorn перед живыми проверками Bitrix/API. Частая ошибка: код изменен, а на `1695` отвечает старый процесс.
- Проверять слушателя так:

```powershell
Get-NetTCPConnection -LocalPort 1695 -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort,State,OwningProcess
```

- На Windows у uvicorn часто есть родительский и дочерний `python.exe`. При рестарте смотреть command line процессов и останавливать только dev-процессы этого проекта.
- Если endpoint внезапно отвечает старым текстом или `404`, сначала подозревать stale server/reload, потом уже ошибку маршрута.
- Live-тест feedback loop: дождаться ответа бота, потом отправлять оценку. Иначе оценка может привязаться не к тому pending answer.

## Отчеты Диагноста

- Быстрый локальный отчет:

```powershell
.\.venv\Scripts\python.exe scripts\error_report.py --since-hours 24 --limit 100 --max-groups 5
```

- API отчет должен идти через `/learning/reports/errors`, а чатовый запрос - через `InternalOrchestrator -> diagnostic_agent`.
- Если в чате ответ похож на уточнение маршрутизации вместо отчета, проверить admin-доступ, регистрацию intent и актуальность запущенного сервера.
- Если видно `ErrorReportService не настроен для Diagnostic Agent`, проверить wiring `learning_recorder` в `InternalOrchestrator.build()` и registry Diagnostic Agent.

## Изменения и коммиты

- Ручные правки делать через `apply_patch`. Для поиска использовать `rg`/`rg --files`.
- Не коммитить `.env.local`, `outputs/`, временные логи, дампы и артефакты живых проверок без отдельной просьбы.
- Перед финальным ответом для кодовых изменений запускать `git diff --check` и релевантные тесты. Для docs-only изменений тесты обычно не нужны, но `git diff --check` полезен.
- В грязном worktree не откатывать чужие изменения. Если файл уже изменен не нами, сначала прочитать diff и встроиться в него.

## Короткий стартовый чеклист

1. Проверить, что открыт настоящий repo AI-server, а не только generated workspace.
2. Выполнить `git status --short --branch`.
3. Прочитать актуальные docs из раздела "Где смотреть контекст".
4. Проверить dev/prod границу: любые live-действия только dev.
5. Для тестов сразу задать `--basetemp` и `cache_dir` внутри workspace.
6. После backend-правок перезапустить dev server и проверить port `1695`.
