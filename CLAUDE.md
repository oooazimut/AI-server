# AI-server

Корпоративный мультиагентный AI-сервер: FastAPI-приложение, принимающее события Bitrix24 (чат-бот, вебхуки) и маршрутизирующее запросы через оркестратор к агентам-специалистам.

## Архитектура

- **Оркестратор** (`internal_orchestrator`) принимает сообщение сотрудника, при необходимости привлекает специалистов и синтезирует ответ.
- **Специалисты**: `bitrix24` (задачи, диск, поиск по порталу), `pto` (производственно-технический отдел), `logistics` (транспорт, учёт машин).
- Манифесты агентов — пакетные, в `backend/ai_server/agents/<id>/` (`manifest.yaml`, `instructions.md`, `skills/`, `knowledge/`). Загрузка — `backend/ai_server/registry.py`.
- Фоновые воркеры: очередь webhook-событий (sqlite, dedupe/retry), индексатор портала, контроль качества задач.
- Голосовые сообщения транскрибируются (`backend/ai_server/transcription.py`): `STT_PROVIDER=openai` (gpt-4o-transcribe, лимит 25 МБ) или `yandex_speechkit` (синхронный API, лимит 1 МБ, требует ffmpeg для конвертации в OggOpus).

## Структура

- `backend/ai_server/` — основной пакет (FastAPI `main.py`, `settings.py` — все env-переменные, `channels/bitrix.py` — обработка событий бота).
- `tests/` — pytest, без сети: внешние вызовы мокаются.
- `var/` — рантайм-данные (sqlite-базы, вложения, индексы) — не коммитится.
- Конфигурация — только через переменные окружения (`.env.local` локально, не коммитится). Полный список — `backend/ai_server/settings.py::get_settings`.

## Команды

```bash
uv sync --extra dev --extra retrieval   # установка зависимостей
uv run pytest -v                        # тесты
uv run ruff check . && uv run ruff format --check .   # линт (обязателен перед коммитом)
uv run uvicorn ai_server.main:app --reload            # локальный запуск
```

В тестах, зависящих от настроек, изолируйте окружение: `monkeypatch.setenv("AI_SERVER_ENV_FILE", "")` плюс явные `setenv` нужных переменных — иначе подтянется `.env.local`.

## Ветки и деплой

- `dev` — основная ветка разработки. Прямые пуши в `dev`/`main` запрещены всем, кроме владельца; остальные работают в feature-ветках и открывают PR в `dev`.
- `main` — продакшен. Попадание кода: PR `dev` → `main` с зелёным CI.
- **Пуш в `main` автоматически деплоится на прод-сервер** (job `Deploy to prod` в `.github/workflows/ci.yml`): `git pull` + `uv sync` + перезапуск systemd-сервиса `ai-server`. Прод живёт в `/opt/ai-server` на ветке `main`, конфиг — `/opt/ai-server/.env.prod` (не в git).

## Конвенции

- Python 3.11+, ruff (длина строки 120, double quotes). Настройки линта — в `pyproject.toml`.
- Секреты (токены, ключи API, пароли) никогда не коммитятся: репозиторий публичный.
- Новые env-переменные добавляются в `settings.py` с дефолтом и, при необходимости, property вида `*_configured`.
