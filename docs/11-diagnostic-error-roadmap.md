# Roadmap: Diagnostic Agent error memory

Документ фиксирует следующий этап работ после feedback loop, incident, diagnostic_report и отчета Диагноста по ошибкам.

Цель этапа - превратить низкие оценки и diagnostic reports в устойчивую память об ошибках: с нормальной карточкой ошибки, классификацией, группировкой похожих случаев, отчетом для Codex и коротким отчетом для Bitrix-чата.

## Главные ограничения

- Не трогать Bitrix prod-бота, prod webhook и prod-процессы.
- Все live-проверки выполнять только через dev-контур.
- Все агенты, включая Diagnostic Agent, вызывать только через `internal_orchestrator`.
- Diagnostic Agent не должен сам становиться отдельной входной точкой для Bitrix-чата.
- Diagnostic Agent не должен напрямую ходить в бизнес-системы. Он анализирует feedback, learning events, trace, actions, tool calls и diagnostic context.
- Не выводить секреты, токены, полный webhook URL и содержимое `.env.local`.

## Текущее состояние

Уже есть базовая цепочка:

```text
ответ бота
-> просьба оценить ответ по 10-балльной шкале
-> feedback
-> incident при низкой оценке или проблемном комментарии
-> Diagnostic Agent через internal_orchestrator
-> diagnostic_report
-> ErrorReportService
-> CLI scripts/error_report.py
-> GET /learning/reports/errors
-> чатовая команда отчета через InternalOrchestrator -> diagnostic_agent
```

Ключевые файлы:

- `backend/ai_server/feedback_loop.py`
- `backend/ai_server/learning.py`
- `backend/ai_server/diagnostics.py`
- `backend/ai_server/agents/diagnostic_agent/error_report.py`
- `backend/ai_server/agents/diagnostic_agent/llm.py`
- `backend/ai_server/routes/learning.py`
- `scripts/error_report.py`

Последние связанные коммиты:

- `d16b1fb Implement feedback diagnostics loop`
- `5b8a144 Add diagnostic group suggestions`
- `a6d8d06 Add diagnostic error reports`

## Целевая модель данных

Не заводить отдельное хранилище ошибок на первом этапе. Источник истины остается append-only `learning_events.jsonl`.

Роли событий:

- `agent_result` - исходный ответ системы пользователю.
- `human_feedback` - оценка и пояснение человека.
- `incident` - фактическая карточка ошибки, созданная из плохого feedback.
- `diagnostic_report` - вывод Diagnostic Agent: классификация, вероятная причина, место исправления и regression test.

### Incident как карточка ошибки

`incident` должен хранить факты, которые уже известны без догадок Диагноста:

- `target_event_id` - исходный learning event ответа.
- `feedback_event_id` - feedback, который создал incident.
- `trace_id` - trace всей цепочки.
- `request` - пользовательский запрос.
- `response` - ответ бота.
- `rating` и `rating_scale`.
- `comment` - пояснение пользователя.
- `target_agent_id` - агент, который дал ответ.
- `target_status` - статус исходного ответа.
- `intent` - распознанный intent, если он есть.
- `handoff_to`, `actions`, `model_usage`.
- `diagnostic_trace`.
- `trace_events` - ограниченный, но достаточный снимок trace.

Нормализованный блок `incident.metadata.classification`:

```json
{
  "category": "user_error|routing_error|agent_error|integration_error|data_error|unknown",
  "subcategory": "",
  "confidence": 0.0
}
```

Нормализованный блок `incident.metadata.routing`:

```json
{
  "intent": "",
  "expected_agent_id": "",
  "actual_agent_id": "",
  "route_reason": ""
}
```

Нормализованный блок `incident.metadata.failure`:

```json
{
  "stage": "input|routing|agent_reasoning|tool_call|integration|data_lookup|response",
  "tool_name": "",
  "error_code": "",
  "external_system": ""
}
```

Нормализованный блок `incident.metadata.grouping`:

```json
{
  "group_key": "",
  "signature": ""
}
```

### Diagnostic report как вывод Диагноста

`diagnostic_report` должен хранить не только текстовый ответ, но и структурированную сводку в `metadata`:

```json
{
  "target_event_id": "",
  "feedback_event_ids": [],
  "incident_event_ids": [],
  "root_cause_category": "",
  "root_cause_summary": "",
  "where_to_fix": "",
  "fix_proposal": "",
  "regression_test": "",
  "related_group_key": "",
  "evidence": []
}
```

Текстовый `answer` остается человекочитаемым отчетом. Структурированные поля нужны для группировки, CLI, API, Codex и Bitrix-чата без парсинга markdown.

## Классификация ошибок

Базовые категории:

- `user_error` - пользователь спросил неоднозначно, запрос невозможен, нет нужных прав или ожидание не соответствует доступным данным.
- `routing_error` - оркестратор выбрал не того агента, не распознал intent или не передал нужный context.
- `agent_error` - агент получил правильную задачу, но ошибся в рассуждении, формате ответа, использовании правил или навыков.
- `integration_error` - сбой внешней системы, API, webhook, tool call, auth, network, rate limit.
- `data_error` - данные отсутствуют, устарели, неполные, противоречивые или лежат не в том источнике.
- `unknown` - evidence недостаточно.

Решение по категории должно опираться на trace и факты:

- выбранный агент и ожидаемый агент;
- intent и route decision;
- tool calls и их статусы;
- external system и error code;
- наличие или отсутствие данных;
- feedback comment;
- финальный ответ и target status.

## Группировка похожих ошибок

Группировка должна быть детерминированной, чтобы повторные отчеты давали стабильные ключи.

Приоритет ключей:

1. `category + stage + target_agent_id + intent`.
2. Для интеграций: `integration_error + external_system + tool_name + error_code`.
3. Для маршрутизации: `routing_error + intent + expected_agent_id + actual_agent_id`.
4. Для данных: `data_error + data_source + entity_type + missing_field`.
5. Для агента: `agent_error + target_agent_id + loaded_rule/skill + response_issue`.
6. Fallback: текущие ключи по `reason`, `target_agent`, `target_status`, rules, skills, tools и tags.

Примеры:

```text
routing_error:intent=task_search:expected=bitrix24:actual=secure_org_data
integration_error:bitrix24:task.item.list:http_403
data_error:secure_org_data:inventory_item:not_found
agent_error:bitrix24:response:missed_requested_field
```

## Этапы работ

### 1. Зафиксировать контракт событий

Статус: нужно сделать.

Цель: добавить явный контракт incident/diagnostic_report и не ломать существующие события.

Сделать:

- описать поля incident и diagnostic_report в docs;
- добавить helper-функции для формирования `classification`, `routing`, `failure`, `grouping`;
- обеспечить backward compatibility для старых incidents без новых полей;
- добавить unit tests на запись incident из feedback.

Критерий готовности:

- старые отчеты продолжают работать;
- новые incidents получают пустые или заполненные нормализованные блоки;
- низкая оценка в тесте создает `human_feedback`, `incident`, затем `diagnostic_report`.

### 2. Добавить первичную классификацию incident

Статус: нужно сделать.

Цель: до вызова Диагноста заполнить то, что можно определить детерминированно.

Сделать:

- извлекать `intent`, `actual_agent_id`, `target_status`, tool status из target event и trace;
- определять очевидные `integration_error` по failed tool calls и external system;
- определять очевидные `routing_error`, если в trace есть route decision и expected/actual mismatch;
- оставлять `unknown`, если evidence недостаточно;
- не заставлять LLM придумывать факты.

Критерий готовности:

- incident содержит `classification.category`;
- `unknown` допустим и не считается ошибкой системы;
- классификация покрыта тестами на routing, integration, data/unknown cases.

### 3. Научить Diagnostic Agent возвращать структурированный вывод

Статус: нужно сделать.

Цель: Diagnostic Agent должен отдавать не только текстовый отчет, но и машинно-читаемые поля.

Сделать:

- расширить JSON-контракт `DiagnosticLLMService.compose`;
- добавить поля `root_cause_category`, `where_to_fix`, `fix_proposal`, `regression_test`, `evidence`;
- сохранить эти поля в `diagnostic_report.metadata`;
- оставить человекочитаемый `answer` для чата и CLI;
- при отсутствии LLM или неполном ответе делать безопасный fallback.

Критерий готовности:

- diagnostic_report можно использовать без markdown-парсинга;
- ErrorReportService берет `where_to_fix`, `fix_proposal`, `regression_test` из metadata;
- markdown-парсер остается только как compatibility fallback.

### 4. Улучшить группировку повторяющихся ошибок

Статус: нужно сделать.

Цель: группы должны показывать не просто агента или reason, а повторяемую причину.

Сделать:

- генерировать `group_key` и `signature` для incident;
- использовать новые ключи в `ErrorReportService`;
- показывать fallback-группы для старых incidents;
- считать `first_seen`, `last_seen`, `count`, `affected_agents`, `example_incident_ids`;
- связывать группу с последними diagnostic reports.

Критерий готовности:

- похожие ошибки собираются в одну стабильную группу;
- отчет показывает top groups по count;
- для группы есть пример, вероятная причина, место исправления и regression test.

### 5. Разделить отчет для Codex и отчет для Bitrix-чата

Статус: нужно сделать.

Цель: Codex получает подробную инженерную карточку, Bitrix-чат - короткую управленческую выжимку.

Codex report:

- group key и count;
- request/response/feedback examples;
- trace_id и incident ids;
- affected agent, intent, tool, external system;
- root cause;
- where to fix;
- fix proposal;
- regression test;
- confidence и gaps.

Bitrix chat report:

- период отчета;
- количество incidents и diagnostic_reports;
- top 3-5 групп;
- короткая причина;
- куда смотреть;
- что проверить дальше;
- без секретов, длинных trace и внутренних payload.

Критерий готовности:

- `scripts/error_report.py --format json` подходит для Codex;
- `scripts/error_report.py --format markdown` остается коротким и читаемым;
- `GET /learning/reports/errors?format=json` возвращает полный report;
- чатовая команда отчета через `InternalOrchestrator -> diagnostic_agent` возвращает короткую версию.

### 6. Реализовать layered report и deep context

Статус: нужно сделать.

Цель: сохранять максимум полезной диагностической информации, но выдавать ее слоями. Базовый отчет должен сразу отвечать на вопросы: какой был запрос, куда он был передан и почему, какие действия сделаны и почему, что пошло не так и как это исправить. Если причины недостаточно понятны, Codex или Diagnostic Agent должен иметь возможность запросить deep context по `incident_id` или `trace_id`.

Архитектурные границы:

- входящие сообщения из Bitrix, API и runner идут только в `internal_orchestrator`;
- specialist-агенты не становятся самостоятельными входными точками;
- specialist-агенты не вызывают друг друга напрямую;
- Diagnostic Agent вызывается только через `internal_orchestrator`;
- ErrorReportService остается read-only сборщиком отчетов, а не отдельным агентом и не прямым маршрутом общения;
- live-проверки выполняются только через dev-контур;
- prod-бот, prod webhook и prod-процессы не изменяются.

Слои отчета:

1. `summary` - короткий отчет для Bitrix-чата.
2. `codex` - основной инженерный отчет для исправления.
3. `deep` - полный диагностический контекст по запросу.

#### Summary report

Показывать в Bitrix-чате:

- период отчета;
- количество incidents;
- top 3-5 групп;
- пример запроса;
- куда запрос был передан;
- вероятную причину;
- где смотреть;
- что проверить дальше;
- ссылочные ids: `incident_id`, `trace_id`, `diagnostic_report_id`;
- без длинных payload, секретов, raw tool outputs и полного trace.

#### Codex report

Показывать Codex по умолчанию:

- `incident_id`, `trace_id`, `target_event_id`, `feedback_event_id`, `diagnostic_report_id`;
- исходный запрос;
- финальный ответ;
- оценку и комментарий пользователя;
- канал и источник;
- распознанный intent;
- выбранного агента;
- expected agent, если Diagnostic Agent уверен в mismatch;
- route decision и route reason;
- почему выбран этот агент;
- какие actions сделал агент;
- какие tool calls были выполнены;
- краткие tool results;
- ошибки tool calls;
- loaded rules/skills оркестратора и агента;
- retrieval hits summary;
- missing requirements;
- wrong facts или unsupported claims, если они найдены;
- root cause;
- evidence;
- gaps;
- where to fix;
- fix proposal;
- regression test.

#### Deep context

Сохранять или уметь подтянуть по `incident_id` / `trace_id`, но не показывать в обычном отчете:

- полный trace payload;
- полный dialog context;
- полный target event;
- полный feedback event;
- полный diagnostic_report event;
- raw orchestrator decision;
- raw specialist decision;
- tool inputs и outputs с маскированием секретов;
- all retrieval hits;
- model usage;
- loaded rule/skill match reasons;
- environment snapshot без секретов;
- branch/commit/config flags без значений секретов.

Сделать:

- добавить `view`/`detail` параметр для отчета: `summary`, `codex`, `deep`;
- в CLI добавить аналогичный флаг, например `--view summary|codex|deep`;
- в API `GET /learning/reports/errors` поддержать `view`;
- для чатовой команды всегда использовать `summary`;
- для Codex и локального CLI по умолчанию использовать `codex`;
- `deep` отдавать только через CLI/API с admin secret, не в обычный Bitrix-чат;
- в deep view маскировать секреты и токены до форматирования ответа;
- сохранить backward compatibility с текущим `format=markdown|json`.

Критерий готовности:

- Bitrix-чату доступен короткий безопасный отчет;
- Codex получает отчет, достаточный для исправления без ручного поиска по всем логам;
- deep context доступен по запросу и не смешивается с обычным report view;
- ни один report view не вызывает Diagnostic Agent напрямую в обход `internal_orchestrator`;
- secrets masking покрыт тестами.

### 7. Добавить regression tests

Статус: нужно сделать.

Цель: изменения нельзя считать готовыми без тестов на полный feedback -> incident -> diagnostic_report -> report flow.

Покрыть:

- parsing оценок `3`, `3/10`, `2 не нашел задачу`;
- хорошая оценка не создает incident;
- плохая оценка создает incident;
- incident содержит target event, feedback, trace и нормализованные блоки;
- Diagnostic Agent вызывается через orchestrator;
- diagnostic_report связан с incident;
- ErrorReportService группирует новые и старые incidents;
- CLI отчет возвращает markdown и json;
- API отчет идет через `/learning/reports/errors`.

Критерий готовности:

- unit tests проходят локально;
- docs-only и backend changes не требуют live prod;
- live smoke при необходимости выполняется только через dev-бота `AI dev`.

### 8. Dev live smoke

Статус: делать только после unit/integration tests.

Цель: проверить реальную dev-цепочку без prod-бота.

Сценарий:

```text
сообщение dev-боту
-> ответ с feedback prompt
-> низкая оценка с комментарием
-> human_feedback
-> incident
-> Diagnostic Agent через internal_orchestrator
-> diagnostic_report
-> команда отчета в dev/admin чате
```

Проверить:

- pending feedback привязан к правильному `dialog_key`;
- оценка не воспринимается как новый обычный запрос;
- acknowledgement в чат не раскрывает внутренности;
- отчет не содержит секреты;
- trace показывает путь через `internal_orchestrator`.

## Порядок реализации

Рекомендуемый порядок:

1. Docs и контракт полей.
2. Backward-compatible schema helpers.
3. Deterministic incident classification.
4. Structured diagnostic_report metadata.
5. Group key/signature и обновление ErrorReportService.
6. Разделение Codex/Bitrix report views.
7. Layered report: summary/codex/deep.
8. Unit/integration tests.
9. Dev live smoke.

## Открытые решения

- Нужно ли вводить отдельный статус incident: `open`, `diagnosed`, `fixed`, `ignored`.
- Нужен ли отдельный endpoint для конкретной карточки ошибки: `GET /learning/incidents/{id}`.
- Нужно ли сохранять full trace snapshot в incident или держать только `trace_id` плюс короткую evidence-сводку.
- Кто подтверждает `fixed`: Codex после PR, человек в Bitrix-чате или отдельная admin-команда.
- Когда переносить память ошибок из JSONL в SQLite/Postgres, если объем станет большим.

## Definition of Done

Этап считается завершенным, когда:

- плохой feedback создает incident с нормализованной карточкой ошибки;
- Diagnostic Agent через оркестратора создает связанный diagnostic_report со структурированными полями;
- похожие ошибки группируются стабильными ключами;
- Codex получает полный JSON-отчет для исправления;
- Bitrix-чат получает короткий безопасный отчет;
- deep context доступен только по явному запросу и с маскированием секретов;
- входящие сообщения, чатовые команды и Diagnostic Agent flow идут через `internal_orchestrator`;
- есть regression tests на основной flow;
- dev live smoke подтвержден без изменений prod-бота.
