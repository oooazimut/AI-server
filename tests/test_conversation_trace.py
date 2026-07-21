from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ai_server.integrations.redis.conversation_trace import RedisConversationTrace
from ai_server.integrations.webhook_utils import sanitize_webhook_payload
from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.routes.bitrix import _validate_trace_secret, conversation_trace_recent


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)


def _settings(**overrides):
    defaults = {
        "conversation_trace_enabled": True,
        "conversation_trace_ttl_hours": 48,
        "conversation_trace_max_text_chars": 8000,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_trace() -> tuple[RedisConversationTrace, MagicMock]:
    trace = RedisConversationTrace("redis://localhost/0", settings=_settings())
    client = MagicMock()
    client.incr = AsyncMock(return_value=1)
    client.zremrangebyscore = AsyncMock()
    trace._client = client
    return trace, client


def _make_pipe(client: MagicMock) -> AsyncMock:
    pipe = AsyncMock()
    pipe.set = MagicMock()
    pipe.zadd = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[True])
    client.pipeline = MagicMock(return_value=pipe)
    return pipe


def _message_payload() -> dict:
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "auth": {"application_token": "top-secret"},
        "data": {
            "bot": {
                "id": 207,
                "auth": {
                    "access_token": "nested-access",
                    "refresh_token": "nested-refresh",
                    "client_secret": "nested-client-secret",
                },
            },
            "chat": {"id": 3669, "dialogId": "chat3669"},
            "message": {"id": 127203, "authorId": 27, "text": "Bitrix покажи склад Борисов"},
            "user": {"id": 27, "name": "Андрей Борисов"},
        },
    }


def test_sanitize_webhook_payload_removes_nested_auth_and_tokens():
    sanitized = sanitize_webhook_payload(_message_payload())
    dumped = json.dumps(sanitized, ensure_ascii=False)

    assert "top-secret" not in dumped
    assert "nested-access" not in dumped
    assert "nested-refresh" not in dumped
    assert "nested-client-secret" not in dumped
    assert not _contains_key(sanitized, "auth")
    assert not _contains_key(sanitized, "access_token")
    assert not _contains_key(sanitized, "refresh_token")


def test_record_inbound_message_writes_safe_searchable_trace():
    trace, client = _make_trace()
    pipe = _make_pipe(client)

    anyio_run(
        trace.record_inbound(
            event_id=618,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_message_payload(),
            inserted=True,
        )
    )

    pipe.set.assert_called_once()
    _, payload_json = pipe.set.call_args.args[:2]
    stored = json.loads(payload_json)
    dumped = json.dumps(stored, ensure_ascii=False)

    assert stored["trace_type"] == "inbound_message"
    assert stored["event_id"] == 618
    assert stored["message_id"] == 127203
    assert stored["user_id"] == 27
    assert stored["dialog_id"] == "chat3669"
    assert stored["text"] == "Bitrix покажи склад Борисов"
    assert "nested-access" not in dumped
    assert "refresh_token" not in dumped
    assert any("ai_server:conversation_trace:user:27" in str(call) for call in pipe.zadd.mock_calls)
    assert any("ai_server:conversation_trace:message:127203" in str(call) for call in pipe.zadd.mock_calls)


def test_record_ingress_keeps_unknown_v2_event_searchable_without_raw_payload():
    trace, client = _make_trace()
    pipe = _make_pipe(client)
    payload = _message_payload()
    payload["event"] = "ONIMBOTV2VOICEADD"
    payload["data"]["message"]["attachments"] = [{"id": 501, "name": "voice.ogg", "type": "audio/ogg"}]

    anyio_run(
        trace.record_ingress(
            event_id=619,
            event_type="ONIMBOTV2VOICEADD",
            payload=payload,
            inserted=True,
        )
    )

    pipe.set.assert_called_once()
    _, payload_json = pipe.set.call_args.args[:2]
    stored = json.loads(payload_json)
    dumped = json.dumps(stored, ensure_ascii=False)

    assert stored["trace_type"] == "ingress_event"
    assert stored["message_event_recognized"] is False
    assert stored["parse_status"] == "parsed"
    assert stored["message_id"] == 127203
    assert stored["user_id"] == 27
    assert stored["dialog_id"] == "chat3669"
    assert stored["attachment_count"] == 1
    assert stored["attachment_type_hints"] == ["audio/ogg"]
    assert stored["audio_attachment_hint"] is True
    assert "top-secret" not in dumped
    assert "nested-access" not in dumped
    assert any("ai_server:conversation_trace:user:27" in str(call) for call in pipe.zadd.mock_calls)
    assert any("ai_server:conversation_trace:message:127203" in str(call) for call in pipe.zadd.mock_calls)


def test_record_agent_result_and_outbound_are_indexed_by_task_and_user():
    trace, client = _make_trace()
    pipe = _make_pipe(client)
    task = AgentTask(
        task_id="task-1",
        request="Покажи склад Борисов",
        user=UserContext(id="27", channel="bitrix24", raw={"message_id": 127203, "chat_id": 3669}),
        context={"dialog_key": "chat:3669:user:27", "dialog_id": "chat3669", "recipient_id": "chat3669"},
    )
    result = AgentResult(status="needs_clarification", agent_id="internal_orchestrator", answer="Нужна авторизация")

    anyio_run(trace.record_agent_result(task=task, result=result, source="orchestrator"))
    anyio_run(
        trace.record_outbound(
            task=task,
            result=result,
            recipient_id="chat3669",
            body="Нужна авторизация",
            status="sent",
        )
    )

    assert pipe.set.call_count == 2
    stored_events = [json.loads(call.args[1]) for call in pipe.set.mock_calls]
    assert [event["trace_type"] for event in stored_events] == ["agent_result", "outbound_message"]
    assert all(event["task_id"] == "task-1" for event in stored_events)
    assert all(event["user_id"] == "27" for event in stored_events)
    assert any("ai_server:conversation_trace:task:" in str(call) for call in pipe.zadd.mock_calls)


def test_record_timing_step_is_indexed_by_message_task_and_dialog():
    trace, client = _make_trace()
    pipe = _make_pipe(client)
    task = AgentTask(
        task_id="task-1",
        request="Покажи склад Борисов",
        user=UserContext(id="27", channel="bitrix24", raw={"message_id": 127203, "chat_id": 3669}),
        context={"dialog_key": "chat:3669:user:27", "dialog_id": "chat3669", "recipient_id": "chat3669"},
    )

    anyio_run(
        trace.record_timing(
            task=task,
            component="bitrix24",
            stage="tool_execute",
            started_at="2026-07-14T14:43:46.000000+03:00",
            elapsed_ms=1234.56,
            status="ok",
            step=2,
            tool="bitrix_warehouse_search",
            details={"rows": 10},
        )
    )

    pipe.set.assert_called_once()
    stored = json.loads(pipe.set.call_args.args[1])
    assert stored["trace_type"] == "timing_step"
    assert stored["component"] == "bitrix24"
    assert stored["stage"] == "tool_execute"
    assert stored["elapsed_ms"] == 1234.6
    assert stored["started_at"] == "2026-07-14T14:43:46.000000+03:00"
    assert stored["status"] == "ok"
    assert stored["step"] == 2
    assert stored["tool"] == "bitrix_warehouse_search"
    assert stored["message_id"] == 127203
    assert stored["task_id"] == "task-1"
    assert any("ai_server:conversation_trace:message:127203" in str(call) for call in pipe.zadd.mock_calls)
    assert any("ai_server:conversation_trace:task:" in str(call) for call in pipe.zadd.mock_calls)


def test_recent_returns_events_from_redis_zset():
    trace, client = _make_trace()
    client.zrevrangebyscore = AsyncMock(return_value=["2", "1"])
    client.get = AsyncMock(
        side_effect=[
            '{"id": 2, "trace_type": "outbound_message"}',
            '{"id": 1, "trace_type": "inbound_message"}',
        ]
    )

    events = anyio_run(trace.recent(limit=10, hours=24))

    assert [event["id"] for event in events] == [2, 1]


def test_conversation_trace_admin_endpoint_requires_secret():
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(conversation_trace_secret="s")))
    )

    with pytest.raises(HTTPException) as exc:
        _validate_trace_secret(request, provided="wrong")

    assert exc.value.status_code == 403


def test_conversation_trace_admin_endpoint_queries_by_user():
    trace = SimpleNamespace(
        enabled=True,
        by_message=AsyncMock(return_value=[]),
        by_task=AsyncMock(return_value=[]),
        by_dialog=AsyncMock(return_value=[]),
        by_user=AsyncMock(return_value=[{"id": 1, "trace_type": "inbound_message"}]),
        recent=AsyncMock(return_value=[]),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(conversation_trace_secret="trace-secret"),
                conversation_trace=trace,
            )
        )
    )

    result = anyio_run(
        conversation_trace_recent(
            request,
            secret="trace-secret",
            hours=24,
            limit=100,
            user_id="27",
        )
    )

    assert result["enabled"] is True
    assert result["count"] == 1
    trace.by_user.assert_awaited_once_with("27", limit=100, hours=24)


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_disabled_trace_is_noop(enabled: bool, expected: int):
    trace = RedisConversationTrace("redis://localhost/0", settings=_settings(conversation_trace_enabled=enabled))
    client = MagicMock()
    client.incr = AsyncMock(return_value=1)
    client.pipeline = MagicMock(return_value=_make_pipe(client))
    trace._client = client

    anyio_run(trace.record({"trace_type": "manual"}))

    assert client.incr.await_count == expected


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return any(str(item_key).lower() == key for item_key in value) or any(
            _contains_key(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False
