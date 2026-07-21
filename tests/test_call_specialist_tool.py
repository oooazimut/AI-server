import asyncio
from unittest.mock import AsyncMock, MagicMock

from ai_server.models import AgentResult, AgentTask, ToolStatus
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool
from ai_server.registry import load_agent_manifests
from tests.fakes import FakeOrchestratorStore


def _fake_specialist(*, status: str = "completed", answer: str = "Готово.") -> MagicMock:
    spec = MagicMock()
    spec.handle = AsyncMock(
        return_value=AgentResult(
            status=status,
            agent_id="fake",
            answer=answer,
            confidence=0.9,
        )
    )
    return spec


def _make_tool(specialists: dict, store=None) -> CallSpecialistTool:
    manifests = load_agent_manifests()
    return CallSpecialistTool(specialists, manifests, store=store)


def test_routes_to_correct_specialist():
    bitrix = _fake_specialist(answer="Битрикс ответил.")
    tool = _make_tool({"bitrix24": bitrix})

    result = asyncio.run(tool.execute({"specialist_id": "bitrix24", "request": "Создай задачу"}))

    assert result.status == ToolStatus.OK
    assert result.data["specialist"] == "bitrix24"
    assert result.data["answer"] == "Битрикс ответил."
    bitrix.handle.assert_awaited_once()


def test_error_when_specialist_not_found():
    tool = _make_tool({"bitrix24": _fake_specialist()})

    result = asyncio.run(tool.execute({"specialist_id": "unknown", "request": "Что-то"}))

    assert result.status == ToolStatus.ERROR
    assert "unknown" in result.error


def test_sets_pending_on_needs_clarification():
    store = FakeOrchestratorStore()
    tool = _make_tool({"bitrix24": _fake_specialist(status="needs_clarification")}, store=store)

    asyncio.run(tool.execute({"specialist_id": "bitrix24", "request": "..."}, dialog_key="dlg1"))

    assert store._kv.get(("dlg1", "pending_specialist")) == "bitrix24"


def test_sets_pending_on_needs_human():
    store = FakeOrchestratorStore()
    tool = _make_tool({"bitrix24": _fake_specialist(status="needs_human")}, store=store)

    asyncio.run(tool.execute({"specialist_id": "bitrix24", "request": "..."}, dialog_key="dlg1"))

    assert store._kv.get(("dlg1", "pending_specialist")) == "bitrix24"


def test_clears_pending_on_completed():
    store = FakeOrchestratorStore()
    store.set_pending("dlg1", "bitrix24")
    tool = _make_tool({"bitrix24": _fake_specialist(status="completed")}, store=store)

    asyncio.run(tool.execute({"specialist_id": "bitrix24", "request": "..."}, dialog_key="dlg1"))

    assert store._kv.get(("dlg1", "pending_specialist")) is None


def test_passes_dialog_key_and_user_id_to_specialist():
    spec = _fake_specialist()
    tool = _make_tool({"bitrix24": spec})

    asyncio.run(
        tool.execute(
            {"specialist_id": "bitrix24", "request": "Запрос"},
            user_id=42,
            dialog_key="dlg1",
            dialog_id="123",
        )
    )

    call_args = spec.handle.call_args[0][0]
    assert isinstance(call_args, AgentTask)
    assert call_args.context.get("dialog_key") == "dlg1"
    assert call_args.context.get("dialog_id") == "123"
    assert call_args.user.id == "42"


def test_explicit_segment_request_wins_over_the_full_dialog_request():
    spec = _fake_specialist()
    tool = _make_tool({"bitrix24": spec})
    task = AgentTask(
        task_id="t1",
        request="Логист покажи машины Битрикс покажи склад Борисова",
        context={
            "t0006_original_request": "Логист покажи машины Битрикс покажи склад Борисова",
            "t0007_explicit_segment_request": "покажи склад Борисова",
        },
    )

    asyncio.run(tool.execute_with_task({"specialist_id": "bitrix24", "request": "покажи склад Борисова"}, task=task))

    assert spec.handle.call_args[0][0].request == "покажи склад Борисова"


def test_definition_includes_all_specialist_ids():
    tool = _make_tool({"bitrix24": _fake_specialist(), "pto": _fake_specialist()})
    defn = tool.definition()

    assert defn.name == "call_specialist"
    enum_vals = defn.parameters["properties"]["specialist_id"]["enum"]
    assert "bitrix24" in enum_vals
    assert "pto" in enum_vals


def test_schedule_fn_called_when_specialist_returns_tasks():
    from ai_server.models import ScheduledTask

    spec = MagicMock()
    spec.handle = AsyncMock(
        return_value=AgentResult(
            status="completed",
            agent_id="fake",
            answer="done",
            scheduled_tasks=[ScheduledTask(job_id="j1", agent_id="bitrix24", trigger={})],
        )
    )
    scheduled: list = []
    tool = _make_tool({"bitrix24": spec})
    tool.schedule_fn = lambda tasks: scheduled.extend(tasks)

    asyncio.run(tool.execute({"specialist_id": "bitrix24", "request": "..."}))

    assert len(scheduled) == 1
    assert scheduled[0].job_id == "j1"
