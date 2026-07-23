import asyncio
import inspect
from pathlib import Path

import pytest

import ai_server.agent_worker as agent_worker
import ai_server.startup as startup
import ai_server.workers.bitrix.task_close_direct_dispatcher as task_close_dispatcher
from ai_server.agents.bitrix24.specialist import Bitrix24Specialist
from ai_server.agents.bitrix24.tools.bitrix_api import BitrixApiTool
from ai_server.agents.bitrix24.tools.task_create import TaskCreateDraftTool
from ai_server.models import AgentManifest, AgentTask
from ai_server.orchestrators.internal import OrchestratorTransportRuntime
from ai_server.orchestrators.plan_authoritative import (
    DeepSeekPlanService,
    PlanAuthoritativeOrchestrator,
)
from ai_server.registry import load_agent_manifests
from ai_server.specialists import build_specialist_registry


def test_production_manifests_have_one_autonomous_agent():
    manifests = load_agent_manifests()
    autonomous = [manifest.id for manifest in manifests if manifest.reasoning_mode == "autonomous"]

    assert autonomous == ["internal_orchestrator"]
    assert all(
        manifest.reasoning_mode == "executor"
        for manifest in manifests
        if manifest.kind == "specialist"
    )


def test_executor_without_structured_entrypoint_is_not_constructed(monkeypatch):
    class UnstructuredSpecialist:
        build_calls = 0

        @classmethod
        def build(cls, manifest, **deps):
            cls.build_calls += 1
            return cls()

    manifest = AgentManifest(
        id="legacy",
        name="Legacy",
        kind="specialist",
        reasoning_mode="executor",
        description="test",
        entrypoint="tests.fake.Legacy",
    )
    monkeypatch.setattr("ai_server.specialists._load_entrypoint", lambda _: UnstructuredSpecialist)

    assert build_specialist_registry([manifest]) == {}
    assert UnstructuredSpecialist.build_calls == 0


def test_transport_base_cannot_process_user_semantics():
    manifest = AgentManifest(
        id="internal_orchestrator",
        name="Orchestrator",
        kind="orchestrator",
        reasoning_mode="autonomous",
        description="test",
    )

    with pytest.raises(RuntimeError, match="PLAN_AUTHORITATIVE_PLANNER_REQUIRED"):
        PlanAuthoritativeOrchestrator.build(manifest, orchestrator_llm=object())
    with pytest.raises(RuntimeError, match="PLAN_AUTHORITATIVE_HANDLER_REQUIRED"):
        asyncio.run(
            OrchestratorTransportRuntime(manifest).handle(
                AgentTask(task_id="t", request="test")
            )
        )


def test_plan_service_requires_explicit_orchestrator_client():
    with pytest.raises(RuntimeError, match="PLAN_AUTHORITATIVE_LLM_CLIENT_REQUIRED"):
        DeepSeekPlanService(None)  # type: ignore[arg-type]


def test_only_worker_runtime_constructs_the_pro_orchestrator():
    startup_source = inspect.getsource(startup)
    worker_source = inspect.getsource(agent_worker)

    assert "DeepSeekPlanService" not in startup_source
    assert "PlanAuthoritativeOrchestrator" not in startup_source
    assert "PlanAuthoritativeOrchestrator.build" in worker_source
    assert "build_orchestrator_llm_client" not in startup_source
    assert "DeepSeekPlanService" in worker_source
    for forbidden in (
        "OrchestratorLLMService(",
        "BitrixLLMService(",
        "LogisticsLLMService(",
        "DiagnostLLMService(",
        "PtoLLMService(",
        "KartotekaLLMService(",
    ):
        assert forbidden not in startup_source
        assert forbidden not in worker_source


def test_bitrix_free_text_guard_never_calls_a_model():
    manifest = next(item for item in load_agent_manifests() if item.id == "bitrix24")
    specialist = Bitrix24Specialist(manifest)

    result = asyncio.run(specialist.handle(AgentTask(task_id="t", request="покажи задачи")))

    assert result.status == "failed"
    assert result.metadata["reason"] == "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"
    assert result.model_usage[0].status == "not_used"


def test_bitrix_executor_has_no_model_injection_surface():
    assert "llm" not in inspect.signature(Bitrix24Specialist).parameters


def test_repository_declares_orchestrator_as_semantic_owner():
    source_root = Path(__file__).resolve().parents[1]
    authority = source_root / "ORCHESTRATOR_AUTHORITY.md"

    assert authority.exists()
    text = authority.read_text(encoding="utf-8")
    assert "internal_orchestrator" in text
    assert "единственный автономный агент" in text


def test_bitrix_tools_have_no_hidden_orchestrator_or_write_fallback_dependencies():
    source_root = Path(__file__).resolve().parents[1]
    bitrix_tools = source_root / "backend" / "ai_server" / "agents" / "bitrix24" / "tools"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in bitrix_tools.rglob("*.py")
    )

    assert "ai_server.orchestrators" not in combined
    assert "_execute_write" not in inspect.getsource(BitrixApiTool)
    assert "normalized_fallback" not in inspect.getsource(BitrixApiTool)
    assert set(inspect.signature(BitrixApiTool).parameters) == {"client", "bitrix_oauth"}
    assert set(inspect.signature(TaskCreateDraftTool).parameters) == {"store"}


def test_bitrix_executor_package_contains_no_user_language_policy():
    source_root = Path(__file__).resolve().parents[1]
    bitrix_package = source_root / "backend" / "ai_server" / "agents" / "bitrix24"
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in bitrix_package.rglob("*.py")
    )

    assert "да, подтверждаю" not in combined
    assert "def matches_draft_confirmation" not in combined
    assert "ai_server.orchestrators.draft_confirmation" not in combined
    assert not (bitrix_package / "draft_confirmation.py").exists()


def test_retired_bitrix_proposal_storage_cannot_return():
    source_root = Path(__file__).resolve().parents[1]
    store_source = (
        source_root / "backend" / "ai_server" / "integrations" / "postgres" / "bitrix_agent.py"
    ).read_text(encoding="utf-8")

    assert "incomplete_proposals" not in store_source
    assert "save_proposal" not in store_source
    assert "incomplete_proposals" not in (source_root / "deploy" / "migrate_sprint11.sql").read_text(
        encoding="utf-8"
    )


def test_task_close_human_templates_exist_only_in_orchestrator():
    source_root = Path(__file__).resolve().parents[1]
    executor_source = (
        source_root / "backend" / "ai_server" / "agents" / "bitrix24" / "tools" / "task_close.py"
    ).read_text(encoding="utf-8")
    formatter_source = (
        source_root / "backend" / "ai_server" / "orchestrators" / "bitrix_formatter.py"
    ).read_text(encoding="utf-8")

    for marker in (
        "1. Выполняемые работы",
        "2. Использовано материалов, оборудование",
        "3. Статус выполнения работ",
        "Внести изменения (укажите пункт или подпункт",
    ):
        assert marker not in executor_source
        assert marker in formatter_source
    assert "format_task_close_draft_message" not in executor_source
    assert "result_templates.example.json" not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (source_root / "backend").rglob("*.py")
    )


def test_root_docs_do_not_advertise_retired_bitrix_runtime():
    source_root = Path(__file__).resolve().parents[1]
    docs = "\n".join(
        (source_root / name).read_text(encoding="utf-8")
        for name in ("README.md", "CLAUDE.md")
    )

    for marker in (
        "QUALITY_CONTROL_WEBHOOK_ENABLED",
        "SUPERVISOR_ENABLED",
        "BitrixLLMService",
        "ProposalStorePort",
        "BitrixSupervisorPort",
        "QualityControlHandlerPort",
    ):
        assert marker not in docs


def test_retired_orchestrator_model_service_source_is_deleted():
    source_root = Path(__file__).resolve().parents[1]

    assert not (
        source_root / "backend" / "ai_server" / "orchestrators" / "orchestrator_llm.py"
    ).exists()


def test_retired_agent_brains_are_physically_deleted():
    source_root = Path(__file__).resolve().parents[1]
    backend_agents = source_root / "backend" / "ai_server" / "agents"

    for retired in ("pto", "kartoteka", "diagnost"):
        assert not any((backend_agents / retired).rglob("*.py"))
    for retired_source in (
        backend_agents / "logistics" / "llm.py",
        source_root / "backend" / "ai_server" / "workers" / "diagnost" / "feedback_receiver.py",
        source_root / "backend" / "ai_server" / "workers" / "diagnost" / "feedback_scheduler.py",
        source_root / "backend" / "ai_server" / "scheduler_worker.py",
    ):
        assert not retired_source.exists()


def test_executors_and_event_workers_cannot_import_the_model_client():
    source_root = Path(__file__).resolve().parents[1] / "backend" / "ai_server"
    for package in ("agents", "workers", "routes"):
        for source_file in (source_root / package).rglob("*.py"):
            content = source_file.read_text(encoding="utf-8")
            assert "from ai_server.llm import" not in content, source_file
            assert "import ai_server.llm" not in content, source_file


def test_operator_manifests_have_no_brain_or_user_routing_surface():
    operators = {
        manifest.id: manifest
        for manifest in load_agent_manifests()
        if manifest.kind == "operator"
    }

    assert set(operators) == {"diagnost"}
    for manifest in operators.values():
        assert manifest.reasoning_mode == "executor"
        assert manifest.entrypoint is None
        assert manifest.instructions_file is None
        assert manifest.skills_path is None
        assert manifest.knowledge_path is None
        assert manifest.capabilities == []
        assert manifest.tools == []
        assert all(not automation.uses_llm for automation in manifest.automations)


def test_logistics_is_a_structured_executor_without_a_brain_surface():
    manifest = next(item for item in load_agent_manifests() if item.id == "logistics")

    assert manifest.kind == "specialist"
    assert manifest.reasoning_mode == "executor"
    assert manifest.entrypoint == "backend.ai_server.agents.logistics.LogisticsSpecialist"
    assert manifest.instructions_file is None
    assert manifest.skills_path is None
    assert manifest.knowledge_path is None
    assert manifest.tools
    assert all(not automation.uses_llm for automation in manifest.automations)


def test_webhook_ingress_has_no_premodel_feedback_interceptor():
    source = inspect.getsource(agent_worker)
    queue_source = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "ai_server"
        / "workers"
        / "bitrix"
        / "webhook_event_queue.py"
    ).read_text(encoding="utf-8")

    assert "FeedbackReceiver" not in source
    assert "feedback_receiver" not in queue_source


def test_bitrix_manifest_and_package_have_no_legacy_semantic_sources():
    manifest = next(item for item in load_agent_manifests() if item.id == "bitrix24")
    package = Path(__file__).resolve().parents[1] / "agents" / "bitrix24"

    assert manifest.skills_path is None
    assert manifest.contracts_path is None
    assert manifest.knowledge_path is None
    assert not (package / "skills").exists() or not any((package / "skills").iterdir())
    assert not (package / "contracts").exists() or not any((package / "contracts").iterdir())
    assert not (package / "knowledge" / "topics").exists() or not any(
        (package / "knowledge" / "topics").iterdir()
    )
    assert "app.agent.runtime_v2" not in (package / "manifest.yaml").read_text(encoding="utf-8")


def test_bitrix_executor_package_has_no_llm_or_business_worker_sources():
    source_root = Path(__file__).resolve().parents[1]
    bitrix_package = source_root / "backend" / "ai_server" / "agents" / "bitrix24"
    worker_package = source_root / "backend" / "ai_server" / "workers" / "bitrix"

    assert not (bitrix_package / "llm.py").exists()
    assert not (bitrix_package / "quality_control.py").exists()
    assert not (worker_package / "supervisor.py").exists()

    forbidden = (
        "BitrixLLMService",
        "FakeBitrixLLM",
        "bitrix_task_supervisor",
        "bitrix_task_quality_control",
    )
    for path in (source_root / "backend", source_root / "agents"):
        for source_file in path.rglob("*"):
            if source_file.suffix not in {".py", ".yaml", ".yml", ".md", ".json"}:
                continue
            if source_file == Path(__file__):
                continue
            content = source_file.read_text(encoding="utf-8")
            for marker in forbidden:
                assert marker not in content, f"{marker} remains in {source_file}"


def test_task_close_event_worker_cannot_execute_or_render_bitrix_actions():
    source = inspect.getsource(task_close_dispatcher)

    assert "orchestrator_handler(task)" in source
    for forbidden in (
        "_execute_task_close",
        "format_task_close_draft_message",
        "build_task_close_draft_from_args",
        "send_bot_message",
        "bitrix.call",
    ):
        assert forbidden not in source
