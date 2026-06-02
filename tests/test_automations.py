from ai_server.registry import get_agent_manifest, summarize_agents
from ai_server.workers.registry import (
    get_automation_manifest,
    load_automation_manifests,
    summarize_automations,
)


def test_bitrix_automations_are_registered():
    automations = load_automation_manifests(agent_id="bitrix24")
    ids = {automation.id for automation in automations}

    assert {
        "bitrix_webhook_event_queue",
        "bitrix_portal_search_indexer",
        "bitrix_search_webhook_indexer",
        "bitrix_reconciler",
        "bitrix_task_supervisor",
        "bitrix_task_quality_control",
        "bitrix_vehicle_usage",
        "bitrix_event_poller",
    }.issubset(ids)
    assert all(automation.owner_agent_id == "bitrix24" for automation in automations)


def test_quality_control_automation_policy_flags():
    automation = get_automation_manifest("bitrix_task_quality_control")

    assert automation is not None
    assert automation.kind == "business_workflow"
    assert automation.trigger == "webhook"
    assert automation.uses_llm is True
    assert automation.requires_oauth_actor is True
    assert automation.human_approval_required is True
    assert "var/quality_control_state.json" in automation.state_paths


def test_agent_summary_exposes_automation_ids():
    manifest = get_agent_manifest("bitrix24")

    assert manifest is not None
    summary = next(item for item in summarize_agents([manifest]) if item.id == "bitrix24")
    assert "bitrix_portal_search_indexer" in summary.automations


def test_summarize_automations():
    summaries = summarize_automations(load_automation_manifests(agent_id="bitrix24"))

    assert summaries
    assert {summary.owner_agent_id for summary in summaries} == {"bitrix24"}
    assert any(summary.kind == "data_pipeline" for summary in summaries)

