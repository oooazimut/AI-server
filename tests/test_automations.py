from ai_server.registry import (
    get_agent_manifest,
    load_automation_manifests,
    summarize_agents,
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
    }.issubset(ids)
    assert "bitrix_vehicle_usage" not in ids
    assert "bitrix_event_poller" not in ids
    assert all(automation.owner_agent_id == "bitrix24" for automation in automations)


def test_logistics_automations_are_registered():
    automations = load_automation_manifests(agent_id="logistics")
    ids = {automation.id for automation in automations}

    assert "logistics_vehicle_usage" in ids
    assert all(automation.owner_agent_id == "logistics" for automation in automations)


def test_retired_bitrix_business_automations_are_not_registered():
    ids = {automation.id for automation in load_automation_manifests(agent_id="bitrix24")}

    assert "bitrix_task_supervisor" not in ids
    assert "bitrix_task_quality_control" not in ids


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
