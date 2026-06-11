from ai_server.registry import get_agent_manifest, load_agent_manifests, summarize_agents


def test_load_agent_manifests():
    manifests = load_agent_manifests()

    assert {agent.id for agent in manifests} >= {
        "internal_orchestrator",
        "bitrix24",
        "logistics",
        "pto",
    }


def test_bitrix_manifest_uses_package_specification():
    manifest = get_agent_manifest("bitrix24")

    assert manifest is not None
    assert manifest.instructions_file == "agents/bitrix24/instructions.md"
    assert manifest.skills_path == "agents/bitrix24/skills"
    assert manifest.knowledge_path == "agents/bitrix24/knowledge/topics"
    assert "portal_search" in manifest.tools


def test_summarize_agents():
    summaries = summarize_agents(load_agent_manifests())

    assert summaries
    assert all(summary.id for summary in summaries)
