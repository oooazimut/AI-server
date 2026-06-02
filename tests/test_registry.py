from ai_server.registry import load_agent_manifests, summarize_agents


def test_load_agent_manifests():
    manifests = load_agent_manifests()

    assert {agent.id for agent in manifests} >= {
        "internal_orchestrator",
        "support_operator",
        "bitrix24",
        "networking",
    }


def test_summarize_agents():
    summaries = summarize_agents(load_agent_manifests())

    assert summaries
    assert all(summary.id for summary in summaries)
