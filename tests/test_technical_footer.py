import anyio

from ai_server.models import AgentResult, ModelUsageRecord
from ai_server.technical_footer import ProviderBalanceSnapshot, TechnicalFooterService, append_footer


def test_technical_footer_shows_model_usage_and_provider_balance_for_allowed_user(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "1,9")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_BALANCE_ENABLED", "true")

    result = AgentResult(
        status="completed",
        agent_id="internal_orchestrator",
        answer="Готово",
        handoff_to=["bitrix24"],
        model_usage=[
            ModelUsageRecord(
                agent_id="internal_orchestrator",
                provider="deepseek",
                model="deepseek-v4-flash",
                status="used",
                input_tokens=5177,
                output_tokens=299,
            ),
            ModelUsageRecord(
                agent_id="internal_orchestrator",
                provider="local",
                model="specialist_answer_passthrough",
                status="skipped",
            ),
        ],
    )
    service = TechnicalFooterService(balance_registry={"deepseek": FakeDeepSeekBalance()})

    async def build_footer():
        return await service.build_for_agent_result(result, user_id=9, channel="bitrix24_chat")

    footer = anyio.run(build_footer)

    assert footer == "оркестр → Bitrix, 5177/299 ток., DeepSeek OK, $12.34"
    assert "specialist_answer_passthrough" not in footer


def test_technical_footer_shows_zero_tokens_and_balance_for_direct_route(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "1,9")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_BALANCE_ENABLED", "true")

    result = AgentResult(
        status="completed",
        agent_id="internal_orchestrator",
        answer="Готово",
        handoff_to=["bitrix24"],
        model_usage=[
            ModelUsageRecord(
                agent_id="internal_orchestrator",
                provider="",
                model="",
                status="skipped",
                notes=["explicit_agent_direct_route"],
            ),
            ModelUsageRecord(agent_id="bitrix24", provider="local", model="bitrix_direct_response", status="skipped"),
        ],
    )
    service = TechnicalFooterService(balance_registry={"deepseek": FakeDeepSeekBalance()})

    async def build_footer():
        return await service.build_for_agent_result(result, user_id=9, channel="bitrix24_chat")

    footer = anyio.run(build_footer)

    assert footer == "оркестр → Bitrix, 0/0 ток., DeepSeek OK, $12.34"


def test_technical_footer_is_hidden_for_non_allowed_user(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "1,9")

    service = TechnicalFooterService(balance_registry={"deepseek": FakeDeepSeekBalance()})
    result = AgentResult(status="completed", agent_id="internal_orchestrator", answer="Готово")

    async def build_footer():
        return await service.build_for_agent_result(result, user_id=5, channel="bitrix24_chat")

    footer = anyio.run(build_footer)

    assert footer == ""


def test_append_footer_keeps_user_answer_readable():
    message = append_footer(
        "Готово",
        "оркестр → Bitrix, 5177/299 ток., DeepSeek OK, $12.34",
    )

    assert message == "Готово\n\n---\nТех: оркестр → Bitrix, 5177/299 ток., DeepSeek OK, $12.34"


class FakeDeepSeekBalance:
    async def snapshot(self):
        return ProviderBalanceSnapshot(
            provider="deepseek",
            status="ok",
            lines=["DeepSeek: доступен; баланс $12.34."],
            available=True,
        )
