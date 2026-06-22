from __future__ import annotations

from .agent_schema import PostgresAgentSchema


class PostgresOrchestratorStore(PostgresAgentSchema):
    """История диалогов оркестратора (схема internal_orchestrator).

    Наследует ensure_schema(), load_turns(), append_turn() из PostgresAgentSchema.
    Никаких дополнительных таблиц: оркестратор хранит только историю диалогов.
    """

    _SCHEMA = "internal_orchestrator"
