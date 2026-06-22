from .agent_schema import PostgresAgentSchema


class PostgresPtoAgentStore(PostgresAgentSchema):
    """PTO agent store: dialog_history in the 'pto' schema.

    Satisfies AgentDialogStorePort via structural typing.
    Agent-specific tables (documents) will be added in a future sprint.
    """

    _SCHEMA = "pto"
