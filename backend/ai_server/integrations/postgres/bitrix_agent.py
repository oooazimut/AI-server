from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.portal_search.search_index import (
    _merge_content_metadata,
    _score_result,
    _should_preserve_content,
)
from ai_server.integrations.bitrix.portal_search.text_utils import (
    body_with_content,
    clean_text,
    content_text_from_body,
    escape_like,
    file_extension,
    flatten_unique,
    increment,
    normalize_extensions,
    normalize_search_text,
    query_term_groups,
    safe_int,
    safe_json,
)
from ai_server.integrations.bitrix.portal_search.types import (
    CONTENT_INDEX_VERSION,
    CONTENT_TERMINAL_STATUSES,
    PortalContentReadiness,
    PortalIndexStats,
    PortalSearchResult,
)
from ai_server.utils import MOSCOW_TZ

from .agent_schema import PostgresAgentSchema

_TABLE = "bitrix24.portal_search_items"


def _draft_actor_user_id(params: dict[str, Any]) -> int | None:
    values = [params.get("_draft_user_id"), params.get("_actor_user_id"), params.get("user_id")]
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    values.extend([fields.get("CREATED_BY"), fields.get("RESPONSIBLE_ID")])
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


class PostgresBitrixAgentStore(PostgresAgentSchema):
    """Bitrix24 agent store: owns all Bitrix24 agent data in the 'bitrix24' schema.

    Tables: dialog_history, incomplete_proposals, portal_search_items.
    Satisfies AgentStorePort (dialogs) and provides portal search methods.
    """

    _SCHEMA = "bitrix24"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()  # creates bitrix24 schema + dialog_history table
        with self._sync_connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.incomplete_proposals (
                    id SERIAL PRIMARY KEY,
                    task_id INTEGER NOT NULL,
                    task_title TEXT,
                    missing_parts TEXT,
                    responsible_id INTEGER,
                    responsible_dialog_id TEXT,
                    responsible_response TEXT,
                    status TEXT NOT NULL DEFAULT 'awaiting_response',
                    created_at TEXT,
                    scheduled_for TEXT
                )
                """
            )
            db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    entity_type      TEXT NOT NULL,
                    entity_id        TEXT NOT NULL,
                    title            TEXT NOT NULL,
                    body             TEXT NOT NULL,
                    url              TEXT NOT NULL,
                    search_text      TEXT NOT NULL,
                    metadata_json    TEXT NOT NULL,
                    source_updated_at TEXT,
                    last_seen_at     TEXT,
                    indexed_at       TEXT NOT NULL,
                    PRIMARY KEY (entity_type, entity_id)
                )
                """
            )
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_type    ON {_TABLE}(entity_type)")
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_indexed ON {_TABLE}(indexed_at)")
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_seen    ON {_TABLE}(last_seen_at)")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.pending_task_draft (
                    dialog_key TEXT PRIMARY KEY,
                    params_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.interactive_drafts (
                    draft_id TEXT PRIMARY KEY,
                    dialog_key TEXT NOT NULL,
                    user_id INTEGER,
                    draft_type TEXT NOT NULL,
                    specialist_id TEXT NOT NULL,
                    original_request TEXT NOT NULL DEFAULT '',
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            db.execute(
                "ALTER TABLE bitrix24.interactive_drafts ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1"
            )
            db.execute("ALTER TABLE bitrix24.interactive_drafts ADD COLUMN IF NOT EXISTS claim_token TEXT")
            db.execute("ALTER TABLE bitrix24.interactive_drafts ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ")
            db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_interactive_drafts_one_active_dialog
                ON bitrix24.interactive_drafts(dialog_key)
                WHERE status = 'active'
                """
            )
            db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_interactive_drafts_one_open_dialog
                ON bitrix24.interactive_drafts(dialog_key)
                WHERE status IN ('active', 'confirming')
                """
            )
            db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_interactive_drafts_one_live_dialog
                ON bitrix24.interactive_drafts(dialog_key)
                WHERE status IN ('active', 'confirming', 'finalizing')
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interactive_drafts_status_expiry
                ON bitrix24.interactive_drafts(status, expires_at)
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.interactive_draft_transitions (
                    id BIGSERIAL PRIMARY KEY,
                    draft_id TEXT NOT NULL,
                    dialog_key TEXT NOT NULL,
                    transition TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    actor_user_id INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interactive_draft_transitions_draft
                ON bitrix24.interactive_draft_transitions(draft_id, created_at)
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_control_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_by INTEGER,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_controlled_users (
                    user_id INTEGER PRIMARY KEY,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    controlled_from TIMESTAMPTZ,
                    updated_by INTEGER,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                "ALTER TABLE bitrix24.task_close_controlled_users ADD COLUMN IF NOT EXISTS controlled_from TIMESTAMPTZ"
            )
            db.execute(
                """
                UPDATE bitrix24.task_close_controlled_users AS controlled
                SET controlled_from = COALESCE(
                        (
                            SELECT CASE
                                WHEN length(setting.value) = 10
                                    THEN setting.value::date::timestamp AT TIME ZONE 'Europe/Moscow'
                                ELSE setting.value::timestamptz
                            END
                            FROM bitrix24.task_close_control_settings AS setting
                            WHERE setting.key = 'control_enabled_from'
                              AND setting.value ~ '^\\d{4}-\\d{2}-\\d{2}($|[T ])'
                            LIMIT 1
                        ),
                        controlled.updated_at,
                        now()
                    )
                WHERE controlled.active IS TRUE AND controlled.controlled_from IS NULL
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_control_operators (
                    user_id INTEGER PRIMARY KEY,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_by INTEGER,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_control_revisions (
                    id SERIAL PRIMARY KEY,
                    actor_user_id INTEGER,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_processing_state (
                    task_id INTEGER NOT NULL,
                    state_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    actor_user_id INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (task_id, state_key)
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_close_processing_state_status
                ON bitrix24.task_close_processing_state(status)
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.task_close_control_events (
                    task_id INTEGER NOT NULL,
                    close_event_key TEXT NOT NULL,
                    responsible_id INTEGER,
                    closed_by_user_id INTEGER,
                    closed_at TIMESTAMPTZ,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (task_id, close_event_key)
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_close_control_events_decision
                ON bitrix24.task_close_control_events(decision)
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_close_control_events_responsible
                ON bitrix24.task_close_control_events(responsible_id)
                """
            )

    def save_proposal(
        self,
        *,
        task_id: int,
        task_title: str = "",
        missing_parts: str = "",
        responsible_id: int | None = None,
        responsible_dialog_id: str = "",
        scheduled_for: str = "",
    ) -> int:
        with self._sync_connect() as db:
            row = db.execute(
                """
                INSERT INTO bitrix24.incomplete_proposals
                    (task_id, task_title, missing_parts, responsible_id, responsible_dialog_id,
                     status, created_at, scheduled_for)
                VALUES (%s, %s, %s, %s, %s, 'awaiting_response', %s, %s)
                RETURNING id
                """,
                (
                    task_id,
                    task_title,
                    missing_parts,
                    responsible_id,
                    responsible_dialog_id,
                    datetime.now(UTC).isoformat(),
                    scheduled_for,
                ),
            ).fetchone()
        return int(row["id"]) if row else 0

    def get_proposal_by_id(self, proposal_id: int) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute("SELECT * FROM bitrix24.incomplete_proposals WHERE id = %s", (proposal_id,)).fetchone()
        return dict(row) if row else None

    def get_proposals_for_manager(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT * FROM bitrix24.incomplete_proposals
                WHERE status IN ('awaiting_response', 'proposed')
                ORDER BY created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM bitrix24.incomplete_proposals
                WHERE responsible_id = %s AND status = 'awaiting_response'
                ORDER BY created_at LIMIT 1
                """,
                (responsible_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_responsible_response(self, proposal_id: int, response_text: str) -> None:
        with self._sync_connect() as db:
            db.execute(
                "UPDATE bitrix24.incomplete_proposals SET responsible_response = %s WHERE id = %s",
                (response_text, proposal_id),
            )

    def mark_status(self, proposal_id: int, status: str) -> None:
        with self._sync_connect() as db:
            db.execute(
                "UPDATE bitrix24.incomplete_proposals SET status = %s WHERE id = %s",
                (status, proposal_id),
            )

    def delete_proposal(self, proposal_id: int) -> None:
        with self._sync_connect() as db:
            db.execute("DELETE FROM bitrix24.incomplete_proposals WHERE id = %s", (proposal_id,))

    # ------------------------------------------------------------------
    # Pending Bitrix write draft
    # ------------------------------------------------------------------

    async def save_task_draft(self, dialog_key: str, params: dict[str, Any]) -> None:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT draft_id, draft_type, original_request, params_json, status, version, created_at, expires_at
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND status IN ('active', 'confirming', 'finalizing')
                FOR UPDATE
                """,
                (dialog_key,),
            )
            active = await cur.fetchone()
            draft_type = str(params.get("_draft_type") or "task_create").strip() or "task_create"
            actor_user_id = _draft_actor_user_id(params)
            specialist_id = str(params.get("_draft_specialist") or "bitrix24").strip() or "bitrix24"
            draft_params = dict(params)
            promoted_created_at = _parse_draft_datetime(draft_params.pop("_promoted_legacy_created_at", None))
            original_request = str(draft_params.get("_original_request") or "")
            now = datetime.now(UTC)
            transition = "created"
            if active and active["status"] == "active" and active["expires_at"] <= now:
                await db.execute(
                    """
                    UPDATE bitrix24.interactive_drafts
                    SET status = 'expired', updated_at = %s
                    WHERE draft_id = %s AND status = 'active'
                    """,
                    (now, active["draft_id"]),
                )
                await db.execute(
                    """
                    INSERT INTO bitrix24.interactive_draft_transitions
                        (draft_id, dialog_key, transition, payload_json, actor_user_id)
                    VALUES (%s, %s, 'expired', %s, %s)
                    """,
                    (active["draft_id"], dialog_key, active["params_json"], actor_user_id),
                )
                active = None
            if active:
                if active["status"] != "active":
                    raise RuntimeError(
                        f"ACTIVE_DRAFT_IN_PROGRESS:{active['draft_id']}:{active['draft_type']}:{active['status']}"
                    )
                active_type = str(active["draft_type"] or "")
                if active_type != draft_type:
                    raise RuntimeError(f"ACTIVE_DRAFT_CONFLICT:{active['draft_id']}:{active_type}:{draft_type}")
                draft_id = str(active["draft_id"])
                created_at = active["created_at"]
                expires_at = active["expires_at"]
                version = int(active["version"] or 1) + 1
                transition = "edited"
                initial_request = str(active["original_request"] or original_request)
            else:
                draft_id = str(uuid.uuid4())
                created_at = min(promoted_created_at, now) if promoted_created_at is not None else now
                expires_at = now + timedelta(minutes=15)
                if promoted_created_at is not None:
                    expires_at = created_at + timedelta(minutes=15)
                version = 1
                initial_request = original_request
            stored_params = {
                **draft_params,
                "_draft_id": draft_id,
                "_draft_type": draft_type,
                "_draft_specialist": specialist_id,
                "_draft_created_at": created_at.isoformat(),
                "_draft_expires_at": expires_at.isoformat(),
                "_draft_version": version,
            }
            if initial_request:
                stored_params["_original_request"] = initial_request
            if actor_user_id is not None:
                stored_params["_draft_user_id"] = actor_user_id
            params.clear()
            params.update(stored_params)
            payload_json = json.dumps(stored_params, ensure_ascii=False)
            transition_payload = dict(stored_params)
            if transition == "edited" and original_request and original_request != initial_request:
                transition_payload["_edit_request"] = original_request
            transition_payload_json = json.dumps(transition_payload, ensure_ascii=False)
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_drafts
                    (draft_id, dialog_key, user_id, draft_type, specialist_id, original_request,
                     params_json, status, version, created_at, updated_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s, %s, %s, %s)
                ON CONFLICT (draft_id) DO UPDATE
                SET user_id = EXCLUDED.user_id,
                    original_request = EXCLUDED.original_request,
                    params_json = EXCLUDED.params_json,
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
                """,
                (
                    draft_id,
                    dialog_key,
                    actor_user_id,
                    draft_type,
                    specialist_id,
                    initial_request,
                    payload_json,
                    version,
                    created_at,
                    now,
                    expires_at,
                ),
            )
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (draft_id, dialog_key, transition, transition_payload_json, actor_user_id),
            )
            await db.execute(
                """
                INSERT INTO bitrix24.pending_task_draft (dialog_key, params_json)
                VALUES (%s, %s)
                ON CONFLICT (dialog_key) DO UPDATE
                SET params_json = EXCLUDED.params_json,
                    created_at = now()
                """,
                (dialog_key, payload_json),
            )

    async def claim_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        claim_token = str(uuid.uuid4())
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET status = 'confirming', claim_token = %s, claimed_at = %s, updated_at = %s
                WHERE dialog_key = %s
                  AND draft_id = %s
                  AND version = %s
                  AND draft_type = %s
                  AND status = 'active'
                  AND expires_at > %s
                RETURNING draft_id, params_json, user_id
                """,
                (claim_token, now, now, dialog_key, expected_draft_id, expected_version, expected_type, now),
            )
            claimed = await cur.fetchone()
            if not claimed:
                return None
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, 'confirming', %s, %s)
                """,
                (claimed["draft_id"], dialog_key, claimed["params_json"], claimed["user_id"]),
            )
        raw = claimed["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_claim_token"] = claim_token
        return result

    async def claim_expired_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        claim_token = str(uuid.uuid4())
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET status = 'finalizing', claim_token = %s, claimed_at = %s, updated_at = %s
                WHERE dialog_key = %s
                  AND draft_id = %s
                  AND version = %s
                  AND draft_type = %s
                  AND status IN ('active', 'expired')
                  AND expires_at <= %s
                RETURNING draft_id, params_json, user_id
                """,
                (claim_token, now, now, dialog_key, expected_draft_id, expected_version, expected_type, now),
            )
            claimed = await cur.fetchone()
            if not claimed:
                return None
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, 'finalizing', %s, %s)
                """,
                (claimed["draft_id"], dialog_key, claimed["params_json"], claimed["user_id"]),
            )
        raw = claimed["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_claim_token"] = claim_token
        return result

    async def reclaim_stale_finalizing_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=max(lease_seconds, 60))
        claim_token = str(uuid.uuid4())
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET claim_token = %s, claimed_at = %s, updated_at = %s
                WHERE dialog_key = %s
                  AND draft_id = %s
                  AND version = %s
                  AND draft_type = %s
                  AND status = 'finalizing'
                  AND COALESCE(claimed_at, updated_at) <= %s
                RETURNING draft_id, params_json, user_id
                """,
                (
                    claim_token,
                    now,
                    now,
                    dialog_key,
                    expected_draft_id,
                    expected_version,
                    expected_type,
                    cutoff,
                ),
            )
            claimed = await cur.fetchone()
            if not claimed:
                return None
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, 'reclaimed_finalizing', %s, %s)
                """,
                (claimed["draft_id"], dialog_key, claimed["params_json"], claimed["user_id"]),
            )
        raw = claimed["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_claim_token"] = claim_token
        result["_reclaimed_finalizing"] = True
        return result

    async def renew_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        claim_token: str,
        expected_status: str,
    ) -> bool:
        if not draft_id or not claim_token or expected_status not in {"confirming", "finalizing"}:
            return False
        now = datetime.now(UTC)
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET claimed_at = %s, updated_at = %s
                WHERE dialog_key = %s AND draft_id = %s
                  AND claim_token = %s AND status = %s
                RETURNING draft_id
                """,
                (now, now, dialog_key, draft_id, claim_token, expected_status),
            )
            return bool(await cur.fetchone())

    async def resolve_stale_confirming_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=max(lease_seconds, 60))
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET status = 'attention', claim_token = NULL, claimed_at = NULL, updated_at = %s
                WHERE dialog_key = %s AND draft_id = %s AND version = %s
                  AND draft_type = %s AND status = 'confirming'
                  AND COALESCE(claimed_at, updated_at) <= %s
                RETURNING params_json, user_id
                """,
                (now, dialog_key, expected_draft_id, expected_version, expected_type, cutoff),
            )
            resolved = await cur.fetchone()
            if not resolved:
                return None
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, 'attention_unknown_outcome', %s, %s)
                """,
                (expected_draft_id, dialog_key, resolved["params_json"], resolved["user_id"]),
            )
        raw = resolved["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_resolution_status"] = "attention"
        return result

    async def release_task_draft(self, dialog_key: str, *, draft_id: str, claim_token: str = "") -> None:
        now = datetime.now(UTC)
        async with await self._connect() as db:
            cur = await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET status = CASE
                        WHEN status = 'finalizing' THEN 'expired'
                        WHEN expires_at > %s THEN 'active'
                        ELSE 'expired'
                    END,
                    claim_token = NULL,
                    claimed_at = NULL,
                    updated_at = %s
                WHERE dialog_key = %s AND draft_id = %s AND status IN ('confirming', 'finalizing')
                  AND (%s = '' OR claim_token = %s)
                RETURNING params_json, user_id, status
                """,
                (now, now, dialog_key, draft_id, claim_token, claim_token),
            )
            released = await cur.fetchone()
            if released:
                await db.execute(
                    """
                    INSERT INTO bitrix24.interactive_draft_transitions
                        (draft_id, dialog_key, transition, payload_json, actor_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (draft_id, dialog_key, released["status"], released["params_json"], released["user_id"]),
                )

    async def finalize_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        params: dict[str, Any],
        claim_token: str = "",
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT draft_type, specialist_id, user_id, version, created_at, expires_at
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND draft_id = %s AND status = 'confirming'
                  AND (%s = '' OR claim_token = %s)
                FOR UPDATE
                """,
                (dialog_key, draft_id, claim_token, claim_token),
            )
            claimed = await cur.fetchone()
            if not claimed:
                return None
            version = int(claimed["version"] or 1) + 1
            status = "active" if claimed["expires_at"] > now else "expired"
            stored = {
                **params,
                "_draft_id": draft_id,
                "_draft_type": str(claimed["draft_type"]),
                "_draft_specialist": str(claimed["specialist_id"]),
                "_draft_user_id": claimed["user_id"],
                "_draft_created_at": claimed["created_at"].isoformat(),
                "_draft_expires_at": claimed["expires_at"].isoformat(),
                "_draft_version": version,
            }
            payload_json = json.dumps(stored, ensure_ascii=False)
            await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET params_json = %s, status = %s, version = %s,
                    claim_token = NULL, claimed_at = NULL, updated_at = %s
                WHERE draft_id = %s AND status = 'confirming'
                """,
                (payload_json, status, version, now, draft_id),
            )
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    draft_id,
                    dialog_key,
                    "processed" if status == "active" else "expired",
                    payload_json,
                    claimed["user_id"],
                ),
            )
            if status == "active":
                await db.execute(
                    """
                    INSERT INTO bitrix24.pending_task_draft (dialog_key, params_json)
                    VALUES (%s, %s)
                    ON CONFLICT (dialog_key) DO UPDATE
                    SET params_json = EXCLUDED.params_json, created_at = now()
                    """,
                    (dialog_key, payload_json),
                )
            else:
                await db.execute(
                    "DELETE FROM bitrix24.pending_task_draft WHERE dialog_key = %s",
                    (dialog_key,),
                )
        return stored

    async def get_task_draft_for_finalizer(self, dialog_key: str) -> dict[str, Any] | None:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT params_json, claim_token
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND status IN ('active', 'expired', 'finalizing')
                ORDER BY created_at
                LIMIT 1
                """,
                (dialog_key,),
            )
            row = await cur.fetchone()
        if not row:
            return None
        raw = row["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_claim_token"] = str(row["claim_token"] or "")
        return result

    async def get_claimed_task_draft(
        self,
        dialog_key: str,
        *,
        expected_type: str,
    ) -> dict[str, Any] | None:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT params_json, claim_token
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND draft_type = %s AND status = 'confirming'
                LIMIT 1
                """,
                (dialog_key, expected_type),
            )
            row = await cur.fetchone()
        if not row:
            return None
        raw = row["params_json"]
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
        result["_draft_claim_token"] = str(row["claim_token"] or "")
        return result

    async def get_task_draft(self, dialog_key: str, *, ttl_minutes: int | None = None) -> dict[str, Any] | None:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT draft_id, params_json, status, created_at, expires_at, user_id
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND status IN ('active', 'confirming', 'finalizing')
                FOR UPDATE
                """,
                (dialog_key,),
            )
            active = await cur.fetchone()
            if active:
                if active["status"] in {"confirming", "finalizing"}:
                    return None
                now = datetime.now(UTC)
                effective_expiry = active["expires_at"]
                if ttl_minutes is not None and ttl_minutes > 0:
                    effective_expiry = min(effective_expiry, active["created_at"] + timedelta(minutes=ttl_minutes))
                if effective_expiry <= now:
                    await db.execute(
                        """
                        UPDATE bitrix24.interactive_drafts
                        SET status = 'expired', updated_at = %s
                        WHERE draft_id = %s AND status IN ('active', 'confirming')
                        """,
                        (now, active["draft_id"]),
                    )
                    await db.execute(
                        """
                        INSERT INTO bitrix24.interactive_draft_transitions
                            (draft_id, dialog_key, transition, payload_json, actor_user_id)
                        VALUES (%s, %s, 'expired', %s, %s)
                        """,
                        (
                            active["draft_id"],
                            dialog_key,
                            active["params_json"],
                            active["user_id"],
                        ),
                    )
                    await db.execute(
                        "DELETE FROM bitrix24.pending_task_draft WHERE dialog_key = %s",
                        (dialog_key,),
                    )
                    return None
                raw = active["params_json"]
                return json.loads(raw) if isinstance(raw, str) else raw
            if ttl_minutes is not None and ttl_minutes > 0:
                await db.execute(
                    """
                    DELETE FROM bitrix24.pending_task_draft
                    WHERE dialog_key = %s
                      AND created_at < now() - make_interval(mins => %s)
                    """,
                    (dialog_key, ttl_minutes),
                )
            cur = await db.execute(
                "SELECT params_json, created_at FROM bitrix24.pending_task_draft WHERE dialog_key = %s",
                (dialog_key,),
            )
            row = await cur.fetchone()
        if not row:
            return None
        raw = row["params_json"]
        legacy = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(legacy, dict):
            return None
        legacy_created_at = row["created_at"]
        legacy_ttl_minutes = ttl_minutes if ttl_minutes is not None and ttl_minutes > 0 else 15
        if legacy_created_at + timedelta(minutes=legacy_ttl_minutes) <= datetime.now(UTC):
            async with await self._connect() as db:
                await db.execute(
                    "DELETE FROM bitrix24.pending_task_draft WHERE dialog_key = %s AND created_at = %s",
                    (dialog_key, legacy_created_at),
                )
            return None
        promoted = dict(legacy)
        promoted["_promoted_legacy_created_at"] = legacy_created_at.isoformat()
        await self.save_task_draft(dialog_key, promoted)
        return await self.get_task_draft(dialog_key, ttl_minutes=ttl_minutes)

    def list_task_drafts(
        self,
        *,
        draft_type: str = "",
        limit: int = 100,
        expired_only: bool = False,
    ) -> list[dict[str, Any]]:
        expiry_operator = "<=" if expired_only else ">"
        status_predicate = "status IN ('active', 'expired', 'finalizing')" if expired_only else "status = 'active'"
        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT dialog_key, params_json, status, created_at, updated_at
                FROM bitrix24.interactive_drafts
                WHERE {status_predicate} AND expires_at {expiry_operator} now()
                ORDER BY created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            raw = row["params_json"]
            params = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(params, dict):
                continue
            if draft_type and str(params.get("_draft_type") or "") != draft_type:
                continue
            result.append(
                {
                    "dialog_key": row["dialog_key"],
                    "params": params,
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return result

    async def delete_task_draft(
        self,
        dialog_key: str,
        *,
        status: str = "cancelled",
        expected_draft_id: str = "",
        expected_version: int | None = None,
        expected_claim_token: str = "",
    ) -> None:
        if status not in {"cancelled", "confirmed", "finalized"}:
            raise ValueError("draft terminal status must be cancelled, confirmed, or finalized")
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT draft_id, params_json, user_id
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s
                  AND status IN ('active', 'confirming', 'expired', 'finalizing')
                  AND (%s = '' OR draft_id = %s)
                  AND (%s IS NULL OR version = %s)
                  AND (%s = '' OR claim_token = %s)
                FOR UPDATE
                """,
                (
                    dialog_key,
                    expected_draft_id,
                    expected_draft_id,
                    expected_version,
                    expected_version,
                    expected_claim_token,
                    expected_claim_token,
                ),
            )
            active = await cur.fetchone()
            if active:
                await db.execute(
                    """
                    UPDATE bitrix24.interactive_drafts
                    SET status = %s, claim_token = NULL, claimed_at = NULL, updated_at = now()
                    WHERE draft_id = %s AND status IN ('active', 'confirming', 'expired', 'finalizing')
                    """,
                    (status, active["draft_id"]),
                )
                await db.execute(
                    """
                    INSERT INTO bitrix24.interactive_draft_transitions
                        (draft_id, dialog_key, transition, payload_json, actor_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (active["draft_id"], dialog_key, status, active["params_json"], active["user_id"]),
                )
            await db.execute(
                """
                DELETE FROM bitrix24.pending_task_draft
                WHERE dialog_key = %s
                  AND (%s = '' OR params_json::jsonb ->> '_draft_id' = %s)
                """,
                (dialog_key, expected_draft_id, expected_draft_id),
            )

    # ------------------------------------------------------------------
    # Bitrix task close control state
    # ------------------------------------------------------------------

    def get_task_close_control_setting(self, key: str) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT key, value, updated_by, updated_at
                FROM bitrix24.task_close_control_settings
                WHERE key = %s
                """,
                (key,),
            ).fetchone()
        return dict(row) if row else None

    def set_task_close_control_setting(self, *, key: str, value: str, updated_by: int | None = None) -> None:
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO bitrix24.task_close_control_settings (key, value, updated_by, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = now()
                """,
                (key, value, updated_by),
            )
            self._record_task_close_control_revision(
                db,
                actor_user_id=updated_by,
                action="set_setting",
                payload={"key": key, "value": value},
            )

    def task_close_operator_ids(self) -> set[int]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT user_id
                FROM bitrix24.task_close_control_operators
                WHERE active IS TRUE
                ORDER BY user_id
                """
            ).fetchall()
        return {int(row["user_id"]) for row in rows if row.get("user_id") is not None}

    def set_task_close_operators(self, *, operator_user_ids: list[int], actor_user_id: int | None) -> list[int]:
        cleaned = sorted({int(user_id) for user_id in operator_user_ids if int(user_id) > 0})
        with self._sync_connect() as db:
            db.execute(
                """
                UPDATE bitrix24.task_close_control_operators
                SET active = FALSE, updated_by = %s, updated_at = now()
                """,
                (actor_user_id,),
            )
            for user_id in cleaned:
                db.execute(
                    """
                    INSERT INTO bitrix24.task_close_control_operators (user_id, active, updated_by, updated_at)
                    VALUES (%s, TRUE, %s, now())
                    ON CONFLICT (user_id) DO UPDATE SET
                        active = TRUE,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = now()
                    """,
                    (user_id, actor_user_id),
                )
            self._record_task_close_control_revision(
                db,
                actor_user_id=actor_user_id,
                action="set_operators",
                payload={"operator_user_ids": cleaned},
            )
        return cleaned

    def upsert_task_close_operator(
        self,
        *,
        user_id: int,
        active: bool = True,
        updated_by: int | None = None,
    ) -> None:
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO bitrix24.task_close_control_operators (user_id, active, updated_by, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (user_id) DO UPDATE SET
                    active = EXCLUDED.active,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = now()
                """,
                (user_id, active, updated_by),
            )
            self._record_task_close_control_revision(
                db,
                actor_user_id=updated_by,
                action="set_operator",
                payload={"user_id": user_id, "active": active},
            )

    def get_task_close_controlled_user(self, user_id: int) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT user_id, active, controlled_from, updated_by, updated_at
                FROM bitrix24.task_close_controlled_users
                WHERE user_id = %s
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def task_close_controlled_user_ids(self) -> set[int]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT user_id
                FROM bitrix24.task_close_controlled_users
                WHERE active IS TRUE
                ORDER BY user_id
                """
            ).fetchall()
        return {int(row["user_id"]) for row in rows if row.get("user_id") is not None}

    def upsert_task_close_controlled_user(
        self,
        *,
        user_id: int,
        active: bool = True,
        updated_by: int | None = None,
    ) -> None:
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO bitrix24.task_close_controlled_users
                    (user_id, active, controlled_from, updated_by, updated_at)
                VALUES (%s, %s, CASE WHEN %s THEN now() ELSE NULL END, %s, now())
                ON CONFLICT (user_id) DO UPDATE SET
                    active = EXCLUDED.active,
                    controlled_from = CASE
                        WHEN EXCLUDED.active AND (
                            bitrix24.task_close_controlled_users.controlled_from IS NULL
                            OR bitrix24.task_close_controlled_users.active IS FALSE
                        )
                        THEN now()
                        ELSE bitrix24.task_close_controlled_users.controlled_from
                    END,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = now()
                """,
                (user_id, active, active, updated_by),
            )
            self._record_task_close_control_revision(
                db,
                actor_user_id=updated_by,
                action="set_controlled_user",
                payload={"user_id": user_id, "active": active},
            )

    def get_task_close_processing_state(self, *, task_id: object, state_key: str) -> dict[str, Any] | None:
        task_id_int = safe_int(task_id)
        if task_id_int is None:
            return None
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT task_id, state_key, status, payload_json, actor_user_id, created_at, updated_at
                FROM bitrix24.task_close_processing_state
                WHERE task_id = %s AND state_key = %s
                """,
                (task_id_int, state_key),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = safe_json(result.pop("payload_json"))
        return result

    async def confirm_admin_change_draft(
        self,
        *,
        dialog_key: str,
        draft_id: str,
        draft_version: int,
        actor_user_id: int,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT params_json
                FROM bitrix24.interactive_drafts
                WHERE dialog_key = %s AND draft_id = %s AND version = %s
                  AND draft_type = 'admin_change' AND status = 'active' AND expires_at > %s
                FOR UPDATE
                """,
                (dialog_key, draft_id, draft_version, now),
            )
            row = await cur.fetchone()
            if not row:
                return {"status": "not_found"}
            raw = row["params_json"]
            draft = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(draft, dict):
                return {"status": "invalid"}
            field = str(draft.get("field") or "")
            target_user_id = safe_int(draft.get("target_user_id"))
            lock_key = f"task_close_control:{field}:{target_user_id or 0}"
            await db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))

            current: object
            if field in {"operator", "controlled_user"} and target_user_id is not None:
                table = (
                    "bitrix24.task_close_control_operators"
                    if field == "operator"
                    else "bitrix24.task_close_controlled_users"
                )
                cur = await db.execute(f"SELECT active FROM {table} WHERE user_id = %s FOR UPDATE", (target_user_id,))
                member = await cur.fetchone()
                current = bool(member and member["active"])
            elif field in {"auto_close_time", "control_enabled_from"}:
                cur = await db.execute(
                    "SELECT value FROM bitrix24.task_close_control_settings WHERE key = %s FOR UPDATE",
                    (field,),
                )
                setting = await cur.fetchone()
                current = str(setting["value"]) if setting else ("20:00" if field == "auto_close_time" else "")
            else:
                return {"status": "invalid"}
            if current != draft.get("old_value"):
                return {"status": "conflict", "current_value": current, "draft": draft}

            new_value = draft.get("new_value")
            if field == "operator" and target_user_id is not None:
                await db.execute(
                    """
                    INSERT INTO bitrix24.task_close_control_operators (user_id, active, updated_by, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        active = EXCLUDED.active, updated_by = EXCLUDED.updated_by, updated_at = EXCLUDED.updated_at
                    """,
                    (target_user_id, bool(new_value), actor_user_id, now),
                )
            elif field == "controlled_user" and target_user_id is not None:
                await db.execute(
                    """
                    INSERT INTO bitrix24.task_close_controlled_users
                        (user_id, active, controlled_from, updated_by, updated_at)
                    VALUES (%s, %s, CASE WHEN %s THEN %s ELSE NULL END, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        active = EXCLUDED.active,
                        controlled_from = CASE
                            WHEN EXCLUDED.active AND (
                                bitrix24.task_close_controlled_users.active IS FALSE
                                OR bitrix24.task_close_controlled_users.controlled_from IS NULL
                            ) THEN EXCLUDED.controlled_from
                            ELSE bitrix24.task_close_controlled_users.controlled_from
                        END,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (target_user_id, bool(new_value), bool(new_value), now, actor_user_id, now),
                )
            else:
                await db.execute(
                    """
                    INSERT INTO bitrix24.task_close_control_settings (key, value, updated_by, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value, updated_by = EXCLUDED.updated_by, updated_at = EXCLUDED.updated_at
                    """,
                    (field, str(new_value or ""), actor_user_id, now),
                )
            revision_payload = {
                "draft_id": draft_id,
                "field": field,
                "target_user_id": target_user_id,
                "old_value": current,
                "new_value": new_value,
            }
            await db.execute(
                """
                INSERT INTO bitrix24.task_close_control_revisions (actor_user_id, action, payload_json)
                VALUES (%s, %s, %s)
                """,
                (actor_user_id, "confirm_admin_change", json.dumps(revision_payload, ensure_ascii=False)),
            )
            await db.execute(
                """
                UPDATE bitrix24.interactive_drafts
                SET status = 'confirmed', updated_at = %s
                WHERE draft_id = %s AND version = %s AND status = 'active'
                """,
                (now, draft_id, draft_version),
            )
            await db.execute(
                """
                INSERT INTO bitrix24.interactive_draft_transitions
                    (draft_id, dialog_key, transition, payload_json, actor_user_id)
                VALUES (%s, %s, 'confirmed', %s, %s)
                """,
                (draft_id, dialog_key, json.dumps(draft, ensure_ascii=False), actor_user_id),
            )
            await db.execute(
                """
                DELETE FROM bitrix24.pending_task_draft
                WHERE dialog_key = %s AND params_json::jsonb ->> '_draft_id' = %s
                """,
                (dialog_key, draft_id),
            )
        return {"status": "confirmed", "draft": draft}

    def list_task_close_processing_states(
        self,
        *,
        statuses: list[str] | None = None,
        state_key_prefix: str = "",
        responsible_id: int | None = None,
        dialog_key: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        where: list[str] = ["TRUE"]
        params: list[Any] = []
        if statuses:
            where.append("status = ANY(%s)")
            params.append(list(statuses))
        if state_key_prefix:
            where.append("state_key LIKE %s ESCAPE '\\'")
            params.append(f"{escape_like(state_key_prefix)}%")
        if responsible_id is not None:
            where.append("(payload_json::jsonb ->> 'responsible_id') = %s")
            params.append(str(responsible_id))
        if dialog_key:
            where.append("(payload_json::jsonb ->> 'dialog_key') = %s")
            params.append(dialog_key)
        params.append(limit)

        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT task_id, state_key, status, payload_json, actor_user_id, created_at, updated_at
                FROM bitrix24.task_close_processing_state
                WHERE {" AND ".join(where)}
                ORDER BY created_at ASC, task_id ASC, state_key ASC
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = safe_json(item.pop("payload_json"))
            result.append(item)
        return result

    def upsert_task_close_processing_state(
        self,
        *,
        task_id: object,
        state_key: str,
        status: str,
        payload: dict[str, Any] | None = None,
        actor_user_id: int | None = None,
    ) -> None:
        task_id_int = safe_int(task_id)
        if task_id_int is None or not state_key:
            return
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO bitrix24.task_close_processing_state
                    (task_id, state_key, status, payload_json, actor_user_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (task_id, state_key) DO UPDATE SET
                    status = EXCLUDED.status,
                    payload_json = EXCLUDED.payload_json,
                    actor_user_id = EXCLUDED.actor_user_id,
                    updated_at = now()
                """,
                (task_id_int, state_key, status, json.dumps(payload or {}, ensure_ascii=False), actor_user_id),
            )

    def get_task_close_control_event(self, *, task_id: object, close_event_key: str) -> dict[str, Any] | None:
        task_id_int = safe_int(task_id)
        if task_id_int is None or not close_event_key:
            return None
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT task_id, close_event_key, responsible_id, closed_by_user_id, closed_at,
                       decision, reason, payload_json, seen_at, updated_at
                FROM bitrix24.task_close_control_events
                WHERE task_id = %s AND close_event_key = %s
                """,
                (task_id_int, close_event_key),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = safe_json(result.pop("payload_json"))
        return result

    def upsert_task_close_control_event(
        self,
        *,
        task_id: object,
        close_event_key: str,
        decision: str,
        reason: str = "",
        closed_at: str | None = None,
        responsible_id: int | None = None,
        closed_by_user_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        task_id_int = safe_int(task_id)
        if task_id_int is None or not close_event_key or not decision:
            return
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO bitrix24.task_close_control_events
                    (task_id, close_event_key, responsible_id, closed_by_user_id, closed_at,
                     decision, reason, payload_json, updated_at)
                VALUES (%s, %s, %s, %s, %s::timestamptz, %s, %s, %s, now())
                ON CONFLICT (task_id, close_event_key) DO UPDATE SET
                    responsible_id = EXCLUDED.responsible_id,
                    closed_by_user_id = EXCLUDED.closed_by_user_id,
                    closed_at = EXCLUDED.closed_at,
                    decision = EXCLUDED.decision,
                    reason = EXCLUDED.reason,
                    payload_json = EXCLUDED.payload_json,
                    updated_at = now()
                """,
                (
                    task_id_int,
                    close_event_key,
                    responsible_id,
                    closed_by_user_id,
                    closed_at,
                    decision,
                    reason,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )

    def _record_task_close_control_revision(
        self,
        db: Any,
        *,
        actor_user_id: int | None,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        db.execute(
            """
            INSERT INTO bitrix24.task_close_control_revisions
                (actor_user_id, action, payload_json, created_at)
            VALUES (%s, %s, %s, now())
            """,
            (actor_user_id, action, json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)),
        )

    # ------------------------------------------------------------------
    # Portal search index
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        try:
            with self._sync_connect() as db:
                row = db.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'bitrix24' AND table_name = 'portal_search_items'",
                ).fetchone()
            return bool(row)
        except Exception:
            return False

    def upsert_item(
        self,
        *,
        entity_type: str,
        entity_id: object,
        title: str,
        body: str = "",
        url: str = "",
        metadata: dict[str, Any] | None = None,
        source_updated_at: str | None = None,
        preserve_content: bool = True,
    ) -> None:
        now = datetime.now(MOSCOW_TZ).isoformat()
        normalized_title = clean_text(title) or f"{entity_type} #{entity_id}"
        normalized_body = clean_text(body)
        normalized_metadata = dict(metadata or {})
        existing = self._get_existing_item(entity_type=entity_type, entity_id=entity_id)
        if (
            preserve_content
            and existing
            and _should_preserve_content(
                existing_metadata=existing["metadata"],
                new_metadata=normalized_metadata,
                existing_source_updated_at=existing["source_updated_at"],
                new_source_updated_at=source_updated_at,
            )
        ):
            normalized_metadata = _merge_content_metadata(base=normalized_metadata, existing=existing["metadata"])
            existing_content = content_text_from_body(existing["body"])
            if existing_content:
                normalized_body = body_with_content(normalized_body, existing_content)

        search_text = normalize_search_text(" ".join([entity_type, str(entity_id), normalized_title, normalized_body]))
        with self._sync_connect() as db:
            db.execute(
                f"""
                INSERT INTO {_TABLE} (
                    entity_type, entity_id, title, body, url,
                    search_text, metadata_json, source_updated_at, last_seen_at, indexed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entity_type, entity_id) DO UPDATE SET
                    title             = EXCLUDED.title,
                    body              = EXCLUDED.body,
                    url               = EXCLUDED.url,
                    search_text       = EXCLUDED.search_text,
                    metadata_json     = EXCLUDED.metadata_json,
                    source_updated_at = EXCLUDED.source_updated_at,
                    last_seen_at      = EXCLUDED.last_seen_at,
                    indexed_at        = EXCLUDED.indexed_at
                """,
                (
                    entity_type,
                    str(entity_id),
                    normalized_title,
                    normalized_body,
                    url,
                    search_text,
                    json.dumps(normalized_metadata, ensure_ascii=False),
                    source_updated_at,
                    now,
                    now,
                ),
            )

    def delete_item(self, *, entity_type: str, entity_id: object) -> bool:
        with self._sync_connect() as db:
            cur = db.execute(
                f"DELETE FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s",
                (entity_type, str(entity_id)),
            )
        return bool(cur.rowcount)

    def delete_stale_items(self, *, entity_types: set[str], seen_before: str) -> int:
        if not entity_types:
            return 0
        with self._sync_connect() as db:
            cur = db.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE entity_type = ANY(%s)
                  AND (last_seen_at IS NULL OR last_seen_at < %s)
                """,
                (list(sorted(entity_types)), seen_before),
            )
        return int(cur.rowcount or 0)

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[PortalSearchResult]:
        term_groups = query_term_groups(query)
        terms = flatten_unique(term_groups)
        if not term_groups:
            return []

        where: list[str] = []
        params: list[object] = []
        for group in term_groups:
            group_where = []
            for term in group:
                group_where.append("search_text LIKE %s ESCAPE '\\'")
                params.append(f"%{escape_like(term)}%")
            where.append("(" + " OR ".join(group_where) + ")")
        if entity_types:
            where.append("entity_type = ANY(%s)")
            params.append(list(sorted(entity_types)))

        query_sql = (
            f"SELECT entity_type, entity_id, title, body, url, metadata_json, search_text "
            f"FROM {_TABLE} WHERE " + " AND ".join(where) + " LIMIT 500"
        )
        with self._sync_connect() as db:
            rows = db.execute(query_sql, params).fetchall()

        normalized_query = normalize_search_text(query)
        scored: list[PortalSearchResult] = []
        for row in rows:
            score = _score_result(
                normalized_query,
                terms,
                title=normalize_search_text(row["title"]),
                body=normalize_search_text(row["body"]),
                search_text=row["search_text"],
            )
            scored.append(
                PortalSearchResult(
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    title=row["title"],
                    body=row["body"],
                    url=row["url"],
                    score=score,
                    metadata=safe_json(row["metadata_json"]),
                )
            )
        return sorted(
            scored,
            key=lambda item: (-item.score, item.entity_type, item.title.casefold(), item.entity_id),
        )[:limit]

    def disk_delta_folder_candidates(
        self,
        *,
        cursor_type: str | None,
        cursor_id: str | None,
        limit: int,
    ) -> tuple[list[PortalSearchResult], str | None, str | None, bool]:
        normalized_type = cursor_type or ""
        normalized_id = safe_int(cursor_id) or 0
        rows = self._select_disk_delta_folder_rows(cursor_type=normalized_type, cursor_id=normalized_id, limit=limit)
        wrapped = False
        if not rows and (normalized_type or normalized_id):
            rows = self._select_disk_delta_folder_rows(cursor_type="", cursor_id=0, limit=limit)
            wrapped = bool(rows)
        candidates = [_row_to_search_result(row) for row in rows]
        next_type: str | None = None
        next_id: str | None = None
        if candidates:
            last = candidates[-1]
            next_type = last.entity_type
            next_id = last.entity_id
        return candidates, next_type, next_id, wrapped

    def children_by_parent_id(self, parent_id: object) -> list[PortalSearchResult]:
        parent = str(parent_id)
        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_file', 'disk_folder')
                  AND (
                    metadata_json LIKE %s ESCAPE '\\'
                    OR metadata_json LIKE %s ESCAPE '\\'
                  )
                """,
                (
                    f'%"parent_id": "{escape_like(parent)}"%',
                    f'%"parent_id": {parent}%',
                ),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_candidates(self, *, limit: int) -> list[PortalSearchResult]:
        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_file', 'task_attachment')
                  AND NOT COALESCE((
                    (
                      metadata_json::jsonb ->> 'content_index_status' = 'indexed'
                      AND metadata_json::jsonb ->> 'content_index_version' = %s
                    )
                    OR (
                      metadata_json::jsonb ->> 'content_index_version' = %s
                      AND metadata_json::jsonb ->> 'content_index_status' = ANY(%s)
                    )
                  ), FALSE)
                ORDER BY indexed_at DESC
                LIMIT %s
                """,
                (
                    CONTENT_INDEX_VERSION,
                    CONTENT_INDEX_VERSION,
                    sorted({*CONTENT_TERMINAL_STATUSES, "unsupported"}),
                    limit,
                ),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_readiness(self, *, allowed_extensions: set[str]) -> PortalContentReadiness:
        normalized_allowed = normalize_extensions(allowed_extensions)
        with self._sync_connect() as db:
            rows = db.execute(
                f"SELECT title, metadata_json FROM {_TABLE} WHERE entity_type IN ('disk_file', 'task_attachment')"
            ).fetchall()

        indexed_by_extension: dict[str, int] = {}
        pending_by_extension: dict[str, int] = {}
        pending_by_status: dict[str, int] = {}
        terminal_by_status: dict[str, int] = {}
        unsupported_by_extension: dict[str, int] = {}
        total_documents = supported_documents = indexed = pending = terminal = unsupported = 0

        for row in rows:
            total_documents += 1
            ext = file_extension(str(row["title"] or "")) or "<none>"
            metadata = safe_json(row["metadata_json"])
            status = str(metadata.get("content_index_status") or "none")
            content_version = str(metadata.get("content_index_version") or "")

            if ext not in normalized_allowed:
                unsupported += 1
                increment(unsupported_by_extension, ext)
                continue

            supported_documents += 1
            if status == "indexed":
                indexed += 1
                increment(indexed_by_extension, ext)
            elif content_version == CONTENT_INDEX_VERSION and status in CONTENT_TERMINAL_STATUSES:
                terminal += 1
                increment(terminal_by_status, status)
            else:
                pending += 1
                increment(pending_by_extension, ext)
                increment(pending_by_status, status)

        return PortalContentReadiness(
            total_documents=total_documents,
            supported_documents=supported_documents,
            indexed=indexed,
            pending=pending,
            terminal=terminal,
            unsupported=unsupported,
            indexed_by_extension=indexed_by_extension,
            pending_by_extension=pending_by_extension,
            pending_by_status=pending_by_status,
            terminal_by_status=terminal_by_status,
            unsupported_by_extension=unsupported_by_extension,
        )

    def update_item_body_metadata(
        self,
        *,
        entity_type: str,
        entity_id: object,
        body: str,
        metadata: dict[str, Any],
    ) -> None:
        now = datetime.now(MOSCOW_TZ).isoformat()
        normalized_body = clean_text(body)
        with self._sync_connect() as db:
            row = db.execute(
                f"SELECT title FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s",
                (entity_type, str(entity_id)),
            ).fetchone()
            if not row:
                return
            search_text = normalize_search_text(" ".join([entity_type, str(entity_id), row["title"], normalized_body]))
            db.execute(
                f"""
                UPDATE {_TABLE}
                SET body = %s, search_text = %s, metadata_json = %s, indexed_at = %s
                WHERE entity_type = %s AND entity_id = %s
                """,
                (
                    normalized_body,
                    search_text,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    entity_type,
                    str(entity_id),
                ),
            )

    def stats(self) -> PortalIndexStats:
        with self._sync_connect() as db:
            total = db.execute(f"SELECT COUNT(*) AS n FROM {_TABLE}").fetchone()["n"]
            by_type_rows = db.execute(
                f"SELECT entity_type, COUNT(*) AS count FROM {_TABLE} GROUP BY entity_type"
            ).fetchall()
            last_row = db.execute(f"SELECT MAX(indexed_at) AS t FROM {_TABLE}").fetchone()
            content_rows = db.execute(
                f"SELECT metadata_json FROM {_TABLE} WHERE entity_type IN ('disk_file', 'task_attachment')"
            ).fetchall()

        content_by_status: dict[str, int] = {}
        for row in content_rows:
            status = safe_json(row["metadata_json"]).get("content_index_status")
            if status:
                k = str(status)
                content_by_status[k] = content_by_status.get(k, 0) + 1

        return PortalIndexStats(
            total_items=int(total),
            by_type={str(r["entity_type"]): int(r["count"]) for r in by_type_rows},
            content_by_status=content_by_status,
            last_indexed_at=str(last_row["t"]) if last_row and last_row["t"] else None,
            path=Path("(postgresql)"),
            exists=True,
        )

    def get_item(self, *, entity_type: str, entity_id: object) -> PortalSearchResult | None:
        with self._sync_connect() as db:
            row = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        return _row_to_search_result(row) if row else None

    def item_snapshot(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        return self._get_existing_item(entity_type=entity_type, entity_id=entity_id)

    def _select_disk_delta_folder_rows(self, *, cursor_type: str, cursor_id: int, limit: int) -> list[dict]:
        with self._sync_connect() as db:
            return db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_storage', 'disk_folder')
                  AND (
                    %s = ''
                    OR entity_type > %s
                    OR (entity_type = %s AND CAST(entity_id AS INTEGER) > %s)
                  )
                ORDER BY entity_type, CAST(entity_id AS INTEGER)
                LIMIT %s
                """,
                (cursor_type, cursor_type, cursor_type, cursor_id, limit),
            ).fetchall()

    def _get_existing_item(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                f"""
                SELECT body, metadata_json, source_updated_at FROM {_TABLE}
                WHERE entity_type = %s AND entity_id = %s
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        if not row:
            return None
        return {
            "body": str(row["body"] or ""),
            "metadata": safe_json(row["metadata_json"]),
            "source_updated_at": row["source_updated_at"],
        }


def _row_to_search_result(row: dict) -> PortalSearchResult:
    return PortalSearchResult(
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        score=0,
        metadata=safe_json(row["metadata_json"]),
    )


def _parse_draft_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
