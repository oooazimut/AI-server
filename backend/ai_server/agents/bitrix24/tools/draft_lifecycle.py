from __future__ import annotations

from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.utils import optional_int


def bitrix_mutation_outcome(error: Exception) -> str:
    code = str(getattr(error, "error", "") or "").upper()
    return "unknown" if code.startswith("HTTP_5") else "rejected"


async def renew_exact_draft_claim(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    draft: dict[str, Any],
    expected_status: str = "confirming",
) -> bool:
    renew = getattr(store, "renew_task_draft_claim", None)
    if not callable(renew):
        return True
    return bool(
        await renew(
            dialog_key,
            draft_id=str(draft.get("_draft_id") or ""),
            claim_token=str(draft.get("_draft_claim_token") or ""),
            expected_status=expected_status,
        )
    )


async def resolve_unknown_draft_claim(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    expected_type: str,
) -> dict[str, Any] | None:
    getter = getattr(store, "get_claimed_task_draft", None)
    if not callable(getter):
        return None
    claimed = await getter(dialog_key, expected_type=expected_type)
    if not isinstance(claimed, dict):
        return None
    resolver = getattr(store, "resolve_stale_confirming_task_draft", None)
    resolved = None
    if callable(resolver):
        draft_id = str(claimed.get("_draft_id") or "")
        version = optional_int(claimed.get("_draft_version"))
        if draft_id and version is not None:
            resolved = await resolver(
                dialog_key,
                expected_draft_id=draft_id,
                expected_version=version,
                expected_type=expected_type,
            )
    if isinstance(resolved, dict):
        return resolved
    return {**claimed, "_draft_resolution_status": "confirming"}


def attach_draft_metadata(
    payload: dict[str, Any],
    *,
    source_args: dict[str, Any],
    user_id: int | None,
) -> dict[str, Any]:
    result = dict(payload)
    for key in (
        "_original_request",
        "_draft_specialist",
        "_direct_close_close_event_key",
        "_direct_close_closed_at",
        "_direct_close_already_closed",
    ):
        if source_args.get(key) not in (None, ""):
            result[key] = source_args[key]
    actor = source_args.get("_draft_user_id", user_id)
    if actor is not None:
        result["_draft_user_id"] = actor
    return result


async def claim_exact_draft(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    draft: dict[str, Any],
    expected_type: str,
) -> dict[str, Any] | None:
    draft_id = str(draft.get("_draft_id") or "")
    try:
        version = int(draft.get("_draft_version"))
    except (TypeError, ValueError):
        return None
    if not draft_id or version <= 0:
        return None
    return await store.claim_task_draft(
        dialog_key,
        expected_draft_id=draft_id,
        expected_version=version,
        expected_type=expected_type,
    )


async def release_exact_draft(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    draft: dict[str, Any],
) -> None:
    draft_id = str(draft.get("_draft_id") or "")
    if draft_id:
        await store.release_task_draft(
            dialog_key,
            draft_id=draft_id,
            claim_token=str(draft.get("_draft_claim_token") or ""),
        )


async def discard_exact_draft(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    draft: dict[str, Any],
    expected_type: str,
) -> bool:
    claimed = await claim_exact_draft(
        store,
        dialog_key=dialog_key,
        draft=draft,
        expected_type=expected_type,
    )
    if claimed is None:
        return False
    await store.delete_task_draft(
        dialog_key,
        status="cancelled",
        expected_draft_id=str(claimed.get("_draft_id") or ""),
        expected_version=optional_int(claimed.get("_draft_version")),
        expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
    )
    return True
