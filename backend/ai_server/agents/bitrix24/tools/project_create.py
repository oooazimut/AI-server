from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixWritePort
from ai_server.utils import compact_text, optional_int, unique

PROJECT_CREATE_DRAFT_TYPE = "project_create"


@dataclass(frozen=True)
class BitrixProjectCreateDraft:
    method: str = "sonet_group.create"
    params: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, str] = field(default_factory=dict)
    contract_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        params = self.params.get("params") if isinstance(self.params.get("params"), dict) else {}
        return not self.contract_errors and bool(params.get("fields"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "preview": self.preview,
            "contract_errors": self.contract_errors,
            "notes": self.notes,
        }


def build_project_create_draft_from_args(
    args: dict[str, Any],
    *,
    user_id: int | None = None,
    actor_name: str = "",
    actor_is_admin: bool = False,
) -> BitrixProjectCreateDraft:
    args = args or {}
    name = compact_text(str(args.get("name") or args.get("title") or "")).strip(" .,:;-")
    description = compact_text(str(args.get("description") or ""))
    personal_for_self = _truthy(args.get("personal_for_self"))
    owner_id = optional_int(args.get("owner_id")) or user_id
    subject_id = optional_int(args.get("subject_id")) or 1
    opened = not _falsey(args.get("opened"))
    visible = not _falsey(args.get("visible"))
    project = not _falsey(args.get("project"))

    errors: list[str] = []
    notes: list[str] = []

    actor_label = compact_text(actor_name)
    if not name:
        errors.append("project_create_draft.name is required")
    if user_id is None:
        errors.append("project_create_draft requires current Bitrix user_id")
    if not actor_is_admin:
        if not personal_for_self:
            errors.append("regular users may prepare only their own personal project")
        elif not _same_personal_project_name(name, actor_label):
            errors.append("personal project name must match the current user's Bitrix surname and name")
    if subject_id is None:
        errors.append("project_create_draft.subject_id is required")
    if owner_id is None:
        errors.append("project_create_draft.owner_id is required")

    fields: dict[str, Any] = {}
    if name:
        fields["NAME"] = name
        fields["DESCRIPTION"] = description or f"Личный проект {name}" if personal_for_self else description
    if subject_id is not None:
        fields["SUBJECT_ID"] = subject_id
    if owner_id is not None:
        fields["OWNER_ID"] = owner_id
    fields["OPENED"] = "Y" if opened else "N"
    fields["VISIBLE"] = "Y" if visible else "N"
    fields["PROJECT"] = "Y" if project else "N"
    fields["INITIATE_PERMS"] = "K"
    fields["SPAM_PERMS"] = "K"

    if personal_for_self:
        notes.append("Личный проект: имя должно совпадать с фамилией и именем текущего пользователя Bitrix.")
    if not actor_is_admin:
        notes.append("Обычный пользователь не может создавать произвольные проекты.")

    payload = {
        "_draft_type": PROJECT_CREATE_DRAFT_TYPE,
        "method": "sonet_group.create",
        "params": {"fields": fields} if fields else {},
        "personal_for_self": personal_for_self,
        "actor_user_id": user_id,
        "actor_name": actor_label,
    }
    return BitrixProjectCreateDraft(
        params=payload,
        preview=_draft_preview(fields, personal_for_self=personal_for_self),
        contract_errors=unique(errors),
        notes=notes,
    )


class ProjectCreateDraftTool:
    name = "project_create_draft"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Prepare a Bitrix project/workgroup creation draft. Use only after project search returned no match. "
                "Regular users may draft only their own personal project; arbitrary projects are admin-only. "
                "The user must confirm before sonet_group.create."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "personal_for_self": {"type": "boolean"},
                    "opened": {"type": "boolean"},
                    "visible": {"type": "boolean"},
                    "project": {"type": "boolean"},
                    "subject_id": {"type": "integer"},
                    "owner_id": {"type": "integer"},
                },
                "required": ["name"],
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        context_error = _write_context_error(user_id=user_id, dialog_id=dialog_id, action="project draft")
        if context_error:
            return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error)
        draft = build_project_create_draft_from_args(
            args,
            user_id=user_id,
            actor_name=str(args.get("_actor_name") or ""),
            actor_is_admin=_truthy(args.get("_actor_is_admin")),
        )
        if not draft.is_ready:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error=f"project_create_draft called outside contract: {'; '.join(draft.contract_errors)}",
                data=draft.as_action_details(),
            )
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="project_create_draft requires dialog_key",
                data=draft.as_action_details(),
            )
        await self._store.save_task_draft(dialog_key, draft.params)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"params": draft.params, "preview": draft.preview, "notes": draft.notes},
        )


class ProjectCreateConfirmTool:
    name = "project_create_confirm"

    def __init__(
        self,
        store: TaskDraftStorePort,
        write_client: BitrixWritePort | None = None,
        *,
        bitrix_oauth: BitrixOAuthService | None = None,
        dry_run: bool = False,
        oauth_required_for_writes: bool = True,
        draft_ttl_minutes: int | None = None,
    ) -> None:
        self._store = store
        self._write_client = write_client
        self._bitrix_oauth = bitrix_oauth
        self._dry_run = dry_run
        self._oauth_required_for_writes = oauth_required_for_writes
        self._draft_ttl_minutes = draft_ttl_minutes

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Confirm and create the pending Bitrix project/workgroup draft as the current OAuth user.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="project_create_confirm requires dialog_key"
            )
        draft = await self._store.get_task_draft(dialog_key, ttl_minutes=self._draft_ttl_minutes)
        if not draft or draft.get("_draft_type") != PROJECT_CREATE_DRAFT_TYPE:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="no pending project creation draft found for this dialog",
            )
        params = draft.get("params") if isinstance(draft.get("params"), dict) else {}
        if not params.get("fields"):
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error="project creation draft has no fields",
                data={"draft": draft},
            )
        sanitized = apply_write_policy("sonet_group.create", params)
        if self._dry_run:
            return ToolResult(status=ToolStatus.DRY_RUN, tool=self.name, data={"params": sanitized, "draft": draft})

        if self._oauth_required_for_writes:
            context_error = _write_context_error(user_id=user_id, dialog_id=dialog_id, action="project creation")
            if context_error:
                return ToolResult(
                    status=ToolStatus.DENIED, tool=self.name, error=context_error, data={"params": sanitized}
                )
            if self._bitrix_oauth is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="Bitrix OAuth is required for project creation.",
                    data={"params": sanitized},
                )
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                result = await oauth_client.call("sonet_group.create", sanitized)
            except BitrixOAuthTokenMissing as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=str(exc), data={"params": sanitized}
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool=self.name,
                    error=str(exc),
                    data={"params": sanitized},
                )
            except Exception as exc:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                    data={"params": sanitized},
                )
            followup_task_draft = await _replace_with_followup_task_draft(
                self._store,
                dialog_key=dialog_key,
                project_draft=draft,
                project_create_result=result,
            )
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data=_project_confirm_data(
                    result=result,
                    params=sanitized,
                    draft=draft,
                    followup_task_draft=followup_task_draft,
                ),
            )

        if self._write_client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="write_client is not configured")
        try:
            result = await self._write_client.call("sonet_group.create", sanitized)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
                data={"params": sanitized},
            )
        followup_task_draft = await _replace_with_followup_task_draft(
            self._store,
            dialog_key=dialog_key,
            project_draft=draft,
            project_create_result=result,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data=_project_confirm_data(
                result=result,
                params=sanitized,
                draft=draft,
                followup_task_draft=followup_task_draft,
            ),
        )


class ProjectCreateDiscardTool:
    name = "project_create_discard"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Discard the pending Bitrix project/workgroup creation draft.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        linked_task = False
        if dialog_key:
            draft = await self._store.get_task_draft(dialog_key)
            linked_task = isinstance(draft, dict) and isinstance(
                draft.get("after_project_create_task_draft"), dict
            )
            await self._store.delete_task_draft(dialog_key)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"discarded": bool(dialog_key), "linked_task": linked_task},
        )


def _draft_preview(fields: dict[str, Any], *, personal_for_self: bool) -> dict[str, str]:
    return {
        "name": compact_text(str(fields.get("NAME") or "проект")),
        "type": "личный проект" if personal_for_self else "проект",
        "visibility": "открытый" if fields.get("OPENED") == "Y" else "закрытый",
        "description": compact_text(str(fields.get("DESCRIPTION") or "")),
    }


async def _replace_with_followup_task_draft(
    store: TaskDraftStorePort,
    *,
    dialog_key: str,
    project_draft: dict[str, Any],
    project_create_result: object,
) -> dict[str, Any] | None:
    followup_task_draft = _followup_task_draft(project_draft, project_create_result)
    if followup_task_draft is None:
        await store.delete_task_draft(dialog_key)
        return None
    await store.save_task_draft(dialog_key, followup_task_draft["params"])
    return followup_task_draft


def _project_confirm_data(
    *,
    result: object,
    params: dict[str, Any],
    draft: dict[str, Any],
    followup_task_draft: dict[str, Any] | None,
) -> dict[str, Any]:
    data = {"result": result, "params": params, "draft": draft}
    if followup_task_draft is not None:
        data["followup_task_draft"] = followup_task_draft
    return data


def _followup_task_draft(project_draft: dict[str, Any], project_create_result: object) -> dict[str, Any] | None:
    followup = project_draft.get("after_project_create_task_draft")
    if not isinstance(followup, dict):
        return None
    params = followup.get("params") if isinstance(followup.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    if not fields:
        return None
    project_id = _created_project_id(project_create_result)
    if project_id is None:
        return None
    project_name = compact_text(str(followup.get("project_name") or ""))
    task_fields = dict(fields)
    task_fields["GROUP_ID"] = project_id
    preview = dict(followup.get("preview")) if isinstance(followup.get("preview"), dict) else {}
    if project_name:
        preview["project"] = project_name
    return {
        "params": {"fields": task_fields},
        "preview": preview,
        "project_id": project_id,
        "project_name": project_name,
    }


def _created_project_id(value: object) -> int | None:
    direct = optional_int(value)
    if direct is not None:
        return direct
    if not isinstance(value, dict):
        return None
    candidates: list[object] = [value.get("id"), value.get("ID"), value.get("group_id"), value.get("GROUP_ID")]
    nested = value.get("result")
    if isinstance(nested, dict):
        candidates.extend([nested.get("id"), nested.get("ID"), nested.get("group_id"), nested.get("GROUP_ID")])
    else:
        candidates.append(nested)
    for candidate in candidates:
        project_id = optional_int(candidate)
        if project_id is not None:
            return project_id
    return None


def _same_personal_project_name(name: str, actor_name: str) -> bool:
    if not name or not actor_name:
        return False
    candidate = _normalize_person_tokens(name)
    actor = _normalize_person_tokens(actor_name)
    if not candidate or not actor:
        return False
    if candidate == actor:
        return True
    if len(actor) >= 2 and candidate == actor[:2]:
        return True
    return len(actor) >= 2 and candidate == [actor[1], actor[0]]


def _normalize_person_tokens(value: str) -> list[str]:
    return compact_text(value).replace("ё", "е").casefold().split()


def _write_context_error(*, user_id: int | None, dialog_id: str | None, action: str) -> str:
    if user_id is None:
        return f"Bitrix {action} denied: current Bitrix user_id is missing."
    if not str(dialog_id or "").strip():
        return f"Bitrix {action} denied: current Bitrix dialog_id is missing."
    return ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on"}
    return bool(value)


def _falsey(value: object) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().casefold() in {"0", "false", "no", "n", "нет", "off"}
    return False


__all__ = [
    "PROJECT_CREATE_DRAFT_TYPE",
    "BitrixProjectCreateDraft",
    "build_project_create_draft_from_args",
    "ProjectCreateDraftTool",
    "ProjectCreateConfirmTool",
    "ProjectCreateDiscardTool",
]
