from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.agents.bitrix24.tools.bitrix_api import _normalize_project_name
from ai_server.agents.bitrix24.tools.draft_lifecycle import (
    attach_draft_metadata,
    bitrix_mutation_outcome,
    claim_exact_draft,
    discard_exact_draft,
    release_exact_draft,
    renew_exact_draft_claim,
    resolve_unknown_draft_claim,
)
from ai_server.agents.bitrix24.tools.project_create import build_project_create_draft_from_args
from ai_server.agents.bitrix24.tools.read_client import resolve_current_user_read_client
from ai_server.agents.bitrix24.tools.tasks import _search_projects_snapshot
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixWritePort
from ai_server.utils import MOSCOW_TZ, compact_text, optional_int, unique

DEFAULT_TASK_DEADLINE_WORKING_DAYS = 3
DEFAULT_TASK_DEADLINE_HOUR = 19
TASK_CREATE_DRAFT_TYPE = "task_create"


@dataclass(frozen=True)
class BitrixTaskCreateDraft:
    method: str = "tasks.task.add"
    params: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    preview: dict[str, str] = field(default_factory=dict)
    contract_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.contract_errors and bool(self.params.get("fields"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "summary": self.summary,
            "preview": self.preview,
            "contract_errors": self.contract_errors,
            "notes": self.notes,
        }


def build_task_create_draft_from_args(args: dict[str, Any], *, user_id: int | None = None) -> BitrixTaskCreateDraft:
    args = args or {}
    title = compact_text(str(args.get("title") or "")).strip(" .,:;-")
    description = compact_text(str(args.get("description") or ""))
    responsible_id = optional_int(args.get("responsible_id"))
    responsible_label = compact_text(str(args.get("responsible_name") or args.get("responsible_label") or ""))
    responsible_note = ""
    group_id = optional_int(args.get("group_id"))
    project_label = compact_text(str(args.get("project_name") or args.get("group_name") or ""))

    if responsible_id is None and _truthy(args.get("responsible_self")):
        responsible_id = user_id
        if responsible_id is not None:
            responsible_note = "Ответственным выбран пользователь Bitrix из текущего диалога."
        else:
            responsible_note = "LLM выбрала текущего пользователя, но канал не передал Bitrix user id."
    if not responsible_label and responsible_id is not None:
        responsible_label = "указанный сотрудник"

    deadline, deadline_note, no_deadline, deadline_error = _deadline_from_args(args)

    contract_errors: list[str] = []
    notes = [note for note in (responsible_note, deadline_note) if note]
    fields: dict[str, Any] = {}

    if title:
        fields["TITLE"] = title
        fields["DESCRIPTION"] = description or _default_description(title)
    else:
        contract_errors.append("task_create_draft.title is required")

    if responsible_id is not None:
        fields["RESPONSIBLE_ID"] = responsible_id
    else:
        if _truthy(args.get("responsible_self")):
            contract_errors.append("task_create_draft.responsible_self requires channel user id")
        else:
            contract_errors.append("task_create_draft requires one of responsible_id or responsible_self")

    if deadline_error:
        contract_errors.append(deadline_error)

    if user_id is not None:
        fields["CREATED_BY"] = user_id
    if group_id is not None:
        fields["GROUP_ID"] = group_id
    if deadline:
        fields["DEADLINE"] = deadline
    elif no_deadline:
        fields["NO_DEADLINE"] = True

    return BitrixTaskCreateDraft(
        params={"_draft_type": TASK_CREATE_DRAFT_TYPE, "fields": fields} if fields else {},
        summary=_summary(title=title, responsible_id=responsible_id, deadline=deadline, group_id=group_id),
        preview=_draft_preview(fields, responsible_label=responsible_label, project_label=project_label)
        if fields
        else {},
        contract_errors=unique(contract_errors),
        notes=notes,
    )


class TaskCreateDraftTool:
    name = "task_create_draft"

    def __init__(
        self,
        store: TaskDraftStorePort,
        project_client: Any | None = None,
        portal_search: Any | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._store = store
        self._project_client = project_client
        self._portal_search = portal_search
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_create_draft",
            description=(
                "Prepare a Bitrix task creation draft from fields already understood by the LLM. "
                "Call only after all required fields are collected. "
                "The draft is saved and will be shown back on the next turn for confirmation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "responsible_id": {"type": "integer"},
                    "responsible_self": {"type": "boolean"},
                    "responsible_name": {"type": "string"},
                    "group_id": {"type": "integer"},
                    "project_name": {"type": "string"},
                    "group_name": {"type": "string"},
                    "deadline_iso": {"type": "string"},
                    "no_deadline": {"type": "boolean"},
                },
                "required": ["title"],
                "allOf": [
                    {"anyOf": [{"required": ["responsible_id"]}, {"required": ["responsible_self"]}]},
                ],
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
        resolved_args, project_note, project_error = await _args_with_resolved_project(
            args,
            project_client=self._project_client,
            portal_search=self._portal_search,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if project_error is not None:
            if (
                project_error.status == ToolStatus.NOT_FOUND
                and project_error.data.get("allow_personal_project_creation") is True
            ):
                project_create_result = await _prepare_missing_default_personal_project(
                    store=self._store,
                    args=resolved_args,
                    user_id=user_id,
                    dialog_key=dialog_key,
                    dialog_id=dialog_id,
                )
                if project_create_result is not None:
                    return project_create_result
            return project_error
        draft = build_task_create_draft_from_args(resolved_args, user_id=user_id)
        if not draft.is_ready:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error=f"task_create_draft called outside contract: {'; '.join(draft.contract_errors)}",
                data=draft.as_action_details(),
            )
        stored_params = attach_draft_metadata(dict(draft.params), source_args=resolved_args, user_id=user_id)
        resolved_project = resolved_args.get("_resolved_project")
        if isinstance(resolved_project, dict):
            stored_params["_resolved_project"] = dict(resolved_project)
        if dialog_key:
            await self._store.save_task_draft(dialog_key, stored_params)
        notes = list(draft.notes)
        if project_note:
            notes.append(project_note)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "summary": draft.summary,
                "params": stored_params,
                "preview": draft.preview,
                "notes": notes,
            },
        )


class TaskCreateConfirmTool:
    name = "task_create_confirm"

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
            name="task_create_confirm",
            description=(
                "Confirm and execute the pending task creation draft. "
                "Call this after the user explicitly confirms they want to create the task. "
                "When OAuth is required, creation executes only as the current Bitrix user."
            ),
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
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_create_confirm requires dialog_key",
            )
        draft_params = await self._store.get_task_draft(dialog_key, ttl_minutes=self._draft_ttl_minutes)
        if not draft_params:
            unresolved = await resolve_unknown_draft_claim(
                self._store, dialog_key=dialog_key, expected_type=TASK_CREATE_DRAFT_TYPE
            )
            if unresolved is not None:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="task creation outcome requires durable audit before retry",
                    data={
                        "mutation_outcome": "unknown",
                        "resolution_status": unresolved.get("_draft_resolution_status"),
                        "draft_id": unresolved.get("_draft_id"),
                    },
                )
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="no pending task draft found for this dialog",
            )
        if draft_params.get("_draft_type") not in (None, TASK_CREATE_DRAFT_TYPE):
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="no pending task creation draft found for this dialog",
                data={"draft_type": draft_params.get("_draft_type")},
            )
        if self._dry_run:
            return ToolResult(
                status=ToolStatus.DRY_RUN,
                tool=self.name,
                data={"params": draft_params},
            )
        write_params = {key: value for key, value in draft_params.items() if not str(key).startswith("_")}
        sanitized = apply_write_policy("tasks.task.add", write_params)
        project_reference = draft_params.get("_resolved_project")

        if self._oauth_required_for_writes:
            context_error = _write_context_error(user_id=user_id, dialog_id=dialog_id)
            if context_error:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool=self.name,
                    error=context_error,
                    data={"params": sanitized},
                )
            if self._bitrix_oauth is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="Bitrix OAuth is required for task creation.",
                    data={"params": sanitized},
                )
            claimed = None
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                project_error = await _validate_resolved_project_reference(oauth_client, project_reference)
                if project_error is not None:
                    return ToolResult(
                        status=project_error.status,
                        tool=self.name,
                        error=project_error.error,
                        data={"params": sanitized},
                    )
                claimed = await claim_exact_draft(
                    self._store, dialog_key=dialog_key, draft=draft_params, expected_type=TASK_CREATE_DRAFT_TYPE
                )
                if claimed is None:
                    return ToolResult(
                        status=ToolStatus.DENIED,
                        tool=self.name,
                        error="task draft changed, expired, or is already being confirmed",
                    )
                if not await renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed):
                    return ToolResult(status=ToolStatus.DENIED, tool=self.name, error="task draft claim was lost")
                result = await oauth_client.call("tasks.task.add", sanitized)
            except BitrixOAuthTokenMissing as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error=str(exc),
                    data={"params": sanitized},
                )
            except BitrixConfigError as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error=str(exc),
                    data={"params": sanitized, "mutation_outcome": "not_started"},
                )
            except BitrixApiError as exc:
                outcome = bitrix_mutation_outcome(exc)
                if claimed is not None and outcome == "rejected":
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=str(exc),
                    data={"params": sanitized, "mutation_outcome": outcome},
                )
            except Exception as exc:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                    data={"params": sanitized},
                )
            await self._store.delete_task_draft(
                dialog_key,
                status="confirmed",
                expected_draft_id=str(claimed.get("_draft_id") or ""),
                expected_version=optional_int(claimed.get("_draft_version")),
                expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
            )
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data={"result": result, "params": sanitized},
            )

        if self._write_client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="write_client is not configured",
            )
        claimed = None
        try:
            project_error = await _validate_resolved_project_reference(self._write_client, project_reference)
            if project_error is not None:
                return ToolResult(
                    status=project_error.status,
                    tool=self.name,
                    error=project_error.error,
                    data={"params": sanitized, "mutation_outcome": "unknown"},
                )
            claimed = await claim_exact_draft(
                self._store, dialog_key=dialog_key, draft=draft_params, expected_type=TASK_CREATE_DRAFT_TYPE
            )
            if claimed is None:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool=self.name,
                    error="task draft changed, expired, or is already being confirmed",
                )
            if not await renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed):
                return ToolResult(status=ToolStatus.DENIED, tool=self.name, error="task draft claim was lost")
            result = await self._write_client.call("tasks.task.add", sanitized)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
                data={"mutation_outcome": "unknown"},
            )
        await self._store.delete_task_draft(
            dialog_key,
            status="confirmed",
            expected_draft_id=str(claimed.get("_draft_id") or ""),
            expected_version=optional_int(claimed.get("_draft_version")),
            expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"result": result, "params": sanitized},
        )


class TaskDraftDiscardTool:
    name = "task_draft_discard"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_draft_discard",
            description="Discard the pending task creation draft. Call when the user cancels or abandons the task.",
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
        if dialog_key:
            draft = await self._store.get_task_draft(dialog_key)
            if isinstance(draft, dict) and draft.get("_draft_type") in (None, TASK_CREATE_DRAFT_TYPE):
                await discard_exact_draft(
                    self._store,
                    dialog_key=dialog_key,
                    draft=draft,
                    expected_type=TASK_CREATE_DRAFT_TYPE,
                )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"discarded": bool(dialog_key)},
        )


async def _args_with_resolved_project(
    args: dict[str, Any],
    *,
    project_client: Any | None,
    portal_search: Any | None,
    bitrix_oauth: BitrixOAuthService | None,
    user_id: int | None,
) -> tuple[dict[str, Any], str, ToolResult | None]:
    resolved_args = dict(args or {})
    if optional_int(resolved_args.get("group_id")) is not None:
        return resolved_args, "", None
    project_name = compact_text(str(resolved_args.get("project_name") or resolved_args.get("group_name") or ""))
    default_personal_project = _truthy(resolved_args.get("_default_personal_project"))
    if not project_name:
        return resolved_args, "", None

    read_client = project_client
    if bitrix_oauth is not None:
        read_client, _access_actor, access_error = await resolve_current_user_read_client(
            TaskCreateDraftTool.name,
            fallback_client=project_client,
            bitrix_oauth=bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return resolved_args, "", access_error

    snapshot_projects = _search_projects_snapshot(portal_search, project_name, limit=10)
    if snapshot_projects is not None:
        exact_snapshot = _matching_projects(snapshot_projects, project_name)
        if len(exact_snapshot) == 1:
            return _bind_resolved_project(resolved_args, project=exact_snapshot[0], project_name=project_name)
        if len(exact_snapshot) > 1:
            return (
                resolved_args,
                "",
                _project_resolution_error(
                    f"exact indexed Bitrix project is ambiguous: {project_name}",
                    resolved_args,
                ),
            )

    if read_client is None:
        label = "default personal project" if default_personal_project else "explicit project"
        return (
            resolved_args,
            "",
            _project_resolution_error(
                f"{label} requires Bitrix project resolver",
                resolved_args,
            ),
        )
    try:
        projects = await read_client.search_projects(project_name, limit=10)
    except BitrixConfigError as exc:
        return (
            resolved_args,
            "",
            _project_resolution_error(
                str(exc),
                resolved_args,
                status=ToolStatus.NOT_CONFIGURED,
            ),
        )
    except BitrixApiError as exc:
        return (
            resolved_args,
            "",
            _project_resolution_error(
                str(exc),
                resolved_args,
                status=ToolStatus.ERROR,
            ),
        )
    except Exception as exc:
        return (
            resolved_args,
            "",
            _project_resolution_error(
                f"{type(exc).__name__}: {exc}",
                resolved_args,
                status=ToolStatus.ERROR,
            ),
        )
    exact_projects = _matching_projects(projects, project_name)
    if len(exact_projects) != 1:
        label = "personal Bitrix project" if default_personal_project else "exact Bitrix project"
        resolution = "not found" if not exact_projects else "ambiguous"
        return (
            resolved_args,
            "",
            _project_resolution_error(
                f"{label} {resolution}: {project_name}",
                resolved_args,
                allow_personal_project_creation=default_personal_project and not exact_projects,
            ),
        )
    return _bind_resolved_project(resolved_args, project=exact_projects[0], project_name=project_name)


def _bind_resolved_project(
    resolved_args: dict[str, Any],
    *,
    project: dict[str, Any],
    project_name: str,
) -> tuple[dict[str, Any], str, ToolResult | None]:
    project_id = optional_int(project.get("ID") or project.get("id"))
    if project_id is None:
        return (
            resolved_args,
            "",
            _project_resolution_error(
                f"Bitrix project has no numeric ID: {project_name}",
                resolved_args,
            ),
        )
    resolved_name = compact_text(str(project.get("NAME") or project.get("name") or project_name))
    resolved_args["group_id"] = project_id
    resolved_args["project_name"] = resolved_name or project_name
    resolved_args["_resolved_project"] = {"id": project_id, "name": resolved_args["project_name"]}
    return resolved_args, f"Точный проект: {resolved_args['project_name']} (#{project_id}).", None


def _project_resolution_error(
    error: str,
    args: dict[str, Any],
    *,
    status: ToolStatus = ToolStatus.NOT_FOUND,
    allow_personal_project_creation: bool = False,
) -> ToolResult:
    return ToolResult(
        status=status,
        tool=TaskCreateDraftTool.name,
        error=error,
        data={
            "args": args,
            "allow_personal_project_creation": allow_personal_project_creation,
        },
    )


async def _validate_resolved_project_reference(client: Any, reference: object) -> ToolResult | None:
    if not isinstance(reference, dict):
        return None
    project_id = optional_int(reference.get("id"))
    project_name = compact_text(str(reference.get("name") or ""))
    if project_id is None or not project_name:
        return _project_validation_error(
            ToolStatus.NOT_FOUND,
            "stored task draft has an invalid resolved project reference",
        )
    try:
        projects = await client.search_projects(project_name, limit=10)
    except BitrixConfigError as exc:
        return _project_validation_error(
            ToolStatus.NOT_CONFIGURED,
            f"resolved Bitrix project validation failed: {exc}",
        )
    except BitrixApiError as exc:
        return _project_validation_error(
            ToolStatus.ERROR,
            f"resolved Bitrix project validation failed: {exc}",
        )
    except Exception as exc:
        return _project_validation_error(
            ToolStatus.ERROR,
            f"resolved Bitrix project validation failed: {type(exc).__name__}: {exc}",
        )
    exact = _matching_projects(projects, project_name)
    if len(exact) != 1:
        return _project_validation_error(
            ToolStatus.NOT_FOUND,
            f"resolved Bitrix project is missing, renamed, or ambiguous: {project_name}",
        )
    live_id = optional_int(exact[0].get("ID") or exact[0].get("id"))
    if live_id != project_id:
        return _project_validation_error(
            ToolStatus.NOT_FOUND,
            f"resolved Bitrix project identity changed: {project_name}",
        )
    return None


def _project_validation_error(status: ToolStatus, error: str) -> ToolResult:
    return ToolResult(status=status, tool=TaskCreateConfirmTool.name, error=error)


async def _prepare_missing_default_personal_project(
    *,
    store: TaskDraftStorePort,
    args: dict[str, Any],
    user_id: int | None,
    dialog_key: str | None,
    dialog_id: str | None,
) -> ToolResult | None:
    if not _truthy(args.get("_default_personal_project")):
        return None
    project_name = compact_text(str(args.get("project_name") or args.get("group_name") or ""))
    if not project_name:
        return None
    if user_id is None:
        return ToolResult(
            status=ToolStatus.DENIED,
            tool=TaskCreateDraftTool.name,
            error="Bitrix project creation denied: current Bitrix user_id is missing.",
        )
    if not str(dialog_id or "").strip():
        return ToolResult(
            status=ToolStatus.DENIED,
            tool=TaskCreateDraftTool.name,
            error="Bitrix project creation denied: current Bitrix dialog_id is missing.",
        )
    if not dialog_key:
        return ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=TaskCreateDraftTool.name,
            error="task_create_draft requires dialog_key to prepare a missing personal project",
        )

    task_draft = build_task_create_draft_from_args(args, user_id=user_id)
    if not task_draft.is_ready:
        return ToolResult(
            status=ToolStatus.CONTRACT_VIOLATION,
            tool=TaskCreateDraftTool.name,
            error=f"task_create_draft called outside contract: {'; '.join(task_draft.contract_errors)}",
            data=task_draft.as_action_details(),
        )

    actor_name = compact_text(str(args.get("responsible_name") or args.get("responsible_label") or project_name))
    project_draft = build_project_create_draft_from_args(
        {"name": project_name, "personal_for_self": True},
        user_id=user_id,
        actor_name=actor_name,
        actor_is_admin=False,
    )
    if not project_draft.is_ready:
        return ToolResult(
            status=ToolStatus.CONTRACT_VIOLATION,
            tool=TaskCreateDraftTool.name,
            error=f"project_create_draft called outside contract: {'; '.join(project_draft.contract_errors)}",
            data=project_draft.as_action_details(),
        )

    stored_project_draft = dict(project_draft.params)
    stored_project_draft["after_project_create_task_draft"] = {
        "params": task_draft.params,
        "preview": task_draft.preview,
        "project_name": project_name,
    }
    await store.save_task_draft(dialog_key, stored_project_draft)
    return ToolResult(
        status=ToolStatus.OK,
        tool=TaskCreateDraftTool.name,
        data={
            "requires_project_creation": True,
            "missing_project_name": project_name,
            "project_draft": {
                "params": project_draft.params,
                "preview": project_draft.preview,
                "notes": project_draft.notes,
            },
            "pending_task_draft": {
                "params": task_draft.params,
                "preview": task_draft.preview,
                "notes": task_draft.notes,
            },
        },
    )


def _matching_projects(projects: object, project_name: str) -> list[dict[str, Any]]:
    if not isinstance(projects, list):
        return []
    normalized = _normalize_project_name(project_name)
    exact: list[dict[str, Any]] = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        name = compact_text(str(project.get("NAME") or project.get("name") or ""))
        if _normalize_project_name(name) == normalized:
            exact.append(project)
    return exact


def _deadline_from_args(args: dict[str, Any]) -> tuple[str | None, str, bool, str]:
    if _truthy(args.get("no_deadline")):
        return None, "LLM explicitly selected no_deadline.", True, ""
    raw_deadline = str(args.get("deadline") or args.get("deadline_iso") or "").strip()
    if not raw_deadline:
        return (
            _default_deadline_iso(),
            "Срок по умолчанию: три рабочих дня от даты создания, 19:00 МСК.",
            False,
            "",
        )
    try:
        datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError:
        return None, "", False, "task_create_draft.deadline_iso must be ISO 8601"
    return raw_deadline, "LLM provided deadline_iso.", False, ""


def _default_deadline_iso(*, now: datetime | None = None) -> str:
    current = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    day = current.date()
    remaining = DEFAULT_TASK_DEADLINE_WORKING_DAYS
    while remaining > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            remaining -= 1
    return datetime.combine(day, time(DEFAULT_TASK_DEADLINE_HOUR, 0), tzinfo=MOSCOW_TZ).isoformat()


def _default_description(title: str) -> str:
    return f"Краткое содержание: {title}"


def _draft_preview(fields: dict[str, Any], *, responsible_label: str = "", project_label: str = "") -> dict[str, str]:
    title = compact_text(str(fields.get("TITLE") or "задача"))
    description = compact_text(str(fields.get("DESCRIPTION") or ""))
    deadline = str(fields.get("DEADLINE") or "")
    no_deadline = _truthy(fields.get("NO_DEADLINE"))
    deadline_label = "без срока" if no_deadline else _format_deadline_for_preview(deadline)
    preview = {
        "title": title,
        "description": description,
        "responsible": responsible_label or "указанный сотрудник",
        "deadline": deadline_label,
    }
    if project_label:
        preview["project"] = project_label
    elif fields.get("GROUP_ID") is not None:
        preview["project"] = "выбранный проект"
    return preview


def _format_deadline_for_preview(deadline: str) -> str:
    if not deadline:
        return "не указан"
    try:
        parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00")).astimezone(MOSCOW_TZ)
    except ValueError:
        return deadline
    return parsed.strftime("%d.%m.%Y %H:%M МСК")


def _summary(*, title: str, responsible_id: int | None, deadline: str | None, group_id: int | None) -> str:
    title_part = title or "задача без названия"
    parts = [f"создать задачу `{title_part}`"]
    if responsible_id is not None:
        parts.append(f"ответственный #{responsible_id}")
    if group_id is not None:
        parts.append(f"проект/группа #{group_id}")
    if deadline:
        parts.append(f"срок {deadline}")
    return ", ".join(parts)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on", "без срока", "бессрочно"}
    return bool(value)


def _write_context_error(*, user_id: int | None, dialog_id: str | None) -> str:
    if user_id is None:
        return "Bitrix task creation denied: current Bitrix user_id is missing."
    if not str(dialog_id or "").strip():
        return "Bitrix task creation denied: current Bitrix dialog_id is missing."
    return ""


__all__ = [
    "BitrixTaskCreateDraft",
    "build_task_create_draft_from_args",
    "TaskCreateDraftTool",
    "TaskCreateConfirmTool",
    "TaskDraftDiscardTool",
]
