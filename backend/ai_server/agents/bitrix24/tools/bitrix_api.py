from __future__ import annotations

import re
from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy, decide_bitrix_method_policy
from ai_server.tools.bitrix_ports import BitrixToolClientPort, BitrixWritePort

PROJECT_WRITE_METHODS_WITH_DEDICATED_TOOLS = {"sonet_group.create"}


class BitrixApiTool:
    name = "bitrix_api"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        *,
        write_client: BitrixWritePort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        dry_run: bool = False,
        oauth_required_for_writes: bool = True,
    ) -> None:
        self._client = client
        self._write_client = write_client
        self._bitrix_oauth = bitrix_oauth
        self._dry_run = dry_run
        self._oauth_required_for_writes = oauth_required_for_writes

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_api",
            description=(
                "Bitrix24 REST API access. Read methods (ending in .get/.list/.search) execute immediately. "
                "Write methods execute after explicit user confirmation in the conversation. "
                "When OAuth is required, writes execute only as the current Bitrix user. "
                "Dangerous methods (user management, bots) are denied."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "params": {"type": "object"},
                    "summary": {"type": "string"},
                },
                "required": ["method", "params"],
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
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        summary = str(args.get("summary") or method).strip()
        return await self._call_api(method, params, summary, user_id=user_id, dialog_id=dialog_id)

    async def _call_api(
        self,
        method: str,
        params: dict[str, Any],
        summary: str,
        *,
        user_id: int | None,
        dialog_id: str | None,
    ) -> ToolResult:
        if self._client is None and self._write_client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        if method.strip().casefold() in PROJECT_WRITE_METHODS_WITH_DEDICATED_TOOLS:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                error="Use project_create_draft/project_create_confirm for Bitrix project creation.",
                data={"method": method},
            )
        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return await self._execute_write(method, params, summary, user_id=user_id, dialog_id=dialog_id)
        read_client = self._client
        access_actor = "configured_client"
        if self._bitrix_oauth is not None:
            if user_id is None:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool="bitrix_api",
                    error="Bitrix read operation denied: current Bitrix user_id is missing.",
                    data={"method": method, "params": params},
                )
            try:
                read_client = await self._bitrix_oauth.client_for_user(user_id)
                access_actor = "oauth_current_user"
            except BitrixOAuthTokenMissing as exc:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
        if read_client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        try:
            result = await read_client.result(method, params)
            if method.casefold() == "sonet_group.get":
                result = await _sonet_group_get_with_normalized_fallback(read_client, params, result)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="bitrix_api",
            data={"method": method, "params": params, "result": result, "access_actor": access_actor},
        )

    async def _execute_write(
        self,
        method: str,
        params: dict[str, Any],
        summary: str,
        *,
        user_id: int | None,
        dialog_id: str | None,
    ) -> ToolResult:
        if not params:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="bitrix_api",
                error="Write methods require non-empty params.",
                data={"method": method},
            )
        if self._dry_run:
            return ToolResult(
                status=ToolStatus.DRY_RUN,
                tool="bitrix_api",
                data={"method": method, "summary": summary, "dry_run": True},
            )
        params = apply_write_policy(method, params)

        if self._oauth_required_for_writes:
            context_error = _write_context_error(user_id=user_id, dialog_id=dialog_id)
            if context_error:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool="bitrix_api",
                    error=context_error,
                    data={"method": method},
                )
            if self._bitrix_oauth is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool="bitrix_api",
                    error="Bitrix OAuth is required for write operations.",
                    data={"method": method},
                )

        # Attempt OAuth per-user write.
        if self._bitrix_oauth is not None and user_id is not None:
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                raw = await oauth_client.call(method, params)
                result = raw.get("result") if isinstance(raw, dict) else raw
                return ToolResult(
                    status=ToolStatus.OK,
                    tool="bitrix_api",
                    data={"method": method, "params": params, "result": result},
                )
            except BitrixOAuthTokenMissing as exc:
                if self._oauth_required_for_writes:
                    return ToolResult(
                        status=ToolStatus.NOT_CONFIGURED,
                        tool="bitrix_api",
                        error=str(exc),
                        data={"method": method, "params": params},
                    )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
            except Exception as exc:
                if self._oauth_required_for_writes:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        tool="bitrix_api",
                        error=f"{type(exc).__name__}: {exc}",
                        data={"method": method, "params": params},
                    )

        if self._oauth_required_for_writes:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_api",
                error="OAuth write client is not available for the current Bitrix user.",
                data={"method": method},
            )

        # Legacy fallback: dedicated write client (BitrixWritePort).
        write_cl = self._write_client
        if write_cl is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_api",
                error="No write client configured for Bitrix write operations.",
                data={"method": method},
            )
        try:
            raw = await write_cl.call(method, params)
            result = raw.get("result") if isinstance(raw, dict) else raw
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(
            status=ToolStatus.OK, tool="bitrix_api", data={"method": method, "params": params, "result": result}
        )


def _write_context_error(*, user_id: int | None, dialog_id: str | None) -> str:
    if user_id is None:
        return "Bitrix write operation denied: current Bitrix user_id is missing."
    if not str(dialog_id or "").strip():
        return "Bitrix write operation denied: current Bitrix dialog_id is missing."
    return ""


async def _sonet_group_get_with_normalized_fallback(
    client: BitrixToolClientPort,
    params: dict[str, Any],
    initial_result: Any,
) -> Any:
    if _extract_sonet_groups(initial_result):
        return initial_result

    query = _extract_sonet_group_query(params)
    if not query:
        return initial_result

    fallback_params = _sonet_group_fallback_params(params)
    fallback_result = await client.result("sonet_group.get", fallback_params)
    matches = _match_sonet_groups(_extract_sonet_groups(fallback_result), query=query, limit=_sonet_group_limit(params))
    if not matches:
        return initial_result
    return _replace_empty_sonet_group_result(initial_result, matches)


def _extract_sonet_group_query(params: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(value: Any, *, parent_key: str = "") -> None:
        if isinstance(value, dict):
            if parent_key.casefold() in {"order", "sort", "select"}:
                return
            for key, item in value.items():
                key_text = str(key)
                normalized_key = key_text.upper().lstrip("%=?")
                if normalized_key in {"NAME", "TITLE", "SEARCH_INDEX", "QUERY", "Q", "SEARCH"}:
                    if isinstance(item, str) and item.strip():
                        values.append(item.strip())
                    continue
                visit(item, parent_key=key_text)
        elif isinstance(value, list):
            for item in value:
                visit(item, parent_key=parent_key)

    visit(params)
    if not values:
        return ""
    return max(values, key=len)


def _sonet_group_fallback_params(params: dict[str, Any]) -> dict[str, Any]:
    fallback = dict(params)
    for filter_key in ("FILTER", "filter"):
        value = fallback.get(filter_key)
        if isinstance(value, dict):
            fallback[filter_key] = _strip_sonet_group_name_filters(value)
    if "ORDER" not in fallback and "order" not in fallback:
        fallback["ORDER"] = {"NAME": "ASC"}
    return fallback


def _strip_sonet_group_name_filters(value: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key).upper().lstrip("%=?")
        if normalized_key in {"NAME", "TITLE", "SEARCH_INDEX", "QUERY", "Q", "SEARCH"}:
            continue
        cleaned[key] = item
    return cleaned


def _sonet_group_limit(params: dict[str, Any]) -> int:
    candidates = [params.get("limit"), params.get("LIMIT")]
    nav = params.get("NAV_PARAMS") or params.get("nav_params") or params.get("NavParams")
    if isinstance(nav, dict):
        candidates.extend([nav.get("nPageSize"), nav.get("nTopCount"), nav.get("pageSize")])
    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return max(1, min(value, 20))
    return 10


def _extract_sonet_groups(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("groups", "workgroups", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_sonet_groups(value)
                if nested:
                    return nested
    return []


def _match_sonet_groups(groups: list[dict[str, Any]], *, query: str, limit: int) -> list[dict[str, Any]]:
    normalized_query = _normalize_project_name(query)
    query_terms = set(normalized_query.split())
    scored: list[tuple[int, dict[str, Any]]] = []
    for group in groups:
        name = str(group.get("NAME") or group.get("name") or group.get("TITLE") or group.get("title") or "")
        normalized_name = _normalize_project_name(name)
        if not normalized_name:
            continue
        score = 0
        if normalized_name == normalized_query:
            score = 100
        elif normalized_query and normalized_query in normalized_name:
            score = 80
        elif query_terms and query_terms.issubset(set(normalized_name.split())):
            score = 70
        if score:
            scored.append((score, group))
    scored.sort(key=lambda item: (-item[0], _normalize_project_name(str(item[1].get("NAME") or ""))))
    return [group for _, group in scored[:limit]]


_PROJECT_ALIASES = {
    "almira": "альмера",
    "almera": "альмера",
    "largus": "ларгус",
    "logan": "логан",
}


def _normalize_project_name(value: str) -> str:
    text = value.casefold().replace("ё", "е")
    for source, target in _PROJECT_ALIASES.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    text = re.sub(r"[-–—_/\\]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _replace_empty_sonet_group_result(initial_result: Any, matches: list[dict[str, Any]]) -> Any:
    if isinstance(initial_result, dict):
        replaced = dict(initial_result)
        for key in ("groups", "workgroups", "items", "result"):
            value = replaced.get(key)
            if isinstance(value, list):
                replaced[key] = matches
                replaced["normalized_fallback"] = True
                return replaced
        return {"groups": matches, "normalized_fallback": True}
    return matches
