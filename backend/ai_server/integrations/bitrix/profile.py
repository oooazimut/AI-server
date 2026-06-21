from __future__ import annotations

from typing import Any

from ai_server.utils import optional_int


def compact_user_profile(user: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw Bitrix user dict into a compact profile used across tools and integrations."""
    user_id = optional_int(user.get("ID") or user.get("id"))
    first_name = str(user.get("NAME") or user.get("name") or "").strip()
    last_name = str(user.get("LAST_NAME") or user.get("lastName") or user.get("last_name") or "").strip()
    second_name = str(user.get("SECOND_NAME") or user.get("secondName") or user.get("second_name") or "").strip()
    full_name = " ".join(part for part in (last_name, first_name, second_name) if part).strip()
    email = str(user.get("EMAIL") or user.get("email") or "").strip()
    work_position = str(user.get("WORK_POSITION") or user.get("workPosition") or "").strip()
    departments = _int_list(user.get("UF_DEPARTMENT") or user.get("ufDepartment") or user.get("department"))
    return {
        "id": user_id,
        "label": full_name or email or (f"Bitrix user #{user_id}" if user_id is not None else "Bitrix user"),
        "active": _truthy(user.get("ACTIVE") or user.get("active"), default=True),
        "is_admin": _truthy(
            user.get("IS_ADMIN") or user.get("ADMIN") or user.get("isAdmin") or user.get("is_admin"),
            default=False,
        ),
        "department_ids": departments,
        "work_position": work_position,
        "user_type": str(user.get("USER_TYPE") or user.get("userType") or "").strip(),
        "raw_policy_fields": {
            key: user[key]
            for key in sorted(user)
            if key
            in {
                "ID",
                "ACTIVE",
                "IS_ADMIN",
                "ADMIN",
                "USER_TYPE",
                "UF_DEPARTMENT",
                "WORK_POSITION",
            }
        },
    }


def _int_list(value: object) -> list[int]:
    raw_values = value if isinstance(value, list) else [value]
    result: list[int] = []
    for raw in raw_values:
        item = optional_int(raw)
        if item is not None and item not in result:
            result.append(item)
    return result


def _truthy(value: object, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "да"}
    return bool(value)
