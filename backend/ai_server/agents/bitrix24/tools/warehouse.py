from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort


class BitrixWarehouseSearchTool:
    name = "bitrix_warehouse_search"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read-only warehouse/store lookup in Bitrix catalog. Use it for requests about warehouses, "
                "stores, stock locations, inventory leftovers, or phrases like 'find warehouse Borisov'. "
                "It calls catalog.store.list and optionally catalog.storeproduct.list."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "include_products": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "product_limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query"],
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
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="BitrixClient is not injected",
            )
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="query is required")

        limit = max(1, min(int(args.get("limit") or 10), 20))
        product_limit = max(1, min(int(args.get("product_limit") or 20), 50))
        include_products = bool(args.get("include_products"))

        try:
            raw_stores = await self._client.result("catalog.store.list", {})
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool=self.name,
                error=str(exc),
                data={"query": query},
            )

        stores = _extract_items(raw_stores, "stores")
        matches = _match_stores(stores, query=query, limit=limit)
        data: dict[str, Any] = {
            "query": query,
            "matches": matches,
            "total_stores_seen": len(stores),
            "summary": _stores_summary(matches, query=query),
        }

        if include_products and matches:
            store_id = matches[0].get("id")
            products_result = await self._store_products(store_id, limit=product_limit)
            data["products"] = products_result

        return ToolResult(status=ToolStatus.OK, tool=self.name, data=data)

    async def _store_products(self, store_id: object, *, limit: int) -> dict[str, Any]:
        if store_id in (None, "") or self._client is None:
            return {"status": "not_available", "items": [], "message": "store id is missing"}

        params = {
            "filter": {"storeId": store_id},
            "select": ["storeId", "productId", "amount"],
        }
        try:
            raw = await self._client.result("catalog.storeproduct.list", params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return {
                "status": "error",
                "method": "catalog.storeproduct.list",
                "error": str(exc),
                "items": [],
            }

        rows = _extract_items(raw, "storeProducts")[:limit]
        product_ids = [_first(row, "productId", "PRODUCT_ID", "product_id") for row in rows]
        product_names = await self._product_names([pid for pid in product_ids if pid not in (None, "")])

        items = []
        for row in rows:
            product_id = _first(row, "productId", "PRODUCT_ID", "product_id")
            items.append(
                {
                    "product_id": product_id,
                    "product_name": product_names.get(str(product_id), ""),
                    "amount": _first(row, "amount", "AMOUNT", "quantity", "QUANTITY"),
                    "raw": row,
                }
            )
        return {"status": "ok", "store_id": store_id, "items": items, "limit": limit}

    async def _product_names(self, product_ids: list[object]) -> dict[str, str]:
        if not product_ids or self._client is None:
            return {}
        try:
            raw = await self._client.result(
                "catalog.product.list",
                {"filter": {"id": product_ids}, "select": ["id", "iblockId", "name"]},
            )
        except (BitrixApiError, BitrixConfigError):
            raw = None
        products = _extract_items(raw, "products")
        names: dict[str, str] = {}
        for product in products:
            product_id = _first(product, "id", "ID")
            name = str(_first(product, "name", "NAME") or "").strip()
            if product_id not in (None, "") and name:
                names[str(product_id)] = name
        missing_ids = [product_id for product_id in product_ids if str(product_id) not in names]
        for product_id in missing_ids[:10]:
            name = await self._product_name(product_id)
            if name:
                names[str(product_id)] = name
        return names

    async def _product_name(self, product_id: object) -> str:
        if self._client is None:
            return ""
        try:
            raw = await self._client.result("catalog.product.get", {"id": product_id})
        except (BitrixApiError, BitrixConfigError):
            return ""
        if isinstance(raw, dict):
            product = raw.get("product") if isinstance(raw.get("product"), dict) else raw
            return str(_first(product, "name", "NAME") or "").strip()
        return ""


def _match_stores(stores: list[dict[str, Any]], *, query: str, limit: int) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for store in stores:
        title = str(_first(store, "title", "TITLE", "name", "NAME") or "")
        address = str(_first(store, "address", "ADDRESS") or "")
        description = str(_first(store, "description", "DESCRIPTION") or "")
        search_text = f"{title} {address} {description}".casefold()
        score = 0
        for term in terms:
            if term in title.casefold():
                score += 10
            elif term in search_text:
                score += 3
        if score > 0:
            scored.append((score, _compact_store(store)))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("title") or "")))
    return [store for _, store in scored[:limit]]


def _compact_store(store: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _first(store, "id", "ID"),
        "title": _first(store, "title", "TITLE", "name", "NAME"),
        "address": _first(store, "address", "ADDRESS"),
        "active": _first(store, "active", "ACTIVE"),
        "is_default": _first(store, "isDefault", "IS_DEFAULT"),
        "raw": store,
    }


def _stores_summary(matches: list[dict[str, Any]], *, query: str) -> str:
    if not matches:
        return f"No Bitrix catalog stores found for query: {query}"
    lines = [f"Found Bitrix catalog stores: {len(matches)}"]
    for store in matches:
        title = str(store.get("title") or f"store #{store.get('id')}")
        address = str(store.get("address") or "").strip()
        suffix = f" — {address}" if address else ""
        lines.append(f"- {title} (id: {store.get('id')}){suffix}")
    return "\n".join(lines)


def _query_terms(query: str) -> list[str]:
    ignored = {
        "warehouse",
        "store",
        "stock",
        "find",
        "search",
        "склад",
        "склада",
        "складе",
        "склады",
        "найди",
        "поиск",
        "битрикс",
        "bitrix",
    }
    terms = []
    for raw in query.replace(",", " ").split():
        term = raw.strip().casefold()
        if len(term) < 2 or term in ignored:
            continue
        terms.append(term)
    return terms or [query.casefold()]


def _extract_items(raw: Any, preferred_key: str) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        keys = (preferred_key, preferred_key.lower(), "items", "result", "products", "stores")
        for key in keys:
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_items(value, preferred_key)
                if nested:
                    return nested
    return []


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None
