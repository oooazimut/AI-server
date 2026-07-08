from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
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
                "It calls catalog.store.list and optionally catalog.storeproduct.list. Product rows include "
                "only items with a positive available amount. Use product_limit=10 by default and product_offset "
                "for follow-up requests asking for the next items."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "include_products": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "product_limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "product_offset": {"type": "integer", "minimum": 0},
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
        product_limit = max(1, min(int(args.get("product_limit") or 10), 50))
        product_offset = max(0, int(args.get("product_offset") or args.get("offset") or 0))
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
            products_result = await self._store_products(store_id, limit=product_limit, offset=product_offset)
            data["products"] = products_result

        return ToolResult(status=ToolStatus.OK, tool=self.name, data=data)

    async def _store_products(self, store_id: object, *, limit: int, offset: int = 0) -> dict[str, Any]:
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

        rows = _extract_items(raw, "storeProducts")
        available_rows = [
            row for row in rows if _positive_amount(_first(row, "amount", "AMOUNT", "quantity", "QUANTITY")) is not None
        ]
        product_ids = [_first(row, "productId", "PRODUCT_ID", "product_id") for row in available_rows]
        product_details = await self._product_details([pid for pid in product_ids if pid not in (None, "")])

        named_items = []
        missing_name_count = 0
        for row in available_rows:
            product_id = _first(row, "productId", "PRODUCT_ID", "product_id")
            product = product_details.get(str(product_id), {})
            product_name = str(product.get("name") or "").strip()
            if not product_name:
                missing_name_count += 1
                continue
            named_items.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "iblock_id": product.get("iblock_id"),
                    "product_url": product.get("url") or "",
                    "amount": _first(row, "amount", "AMOUNT", "quantity", "QUANTITY"),
                    "raw": row,
                }
            )
        items = named_items[offset : offset + limit]
        return {
            "status": "ok",
            "store_id": store_id,
            "items": items,
            "limit": limit,
            "offset": offset,
            "total_rows_seen": len(rows),
            "available_items_seen": len(available_rows),
            "available_items_with_names": len(named_items),
            "filtered_non_positive_count": len(rows) - len(available_rows),
            "filtered_missing_name_count": missing_name_count,
            "has_more": offset + len(items) < len(named_items),
        }

    async def _product_details(self, product_ids: list[object]) -> dict[str, dict[str, Any]]:
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
        details: dict[str, dict[str, Any]] = {}
        for product in products:
            detail = _compact_product(product)
            product_id = detail.get("id")
            if product_id not in (None, "") and detail.get("name"):
                details[str(product_id)] = detail
        missing_ids = [product_id for product_id in product_ids if str(product_id) not in details]
        for product_id in missing_ids[:10]:
            detail = await self._product_detail(product_id)
            if detail.get("name"):
                details[str(product_id)] = detail
        return details

    async def _product_detail(self, product_id: object) -> dict[str, Any]:
        if self._client is None:
            return {}
        try:
            raw = await self._client.result("catalog.product.get", {"id": product_id})
        except (BitrixApiError, BitrixConfigError):
            return {}
        if isinstance(raw, dict):
            product = raw.get("product") if isinstance(raw.get("product"), dict) else raw
            return _compact_product(product)
        return {}


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


def _compact_product(product: dict[str, Any]) -> dict[str, Any]:
    product_id = _first(product, "id", "ID")
    iblock_id = _first(product, "iblockId", "IBLOCK_ID", "iblock_id")
    name = str(_first(product, "name", "NAME") or "").strip()
    return {
        "id": product_id,
        "name": name,
        "iblock_id": iblock_id,
        "url": _catalog_product_url(iblock_id, product_id),
        "raw": product,
    }


def _catalog_product_url(iblock_id: object, product_id: object) -> str:
    if iblock_id in (None, "") or product_id in (None, ""):
        return ""
    return f"/shop/documents-catalog/{iblock_id}/product/{product_id}/"


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


_AMOUNT_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _positive_amount(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, int | float | Decimal):
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            return None
        return amount if amount > 0 else None
    text = str(value).strip()
    if not text:
        return None
    compact = text.replace("\xa0", "").replace(" ", "")
    match = _AMOUNT_RE.search(compact)
    if not match:
        return None
    try:
        amount = Decimal(match.group(0).replace(",", "."))
    except InvalidOperation:
        return None
    return amount if amount > 0 else None
