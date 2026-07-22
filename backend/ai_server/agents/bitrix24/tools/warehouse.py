from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from ai_server.agents.bitrix24.tools.read_client import resolve_current_user_read_client
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort
from ai_server.tools.bitrix_search import PortalSearchPort


class BitrixWarehouseSearchTool:
    name = "bitrix_warehouse_search"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        portal_search: PortalSearchPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._client = client
        self._portal_search = portal_search
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read-only warehouse/store lookup in Bitrix catalog. Use it for requests about warehouses, "
                "stores, stock locations, inventory leftovers, or phrases like 'find warehouse Borisov'. "
                "It calls catalog.store.list and optionally catalog.storeproduct.list. Product rows include "
                "only items with a positive available amount. Use product_limit=50 by default and product_offset "
                "for follow-up requests asking for the next items."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "product_query": {
                        "type": "string",
                        "description": "Optional product-name filter inside the matched warehouse.",
                    },
                    "include_products": {"type": "boolean"},
                    "list_all": {
                        "type": "boolean",
                        "description": "Return the warehouse list instead of treating query as a warehouse name.",
                    },
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
        product_query = str(args.get("product_query") or "").strip()
        list_all = bool(args.get("list_all"))
        if not query and not list_all:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="query is required")
        if list_all:
            query = query or "все"

        limit = max(1, min(int(args.get("limit") or 10), 20))
        product_limit = max(1, min(int(args.get("product_limit") or 50), 50))
        product_offset = max(0, int(args.get("product_offset") or args.get("offset") or 0))
        include_products = bool(args.get("include_products"))

        read_client, access_actor, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return access_error

        snapshot_data = None
        # Stock amounts are operational data: never serve them solely from the
        # periodic PostgreSQL snapshot.  The snapshot may resolve a warehouse,
        # while every contents/product request is verified through live Bitrix.
        if not list_all and not include_products:
            snapshot_data = self._snapshot_search(
                query=query,
                include_products=include_products,
                limit=limit,
                product_limit=product_limit,
                product_offset=product_offset,
                product_query=product_query,
                access_actor=access_actor,
            )
        if snapshot_data is not None:
            return ToolResult(status=ToolStatus.OK, tool=self.name, data=snapshot_data)

        try:
            raw_stores = await read_client.result("catalog.store.list", {})
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool=self.name,
                error=str(exc),
                data={"query": query},
            )

        stores = _extract_items(raw_stores, "stores")
        matches = (
            [_compact_store(store) for store in stores[:limit]]
            if list_all
            else _match_stores(stores, query=query, limit=limit)
        )
        data: dict[str, Any] = {
            "query": query,
            "source": "live_bitrix_rest",
            "access_actor": access_actor,
            "matches": matches,
            "total_stores_seen": len(stores),
            "summary": _stores_summary(matches, query=query),
            "list_all": list_all,
        }

        if include_products and matches:
            store_id = matches[0].get("id")
            products_result = await self._store_products(
                read_client,
                store_id,
                limit=product_limit,
                offset=product_offset,
                product_query=product_query,
            )
            data["products"] = products_result

        return ToolResult(status=ToolStatus.OK, tool=self.name, data=data)

    def _snapshot_search(
        self,
        *,
        query: str,
        include_products: bool,
        limit: int,
        product_limit: int,
        product_offset: int,
        product_query: str,
        access_actor: str,
    ) -> dict[str, Any] | None:
        if self._portal_search is None:
            return None
        try:
            if not self._portal_search.stats().exists:
                return None
            store_results = self._portal_search.search(query, entity_types={"catalog_store"}, limit=limit)
            stock_seed_results = self._portal_search.search(
                query,
                entity_types={"catalog_store_stock"},
                limit=max(limit, product_limit + product_offset),
            )
        except Exception:
            return None

        matches = [_snapshot_store_match(item) for item in store_results]
        if not matches and stock_seed_results:
            matches = [_snapshot_store_match_from_stock(item) for item in stock_seed_results[:limit]]
        matches = _dedupe_store_matches(matches)[:limit]
        if not matches:
            return None

        data: dict[str, Any] = {
            "query": query,
            "matches": matches,
            "total_stores_seen": len(matches),
            "source": "postgres_portal_snapshot",
            "access_actor": access_actor,
            "summary": _stores_summary(matches, query=query),
        }
        if include_products:
            data["products"] = self._snapshot_store_products(
                matches[0],
                limit=product_limit,
                offset=product_offset,
                product_query=product_query,
            )
        return data

    def _snapshot_store_products(
        self, store: dict[str, Any], *, limit: int, offset: int, product_query: str = ""
    ) -> dict[str, Any]:
        if self._portal_search is None:
            return {"status": "not_available", "items": [], "message": "portal search index is missing"}
        store_id = store.get("id")
        store_title = str(store.get("title") or "")
        query = store_title or str(store_id or "")
        try:
            rows = self._portal_search.search(
                query,
                entity_types={"catalog_store_stock"},
                limit=max(1000, offset + limit),
            )
        except Exception as exc:
            return {"status": "error", "error": str(exc), "items": []}

        stock_items = []
        for row in rows:
            metadata = row.metadata or {}
            if store_id not in (None, "") and str(metadata.get("store_id")) != str(store_id):
                continue
            product_name = str(metadata.get("product_name") or row.title).strip()
            if not product_name:
                continue
            if not _matches_product_query(product_name, product_query):
                continue
            stock_items.append(
                {
                    "product_id": metadata.get("product_id"),
                    "product_name": product_name,
                    "iblock_id": metadata.get("iblock_id"),
                    "product_url": metadata.get("product_url") or row.url,
                    "amount": metadata.get("amount"),
                    "raw": metadata,
                    "source": "postgres_portal_snapshot",
                }
            )

        total = len(stock_items)
        items = stock_items[offset : offset + limit]
        return {
            "status": "ok",
            "store_id": store_id,
            "items": items,
            "limit": limit,
            "offset": offset,
            "total_rows_seen": total,
            "available_items_seen": total,
            "available_items_with_names": total,
            "filtered_non_positive_count": 0,
            "filtered_missing_name_count": 0,
            "has_more": offset + len(items) < total,
            "source": "postgres_portal_snapshot",
            "product_query": product_query,
        }

    async def _store_products(
        self,
        client: BitrixToolClientPort,
        store_id: object,
        *,
        limit: int,
        offset: int = 0,
        product_query: str = "",
    ) -> dict[str, Any]:
        if store_id in (None, ""):
            return {"status": "not_available", "items": [], "message": "store id is missing"}

        params = {
            "filter": {"storeId": store_id},
            "select": ["storeId", "productId", "amount"],
        }
        try:
            raw = await client.result("catalog.storeproduct.list", params)
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
        product_details = await self._product_details(client, [pid for pid in product_ids if pid not in (None, "")])

        named_items = []
        missing_name_count = 0
        for row in available_rows:
            product_id = _first(row, "productId", "PRODUCT_ID", "product_id")
            product = product_details.get(str(product_id), {})
            product_name = str(product.get("name") or "").strip()
            if not product_name:
                missing_name_count += 1
                continue
            if not _matches_product_query(product_name, product_query):
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
            "product_query": product_query,
        }

    async def _product_details(
        self, client: BitrixToolClientPort, product_ids: list[object]
    ) -> dict[str, dict[str, Any]]:
        if not product_ids:
            return {}
        try:
            raw = await client.result(
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
            detail = await self._product_detail(client, product_id)
            if detail.get("name"):
                details[str(product_id)] = detail
        return details

    async def _product_detail(self, client: BitrixToolClientPort, product_id: object) -> dict[str, Any]:
        try:
            raw = await client.result("catalog.product.get", {"id": product_id})
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


def _snapshot_store_match(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "metadata", {}) or {}
    return {
        "id": getattr(item, "entity_id", ""),
        "title": getattr(item, "title", ""),
        "address": _first_text(getattr(item, "body", "")),
        "active": metadata.get("active"),
        "is_default": metadata.get("is_default"),
        "source": "postgres_portal_snapshot",
    }


def _snapshot_store_match_from_stock(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "metadata", {}) or {}
    return {
        "id": metadata.get("store_id"),
        "title": metadata.get("store_title"),
        "address": metadata.get("store_address"),
        "active": None,
        "is_default": None,
        "source": "postgres_portal_snapshot",
    }


def _dedupe_store_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for match in matches:
        key = str(match.get("id") or match.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(match)
    return result


def _first_text(value: object) -> str:
    for line in str(value or "").splitlines():
        text = line.strip()
        if text:
            return text
    return ""


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


def _matches_product_query(product_name: str, product_query: str) -> bool:
    """Keep a warehouse product filter deterministic and transparent."""
    terms = [term for term in re.findall(r"[\w-]+", product_query.casefold()) if len(term) >= 2]
    if not terms:
        return True
    normalized_name = product_name.casefold()
    return all(term in normalized_name for term in terms)


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
