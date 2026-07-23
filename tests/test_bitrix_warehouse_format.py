import re

from ai_server.models import ToolResult
from ai_server.orchestrators.bitrix_formatter import (
    _format_warehouse_answer,
    direct_tool_results_response,
)


def test_warehouse_page_uses_absolute_item_numbers_after_offset():
    answer = _format_warehouse_answer(
        {
            "matches": [{"title": "Борисов"}],
            "products": {
                "status": "ok",
                "offset": 50,
                "limit": 50,
                "available_items_with_names": 97,
                "items": [
                    {"product_name": "Позиция 51", "amount": 1},
                    {"product_name": "Позиция 52", "amount": 1},
                ],
            },
        }
    )

    lines = answer.splitlines()
    assert any(line.startswith("51. Позиция 51") for line in lines)
    assert any(line.startswith("52. Позиция 52") for line in lines)
    assert not any(line.startswith("1. Позиция 51") for line in lines)


def test_multi_warehouse_response_has_one_global_fifty_item_budget():
    def result(name: str, available: int, returned: int) -> ToolResult:
        return ToolResult(
            status="ok",
            tool="bitrix_warehouse_search",
            data={
                "matches": [{"title": name}],
                "products": {
                    "status": "ok",
                    "offset": 0,
                    "limit": 50,
                    "available_items_with_names": available,
                    "has_more": available > returned,
                    "items": [
                        {"product_name": f"{name} товар {index:03d}", "amount": 1}
                        for index in range(1, returned + 1)
                    ],
                },
            },
        )

    rendered = direct_tool_results_response(
        agent_id="internal_orchestrator",
        tool_results=[
            result("Борисов", 100, 50),
            result("Карасев", 100, 50),
            result("Гараж", 4, 4),
        ],
    )

    item_lines = [line for line in rendered.answer.splitlines() if re.match(r"^\d+\. ", line)]
    assert len(item_lines) == 50
    assert "Показаны первые 23 позиции из 100" in rendered.answer
    assert rendered.answer.count("Показаны первые 23 позиции из 100") == 2
    assert "Показаны все 4 позиции" in rendered.answer
    assert "Источник bitrix24" not in rendered.answer
