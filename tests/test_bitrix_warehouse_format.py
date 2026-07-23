from ai_server.orchestrators.bitrix_formatter import _format_warehouse_answer


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
