# Складской учёт и каталог товаров

## Когда использовать
Вопросы о товарах, остатках на складе, ценах, складах — любые запросы к разделу «Складской учёт» или «Каталог товаров» Bitrix24.

## Алгоритм поиска товара

1. Сначала поиск по индексу портала (`portal_search`) — быстро, находит по названию. Результаты с `entity_type=catalog_product` — это товары.
2. Если не найдено или нужны актуальные данные: получить `iblockId` через `catalog.catalog.list`, затем `catalog.product.list` с фильтром по имени (`filter[%NAME]=...`).

## Алгоритм получения остатков

```
catalog.catalog.list → iblockId
catalog.product.list filter[iblockId]=X, filter[%NAME]=<название> → id товара
catalog.storeproduct.list filter[PRODUCT_ID]=<id> → [{ storeId, amount }]
catalog.store.list → { id: title } для расшифровки storeId
```

## Примеры вызовов через bitrix_api

```json
{"method": "catalog.catalog.list", "params": {}}

{"method": "catalog.product.list", "params": {
  "filter": {"iblockId": 14, "%NAME": "амортизатор"},
  "select": ["id", "name", "iblockId"]
}}

{"method": "catalog.storeproduct.list", "params": {
  "filter": {"PRODUCT_ID": 1234},
  "select": ["productId", "storeId", "amount"]
}}

{"method": "catalog.store.list", "params": {}}
```

## Фильтры catalog.product.list

- `%NAME` — частичное совпадение по имени (нечувствительно к регистру)
- `iblockId` — обязателен
- `=NAME` — точное совпадение
- `ACTIVE=Y` — только активные товары

## Частые ошибки

- `catalog.product.get` без `iblockId` → ошибка. Всегда передавай оба поля: `id` и `iblockId`.
- Для поиска по частичному имени используй `%NAME`, не `NAME`.
- Остатки (`catalog.storeproduct.list`) не хранятся в индексе — всегда вызывай API.
