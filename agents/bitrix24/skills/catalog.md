# Складской учёт и каталог товаров

## Когда использовать
Вопросы о товарах, остатках на складе, ценах, складах — любые запросы к разделу «Складской учёт» или «Каталог товаров» Bitrix24.

## ВАЖНО: источники данных

**Для остатков на складе — ТОЛЬКО catalog API, никогда не portal_search.**
Portal_search индексирует задачи, файлы, проекты — там нет данных об остатках. Если искать там по имени человека или складу, можно получить CRM-сделки или задачи с похожим именем — это ошибка.

## Алгоритм: что есть на конкретном складе

```
1. catalog.store.list → найти склад по названию, взять id
2. catalog.storeproduct.list filter[STORE_ID]=<id>, filter[>AMOUNT]=0 → список { productId, amount }
3. catalog.catalog.list → взять iblockId каталога
4. catalog.product.list filter[iblockId]=X, filter[ID]=<список productId> → названия товаров
5. Собрать: название товара + количество + ссылка /shop/documents-catalog/{iblockId}/product/{id}/
```

## Алгоритм: где находится конкретный товар

```
1. catalog.catalog.list → iblockId
2. catalog.product.list filter[iblockId]=X, filter[%NAME]=<название> → id товара
3. catalog.storeproduct.list filter[PRODUCT_ID]=<id>, filter[>AMOUNT]=0 → { storeId, amount }
4. catalog.store.list → расшифровать storeId в названия складов
```

## Примеры вызовов через bitrix_api

```json
{"method": "catalog.store.list", "params": {}}

{"method": "catalog.storeproduct.list", "params": {
  "filter": {"STORE_ID": 3, ">AMOUNT": 0},
  "select": ["productId", "storeId", "amount"]
}}

{"method": "catalog.catalog.list", "params": {}}

{"method": "catalog.product.list", "params": {
  "filter": {"iblockId": 15, "ID": [101, 102, 103]},
  "select": ["id", "name", "iblockId"]
}}

{"method": "catalog.product.list", "params": {
  "filter": {"iblockId": 15, "%NAME": "амортизатор"},
  "select": ["id", "name", "iblockId"]
}}
```

## Фильтры

- `filter[>AMOUNT]=0` — только товары с ненулевым остатком
- `filter[%NAME]` — частичное совпадение по имени (нечувствительно к регистру)
- `filter[ID]=[1,2,3]` — несколько ID сразу (получить названия пачкой)
- `iblockId` — обязателен для catalog.product.list

## Частые ошибки

- `catalog.product.get` без `iblockId` → ошибка. Всегда передавай оба поля: `id` и `iblockId`.
- Не искать складские данные через `portal_search` — там нет остатков, только задачи/файлы/CRM.
- Не путать склад (catalog.store) с сотрудником или CRM-сделкой — это разные сущности.
