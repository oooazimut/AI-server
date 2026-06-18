# Складской учёт и каталог товаров

## Когда использовать
Вопросы о товарах, остатках на складе, ценах, складах — любые запросы к разделу «Складской учёт» или «Каталог товаров» Bitrix24.

## ВАЖНО: источники данных

**Для остатков на складе — ТОЛЬКО catalog API, никогда не portal_search.**
Portal_search индексирует задачи, файлы, проекты — там нет данных об остатках. Если искать там по имени человека или складу, можно получить CRM-сделки или задачи с похожим именем — это ошибка.

## Алгоритм: что есть на конкретном складе

```
1. catalog.store.list → найти склад по названию, взять id
2. catalog.storeproduct.list filter[storeId]=<id> → список { productId, amount }
   ВАЖНО: фильтр storeId (camelCase) — при STORE_ID фильтр игнорируется, API вернёт всё подряд
3. batch — получить имена всех товаров за один вызов:
   cmd: { "p<ID>": "catalog.product.get?id=<ID>" } для каждого productId
   (iblockId в запросе не нужен; product.get возвращает name и iblockId)
4. Собрать: название + количество + ссылка /shop/documents-catalog/{iblockId}/product/{id}/
```

## Алгоритм: где находится конкретный товар

```
1. catalog.catalog.list → iblockId
2. catalog.product.list filter[iblockId]=X, filter[%NAME]=<название>, select[id,iblockId,name] → id товара
   ВАЖНО: select обязан содержать "iblockId", иначе API вернёт ошибку 400
3. catalog.storeproduct.list filter[productId]=<id> → { storeId, amount }
   ВАЖНО: фильтр productId (camelCase), не PRODUCT_ID
4. catalog.store.list → расшифровать storeId в названия складов
```

## Примеры вызовов через bitrix_api

```json
{"method": "catalog.store.list", "params": {}}

{"method": "catalog.storeproduct.list", "params": {
  "filter": {"storeId": 21},
  "select": ["productId", "storeId", "amount"]
}}

{"method": "catalog.product.get", "params": {"id": 857}}

{"method": "batch", "params": {
  "halt": 0,
  "cmd": {
    "p857": "catalog.product.get?id=857",
    "p917": "catalog.product.get?id=917",
    "p937": "catalog.product.get?id=937"
  }
}}

{"method": "catalog.catalog.list", "params": {}}

{"method": "catalog.product.list", "params": {
  "filter": {"iblockId": 15, "%NAME": "амортизатор"},
  "select": ["id", "iblockId", "name"]
}}
```

## Фильтры

- `filter[%NAME]` — частичное совпадение по имени (нечувствительно к регистру)
- `filter[ID]=[1,2,3]` — несколько ID сразу (получить названия пачкой)
- `iblockId` — обязателен для catalog.product.list

## Частые ошибки

- `catalog.product.get` без `iblockId` → ошибка. Всегда передавай оба поля: `id` и `iblockId`.
- Не искать складские данные через `portal_search` — там нет остатков, только задачи/файлы/CRM.
- Не путать склад (catalog.store) с сотрудником или CRM-сделкой — это разные сущности.
