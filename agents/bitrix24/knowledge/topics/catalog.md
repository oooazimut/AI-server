# Складской учёт и каталог товаров Bitrix24

## Структура

- **Каталог** (`catalog.catalog.list`) — список каталогов, у каждого есть `iblockId`. Большинство порталов имеют один каталог товаров.
- **Товары** (`catalog.product.list`) — номенклатура. Требует `iblockId`. Поля: `id`, `name`, `previewText`, `detailText`, `iblockId`.
- **Разделы каталога** (`catalog.section.list`) — категории товаров, тоже требуют `iblockId`.
- **Склады** (`catalog.store.list`) — список складов. Поля: `id`, `title`, `address`, `active`, `isDefault`.
- **Остатки** (`catalog.storeproduct.list`) — количество товара на складе. Требует фильтр `PRODUCT_ID` или `STORE_ID`. Поля: `productId`, `storeId`, `amount`.
- **Цены** (`catalog.price.list`) — цены на товары. Фильтр по `PRODUCT_ID`.
- **Единицы измерения** (`catalog.measure.list`) — справочник единиц.

## Типичные сценарии

### Найти товары по названию
Для поиска товаров использовать `catalog.product.list` с `filter[%NAME]`. Portal_search можно использовать для товаров (entity_type=`catalog_product`), но **никогда** не для остатков и складов.

### Узнать остатки конкретного товара
1. Найти `id` товара через `catalog.product.list`
2. Вызвать `catalog.storeproduct.list` с `filter[productId]=<id>` (camelCase — не PRODUCT_ID)
3. Результат содержит `storeId` и `amount` по каждому складу
4. Для названий складов — `catalog.store.list`

### Узнать все товары на конкретном складе
1. `catalog.store.list` → найти нужный склад по названию, взять `id`
2. `catalog.storeproduct.list` с `filter[storeId]=<id>` → список `{ productId, amount }`
   **`storeId` — camelCase, не `STORE_ID`**: при `STORE_ID` фильтр молча игнорируется, API возвращает все 1600+ записей подряд.
3. Для каждого `productId`: `catalog.product.get id=<X>` → `name`, `iblockId` (iblockId в запросе не нужен)
4. Собрать: название + количество + ссылка `/shop/documents-catalog/{iblockId}/product/{id}/`
5. **Не искать склад через portal_search** — portal_search может вернуть CRM-сделки или задачи с похожим именем.

### Важные детали API
- `catalog.product.list` требует `iblockId` в массиве `select`, иначе HTTP 400.
- `filter[ID]=[список]` в `catalog.product.list` игнорируется — получить несколько товаров можно только через цикл `catalog.product.get`.
- `filter[storeId]` и `filter[productId]` в `catalog.storeproduct.list` — camelCase.

### Список всех складов
`catalog.store.list`.

### Получить iblockId каталога
`catalog.catalog.list` — первый элемент обычно и есть основной каталог товаров. Поле `iblockId`.

## Важные детали

- `catalog.product.get` требует обязательные поля `id` **и** `iblockId` — без них вернёт ошибку.
- `catalog.product.list` требует только `iblockId` в filter.
- `catalog.storeproduct.list` возвращает `amount` как строку (например `"5.0000"`).
- Остатки изменяются часто — в поисковом индексе их нет, всегда запрашивать через API.
- URL карточки товара: `{portal}/shop/documents-catalog/{iblockId}/product/{id}/`
- Разделы каталога (секции) — это категории/группы товаров, не путать со складами.
