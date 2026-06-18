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
2. Вызвать `catalog.storeproduct.list` с `filter[PRODUCT_ID]=<id>`, `filter[>AMOUNT]=0`
3. Результат содержит `storeId` и `amount` по каждому складу
4. Для названий складов — `catalog.store.list`

### Узнать все товары на конкретном складе
1. `catalog.store.list` → найти нужный склад по названию, взять `id`
2. `catalog.storeproduct.list` с `filter[STORE_ID]=<id>`, `filter[>AMOUNT]=0` → список `{ productId, amount }`
3. `catalog.catalog.list` → получить `iblockId`
4. `catalog.product.list` с `filter[iblockId]=X`, `filter[ID]=[список productId]` → имена товаров
5. **Не искать склад через portal_search** — portal_search может вернуть CRM-сделки или задачи с похожим именем.

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
