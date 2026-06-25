# Складской учёт и каталог товаров Bitrix24

## Структура

- **Каталог** (`catalog.catalog.list`) — список каталогов, у каждого есть `iblockId`. Большинство порталов имеют один каталог товаров.
- **Товары** (`catalog.product.list`) — номенклатура. Требует `iblockId`. Поля: `id`, `name`, `previewText`, `detailText`, `iblockId`.
- **Разделы каталога** (`catalog.section.list`) — категории товаров, тоже требуют `iblockId`.
- **Склады** (`catalog.store.list`) — список складов. Поля: `id`, `title`, `address`, `active`, `isDefault`.
- **Остатки** (`catalog.storeproduct.list`) — количество товара на складе. Требует фильтр `PRODUCT_ID` или `STORE_ID`. Поля: `productId`, `storeId`, `amount`.
- **Цены** (`catalog.price.list`) — цены на товары. Фильтр по `PRODUCT_ID`.
- **Единицы измерения** (`catalog.measure.list`) — справочник единиц.

## Важные детали

- `catalog.product.get` требует обязательные поля `id` **и** `iblockId` — без них вернёт ошибку.
- `catalog.product.list` требует только `iblockId` в filter.
- `catalog.storeproduct.list` возвращает `amount` как строку (например `"5.0000"`).
- Остатки изменяются часто — в поисковом индексе их нет, всегда запрашивать через API.
- URL карточки товара: `{portal}/shop/documents-catalog/{iblockId}/product/{id}/`
- Разделы каталога (секции) — это категории/группы товаров, не путать со складами.
