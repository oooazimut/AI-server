from __future__ import annotations

from urllib.parse import urlsplit

from ai_server.integrations.bitrix.portal_search.text_utils import make_snippet
from ai_server.integrations.bitrix.portal_search.types import (
    PortalContentSyncStats,
    PortalDeltaSyncStats,
    PortalIndexStats,
    PortalSearchResult,
    PortalSyncStats,
)


def format_portal_sync_stats(stats: PortalSyncStats) -> str:
    lines = [
        "Индекс портала обновлён.",
        f"Задачи: {stats.tasks}",
        f"Проекты: {stats.projects}",
        f"Диск: {stats.disk_items} объектов в {stats.storages} хранилищах",
        f"Вложения задач: {stats.task_attachments}",
        f"Всего: {stats.total}",
    ]
    if stats.catalog_products or stats.catalog_stores or stats.catalog_stock_rows:
        lines.extend(
            [
                f"Catalog products: {stats.catalog_products}",
                f"Catalog stores: {stats.catalog_stores}",
                f"Catalog stock rows: {stats.catalog_stock_rows}",
            ]
        )
    if stats.stale_deleted:
        lines.append(f"Удалено из индекса как исчезнувшее: {stats.stale_deleted}")
    if stats.prune_skipped:
        lines.append("")
        lines.append("Удаление пропущено для неполных обходов:")
        lines.extend(f"- {item}" for item in stats.prune_skipped)
    if stats.content:
        lines.append("")
        lines.extend(format_portal_content_sync_stats(stats.content).splitlines())
    if stats.errors:
        lines.append("")
        lines.append("Есть предупреждения:")
        lines.extend(f"- {error}" for error in stats.errors)
    return "\n".join(lines)


def format_portal_content_sync_stats(stats: PortalContentSyncStats) -> str:
    lines = [
        "Содержимое документов обработано.",
        f"Кандидатов: {stats.candidates}",
        f"Скачано: {stats.downloaded}",
        f"Текст добавлен: {stats.indexed}",
        f"Пропущено: {stats.skipped}",
        f"Неподдерживаемый формат: {stats.unsupported}",
        f"Ошибок: {stats.failed}",
    ]
    if stats.errors:
        lines.append("")
        lines.append("Ошибки по файлам:")
        lines.extend(f"- {error}" for error in stats.errors[:10])
    return "\n".join(lines)


def format_portal_delta_sync_stats(stats: PortalDeltaSyncStats) -> str:
    lines = [
        "Дельта-индексация Диска выполнена.",
        f"Папок проверено: {stats.folders_scanned}",
        f"Объектов увидено: {stats.items_seen}",
        f"Изменений: {stats.items_changed}",
        f"Изменённых файлов: {stats.files_changed}",
        f"Изменённых папок: {stats.folders_changed}",
        f"Удалено из индекса: {stats.deleted}",
    ]
    if stats.wrapped:
        lines.append("Курсор дошёл до конца списка папок и начал новый круг.")
    if stats.errors:
        lines.append("")
        lines.append("Есть предупреждения:")
        lines.extend(f"- {error}" for error in stats.errors[:10])
    return "\n".join(lines)


def format_portal_search_results(results: list[PortalSearchResult], *, query: str) -> str:
    if not results:
        return f"В индексе портала ничего не нашёл по запросу: {query}"

    lines = [f"Нашёл по порталу: {len(results)}"]
    for result in results:
        label = _entity_type_label(result.entity_type)
        title = f"[{result.title}]({result.url})" if result.url else result.title
        snippet = make_snippet(result.body, query=query)
        lines.append(f"- {label}: {title}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


def format_portal_index_stats(stats: PortalIndexStats) -> str:
    if not stats.exists:
        return f"Локальный индекс портала пока не найден: {stats.path}"
    lines = [
        f"В индексе портала: {stats.total_items} объектов.",
        f"Последнее обновление: {stats.last_indexed_at or 'нет данных'}",
    ]
    for entity_type, count in sorted(stats.by_type.items()):
        lines.append(f"- {_entity_type_label(entity_type)}: {count}")
    if stats.content_by_status:
        lines.append("")
        lines.append("Текст документов:")
        for status, count in sorted(stats.content_by_status.items()):
            lines.append(f"- {_content_status_label(status)}: {count}")
    return "\n".join(lines)


def entity_types_for_scope(scope: str) -> set[str] | None:
    normalized = scope.strip().lower()
    if normalized in {"", "all"}:
        return None
    return {
        "documents": {"disk_file", "task_attachment"},
        "files": {"disk_file", "disk_folder", "disk_storage", "task_attachment"},
        "tasks": {"task", "task_attachment"},
        "projects": {"project"},
        "catalog": {"catalog_product", "catalog_store", "catalog_store_stock"},
        "stores": {"catalog_store", "catalog_store_stock"},
        "products": {"catalog_product", "catalog_store_stock"},
        "stock": {"catalog_store_stock"},
    }.get(normalized)


def _entity_type_label(entity_type: str) -> str:
    return {
        "task": "Задача",
        "task_attachment": "Вложение задачи",
        "project": "Проект",
        "disk_storage": "Хранилище",
        "disk_folder": "Папка",
        "disk_file": "Файл",
        "catalog_store": "Склад",
        "catalog_product": "Товар",
        "catalog_store_stock": "Stock",
    }.get(entity_type, entity_type)


def _content_status_label(status: str) -> str:
    return {
        "none": "ещё не брал в обработку",
        "indexed": "текст извлечён",
        "unsupported": "формат пока не поддержан",
        "too_large": "слишком большой файл",
        "empty": "текст не найден",
        "failed": "ошибка извлечения",
        "no_download_url": "нет ссылки скачивания",
    }.get(status, status)


def _domain_from_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    parts = urlsplit(cleaned)
    return parts.netloc.strip().rstrip("/")
