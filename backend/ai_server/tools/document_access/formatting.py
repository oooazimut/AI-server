from __future__ import annotations

import math

from ai_server.integrations.bitrix.portal_search import PortalSearchResult
from ai_server.tools.document_access.types import DocumentCompareReport, FieldDifference


def format_document_comparison_report(report: DocumentCompareReport, *, limit: int = 20) -> str:
    if report.errors:
        lines = list(report.errors)
        candidates = report.first_candidates or report.second_candidates
        if candidates:
            lines.extend(["", "Похожие табличные документы:"])
            lines.extend(f"- {_document_link(item)} ({item.entity_type} #{item.entity_id})" for item in candidates[:8])
        return "\n".join(lines)
    if not report.first or not report.second:
        return "Не смог сравнить документы: не определил оба файла."

    lines = [
        "Сравнил документы:",
        f"1. {_format_document_link(report.first.title, report.first.url)}",
        f"2. {_format_document_link(report.second.title, report.second.url)}",
        "",
        f"Совпавших позиций: {report.common_rows}",
        f"Отличий по значениям: {len(report.changed)}",
        f"Только в первом документе: {len(report.only_first)}",
        f"Только во втором документе: {len(report.only_second)}",
    ]
    if report.changed:
        lines.extend(["", "Изменившиеся позиции:"])
        for difference in report.changed[:limit]:
            lines.append(
                f"- {difference.key} "
                f"({difference.first_sheet}!{difference.first_row} -> {difference.second_sheet}!{difference.second_row})"
            )
            for field in difference.fields:
                lines.append(f"  {field.first.label}: {_format_field_difference(field)}")
    if report.only_first:
        lines.extend(["", "Есть только в первом документе:"])
        lines.extend(f"- {entry.key} ({entry.sheet}!{entry.row_number})" for entry in report.only_first[:limit])
    if report.only_second:
        lines.extend(["", "Есть только во втором документе:"])
        lines.extend(f"- {entry.key} ({entry.sheet}!{entry.row_number})" for entry in report.only_second[:limit])
    if not report.changed and not report.only_first and not report.only_second:
        lines.extend(["", "По распознанным табличным позициям различий не нашёл."])
    return "\n".join(lines)


def _format_field_difference(difference: FieldDifference) -> str:
    if difference.first.number is not None and difference.second.number is not None:
        delta = _format_signed_number(difference.delta or 0)
        percent = f"; {_format_signed_number(difference.percent)}%" if difference.percent is not None else ""
        return f"{_format_number(difference.first.number)} -> {_format_number(difference.second.number)} ({delta}{percent})"
    return f"{difference.first.raw} -> {difference.second.raw}"


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=0.001):
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _format_signed_number(value: float | None) -> str:
    number = value or 0.0
    return ("+" if number > 0 else "") + _format_number(number)


def _document_link(item: PortalSearchResult) -> str:
    return _format_document_link(item.title, item.url)


def _format_document_link(title: str, url: str) -> str:
    return f"[{title}]({url})" if url else title
