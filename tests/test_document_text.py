"""Tests for document_text.extract_text_from_file."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ai_server.document_text import (
    SUPPORTED_EXTENSIONS,
    extract_text_from_file,
)

# ---------------------------------------------------------------------------
# Helpers to create test files
# ---------------------------------------------------------------------------


def _write_txt(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def _write_docx(path: Path, paragraphs: list[str]) -> Path:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    root = ElementTree.Element(f"{{{ns}}}document")
    body = ElementTree.SubElement(root, f"{{{ns}}}body")
    for text in paragraphs:
        p = ElementTree.SubElement(body, f"{{{ns}}}p")
        r = ElementTree.SubElement(p, f"{{{ns}}}r")
        t = ElementTree.SubElement(r, f"{{{ns}}}t")
        t.text = text
    xml_bytes = ElementTree.tostring(root, encoding="unicode").encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml_bytes)
    return path


def _write_xlsx(path: Path, rows: list[list[str]]) -> Path:
    ns_ss = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ElementTree.Element(f"{{{ns_ss}}}workbook")
    sheets_el = ElementTree.SubElement(root, f"{{{ns_ss}}}sheets")
    ElementTree.SubElement(sheets_el, f"{{{ns_ss}}}sheet", attrib={"name": "Sheet1", "sheetId": "1", "r:id": "rId1"})

    ws_root = ElementTree.Element(f"{{{ns_ss}}}worksheet")
    sheet_data = ElementTree.SubElement(ws_root, f"{{{ns_ss}}}sheetData")
    for row_vals in rows:
        row_el = ElementTree.SubElement(sheet_data, f"{{{ns_ss}}}row")
        for val in row_vals:
            c = ElementTree.SubElement(row_el, f"{{{ns_ss}}}c", attrib={"t": "inlineStr"})
            is_el = ElementTree.SubElement(c, f"{{{ns_ss}}}is")
            t_el = ElementTree.SubElement(is_el, f"{{{ns_ss}}}t")
            t_el.text = val

    ws_xml = ElementTree.tostring(ws_root, encoding="unicode").encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", ws_xml)
        archive.writestr("[Content_Types].xml", b"")
    return path


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def test_extract_txt_utf8(tmp_path):
    f = _write_txt(tmp_path / "doc.txt", "Привет мир!\nВторая строка.")
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "Привет" in result.text


def test_extract_txt_cp1251(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_bytes("Текст кириллицей на cp1251".encode("cp1251"))
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "кириллицей" in result.text


def test_extract_txt_bom_utf8(tmp_path):
    f = tmp_path / "bom.txt"
    f.write_bytes("﻿Текст с BOM".encode("utf-8-sig"))
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "BOM" in result.text


def test_extract_txt_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    result = extract_text_from_file(f)
    assert result.status == "empty"


def test_extract_txt_whitespace_only(tmp_path):
    f = _write_txt(tmp_path / "spaces.txt", "   \n\n   ")
    result = extract_text_from_file(f)
    assert result.status == "empty"


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_extract_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_bytes("Имя;Возраст\nИван;30\nОлег;25\n".encode())
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "Иван" in result.text


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_extract_docx_paragraphs(tmp_path):
    f = _write_docx(tmp_path / "doc.docx", ["Первый абзац", "Второй абзац"])
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "Первый абзац" in result.text
    assert "Второй абзац" in result.text


def test_extract_docx_bad_zip(tmp_path):
    f = tmp_path / "bad.docx"
    f.write_bytes(b"not a zip file content")
    result = extract_text_from_file(f)
    assert result.status == "failed"


def test_extract_docx_empty_paragraphs(tmp_path):
    f = _write_docx(tmp_path / "empty.docx", [])
    result = extract_text_from_file(f)
    assert result.status == "empty"


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def test_extract_xlsx_rows(tmp_path):
    f = _write_xlsx(tmp_path / "data.xlsx", [["Наименование", "Сумма"], ["Кабель", "1000"]])
    result = extract_text_from_file(f)
    assert result.status == "indexed"
    assert "Наименование" in result.text
    assert "Кабель" in result.text


def test_extract_xlsx_bad_zip(tmp_path):
    f = tmp_path / "bad.xlsx"
    f.write_bytes(b"not xlsx")
    result = extract_text_from_file(f)
    assert result.status == "failed"


# ---------------------------------------------------------------------------
# Unsupported / unknown extension
# ---------------------------------------------------------------------------


def test_extract_unsupported_extension(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_bytes(b"some data")
    result = extract_text_from_file(f)
    assert result.status == "unsupported"


def test_extract_uses_original_name_extension(tmp_path):
    f = tmp_path / "tmpfile_no_ext"
    f.write_bytes("Содержимое файла txt.".encode())
    result = extract_text_from_file(f, original_name="doc.txt")
    assert result.status == "indexed"


# ---------------------------------------------------------------------------
# max_chars truncation
# ---------------------------------------------------------------------------


def test_extract_truncates_at_max_chars(tmp_path):
    long_text = "А" * 10_000
    f = _write_txt(tmp_path / "long.txt", long_text)
    result = extract_text_from_file(f, max_chars=100)
    assert result.status == "indexed"
    assert len(result.text) <= 100


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS constant
# ---------------------------------------------------------------------------


def test_supported_extensions_includes_common_formats():
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
