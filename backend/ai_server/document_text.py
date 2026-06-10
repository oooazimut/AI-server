from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class ExtractedDocumentText:
    status: str
    text: str = ""
    reason: str = ""


BASE_SUPPORTED_EXTENSIONS = {".txt", ".csv", ".docx", ".xlsx"}
SUPPORTED_EXTENSIONS = BASE_SUPPORTED_EXTENSIONS | {".doc", ".xls", ".pdf"}


def extract_text_from_file(
    path: Path,
    *,
    original_name: str | None = None,
    max_chars: int = 40_000,
) -> ExtractedDocumentText:
    extension = _extension(original_name or path.name)
    try:
        if extension in {".txt", ".csv"}:
            text = _decode_text(path.read_bytes())
        elif extension == ".docx":
            text = _extract_docx(path)
        elif extension == ".doc":
            text = _extract_doc(path)
        elif extension == ".xlsx":
            text = _extract_xlsx(path)
        elif extension == ".xls":
            text = _extract_xls(path)
        elif extension == ".pdf":
            text = _extract_pdf_optional(path)
        else:
            return ExtractedDocumentText("unsupported", reason=f"unsupported extension {extension}")
    except zipfile.BadZipFile:
        return ExtractedDocumentText("failed", reason="invalid zip-based document")
    except Exception as exc:
        return ExtractedDocumentText("failed", reason=type(exc).__name__)

    cleaned = _clean_text(text)
    if not cleaned:
        return ExtractedDocumentText("empty", reason="no text extracted")
    return ExtractedDocumentText("indexed", text=cleaned[:max_chars])


def default_supported_extensions() -> set[str]:
    return set(SUPPORTED_EXTENSIONS)


def _extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter():
        if _local_name(paragraph.tag) != "p":
            continue
        parts: list[str] = []
        for node in paragraph.iter():
            local_name = _local_name(node.tag)
            if local_name == "t" and node.text:
                parts.append(node.text)
            elif local_name == "tab":
                parts.append("\t")
            elif local_name in {"br", "cr"}:
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def _extract_doc(path: Path) -> str:
    return _extract_binary_strings(path.read_bytes())


def _extract_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        worksheet_names = sorted(
            name for name in archive.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml")
        )
        rows: list[str] = []
        for worksheet_name in worksheet_names:
            root = ElementTree.fromstring(archive.read(worksheet_name))
            for row in root.iter():
                if _local_name(row.tag) != "row":
                    continue
                values = [
                    value
                    for cell in row
                    if _local_name(cell.tag) == "c"
                    for value in [_xlsx_cell_value(cell, shared_strings)]
                    if value
                ]
                if values:
                    rows.append(" | ".join(values))
        return "\n".join(rows)


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root:
        if _local_name(item.tag) != "si":
            continue
        strings.append("".join(node.text or "" for node in item.iter() if _local_name(node.tag) == "t").strip())
    return strings


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _local_name(node.tag) == "t").strip()

    raw_value = ""
    for child in cell:
        if _local_name(child.tag) == "v" and child.text:
            raw_value = child.text.strip()
            break

    if not raw_value:
        return ""
    if cell_type == "s" and raw_value.isdigit():
        index = int(raw_value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return raw_value


def _extract_pdf_optional(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            raise RuntimeError("pdf parser is not installed") from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_xls(path: Path) -> str:
    try:
        import xlrd
    except Exception:
        return _extract_binary_strings(path.read_bytes())

    try:
        workbook = xlrd.open_workbook(filename=str(path), on_demand=True)
    except Exception:
        return _extract_binary_strings(path.read_bytes())

    lines: list[str] = []
    try:
        for sheet in workbook.sheets():
            lines.append(sheet.name)
            for row_index in range(sheet.nrows):
                values = [
                    _format_xlrd_value(sheet.cell_value(row_index, column_index)) for column_index in range(sheet.ncols)
                ]
                row_text = " | ".join(value for value in values if value)
                if row_text:
                    lines.append(row_text)
    finally:
        workbook.release_resources()
    return "\n".join(lines)


def _format_xlrd_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _extract_binary_strings(data: bytes) -> str:
    parts: list[str] = []
    for encoding in ("utf-16le", "cp1251", "utf-8", "latin-1"):
        decoded = data.decode(encoding, errors="ignore")
        parts.extend(
            match.group(0)
            for match in re.finditer(
                r"[A-Za-zА-Яа-яЁё0-9№.,:;!?/\\|()#%+=\-\s]{4,}",
                decoded,
            )
        )
    return "\n".join(_dedupe_strings(parts))


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _clean_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", value)).strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_binary_string(value)
        if len(cleaned) < 4 or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _clean_binary_string(value: str) -> str:
    printable = []
    for char in value:
        if char in "\n\t":
            printable.append(" ")
        elif _is_allowed_binary_text_char(char):
            printable.append(char)
    cleaned = re.sub(r"\s+", " ", "".join(printable)).strip()
    if _mostly_noise(cleaned):
        return ""
    return cleaned


def _is_private_or_surrogate(char: str) -> bool:
    code = ord(char)
    return 0xD800 <= code <= 0xDFFF or 0xE000 <= code <= 0xF8FF


def _is_allowed_binary_text_char(char: str) -> bool:
    if not char.isprintable() or _is_private_or_surrogate(char):
        return False
    return bool(re.match(r"[A-Za-zА-Яа-яЁё0-9№.,:;!?/\\|()#%+=\- ]", char))


def _mostly_noise(value: str) -> bool:
    if not value:
        return True
    useful = sum(1 for char in value if _is_allowed_binary_text_char(char))
    return useful / max(len(value), 1) < 0.75


def _extension(name: str) -> str:
    return Path(name).suffix.lower()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
