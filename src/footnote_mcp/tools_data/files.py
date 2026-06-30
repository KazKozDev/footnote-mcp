"""File/table extraction: HTML tables, downloads, CSV/XLSX/XLS/PDF, JSON fetch."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..fetch import _get, fetch_page

from .cache import _read_cache, _write_cache


def _fetch_bytes(url: str, lang: str = "en", timeout: int = 20) -> tuple[bytes | None, str, str | None]:
    try:
        resp = _get(url, lang=lang, timeout=timeout, max_retries=1)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code != 200:
            return None, content_type, f"HTTP {resp.status_code}"
        return bytes(resp.content), content_type, None
    except Exception as exc:
        return None, "", str(exc)


def _table_to_rows(table) -> dict:
    caption_el = table.find("caption")
    caption = caption_el.get_text(" ", strip=True) if caption_el else ""
    rows = []
    max_cols = 0
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [cell.get_text(" ", strip=True) for cell in cells]
        if row:
            rows.append(row)
            max_cols = max(max_cols, len(row))
    if not rows:
        return {"caption": caption, "columns": [], "rows": []}

    first = rows[0]
    header_like = table.find("th") is not None or len(set(first)) == len(first)
    if header_like:
        columns = [col or f"column_{idx + 1}" for idx, col in enumerate(first)]
        data_rows = rows[1:]
    else:
        columns = [f"column_{idx + 1}" for idx in range(max_cols)]
        data_rows = rows

    normalized = []
    for row in data_rows:
        padded = row + [""] * (len(columns) - len(row))
        normalized.append({columns[idx]: padded[idx] for idx in range(len(columns))})
    return {"caption": caption, "columns": columns, "rows": normalized}


def web_extract_tables(url: str, lang: str = "en", max_tables: int = 8, max_rows: int = 80, use_cache: bool = True) -> dict:
    cached = _read_cache(url) if use_cache else None
    if cached and "tables" in cached:
        return {"url": url, "cached": True, "tables": cached["tables"], "table_count": len(cached["tables"])}

    fetched_url, html, pub_date, error = fetch_page(url, lang=lang)
    if error or not html:
        result = {"url": url, "error": error or "Empty response body", "tables": [], "table_count": 0}
        _write_cache(url, result)
        return result

    soup = BeautifulSoup(html, "html.parser")
    tables = []
    for idx, table in enumerate(soup.find_all("table")[:max_tables], 1):
        parsed = _table_to_rows(table)
        parsed["table_index"] = idx
        parsed["rows"] = parsed["rows"][:max_rows]
        parsed["row_count"] = len(parsed["rows"])
        tables.append(parsed)

    result = {
        "url": fetched_url,
        "published": str(pub_date) if pub_date else None,
        "tables": tables,
        "table_count": len(tables),
        "cached": False,
    }
    _write_cache(url, result)
    return result


def web_detect_downloads(url: str, lang: str = "en", max_links: int = 50) -> dict:
    fetched_url, html, _, error = fetch_page(url, lang=lang)
    if error or not html:
        return {"url": url, "error": error or "Empty response body", "downloads": []}

    soup = BeautifulSoup(html, "html.parser")
    exts = (".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".json", ".xml")
    downloads = []
    for a in soup.find_all("a", href=True):
        href = urljoin(fetched_url, a["href"])
        text = a.get_text(" ", strip=True)
        path = urlparse(href).path.lower()
        if path.endswith(exts) or any(token in text.lower() for token in ("csv", "xlsx", "excel", "pdf", "download")):
            downloads.append({"url": href, "text": text, "extension": Path(path).suffix.lower()})
        if len(downloads) >= max_links:
            break
    return {"url": fetched_url, "downloads": downloads, "count": len(downloads)}


def _parse_csv_bytes(data: bytes, max_rows: int) -> dict:
    text = data.decode("utf-8-sig", errors="replace")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = []
    for row in reader:
        rows.append(dict(row))
        if len(rows) >= max_rows:
            break
    return {"tables": [{"table_index": 1, "columns": reader.fieldnames or [], "rows": rows, "row_count": len(rows)}]}


def _parse_xlsx_bytes(data: bytes, max_rows: int) -> dict:
    try:
        import openpyxl
    except ImportError:
        return {"error": "openpyxl is not installed; install requirements.txt to parse XLS/XLSX files"}

    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    tables = []
    for sheet_index, sheet in enumerate(workbook.worksheets, 1):
        rows_iter = sheet.iter_rows(values_only=True)
        try:
            header = next(rows_iter)
        except StopIteration:
            continue
        columns = [str(value) if value is not None else f"column_{idx + 1}" for idx, value in enumerate(header)]
        rows = []
        for values in rows_iter:
            rows.append({columns[idx]: values[idx] if idx < len(values) else None for idx in range(len(columns))})
            if len(rows) >= max_rows:
                break
        tables.append({"table_index": sheet_index, "sheet": sheet.title, "columns": columns, "rows": rows, "row_count": len(rows)})
    return {"tables": tables}


def _parse_xls_bytes(data: bytes, max_rows: int) -> dict:
    try:
        import xlrd
    except ImportError:
        return {"error": "xlrd is not installed; install requirements.txt to parse legacy XLS files"}

    workbook = xlrd.open_workbook(file_contents=data)
    tables = []
    for sheet_index in range(workbook.nsheets):
        sheet = workbook.sheet_by_index(sheet_index)
        if sheet.nrows == 0:
            continue
        header = sheet.row_values(0)
        columns = [str(value) if value not in (None, "") else f"column_{idx + 1}" for idx, value in enumerate(header)]
        rows = []
        for row_idx in range(1, min(sheet.nrows, max_rows + 1)):
            values = sheet.row_values(row_idx)
            rows.append({columns[idx]: values[idx] if idx < len(values) else None for idx in range(len(columns))})
        tables.append(
            {
                "table_index": sheet_index + 1,
                "sheet": sheet.name,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
        )
    return {"tables": tables}


def _parse_pdf_bytes(data: bytes, max_rows: int) -> dict:
    try:
        import pdfplumber

        tables = []
        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_index, page in enumerate(pdf.pages[:10], 1):
                text = page.extract_text() or ""
                pages.append({"page": page_index, "text": text[:4000]})
                for raw_table in page.extract_tables() or []:
                    if not raw_table:
                        continue
                    header = raw_table[0] or []
                    columns = [str(value).strip() if value not in (None, "") else f"column_{idx + 1}" for idx, value in enumerate(header)]
                    rows = []
                    for raw_row in raw_table[1 : max_rows + 1]:
                        row_values = raw_row or []
                        rows.append({columns[idx]: row_values[idx] if idx < len(row_values) else "" for idx in range(len(columns))})
                    tables.append(
                        {
                            "table_index": len(tables) + 1,
                            "page": page_index,
                            "columns": columns,
                            "rows": rows,
                            "row_count": len(rows),
                        }
                    )
        if not any(page.get("text") for page in pages):
            ocr = _ocr_pdf_bytes(data, max_pages=3)
            if ocr.get("pages"):
                pages = ocr["pages"]
        return {"pages": pages, "page_count": len(pages), "tables": tables}
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"error": "pypdf is not installed; install requirements.txt to parse PDF text"}

    reader = PdfReader(io.BytesIO(data))
    pages = []
    rows = []
    for idx, page in enumerate(reader.pages[:10], 1):
        text = page.extract_text() or ""
        pages.append({"page": idx, "text": text[:4000]})
        for line_number, line in enumerate(text.splitlines()[:max_rows], 1):
            if line.strip():
                rows.append({"page": idx, "line_number": line_number, "text": line.strip()})
    if not any(page.get("text") for page in pages):
        ocr = _ocr_pdf_bytes(data, max_pages=3)
        if ocr.get("pages"):
            pages = ocr["pages"]
            rows = []
            for page in pages:
                for line_number, line in enumerate(page.get("text", "").splitlines()[:max_rows], 1):
                    if line.strip():
                        rows.append({"page": page["page"], "line_number": line_number, "text": line.strip()})
    return {
        "pages": pages,
        "page_count": len(reader.pages),
        "tables": [
            {
                "table_index": 1,
                "columns": ["page", "line_number", "text"],
                "rows": rows[:max_rows],
                "row_count": min(len(rows), max_rows),
                "fallback": "pdf_text_lines",
            }
        ],
    }


def _ocr_pdf_bytes(data: bytes, max_pages: int = 3) -> dict:
    try:
        import pdfplumber
        import pytesseract
    except ImportError:
        return {"ocr_available": False, "pages": [], "reason": "pdfplumber or pytesseract is not installed"}

    pages = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_index, page in enumerate(pdf.pages[:max_pages], 1):
                image = page.to_image(resolution=200).original
                text = pytesseract.image_to_string(image) or ""
                pages.append({"page": page_index, "text": text[:4000], "extraction": "ocr"})
        return {"ocr_available": True, "pages": pages}
    except Exception as exc:
        return {"ocr_available": False, "pages": [], "reason": str(exc)}


def web_parse_file(url: str, lang: str = "en", max_rows: int = 200, use_cache: bool = True) -> dict:
    cached = _read_cache(url) if use_cache else None
    if cached and cached.get("parsed_file"):
        return {"url": url, "cached": True, **cached["parsed_file"]}

    data, content_type, error = _fetch_bytes(url, lang=lang)
    if error or data is None:
        return {"url": url, "error": error or "Download failed", "content_type": content_type}

    path = urlparse(url).path.lower()
    if path.endswith((".csv", ".tsv")) or "csv" in content_type:
        parsed = _parse_csv_bytes(data, max_rows=max_rows)
        file_type = "csv"
    elif path.endswith(".xlsx") or "spreadsheet" in content_type:
        parsed = _parse_xlsx_bytes(data, max_rows=max_rows)
        file_type = "xlsx"
    elif path.endswith(".xls") or "excel" in content_type:
        parsed = _parse_xls_bytes(data, max_rows=max_rows)
        file_type = "xls"
    elif path.endswith(".pdf") or "pdf" in content_type:
        parsed = _parse_pdf_bytes(data, max_rows=max_rows)
        file_type = "pdf"
    elif path.endswith(".json") or "json" in content_type:
        parsed = {"json": json.loads(data.decode("utf-8", errors="replace"))}
        file_type = "json"
    else:
        parsed = {"error": f"Unsupported or unknown file type: {content_type or path}"}
        file_type = "unknown"

    result = {"url": url, "file_type": file_type, "content_type": content_type, **parsed}
    _write_cache(url, {"parsed_file": result})
    return result


def web_fetch_json(url: str, lang: str = "en", use_cache: bool = True, timeout: int = 20) -> dict:
    cached = _read_cache(url) if use_cache else None
    if cached and cached.get("fetched_json"):
        return {"url": url, **cached["fetched_json"], "cached": True}

    data, content_type, error = _fetch_bytes(url, lang=lang, timeout=timeout)
    if error or data is None:
        result = {"url": url, "error": error or "Download failed", "content_type": content_type}
        _write_cache(url, {"fetched_json": result})
        return result

    try:
        parsed = json.loads(data.decode("utf-8-sig", errors="replace"))
    except json.JSONDecodeError as exc:
        result = {
            "url": url,
            "error": f"Invalid JSON: {exc}",
            "content_type": content_type,
            "text_sample": data[:1000].decode("utf-8", errors="replace"),
        }
        _write_cache(url, {"fetched_json": result})
        return result

    result = {"url": url, "content_type": content_type, "json": parsed, "cached": False}
    _write_cache(url, {"fetched_json": result})
    return result
