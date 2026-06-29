"""Structured data tools for source-grounded research."""

from __future__ import annotations

import csv
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from fetch import _get, fetch_page


CACHE_DIR = Path(os.getenv("FOOTNOTE_SOURCE_CACHE", "~/.footnote-mcp/source_cache")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECIPE_STORE_PATH = Path(os.getenv("FOOTNOTE_RECIPE_STORE", "~/.footnote-mcp/extraction_recipes.json")).expanduser()


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{_cache_key(url)}.json"


def _read_cache(url: str) -> dict | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cache(url: str, data: dict) -> None:
    existing = _read_cache(url) or {}
    payload = dict(existing)
    payload.update(data)
    payload["cached_at"] = datetime.now().isoformat(timespec="seconds")
    payload["url"] = url
    _cache_path(url).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


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


def _recipe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:80] or "extraction-recipe"


def tool_spec_propose(
    task: str,
    source_url: str = "",
    observed_failure: str = "",
    desired_output: str = "rows with date, value, unit, and source_url",
) -> dict:
    name = _recipe_slug(task)
    return {
        "name": name,
        "task": task,
        "source_url": source_url,
        "observed_failure": observed_failure,
        "why_existing_tools_insufficient": observed_failure or "Generic extraction did not return structured rows.",
        "code_contract": {
            "entrypoint": "extract(source_text, input_payload)",
            "input_payload": {
                "source_url": "optional provenance URL",
                "expected_unit_or_pair": "optional expected unit or currency pair",
                "notes": "optional task-specific notes",
            },
            "output_schema": {
                "rows": [{"date": "YYYY-MM-DD", "value": "number or string", "unit": "string", "source_url": "string"}],
                "row_count": "integer",
                "notes": "optional string",
            },
        },
        "safety_contract": {
            "allowed_imports": sorted(_ALLOWED_RECIPE_IMPORTS),
            "forbidden": [
                "filesystem reads/writes",
                "subprocesses",
                "network access from recipe code",
                "environment/secrets access",
                "dynamic code execution",
            ],
        },
        "desired_output": desired_output,
        "next_steps": ["tool_code_generate", "tool_code_validate", "tool_code_run_sandboxed", "tool_promote"],
    }


def tool_code_generate(spec: dict | None = None) -> dict:
    spec = spec or {}
    source_url = spec.get("source_url", "")
    code = '''import re


def extract(source_text, input_payload):
    source_url = input_payload.get("source_url", "")
    expected_unit = input_payload.get("expected_unit_or_pair", "")
    rows = []
    pattern = re.compile(
        r"(?P<date>\\d{4}-\\d{2}-\\d{2})[^\\n\\d-]{0,80}(?P<value>-?\\d+(?:[.,]\\d+)?)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(source_text or ""):
        rows.append(
            {
                "date": match.group("date"),
                "value": match.group("value").replace(",", "."),
                "unit": expected_unit,
                "source_url": source_url,
            }
        )
    return {"rows": rows, "row_count": len(rows), "notes": "regex date/value extraction"}
'''
    return {
        "spec": spec,
        "code": code,
        "entrypoint": "extract(source_text, input_payload)",
        "source_url": source_url,
        "validation": tool_code_validate(code),
    }


_ALLOWED_RECIPE_IMPORTS = {
    "csv",
    "datetime",
    "html",
    "json",
    "math",
    "re",
    "statistics",
}
_FORBIDDEN_RECIPE_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
_FORBIDDEN_RECIPE_NODES = (ast.AsyncFunctionDef, ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith)


def _import_root(name: str) -> str:
    return (name or "").split(".", 1)[0]


def tool_code_validate(code: str, max_chars: int = 12000) -> dict:
    errors = []
    warnings = []
    if not isinstance(code, str) or not code.strip():
        return {"valid": False, "errors": ["code is empty"], "warnings": [], "allowed_imports": sorted(_ALLOWED_RECIPE_IMPORTS)}
    if len(code) > max_chars:
        errors.append(f"code exceeds max_chars={max_chars}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "valid": False,
            "errors": [f"syntax error: {exc}"],
            "warnings": warnings,
            "allowed_imports": sorted(_ALLOWED_RECIPE_IMPORTS),
        }

    has_extract = False
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_RECIPE_NODES):
            errors.append(f"forbidden AST node: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node.name == "extract":
            has_extract = True
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _FORBIDDEN_RECIPE_CALLS:
                errors.append(f"forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                errors.append(f"forbidden attribute: {node.attr}")
            if root not in _ALLOWED_RECIPE_IMPORTS:
                errors.append(f"forbidden module attribute access: {root}.{node.attr}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_RECIPE_CALLS:
                errors.append(f"forbidden call: {func.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _import_root(alias.name)
                if root not in _ALLOWED_RECIPE_IMPORTS:
                    errors.append(f"import not allowed: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            root = _import_root(node.module or "")
            if root not in _ALLOWED_RECIPE_IMPORTS:
                errors.append(f"import-from not allowed: {node.module}")
    if not has_extract:
        errors.append("missing required function: extract(source_text, input_payload)")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "allowed_imports": sorted(_ALLOWED_RECIPE_IMPORTS),
        "entrypoint": "extract(source_text, input_payload)",
    }


def _recipe_preexec_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    except Exception:
        return


def tool_code_run_sandboxed(
    code: str,
    source_text: str = "",
    input_payload: dict | None = None,
    timeout: int = 5,
    max_output_chars: int = 20000,
) -> dict:
    validation = tool_code_validate(code)
    if not validation.get("valid"):
        return {"ok": False, "validation": validation, "error": "validation failed"}

    wrapper = r'''
import builtins
import importlib
import json
import sys
import traceback

ALLOWED_IMPORTS = set(__ALLOWED_IMPORTS__)
payload = json.loads(sys.stdin.read())
code = payload["code"]
source_text = payload.get("source_text", "")
input_payload = payload.get("input_payload") or {}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = (name or "").split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import not allowed: {name}")
    return importlib.import_module(name)

safe_builtins = {
    "__import__": safe_import,
    "abs": builtins.abs,
    "all": builtins.all,
    "any": builtins.any,
    "bool": builtins.bool,
    "dict": builtins.dict,
    "enumerate": builtins.enumerate,
    "float": builtins.float,
    "int": builtins.int,
    "isinstance": builtins.isinstance,
    "len": builtins.len,
    "list": builtins.list,
    "max": builtins.max,
    "min": builtins.min,
    "range": builtins.range,
    "round": builtins.round,
    "set": builtins.set,
    "sorted": builtins.sorted,
    "str": builtins.str,
    "sum": builtins.sum,
    "tuple": builtins.tuple,
    "zip": builtins.zip,
}
namespace = {"__builtins__": safe_builtins}
try:
    exec(compile(code, "<recipe>", "exec"), namespace)
    extract = namespace.get("extract")
    if not callable(extract):
        raise RuntimeError("extract is not callable")
    result = extract(source_text, input_payload)
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=3)}, ensure_ascii=False))
    sys.exit(1)
'''.replace("__ALLOWED_IMPORTS__", repr(sorted(_ALLOWED_RECIPE_IMPORTS)))
    payload = {"code": code, "source_text": source_text, "input_payload": input_payload or {}}
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", wrapper],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=max(1, min(int(timeout), 15)),
            preexec_fn=_recipe_preexec_limits if os.name == "posix" else None,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "validation": validation, "error": "recipe timed out"}

    stdout = (completed.stdout or "")[:max_output_chars]
    stderr = (completed.stderr or "")[:4000]
    try:
        parsed = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        parsed = {"ok": False, "error": "recipe returned non-JSON output", "stdout": stdout}
    parsed["validation"] = validation
    parsed["returncode"] = completed.returncode
    if stderr:
        parsed["stderr"] = stderr
    return parsed


def _load_recipe_store() -> dict:
    if not RECIPE_STORE_PATH.exists():
        return {"recipes": {}}
    try:
        data = json.loads(RECIPE_STORE_PATH.read_text())
        return data if isinstance(data, dict) and isinstance(data.get("recipes"), dict) else {"recipes": {}}
    except Exception:
        return {"recipes": {}}


def _save_recipe_store(data: dict) -> None:
    RECIPE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECIPE_STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _recipe_rows(result) -> list:
    """Normalize a recipe's return value into a list of rows.

    A recipe's ``extract()`` may return either a bare list of rows or a dict
    such as ``{"rows": [...]}`` (the shape produced by the generated template).
    """
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []
    return []


def tool_promote(
    name: str,
    spec: dict | None,
    code: str,
    sample_source_text: str = "",
    input_payload: dict | None = None,
    expected_min_rows: int = 0,
) -> dict:
    validation = tool_code_validate(code)
    if not validation.get("valid"):
        return {"promoted": False, "validation": validation, "error": "validation failed"}
    smoke_result = None
    if sample_source_text:
        smoke_result = tool_code_run_sandboxed(code, sample_source_text, input_payload or {}, timeout=5)
        rows = _recipe_rows(smoke_result.get("result")) if smoke_result.get("ok") else []
        if not smoke_result.get("ok") or len(rows) < expected_min_rows:
            return {
                "promoted": False,
                "validation": validation,
                "smoke_result": smoke_result,
                "error": f"smoke test produced fewer than expected_min_rows={expected_min_rows}",
            }

    store = _load_recipe_store()
    recipe_id = _recipe_slug(name) + "-" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:10]
    store["recipes"][recipe_id] = {
        "id": recipe_id,
        "name": name,
        "spec": spec or {},
        "code": code,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "plays": 0,
        "wins": 0,
    }
    _save_recipe_store(store)
    return {"promoted": True, "recipe_id": recipe_id, "validation": validation, "smoke_result": smoke_result}


def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    return date(year, month, 1)


def _period_key(value: date, granularity: str) -> str:
    if granularity == "day":
        return value.isoformat()
    if granularity == "week":
        iso = value.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if granularity == "month":
        return f"{value.year:04d}-{value.month:02d}"
    return value.isoformat()


def _extract_periods(text: str, granularity: str) -> set[str]:
    periods = set()
    if granularity == "week":
        periods.update(match.upper() for match in re.findall(r"\d{4}-W\d{2}", text, flags=re.IGNORECASE))
    if granularity == "month":
        periods.update(re.findall(r"\d{4}-\d{2}", text))
    for match in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        parsed = date.fromisoformat(match)
        periods.add(_period_key(parsed, granularity))
    return periods


def _calendar_holidays(calendar: str, start: date, end: date) -> set[str]:
    try:
        import holidays as holidays_lib
    except ImportError:
        return set()

    calendar_map = {
        "us_business_day": "US",
        "ru_business_day": "RU",
    }
    country = calendar_map.get(calendar)
    if not country:
        return set()
    years = range(start.year, end.year + 1)
    try:
        return {day.isoformat() for day in holidays_lib.country_holidays(country, years=years)}
    except Exception:
        return set()


def check_date_completeness(
    start_date: str,
    end_date: str,
    actual_items: list[str],
    granularity: str = "day",
    calendar: str = "calendar",
    holidays: list[str] | None = None,
) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        return {"error": "end_date is before start_date"}
    if granularity not in {"day", "week", "month"}:
        return {"error": "Supported granularities: day, week, month"}
    supported_calendars = {"calendar", "business_day", "crypto_24_7", "forex_weekday", "us_business_day", "ru_business_day"}
    if calendar not in supported_calendars:
        return {"error": f"Supported calendars: {', '.join(sorted(supported_calendars))}"}
    holiday_set = set(holidays or []) | _calendar_holidays(calendar, start, end)

    expected = []
    if granularity == "month":
        current = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)
        while current <= end_month:
            expected.append(_period_key(current, granularity))
            current = _add_month(current)
    else:
        current = start
        seen = set()
        step = timedelta(days=1 if granularity == "day" else 7)
        while current <= end:
            key = _period_key(current, granularity)
            include = True
            if granularity == "day" and calendar in {"business_day", "forex_weekday", "us_business_day", "ru_business_day"}:
                include = current.weekday() < 5 and current.isoformat() not in holiday_set
            if include and key not in seen:
                expected.append(key)
                seen.add(key)
            current += step

    actual = set()
    for item in actual_items:
        text = str(item)
        actual.update(_extract_periods(text, granularity))
    missing = [item for item in expected if item not in actual]
    extra = sorted(actual - set(expected))
    return {
        "start_date": start_date,
        "end_date": end_date,
        "granularity": granularity,
        "calendar": calendar,
        "holidays": sorted(holiday_set),
        "expected_count": len(expected),
        "actual_count": len(actual & set(expected)),
        "missing_items": missing,
        "extra_items": extra,
        "complete": not missing,
    }


def classify_source(url: str, status_code: int | None = None, content_type: str = "", text_sample: str = "") -> dict:
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    source_type = "aggregator"
    reasons = []
    official_tlds = (".gov", ".int")
    forum_hosts = ("reddit.", "stackoverflow.", "quora.", "forum", "bitcointalk")
    blog_markers = ("blog", "medium.com", "substack.com", "wordpress", "blogspot")

    if host.endswith(official_tlds):
        source_type = "official"
        reasons.append("official domain")
    elif any(marker in host for marker in forum_hosts):
        source_type = "forum"
        reasons.append("forum/community host")
    elif any(marker in host or marker in path for marker in blog_markers):
        source_type = "blog"
        reasons.append("blog-like host/path")
    if status_code and status_code >= 400:
        source_type = "blocked" if status_code in (401, 403, 429) else "error"
        reasons.append(f"HTTP {status_code}")
    if "javascript" in text_sample.lower() and "enable" in text_sample.lower():
        source_type = "interactive"
        reasons.append("requires JavaScript")
    return {"url": url, "host": host, "source_type": source_type, "reasons": reasons}


def generate_search_queries(task: str, requirements: dict | None = None, max_queries: int = 8) -> dict:
    requirements = requirements or {}
    target = requirements.get("target") or task
    unit = requirements.get("unit_or_pair") or ""
    hints = requirements.get("search_hints") or []
    base = f"{target} {unit}".strip()
    templates = [
        "{base} official data",
        "{base} historical data table",
        "{base} csv",
        "{base} xlsx",
        "{base} API",
        "{base} filetype:csv",
        "{base} filetype:xlsx",
        "{base} filetype:json",
        "{base} downloadable dataset",
    ]
    queries = []
    for hint in hints:
        if hint and hint not in queries:
            queries.append(str(hint))
    for template in templates:
        query = re.sub(r"\s+", " ", template.format(base=base)).strip()
        if query not in queries:
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return {"task": task, "queries": queries[:max_queries], "count": len(queries[:max_queries])}


def resolve_units(text: str) -> dict:
    upper = text.upper()
    aliases = {
        "EURO": "EUR",
        "EUROS": "EUR",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
        "ROUBLE": "RUB",
        "ROUBLES": "RUB",
        "RUBLE": "RUB",
        "RUBLES": "RUB",
        "POUND": "GBP",
        "YEN": "JPY",
        "YUAN": "CNY",
    }
    for word, code in aliases.items():
        upper = re.sub(rf"\b{word}\b", code, upper)
    slash_pairs = re.findall(r"\b([A-Z]{3})\s*/\s*([A-Z]{3})\b", upper)
    compact_pairs = re.findall(r"\b(USD|EUR|RUB|GBP|JPY|CNY|CHF|BTC|ETH|USDT)(USD|EUR|RUB|GBP|JPY|CNY|CHF|BTC|ETH|USDT)\b", upper)
    to_pairs = re.findall(r"\b(USD|EUR|RUB|GBP|JPY|CNY|CHF|BTC|ETH|USDT)\s+(?:TO|IN|PER)\s+(USD|EUR|RUB|GBP|JPY|CNY|CHF|BTC|ETH|USDT)\b", upper)
    pair_tuples = slash_pairs + compact_pairs + to_pairs
    pairs = sorted({f"{base}/{quote}" for base, quote in pair_tuples if base != quote})
    currencies = sorted(set(re.findall(r"\b(USD|EUR|RUB|GBP|JPY|CNY|CHF|BTC|ETH|USDT)\b", upper)))
    units = []
    for marker in ["%", "PERCENT", "KG", "KILOGRAM", "USD", "EUR", "RUB", "BTC", "RATE", "INDEX"]:
        if marker in upper:
            units.append(marker)
    primary_pair = pairs[0] if pairs else None
    base_currency, quote_currency = primary_pair.split("/") if primary_pair else (None, None)
    return {
        "currency_pairs": pairs,
        "primary_pair": primary_pair,
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "currencies": currencies,
        "units": sorted(set(units)),
        "text_sample": text[:500],
    }


def validate_unit_rows(rows: list[dict], expected_unit_or_pair: str, text_fields: list[str] | None = None) -> dict:
    expected = resolve_units(expected_unit_or_pair)
    expected_pair = expected.get("primary_pair")
    expected_currencies = set(expected.get("currencies", []))
    accepted = []
    rejected = []
    for idx, row in enumerate(rows or [], 1):
        if not isinstance(row, dict):
            rejected.append({"row_index": idx, "row": row, "reason": "row is not an object"})
            continue
        fields = text_fields or list(row.keys())
        row_text = " ".join(str(row.get(field, "")) for field in fields)
        row_units = resolve_units(row_text)
        row_pairs = set(row_units.get("currency_pairs", []))
        row_currencies = set(row_units.get("currencies", []))
        compatible = True
        reason = "compatible"
        if expected_pair:
            if row_pairs and expected_pair not in row_pairs:
                compatible = False
                reason = f"expected {expected_pair}, found {sorted(row_pairs)}"
            elif not row_pairs and row_currencies and not expected_currencies.issubset(row_currencies):
                compatible = False
                reason = f"expected currencies {sorted(expected_currencies)}, found {sorted(row_currencies)}"
        elif expected_currencies and row_currencies and not row_currencies.issubset(expected_currencies):
            compatible = False
            reason = f"unexpected currencies {sorted(row_currencies - expected_currencies)}"
        item = {"row_index": idx, "row": row, "detected": row_units, "reason": reason}
        if compatible:
            accepted.append(item)
        else:
            rejected.append(item)
    return {
        "expected": expected,
        "accepted_rows": [item["row"] for item in accepted],
        "rejected_rows": rejected,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
    return {}


def _heuristic_entailment(claim: str, source_excerpt: str) -> dict:
    claim_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", claim)}
    source_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", source_excerpt)}
    if not claim_tokens:
        return {"status": "unsupported", "score": 0.0, "reason": "empty claim", "backend": "heuristic"}
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", claim))
    source_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", source_excerpt))
    missing_numbers = [number for number in numbers if number not in source_numbers]
    claim_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", claim))
    source_dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", source_excerpt))
    if claim_dates and claim_dates & source_dates and missing_numbers and source_numbers:
        return {
            "status": "contradicted",
            "score": round(overlap, 3),
            "reason": f"same dated evidence contains different numbers: {missing_numbers}",
            "backend": "heuristic",
        }
    if missing_numbers:
        return {
            "status": "unsupported",
            "score": round(overlap, 3),
            "reason": f"numbers missing from source: {missing_numbers}",
            "backend": "heuristic",
        }
    if overlap >= 0.75:
        status = "supported"
    elif overlap >= 0.45:
        status = "partially_supported"
    else:
        status = "unsupported"
    return {"status": status, "score": round(overlap, 3), "reason": "token overlap heuristic", "backend": "heuristic"}


def _local_nli_entailment(claim: str, source_excerpt: str, model: str | None = None) -> dict:
    model = model or os.getenv("FOOTNOTE_NLI_MODEL") or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    try:
        from transformers import pipeline
    except ImportError:
        return {"error": "transformers is not installed; install requirements-nli.txt", "backend": "local_nli", "model": model}

    try:
        classifier = pipeline("text-classification", model=model, top_k=None)
        outputs = classifier({"text": source_excerpt[:4000], "text_pair": claim[:1000]})
    except Exception as exc:
        return {"error": f"local NLI failed: {exc}", "backend": "local_nli", "model": model}

    rows = outputs[0] if outputs and isinstance(outputs[0], list) else outputs
    scores = {}
    for row in rows or []:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        if "entail" in label:
            scores["entailment"] = max(scores.get("entailment", 0.0), score)
        elif "contrad" in label:
            scores["contradiction"] = max(scores.get("contradiction", 0.0), score)
        elif "neutral" in label:
            scores["neutral"] = max(scores.get("neutral", 0.0), score)
    entailment = scores.get("entailment", 0.0)
    contradiction = scores.get("contradiction", 0.0)
    neutral = scores.get("neutral", 0.0)
    if contradiction >= 0.6 and contradiction > entailment:
        status = "contradicted"
        score = contradiction
    elif entailment >= 0.7:
        status = "supported"
        score = entailment
    elif entailment >= 0.35 and entailment >= neutral:
        status = "partially_supported"
        score = entailment
    else:
        status = "unsupported"
        score = max(neutral, 1.0 - entailment)
    return {
        "status": status,
        "score": round(score, 3),
        "reason": f"local NLI scores: {scores}",
        "backend": "local_nli",
        "model": model,
    }


def _ollama_entailment(claim: str, source_excerpt: str, model: str | None = None, timeout: int = 25) -> dict:
    model = model or os.getenv("FOOTNOTE_ENTAILMENT_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"
    endpoint = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
    system = """You are a strict evidence entailment judge.
Use only the source excerpt.
Return JSON only with:
{"status":"supported|partially_supported|unsupported|contradicted","score":0.0-1.0,"reason":"short reason"}
Definitions:
- supported: the source directly entails the whole claim.
- partially_supported: the source supports part of the claim but leaves a material part unstated.
- unsupported: the source does not provide enough evidence for the claim.
- contradicted: the source states facts that conflict with the claim.
Do not use outside knowledge."""
    user = f"CLAIM:\n{claim[:2000]}\n\nSOURCE_EXCERPT:\n{source_excerpt[:6000]}\n\nJudge entailment."
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    req = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    content = (payload.get("message") or {}).get("content", "")
    parsed = _extract_json_object(content)
    status = str(parsed.get("status", "")).lower()
    if status not in {"supported", "partially_supported", "unsupported", "contradicted"}:
        return {"error": "Ollama judge returned invalid status", "raw": content[:1000], "backend": "ollama", "model": model}
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "status": status,
        "score": max(0.0, min(1.0, score)),
        "reason": str(parsed.get("reason", ""))[:500],
        "backend": "ollama",
        "model": model,
    }


def evidence_entailment(claim: str, source_excerpt: str, backend: str = "auto", model: str | None = None) -> dict:
    backend = (backend or "auto").lower()
    heuristic = _heuristic_entailment(claim, source_excerpt)
    if backend == "heuristic":
        return heuristic
    if backend not in {"auto", "ollama", "local_nli"}:
        return {"status": "unsupported", "score": 0.0, "reason": f"unknown backend: {backend}", "backend": backend}
    if backend == "auto" and heuristic["status"] in {"supported", "contradicted"} and heuristic["score"] >= 0.75:
        return heuristic
    if backend == "local_nli":
        judged = _local_nli_entailment(claim=claim, source_excerpt=source_excerpt, model=model)
        if judged.get("error"):
            return {"status": "unsupported", "score": 0.0, "reason": judged["error"], "backend": "local_nli", "fallback": heuristic}
        judged["heuristic_precheck"] = heuristic
        return judged
    try:
        judged = _ollama_entailment(claim=claim, source_excerpt=source_excerpt, model=model)
        if judged.get("error"):
            if backend == "ollama":
                return {"status": "unsupported", "score": 0.0, "reason": judged["error"], "backend": "ollama", "fallback": heuristic}
            return {**heuristic, "fallback_reason": judged["error"]}
        judged["heuristic_precheck"] = heuristic
        return judged
    except Exception as exc:
        if backend == "ollama":
            return {"status": "unsupported", "score": 0.0, "reason": f"Ollama entailment failed: {exc}", "backend": "ollama", "fallback": heuristic}
        return {**heuristic, "fallback_reason": f"Ollama entailment unavailable: {exc}"}


def source_cache_get(url: str) -> dict:
    cached = _read_cache(url)
    return {"url": url, "found": cached is not None, "cache": cached}


def source_cache_put(url: str, payload: dict) -> dict:
    _write_cache(url, payload)
    return {"url": url, "ok": True, "cache_path": str(_cache_path(url))}


def build_research_debug_report(
    task: str,
    requirements: dict | None = None,
    search_memory: dict | None = None,
    sources: list[dict] | None = None,
    verification: dict | None = None,
) -> dict:
    search_memory = search_memory or {}
    sources = sources or []
    verification = verification or {}
    source_rows = []
    for idx, source in enumerate(sources, 1):
        url = source.get("url", "")
        cached = _read_cache(url) if url else None
        quality = cached.get("source_quality") if isinstance(cached, dict) else None
        source_rows.append(
            {
                "source_id": idx,
                "title": source.get("title", ""),
                "url": url,
                "kind": source.get("kind", ""),
                "quality": quality,
                "content_chars": len(source.get("content", "") or ""),
            }
        )
    return {
        "task": task,
        "requirements": requirements or {},
        "attempted_queries": search_memory.get("attempted_queries", []),
        "read_urls": search_memory.get("read_urls", []),
        "failed_urls": search_memory.get("failed_urls", []),
        "search_rounds": search_memory.get("search_rounds", []),
        "sources": source_rows,
        "verification": {
            "task_complete": verification.get("task_complete"),
            "insufficient_evidence": verification.get("insufficient_evidence"),
            "gaps": verification.get("gaps", []),
            "coverage": verification.get("coverage", {}),
        },
    }


def startup_health_check() -> dict:
    checks = {}
    for module_name in ["bs4", "trafilatura", "playwright", "openpyxl", "pypdf", "xlrd", "pdfplumber", "pytesseract"]:
        try:
            __import__(module_name)
            checks[module_name] = {"ok": True}
        except Exception as exc:
            checks[module_name] = {"ok": False, "error": str(exc)}
    try:
        import shutil

        tesseract_path = shutil.which("tesseract")
        checks["tesseract_binary"] = {"ok": bool(tesseract_path), "path": tesseract_path}
    except Exception as exc:
        checks["tesseract_binary"] = {"ok": False, "error": str(exc)}
    checks["cache_dir"] = {"ok": CACHE_DIR.exists(), "path": str(CACHE_DIR)}
    return {"ok": all(item.get("ok") for item in checks.values()), "checks": checks}
