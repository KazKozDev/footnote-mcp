"""Extraction-recipe synthesis, validation, sandboxed execution, and promotion."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RECIPE_STORE_PATH = Path(os.getenv("FOOTNOTE_RECIPE_STORE", "~/.footnote-mcp/extraction_recipes.json")).expanduser()


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
