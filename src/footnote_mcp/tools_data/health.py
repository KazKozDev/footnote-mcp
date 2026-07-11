"""Research debug reports and startup health checks."""

from __future__ import annotations


from .cache import _read_cache
from . import cache


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
    checks["cache_dir"] = {"ok": cache.CACHE_DIR.exists(), "path": str(cache.CACHE_DIR)}
    return {"ok": all(item.get("ok") for item in checks.values()), "checks": checks}
