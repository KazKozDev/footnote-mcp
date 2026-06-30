"""Source-cache read/write and the cache-backed MCP tools."""

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


CACHE_DIR = Path(os.getenv("FOOTNOTE_SOURCE_CACHE", "~/.footnote-mcp/source_cache")).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def source_cache_get(url: str) -> dict:
    cached = _read_cache(url)
    return {"url": url, "found": cached is not None, "cache": cached}


def source_cache_put(url: str, payload: dict) -> dict:
    _write_cache(url, payload)
    return {"url": url, "ok": True, "cache_path": str(_cache_path(url))}
