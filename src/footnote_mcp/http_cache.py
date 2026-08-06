"""Validator cache for conditional HTTP requests.

Stores a page body only when the server handed us a validator (``ETag`` or
``Last-Modified``). The next fetch of that URL sends ``If-None-Match`` /
``If-Modified-Since``, and a ``304 Not Modified`` costs no body transfer — most
hosts do not count it against a rate limit at all.

Deliberately independent of ``tools_data.cache``: importing that package from
``fetch`` would close an import cycle (tools_data → files → fetch). Both simply
read the same ``FOOTNOTE_SOURCE_CACHE`` root.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _enabled() -> bool:
    return os.getenv("FOOTNOTE_HTTP_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _max_bytes() -> int:
    try:
        return max(0, int(os.getenv("FOOTNOTE_HTTP_CACHE_MAX_BYTES", "1000000")))
    except ValueError:
        return 1_000_000


def _cache_dir() -> Path:
    root = os.getenv("FOOTNOTE_SOURCE_CACHE", "").strip() or "~/.footnote-mcp/source_cache"
    return Path(root).expanduser() / "http"


def _entry_path(url: str) -> Path:
    return _cache_dir() / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def _read(url: str) -> dict | None:
    if not _enabled():
        return None
    try:
        path = _entry_path(url)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def conditional_headers(url: str) -> dict:
    """Validator headers to attach to the next request for ``url``."""
    entry = _read(url)
    if not entry or not entry.get("body"):
        return {}
    headers = {}
    if entry.get("etag"):
        headers["If-None-Match"] = str(entry["etag"])
    if entry.get("last_modified"):
        headers["If-Modified-Since"] = str(entry["last_modified"])
    return headers


def load(url: str) -> str | None:
    """The stored body for ``url``, or None when nothing usable is cached."""
    entry = _read(url)
    body = entry.get("body") if entry else None
    return body if isinstance(body, str) and body else None


def store(url: str, response) -> bool:
    """Persist the body when the response carries a validator. Returns whether it did.

    Without a validator the entry could never produce a 304, so storing it would
    only grow the cache directory.
    """
    if not _enabled():
        return False
    headers = getattr(response, "headers", None) or {}
    try:
        etag = str(headers.get("ETag") or headers.get("etag") or "").strip()
        last_modified = str(headers.get("Last-Modified") or headers.get("last-modified") or "").strip()
    except Exception:
        return False
    if not etag and not last_modified:
        return False

    body = getattr(response, "text", "") or ""
    if not body or len(body.encode("utf-8", errors="ignore")) > _max_bytes():
        return False

    try:
        path = _entry_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"url": url, "etag": etag, "last_modified": last_modified, "body": body},
                ensure_ascii=False,
            )
        )
        return True
    except Exception:
        return False
