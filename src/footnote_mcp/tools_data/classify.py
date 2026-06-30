"""Source classification, query generation, and unit resolution/validation."""

from __future__ import annotations

import re
from urllib.parse import urlparse


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
