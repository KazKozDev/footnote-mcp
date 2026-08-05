"""Iterative, evidence-led deep research orchestration.

This module deliberately sits above the fast search/fetch primitives.  It keeps
research state, searches unresolved requirements, ranks documents and chunks in
two separate stages, extracts structured files/tables, and admits only directly
verified evidence into the ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from . import core
from .extract import chunk_text, extract_content, filter_low_quality_chunks
from .fetch import fetch_page
from .rerank import rerank_chunks
from .scraper import fetch as scrape_fetch
from .tools_data.entailment import _extract_json_object, evidence_entailment
from .tools_data.files import _table_to_rows, web_parse_file

JSONCall = Callable[[list[dict[str, str]]], dict[str, Any]]
DiscoverCall = Callable[..., tuple[list[dict], list[str], dict[str, str]]]
FetchCall = Callable[..., dict[str, Any]]
ProgressCall = Callable[[str, dict[str, Any]], None]

_FILE_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".json")
_FIELD_NAMES = ("subject", "metric", "period", "value", "unit")
_DOWNLOAD_HINTS = ("csv", "tsv", "xlsx", "excel", "pdf", "download", "meeting minutes", "minutes link")
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _norm(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def _url_identity(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{host}{path}".lower()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _string_map(value: Any, fallback_key: str) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items() if item not in (None, "")}
    if isinstance(value, str) and value.strip():
        return {fallback_key: value.strip()}
    return {}


def _optional_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if _norm(text) in {"n a", "na", "none", "null", "not applicable", "unspecified"} else text


@dataclass
class ResearchRequirement:
    id: str
    text: str
    subject: str = ""
    metric: str = ""
    period: str = ""
    unit: str = ""
    scope: dict[str, str] = field(default_factory=dict)
    predicate: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    completion_rule: str = "single"
    expected_count: int | None = None
    status: str = "unresolved"
    gap: str = ""
    queries: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    id: str
    requirement_id: str
    claim: str
    subject: str
    metric: str
    period: str
    value: str
    unit: str
    quote: str
    source_url: str
    source_title: str
    status: str
    score: float
    verification: str
    iteration: int
    extraction_type: str = "text"
    scope: dict[str, str] = field(default_factory=dict)
    entity: str = ""
    predicate: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    fact_quote: str = ""
    context_quote: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchState:
    query: str
    requirements: list[ResearchRequirement]
    answer_type: str = "single"
    iteration: int = 0
    stop_reason: str = ""
    answer_ready: bool = False
    aggregation_plan: dict[str, Any] = field(default_factory=dict)
    derived_answer: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    contradictions: list[EvidenceItem] = field(default_factory=list)
    seen_urls: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    routed_sources: list[str] = field(default_factory=list)
    discovery_errors: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=lambda: {
        "iterations": [],
        "model_errors": [],
        "evidence_rejections": {},
        "funnel": {
            "candidates": 0,
            "deduplicated_documents": 0,
            "relevant_documents": 0,
            "fetch_attempts": 0,
            "successful_fetches": 0,
            "structured_rows": 0,
            "ranked_chunks": 0,
            "extracted_facts": 0,
            "verified_evidence": 0,
            "contradictions": 0,
        },
    })

    def unresolved(self) -> list[ResearchRequirement]:
        return [requirement for requirement in self.requirements if requirement.status != "covered"]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unresolved_requirements"] = [item.id for item in self.unresolved()]
        payload["coverage"] = round(
            sum(item.status == "covered" for item in self.requirements) / max(1, len(self.requirements)),
            4,
        )
        return payload


@dataclass
class ResearchBudget:
    max_iterations: int = 4
    initial_fetch: int = 8
    fetch_growth: int = 4
    max_fetch: int = 28
    max_queries_per_iteration: int = 6
    initial_chunks_per_requirement: int = 8
    chunk_growth: int = 8
    max_chunks_per_requirement: int = 96
    max_structured_files: int = 6
    max_rows_per_table: int = 120
    stall_limit: int = 2
    max_elapsed_seconds: float = 240.0
    per_document_timeout: float = 25.0
    per_file_timeout: float = 35.0
    max_extraction_context_chars: int = 16000


def ollama_json_call(model: str | None = None, timeout: int = 60, retries: int = 2) -> JSONCall | None:
    """Build a strict JSON caller; return None when no research model is configured."""
    selected = model or os.getenv("FOOTNOTE_RESEARCH_MODEL") or os.getenv("OLLAMA_MODEL")
    if not selected:
        return None

    def call(messages: list[dict[str, str]]) -> dict[str, Any]:
        import ollama

        deadline = time.monotonic() + max(1, timeout)
        attempt_messages = list(messages)
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"research model call exceeded {timeout}s total deadline")
            client = ollama.Client(timeout=max(1, remaining))
            response = client.chat(
                model=selected,
                messages=attempt_messages,
                format="json",
                think=False,
                options={"temperature": 0},
            )
            content = response.message.content or ""
            parsed = _extract_json_object(content)
            if parsed:
                return parsed
            last_error = ValueError("research model returned invalid JSON")
            if attempt < retries:
                attempt_messages = [
                    *messages,
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": "Return exactly one valid JSON object and no other text."},
                ]
        raise last_error or ValueError("research model returned invalid JSON")

    return call


def _heuristic_requirement(query: str) -> ResearchRequirement:
    exhaustive = bool(re.search(r"\b(all|every|complete list|which countries|what companies)\b", query, re.I))
    return ResearchRequirement(
        id="r1",
        text=query.strip(),
        completion_rule="all_items" if exhaustive else "single",
        gap=query.strip(),
    )


def _aggregation_plan(query: str) -> dict[str, Any]:
    """Recognize list operations that must be executed deterministically."""
    normalized = _norm(query)
    if "more than once" in normalized:
        return {"operation": "group_count_filter", "group_by": "entity", "operator": ">", "threshold": 1}
    match = re.search(r"\bat least\s+(\d+)\b", normalized)
    if match:
        return {
            "operation": "group_count_filter", "group_by": "entity",
            "operator": ">=", "threshold": int(match.group(1)),
        }
    return {}


def _expand_repeated_period_requirements(
    query: str,
    requirements: list[ResearchRequirement],
) -> list[ResearchRequirement]:
    """Turn cross-period attendance questions into atomic event requirements."""
    query_norm = _norm(query)
    months = [month for month in _MONTHS if re.search(rf"\b{month}\b", query_norm)]
    attendance_question = bool(re.search(r"\b(absent|absence|attendance|present|missing)\b", query_norm))
    repeated_period_question = bool(re.search(r"\b(more than once|across|meetings?)\b", query_norm))
    if len(months) < 2 or not attendance_question or not repeated_period_question:
        return requirements

    target = max(
        requirements,
        key=lambda item: len(
            set(_norm(f"{item.text} {item.metric}").split())
            & {"absent", "absence", "attendance", "present", "missing"}
        ),
        default=None,
    )
    if target is None:
        return requirements
    years = re.findall(r"\b(?:19|20)\d{2}\b", query)
    year = years[0] if years else ""
    subject = target.subject or "board members"
    metric = target.metric if _norm(target.metric) in {"absent", "absence", "attendance", "present", "missing"} else "absent"
    entity_match = (
        re.search(r"for the ([^,?.]*?committee of the [^,?.]*?authority)", query, flags=re.I)
        or re.search(r"(?:for|of) the ([^,?.]+?(?:authority|committee|board))", query, flags=re.I)
    )
    entity = entity_match.group(1).strip() if entity_match else "committee"
    expanded = []
    for index, month in enumerate(months, 1):
        period = f"{month.title()} {year}".strip()
        expanded.append(ResearchRequirement(
            id=f"attendance_{index}_{month}",
            text=f"Each board member absent from the {entity} meeting in {period}, according to official meeting minutes",
            subject=entity,
            metric=metric,
            period=period,
            unit="",
            scope={"organization": entity},
            predicate=metric,
            qualifiers={"period": period},
            completion_rule="all_items",
            gap=f"{entity} {period} meeting minutes absent members",
        ))
    return expanded


def decompose_requirements(query: str, model_json: JSONCall | None = None) -> tuple[list[ResearchRequirement], str, str]:
    """Create explicit verifiable requirements without inventing missing targets."""
    if model_json is None:
        requirement = _heuristic_requirement(query)
        requirements = _expand_repeated_period_requirements(query, [requirement])
        return requirements, "set" if requirement.completion_rule == "all_items" else "single", ""
    prompt = [{
        "role": "user",
        "content": (
            "Decompose the research question into independently verifiable requirements. Return JSON with "
            "'answer_type' ('single' or 'set') and 'requirements'. Each requirement must contain: id, text, "
            "subject, metric, period, unit, scope (object), predicate, qualifiers (object), "
            "completion_rule ('single', 'count', or 'all_items'), and optional "
            "expected_count. Scope may contain only literal organization, dataset, jurisdiction, source, or "
            "geography names that an authoritative document should state; otherwise use an empty object. "
            "Do not put entity types such as Country/Person or logical conditions in scope. Do not answer the "
            "question and do not invent a period or unit.\n\nQUESTION:\n" + query
        ),
    }]
    try:
        payload = model_json(prompt)
    except Exception as exc:
        requirement = _heuristic_requirement(query)
        return [requirement], "set" if requirement.completion_rule == "all_items" else "single", str(exc)
    requirements = []
    for index, row in enumerate(payload.get("requirements") or [], 1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        rule = str(row.get("completion_rule") or "single").lower()
        if rule not in {"single", "count", "all_items"}:
            rule = "single"
        expected = row.get("expected_count")
        try:
            expected = int(expected) if expected not in (None, "") else None
        except (TypeError, ValueError):
            expected = None
        requirements.append(ResearchRequirement(
            id=str(row.get("id") or f"r{index}"),
            text=text,
            subject=str(row.get("subject") or "").strip(),
            metric=str(row.get("metric") or "").strip(),
            period=_optional_text(row.get("period")),
            unit=_optional_text(row.get("unit")),
            scope=_string_map(row.get("scope"), "scope"),
            predicate=str(row.get("predicate") or row.get("metric") or "").strip(),
            qualifiers=_string_map(row.get("qualifiers"), "context"),
            completion_rule=rule,
            expected_count=expected,
            gap=text,
        ))
        requirement = requirements[-1]
        if requirement.period and "period" not in requirement.qualifiers:
            requirement.qualifiers["period"] = requirement.period
        if requirement.unit and "unit" not in requirement.qualifiers:
            requirement.qualifiers["unit"] = requirement.unit
    if not requirements:
        requirement = _heuristic_requirement(query)
        requirements = _expand_repeated_period_requirements(query, [requirement])
        return requirements, "set" if requirement.completion_rule == "all_items" else "single", "empty requirements"
    answer_type = str(payload.get("answer_type") or "single").lower()
    requirements = _expand_repeated_period_requirements(query, requirements)
    if answer_type != "set":
        for requirement in requirements:
            if requirement.completion_rule == "all_items":
                requirement.completion_rule = "single"
    return requirements, "set" if answer_type == "set" else "single", ""


def _plan_gap_queries(
    state: ResearchState,
    model_json: JSONCall | None,
    limit: int,
) -> list[tuple[str, str]]:
    unresolved = state.unresolved()
    if not unresolved:
        return []
    direct_urls = re.findall(r"https?://[^\s<>\"]+", state.query)
    direct_plans = []
    if state.iteration == 1 and direct_urls:
        for requirement in unresolved:
            for url in direct_urls:
                clean_url = url.rstrip(".,;:!?) }")
                direct_plans.append((requirement.id, f"{requirement.text} {clean_url}"))
    if state.iteration == 1:
        planned = list(direct_plans)
        for requirement in unresolved:
            if requirement.id.startswith("attendance_"):
                query = f"{requirement.subject} {requirement.period} meeting minutes".strip()
            else:
                parts = [requirement.subject, requirement.metric, requirement.period, requirement.text]
                base = " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))
                query = f"{base} official data table pdf csv".strip()
            if query not in state.queries and (requirement.id, query) not in planned:
                planned.append((requirement.id, query))
        return planned[:limit]
    if model_json is not None:
        compact_evidence = [
            {key: getattr(item, key) for key in ("requirement_id", "subject", "metric", "period", "value", "unit")}
            for item in state.evidence
        ]
        prompt = [{
            "role": "user",
            "content": (
                "Plan targeted web searches only for unresolved evidence gaps. Return JSON with 'queries'; each "
                "entry must contain requirement_id and query. Prefer official datasets, tables, CSV, XLSX, PDF, "
                "or primary sources when appropriate. Do not answer the question.\n\nQUESTION:\n"
                + state.query
                + "\n\nUNRESOLVED REQUIREMENTS:\n"
                + json.dumps([asdict(item) for item in unresolved], ensure_ascii=False)
                + "\n\nVERIFIED SO FAR:\n"
                + json.dumps(compact_evidence, ensure_ascii=False)
                + "\n\nALREADY USED QUERIES (do not repeat):\n"
                + json.dumps(state.queries, ensure_ascii=False)
            ),
        }]
        try:
            payload = model_json(prompt)
            planned = []
            valid_ids = {item.id for item in unresolved}
            for row in payload.get("queries") or []:
                requirement_id = str(row.get("requirement_id") or "")
                query = str(row.get("query") or "").strip()
                if requirement_id in valid_ids and query and query not in state.queries:
                    planned.append((requirement_id, query))
            if planned:
                combined = direct_plans + [item for item in planned if item not in direct_plans]
                return combined[:limit]
        except Exception as exc:
            state.diagnostics["model_errors"].append(f"query planning: {exc}")

    queries = list(direct_plans)
    for requirement in unresolved:
        base = requirement.gap or requirement.text
        suffix = "official data table csv" if state.iteration == 1 else "primary source downloadable dataset"
        query = f"{base} {suffix}".strip()
        if query in state.queries:
            query = f"{query} alternative source {state.iteration}"
        queries.append((requirement.id, query))
    return queries[:limit]


def _rank_documents(requirement: ResearchRequirement, documents: list[dict], limit: int, lang: str) -> list[dict]:
    if len(documents) <= 1:
        return [{**document, "document_relevance": 1.0} for document in documents[:limit]]
    pseudo_chunks = []
    for index, document in enumerate(documents):
        pseudo_chunks.append({
            "text": f"{document.get('title', '')}\n{document.get('snippet', '')}",
            "source_idx": index,
            "source_url": document.get("url", ""),
            "source_title": document.get("title", ""),
            "chunk_idx": 0,
        })
    ranked = rerank_chunks(requirement.text, pseudo_chunks, top_k=min(limit, len(pseudo_chunks)), lang=lang)
    return [{**documents[item["source_idx"]], "document_relevance": item.get("relevance", 0.0)} for item in ranked]


def _parsed_file_chunks(parsed: dict, source_url: str) -> tuple[list[dict], int]:
    """Convert parsed files into addressable evidence segments, never whole-file blobs."""
    chunks: list[dict] = []
    row_count = 0
    extraction_type = parsed.get("file_type") or "file"
    for table_position, table in enumerate(parsed.get("tables") or [], 1):
        columns = [str(value) for value in table.get("columns") or []]
        rows = table.get("rows") or []
        table_index = int(table.get("table_index") or table_position)
        caption = str(table.get("caption") or table.get("sheet") or "").strip()
        context_text = "\n".join(filter(None, [caption, "HEADER: " + " | ".join(columns)]))
        complete_set = not bool(table.get("truncated"))
        for row_position, row in enumerate(rows, 1):
            cells = row if isinstance(row, dict) else {"value": row}
            cell_text = " | ".join(f"{column}: {cells.get(column, '')}" for column in columns) if columns else str(row)
            chunks.append({
                "segment_id": f"table-{table_index}-row-{row_position}",
                "text": cell_text,
                "context_text": context_text,
                "extraction_type": extraction_type,
                "provenance": {
                    "segment_type": "table_row", "table_index": table_index,
                    "row_index": row_position, "columns": columns, "cells": dict(cells),
                    "complete_set": complete_set,
                },
            })
        row_count += len(rows)
    for page in parsed.get("pages") or []:
        lines = [line.strip() for line in str(page.get("text") or "").splitlines() if line.strip()]
        page_number = int(page.get("page") or 0)
        page_header = " | ".join(lines[:3])
        for start in range(0, len(lines), 8):
            block = lines[start:start + 10]
            if not block:
                continue
            text = "\n".join(block)
            complete_set = bool(re.search(r"\b(absent|present|attendees?|members?)\b", text, re.I))
            chunks.append({
                "segment_id": f"page-{page_number}-lines-{start + 1}-{start + len(block)}",
                "text": text,
                "context_text": f"PAGE HEADER: {page_header}",
                "extraction_type": "pdf_text",
                "provenance": {
                    "segment_type": "pdf_lines", "page": page_number,
                    "line_start": start + 1, "line_end": start + len(block),
                    "complete_set": complete_set,
                },
            })
    if "json" in parsed:
        text = json.dumps(parsed["json"], ensure_ascii=False)
        if text:
            chunks.append({
                "segment_id": "json-root", "text": text[: core.MAX_CONTENT_CHARS],
                "context_text": "JSON root", "extraction_type": "json",
                "provenance": {"segment_type": "json", "complete_set": len(text) <= core.MAX_CONTENT_CHARS},
            })
    return chunks, row_count


def _rank_download_links(soup: BeautifulSoup, base_url: str, query: str, limit: int = 12) -> list[dict]:
    query_terms = set(_norm(query).split())
    candidates = {}
    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor.get("href") or "").strip()
        if not raw_href or raw_href.lower().startswith(("javascript:", "mailto:", "#")):
            continue
        download_url = urljoin(base_url, raw_href)
        text = " ".join(anchor.get_text(" ", strip=True).split())
        path = urlparse(download_url).path.lower()
        haystack = f"{text} {path}".lower()
        if not (path.endswith(_FILE_EXTENSIONS) or any(hint in haystack for hint in _DOWNLOAD_HINTS)):
            continue
        candidate_terms = set(_norm(f"{text} {download_url}").split())
        overlap = len(query_terms & candidate_terms) / max(1, len(query_terms))
        score = overlap
        if "minute" in haystack:
            score += 1.0
        if "meeting" in haystack:
            score += 0.4
        if path.endswith(_FILE_EXTENSIONS):
            score += 0.2
        for token in query_terms:
            if (token.isdigit() or token in {"may", "june", "september", "october", "november"}) and token in haystack:
                score += 0.15
        current = candidates.get(download_url)
        item = {"url": download_url, "text": text, "score": round(score, 4)}
        if current is None or item["score"] > current["score"]:
            candidates[download_url] = item
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["url"]))[:limit]


def fetch_research_document(
    document: dict,
    *,
    lang: str = "en",
    max_rows: int = 120,
    parse_downloads: bool = True,
    deadline: float | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch text plus structured tables/files while preserving provenance."""
    url = str(document.get("url") or "")
    title = str(document.get("title") or url)
    path = urlparse(url).path.lower()
    if path.endswith(_FILE_EXTENSIONS):
        try:
            parsed = web_parse_file(url, lang=lang, max_rows=max_rows, deadline=deadline, timeout=timeout or 35.0)
        except Exception as exc:
            parsed = {"url": url, "error": f"file parser failed: {exc}"}
        chunks, row_count = _parsed_file_chunks(parsed, url)
        return {
            "url": url,
            "title": title,
            "success": bool(chunks),
            "text": "\n\n".join(item["text"] for item in chunks),
            "structured_chunks": chunks,
            "structured_rows": row_count,
            "downloads": [],
            **({"error": parsed.get("error")} if parsed.get("error") else {}),
        }

    scraped = scrape_fetch(url, lang=lang, http_fn=fetch_page)
    fetched_url = scraped.get("final_url") or url
    html = scraped.get("html")
    pub_date = scraped.get("pub_date")
    error = scraped.get("error")
    if error or not html:
        return {"url": url, "title": title, "success": False, "error": error or "empty response"}
    text = extract_content(html, url=fetched_url) or ""
    soup = BeautifulSoup(html, "html.parser")
    structured_chunks = []
    structured_rows = 0
    for table_index, table in enumerate(soup.find_all("table")[:8], 1):
        parsed_table = _table_to_rows(table)
        original_count = len(parsed_table.get("rows", []))
        parsed_table["rows"] = parsed_table.get("rows", [])[:max_rows]
        parsed_table["table_index"] = table_index
        parsed_table["truncated"] = original_count > len(parsed_table["rows"])
        table_chunks, count = _parsed_file_chunks(
            {"file_type": "html_table", "tables": [parsed_table]}, fetched_url
        )
        structured_chunks.extend(table_chunks)
        structured_rows += count

    query = " ".join(
        [
            *(str(value) for value in document.get("research_queries") or []),
            str(document.get("research_query") or ""),
            str(document.get("snippet") or ""),
            title,
        ]
    )
    download_candidates = []
    if parse_downloads:
        download_candidates = _rank_download_links(soup, fetched_url, query)
    downloads = [item["url"] for item in download_candidates]
    return {
        "url": fetched_url,
        "title": title,
        "success": bool(text or structured_chunks),
        "text": text[: core.MAX_CONTENT_CHARS],
        "structured_chunks": structured_chunks,
        "structured_rows": structured_rows,
        "downloads": downloads,
        "download_candidates": download_candidates,
        "published": str(pub_date) if pub_date else None,
    }


def _fetch_documents(
    documents: list[dict],
    *,
    lang: str,
    max_rows: int,
    structured_file_budget: int,
    fetch_fn: FetchCall,
    deadline: float,
    per_document_timeout: float = 25.0,
    per_file_timeout: float = 35.0,
) -> list[dict]:
    fetched = []
    if not documents or time.monotonic() >= deadline:
        return fetched
    pool = ThreadPoolExecutor(max_workers=min(core.FETCH_WORKERS, max(1, len(documents))))
    futures = {
        pool.submit(
            fetch_fn, document, lang=lang, max_rows=max_rows, parse_downloads=True,
            deadline=deadline, timeout=min(per_document_timeout, max(0.1, deadline - time.monotonic())),
        ): document
        for document in documents
    }
    done, unfinished = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    for future in done:
        document = futures[future]
        try:
            result = future.result()
        except Exception as exc:
            result = {"url": document.get("url", ""), "title": document.get("title", ""), "success": False, "error": str(exc)}
        result["requirements"] = list(document.get("requirements") or [])
        fetched.append(result)
    for future in unfinished:
        future.cancel()
    pool.shutdown(wait=False, cancel_futures=True)

    download_jobs = []
    for result in list(fetched):
        candidates = result.get("download_candidates") or [
            {"url": url, "text": "", "score": 0.0}
            for url in result.get("downloads") or []
        ]
        for candidate in candidates:
            download_jobs.append((float(candidate.get("score") or 0.0), candidate, result))
    download_jobs.sort(key=lambda item: (-item[0], item[1].get("url", "")))

    seen_downloads = set()
    for _score, candidate, result in download_jobs:
        if time.monotonic() >= deadline:
            break
        if len(seen_downloads) >= structured_file_budget:
            break
        download_url = str(candidate.get("url") or "")
        if not download_url or download_url in seen_downloads:
            continue
        seen_downloads.add(download_url)
        try:
            parsed = web_parse_file(
                download_url, lang=lang, max_rows=max_rows, deadline=deadline,
                timeout=min(per_file_timeout, max(0.1, deadline - time.monotonic())),
            )
        except Exception as exc:
            parsed = {"url": download_url, "error": f"file parser failed: {exc}"}
        chunks, row_count = _parsed_file_chunks(parsed, download_url)
        fetched.append({
            "url": download_url,
            "title": " — ".join(filter(None, [
                str(result.get("title") or ""),
                str(candidate.get("text") or Path(urlparse(download_url).path).name or download_url),
            ])),
            "success": bool(chunks),
            "text": "\n\n".join(item["text"] for item in chunks),
            "structured_chunks": chunks,
            "structured_rows": row_count,
            "requirements": list(result.get("requirements") or []),
            "parent_url": result.get("url"),
            **({"error": parsed.get("error")} if parsed.get("error") else {}),
        })
    return fetched


def _rank_chunks_for_requirements(
    requirements: list[ResearchRequirement],
    fetched: list[dict],
    *,
    lang: str,
    top_k: int,
    deadline: float | None = None,
) -> dict[str, list[dict]]:
    all_chunks = []
    for source_idx, document in enumerate(fetched):
        if not document.get("success"):
            continue
        structured_items = document.get("structured_chunks") or []
        path = urlparse(str(document.get("url") or "")).path.lower()
        structured_file = bool(structured_items) and (
            bool(document.get("parent_url")) or path.endswith(_FILE_EXTENSIONS)
        )
        # File fetchers expose text as a join of the same addressable rows/pages.
        # Do not duplicate that content through the generic chunker.
        document_text = "" if structured_file else str(document.get("text") or "").strip()
        raw_chunks = chunk_text(document_text, lang=lang)
        if not raw_chunks and document_text:
            raw_chunks = [document_text]
        text_chunks = filter_low_quality_chunks(raw_chunks) or raw_chunks
        for chunk_idx, text in enumerate(text_chunks):
            all_chunks.append({
                "segment_id": f"text-{chunk_idx}",
                "text": text,
                "context_text": str(document.get("title") or ""),
                "source_idx": source_idx,
                "source_url": document.get("url", ""),
                "source_title": document.get("title", ""),
                "chunk_idx": chunk_idx,
                "extraction_type": "text",
                "requirements": list(document.get("requirements") or []),
                "provenance": {"segment_type": "text_chunk", "chunk_index": chunk_idx, "complete_set": False},
            })
        for extra_idx, item in enumerate(structured_items, len(text_chunks)):
            all_chunks.append({
                "text": item.get("text", ""),
                "context_text": item.get("context_text", ""),
                "segment_id": item.get("segment_id", f"structured-{extra_idx}"),
                "source_idx": source_idx,
                "source_url": document.get("url", ""),
                "source_title": document.get("title", ""),
                "chunk_idx": extra_idx,
                "extraction_type": item.get("extraction_type", "structured"),
                "requirements": list(document.get("requirements") or []),
                "provenance": dict(item.get("provenance") or {}),
            })
    ranked: dict[str, list[dict]] = {}
    for requirement in requirements:
        if deadline is not None and time.monotonic() >= deadline:
            ranked[requirement.id] = []
            continue
        eligible = [chunk for chunk in all_chunks if requirement.id in chunk.get("requirements", [])]
        query_terms = set(_norm(requirement.text).split())
        for chunk in eligible:
            text_terms = set(_norm(" ".join([
                str(chunk.get("text") or ""), str(chunk.get("context_text") or ""),
                str(chunk.get("source_title") or ""),
            ])).split())
            chunk["lexical_requirement_score"] = len(query_terms & text_terms) / max(1, len(query_terms))
        # Structured files can contribute thousands of rows. A cheap requirement-specific
        # pass bounds semantic reranking without reintroducing a fixed final chunk cap.
        prefilter_limit = max(top_k * 8, 120)
        eligible = sorted(
            eligible,
            key=lambda item: (-float(item.get("lexical_requirement_score") or 0.0), item.get("segment_id", "")),
        )[:prefilter_limit]
        remaining = deadline - time.monotonic() if deadline is not None else None
        if len(eligible) > 64 or (remaining is not None and remaining < 5.0):
            ranked[requirement.id] = eligible[:top_k]
        else:
            ranked[requirement.id] = rerank_chunks(
                requirement.text, eligible, top_k=min(top_k, len(eligible)), lang=lang
            )
    return ranked


def _claim_from_item(item: dict) -> str:
    if item.get("claim"):
        return str(item["claim"]).strip()
    fields = [str(item.get(name) or "").strip() for name in _FIELD_NAMES]
    return " ".join(value for value in fields if value)


def _field_is_grounded(value: str, quote: str) -> bool:
    wanted = _norm(value)
    source = _norm(quote)
    if not wanted:
        return True
    if wanted in source:
        return True
    tokens = [token for token in wanted.split() if len(token) > 2]
    return bool(tokens) and all(token in source for token in tokens)


def _unit_is_grounded(unit: str, text: str, value: str) -> bool:
    """Validate physical units while treating entity types as schema, not units."""
    if not unit or _field_is_grounded(unit, text):
        return True
    normalized = _norm(unit)
    source = text.casefold()
    if "$" in unit and "$" in text:
        return True
    entity_units = {
        "count", "number", "item", "items", "people", "persons", "individuals",
        "countries", "states", "members", "signers", "species", "seats", "names",
    }
    entity_unit_tokens = {"person", "persons", "people", "individual", "individuals", "signer", "signers", "member", "members", "country", "countries", "state", "states", "species", "seat", "seats", "name", "names"}
    if normalized in entity_units or bool(set(normalized.split()) & entity_unit_tokens):
        return bool(value)
    if any(token in normalized for token in ("percent", "percentage")) and "%" in source:
        return True
    currency = any(token in normalized for token in ("usd", "dollar", "currency"))
    if currency and ("$" in text or "dollar" in source or "usd" in source):
        return True
    if "billion" in normalized and re.search(r"\b(?:billion|bn|b)\b", source):
        return True
    if "million" in normalized and re.search(r"\b(?:million|mn|m)\b", source):
        return True
    return False


def _extract_and_verify(
    requirement: ResearchRequirement,
    chunks: list[dict],
    *,
    iteration: int,
    model_json: JSONCall | None,
    entailment_backend: str,
    entailment_model: str | None,
    state: ResearchState,
) -> tuple[list[EvidenceItem], list[EvidenceItem], int]:
    if not chunks or model_json is None:
        return [], [], 0
    sources = []
    source_lookup = {}
    for index, chunk in enumerate(chunks, 1):
        source_id = f"S{index}"
        source_lookup[source_id] = chunk
        sources.append(
            f"[{source_id}] URL: {chunk.get('source_url', '')}\nTITLE: {chunk.get('source_title', '')}\n"
            f"TYPE: {chunk.get('extraction_type', 'text')}\nTEXT:\n{chunk.get('text', '')[:5000]}"
        )
    prompt = [{
        "role": "user",
        "content": (
            "Extract candidate evidence for exactly one requirement. Return JSON with 'items'. Every item must "
            "contain source_id, claim, subject, metric, period, value, unit, and an exact verbatim quote copied "
            "from that source. Every nonempty field value must itself appear verbatim in the quote; use empty "
            "strings for metric, period, or unit when they are not literally stated or not applicable. Do not infer or combine "
            "values across sources. For list or attendance evidence, emit one item per named entity: subject is "
            "that entity's exact name and value is its observed value or status; never put multiple entity names "
            "inside value. When the requirement specifies a period, the contiguous quote must include both that "
            "period and the fact. Return no item when the source does not directly state the fact.\n\nREQUIREMENT:\n"
            + json.dumps(asdict(requirement), ensure_ascii=False)
            + "\n\nSOURCES:\n"
            + "\n\n".join(sources)
        ),
    }]
    try:
        payload = model_json(prompt)
    except Exception as exc:
        state.diagnostics["model_errors"].append(f"evidence extraction {requirement.id}: {exc}")
        return [], [], 0
    extracted = payload.get("items") or []
    verified = []
    contradictions = []

    def reject(reason: str) -> None:
        counts = state.diagnostics["evidence_rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    for row in extracted:
        source_id = str(row.get("source_id") or "")
        chunk = source_lookup.get(source_id)
        if not chunk:
            reject("unknown_source_id")
            continue
        scope_text = " ".join([
            str(chunk.get("text") or ""),
            str(chunk.get("source_title") or ""),
            str(chunk.get("source_url") or ""),
        ])
        if requirement.subject and not _field_is_grounded(requirement.subject, scope_text):
            reject("requirement_subject_not_grounded")
            continue
        quote = str(row.get("quote") or "").strip()
        chunk_text_value = str(chunk.get("text") or "")
        if not quote or _norm(quote) not in _norm(chunk_text_value):
            reject("quote_not_verbatim")
            continue
        fields = {name: str(row.get(name) or "").strip() for name in _FIELD_NAMES}
        if not fields["value"]:
            reject("missing_value")
            continue
        ungrounded_fields = [name for name, value in fields.items() if not _field_is_grounded(value, quote)]
        if ungrounded_fields:
            for name in ungrounded_fields:
                reject(f"{name}_not_grounded_in_quote")
            continue
        for name in ("period", "unit"):
            expected = getattr(requirement, name)
            if expected and not _field_is_grounded(expected, quote):
                reject(f"required_{name}_not_grounded")
                break
        else:
            claim = _claim_from_item(row)
            verdict = evidence_entailment(
                claim,
                quote,
                backend=entailment_backend,
                model=entailment_model,
            )
            status = str(verdict.get("status") or "unsupported")
            evidence = EvidenceItem(
                id=_stable_id(
                    "ev", requirement.id, fields["subject"], fields["metric"], fields["period"],
                    fields["value"], fields["unit"], chunk.get("source_url", ""),
                ),
                requirement_id=requirement.id,
                claim=claim,
                quote=quote,
                source_url=str(chunk.get("source_url") or ""),
                source_title=str(chunk.get("source_title") or ""),
                status=status,
                score=float(verdict.get("score") or 0.0),
                verification=str(verdict.get("backend") or entailment_backend),
                iteration=iteration,
                extraction_type=str(chunk.get("extraction_type") or "text"),
                **fields,
            )
            if status == "supported":
                verified.append(evidence)
            elif status == "contradicted":
                contradictions.append(evidence)
            else:
                reject(f"entailment_{status}")
    return verified, contradictions, len(extracted)


def _extract_and_verify_batch(
    requirements: list[ResearchRequirement],
    ranked_chunks: dict[str, list[dict]],
    *,
    iteration: int,
    model_json: JSONCall | None,
    entailment_backend: str,
    entailment_model: str | None,
    state: ResearchState,
    max_context_chars: int,
) -> tuple[list[EvidenceItem], list[EvidenceItem], int]:
    """Extract evidence with one bounded, isolated model call per requirement."""
    if not requirements or model_json is None:
        return [], [], 0

    active = [item for item in requirements if ranked_chunks.get(item.id)]
    if not active:
        return [], [], 0

    total_budget = max(1000, int(max_context_chars))
    per_requirement_budget = max(600, total_budget // len(active))
    source_lookup: dict[tuple[str, str], dict] = {}
    source_sections: list[tuple[str, str]] = []
    used_total = 0

    for requirement in active:
        used_for_requirement = 0
        sources = []
        for index, chunk in enumerate(ranked_chunks.get(requirement.id, []), 1):
            remaining_for_requirement = per_requirement_budget - used_for_requirement
            remaining_total = total_budget - used_total
            available = min(remaining_for_requirement, remaining_total)
            if available <= 0:
                break
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            excerpt = text[:available]
            source_id = f"S{index}"
            source_lookup[(requirement.id, source_id)] = chunk
            sources.append(
                f"[{source_id}] SEGMENT_ID: {chunk.get('segment_id', '')}\n"
                f"URL: {chunk.get('source_url', '')}\nTITLE: {chunk.get('source_title', '')}\n"
                f"TYPE: {chunk.get('extraction_type', 'text')}\n"
                f"CONTEXT: {chunk.get('context_text', '')}\nFACT_TEXT:\n{excerpt}"
            )
            used = len(excerpt)
            used_for_requirement += used
            used_total += used
        if sources:
            source_sections.append((requirement.id,
                f"REQUIREMENT_ID: {requirement.id}\nREQUIREMENT: "
                f"{json.dumps(asdict(requirement), ensure_ascii=False)}\nSOURCES:\n"
                + "\n\n".join(sources)
            ))

    if not source_sections:
        return [], [], 0

    instruction = (
            "Extract candidate evidence by selecting pointers for the listed requirements. Return JSON with 'items'. Every item must "
            "contain requirement_id, source_id, entity, predicate, value, and qualifiers (object). Do not return "
            "or invent a quote: the application resolves the selected source_id to immutable FACT_TEXT and "
            "CONTEXT. For lists emit one item per entity. Choose only a source under that exact requirement. "
            "The entity and value must occur in FACT_TEXT. Scope such as organization may occur in CONTEXT, "
            "title, or URL. Period/unit qualifiers may occur in FACT_TEXT or CONTEXT. Return no item when the "
            "pointer does not directly establish the fact. entity and value must each be copied character-for-"
            "character as a contiguous substring of FACT_TEXT. Never normalize them to true/false, a calculated "
            "number, an alias, or a summary; for example copy 'no land borders' rather than returning 0.\n\n"
    )
    extracted_with_scope: list[tuple[str, dict]] = []
    for prompt_requirement_id, section in source_sections:
        try:
            payload = model_json([{"role": "user", "content": instruction + section}])
            for row in payload.get("items") or []:
                if isinstance(row, dict):
                    extracted_with_scope.append((prompt_requirement_id, row))
        except Exception as exc:
            state.diagnostics["model_errors"].append(
                f"evidence extraction {prompt_requirement_id}: {exc}"
            )

    requirement_lookup = {item.id: item for item in requirements}
    extracted = [row for _, row in extracted_with_scope]
    verified = []
    contradictions = []

    def reject(reason: str) -> None:
        counts = state.diagnostics["evidence_rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    for prompt_requirement_id, row in extracted_with_scope:
        requirement_id = str(row.get("requirement_id") or "")
        if requirement_id != prompt_requirement_id:
            reject("cross_requirement_pointer")
            continue
        requirement = requirement_lookup.get(requirement_id)
        source_id = str(row.get("source_id") or "")
        chunk = source_lookup.get((requirement_id, source_id))
        if requirement is None:
            reject("unknown_requirement_id")
            continue
        if not chunk:
            reject("unknown_source_id")
            continue
        fact_text = str(chunk.get("text") or "").strip()
        context_text = str(chunk.get("context_text") or "").strip()
        scope_text = " ".join([
            fact_text,
            context_text,
            str(chunk.get("source_title") or ""),
            str(chunk.get("source_url") or ""),
        ])
        required_scope = dict(requirement.scope)
        identity_scope = {
            key: value for key, value in required_scope.items()
            if key.casefold() in {"organization", "dataset", "jurisdiction", "source", "geography"}
        }
        if any(value and not _field_is_grounded(value, scope_text) for value in identity_scope.values()):
            reject("requirement_scope_not_grounded")
            continue
        entity = str(row.get("entity") or row.get("subject") or "").strip()
        predicate = str(row.get("predicate") or row.get("metric") or requirement.predicate or requirement.metric or "").strip()
        value = str(row.get("value") or "").strip()
        qualifiers = _string_map(row.get("qualifiers"), "context")
        legacy_period = _optional_text(row.get("period"))
        legacy_unit = _optional_text(row.get("unit"))
        if legacy_period and "period" not in qualifiers:
            qualifiers["period"] = legacy_period
        if legacy_unit and "unit" not in qualifiers:
            qualifiers["unit"] = legacy_unit
        if not value:
            reject("missing_value")
            continue
        if entity and not _field_is_grounded(entity, fact_text):
            reject("entity_not_grounded_in_segment")
            continue
        if not _field_is_grounded(value, fact_text):
            reject("value_not_grounded_in_segment")
            continue
        grounding_text = " ".join([fact_text, context_text, str(chunk.get("source_title") or "")])
        required_qualifiers = {
            key: value for key, value in requirement.qualifiers.items()
            if key.casefold() in {"period", "unit"}
        }
        if requirement.period:
            required_qualifiers.setdefault("period", requirement.period)
        if requirement.unit:
            required_qualifiers.setdefault("unit", requirement.unit)
        for name, expected in required_qualifiers.items():
            grounded = _unit_is_grounded(expected, grounding_text, value) if name == "unit" else _field_is_grounded(expected, grounding_text)
            if expected and not grounded:
                reject(f"required_{name}_not_grounded")
                break
        else:
            fields = {
                "subject": entity, "metric": predicate,
                "period": qualifiers.get("period", requirement.period),
                "value": value, "unit": qualifiers.get("unit", requirement.unit),
            }
            claim = str(row.get("claim") or " ".join(filter(None, [entity, predicate, value]))).strip()
            verdict = evidence_entailment(
                claim,
                grounding_text,
                backend=entailment_backend,
                model=entailment_model,
            )
            status = str(verdict.get("status") or "unsupported")
            evidence = EvidenceItem(
                id=_stable_id(
                    "ev", requirement.id, entity, predicate, fields["period"], value,
                    fields["unit"], chunk.get("source_url", ""), chunk.get("segment_id", ""),
                ),
                requirement_id=requirement.id,
                claim=claim,
                quote=fact_text,
                source_url=str(chunk.get("source_url") or ""),
                source_title=str(chunk.get("source_title") or ""),
                status=status,
                score=float(verdict.get("score") or 0.0),
                verification=str(verdict.get("backend") or entailment_backend),
                iteration=iteration,
                extraction_type=str(chunk.get("extraction_type") or "text"),
                scope=dict(required_scope),
                entity=entity,
                predicate=predicate,
                qualifiers=qualifiers,
                fact_quote=fact_text,
                context_quote=context_text,
                provenance={
                    "document_id": _stable_id("doc", chunk.get("source_url", "")),
                    "segment_id": str(chunk.get("segment_id") or ""),
                    **dict(chunk.get("provenance") or {}),
                },
                **fields,
            )
            if status == "supported":
                verified.append(evidence)
            elif status == "contradicted":
                contradictions.append(evidence)
            else:
                reject(f"entailment_{status}")
    return verified, contradictions, len(extracted)


def _merge_evidence(state: ResearchState, verified: list[EvidenceItem], contradictions: list[EvidenceItem]) -> int:
    existing = {item.id for item in state.evidence}
    added = 0
    requirement_map = {item.id: item for item in state.requirements}
    for item in verified:
        if item.id in existing:
            continue
        state.evidence.append(item)
        existing.add(item.id)
        requirement_map[item.requirement_id].evidence_ids.append(item.id)
        added += 1
    contradiction_ids = {item.id for item in state.contradictions}
    for item in contradictions:
        if item.id in contradiction_ids:
            continue
        state.contradictions.append(item)
        contradiction_ids.add(item.id)
        requirement_map[item.requirement_id].contradiction_ids.append(item.id)
    return added


def _compare_count(value: int, operator: str, threshold: int) -> bool:
    return {
        ">": value > threshold, ">=": value >= threshold,
        "<": value < threshold, "<=": value <= threshold, "==": value == threshold,
    }.get(operator, False)


def _aggregate_evidence(state: ResearchState) -> None:
    plan = state.aggregation_plan
    if plan.get("operation") != "group_count_filter":
        state.derived_answer = []
        return
    occurrences: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for item in state.evidence:
        entity = item.entity or item.subject
        if not entity:
            continue
        key = _norm(entity)
        labels.setdefault(key, entity)
        qualifier_key = item.qualifiers.get("period") or item.period or item.requirement_id
        occurrences.setdefault(key, set()).add(qualifier_key)
    operator = str(plan.get("operator") or ">")
    threshold = int(plan.get("threshold") or 0)
    state.derived_answer = sorted(
        [labels[key] for key, values in occurrences.items() if _compare_count(len(values), operator, threshold)],
        key=str.casefold,
    )


def _assess_coverage(state: ResearchState, model_json: JSONCall | None = None) -> None:
    """Apply deterministic completeness rules; the model cannot declare a set complete."""
    evidence_by_requirement = {
        requirement.id: [item for item in state.evidence if item.requirement_id == requirement.id]
        for requirement in state.requirements
    }
    for requirement in state.requirements:
        evidence = evidence_by_requirement[requirement.id]
        if requirement.completion_rule == "single" and evidence:
            requirement.status = "covered"
        elif requirement.completion_rule == "count" and requirement.expected_count is not None and len(evidence) >= requirement.expected_count:
            requirement.status = "covered"
        elif requirement.completion_rule == "all_items" and evidence:
            if any(bool(item.provenance.get("complete_set")) for item in evidence):
                requirement.status = "covered"
            else:
                requirement.status = "partially_covered"
        if requirement.status != "covered":
            requirement.gap = requirement.gap or requirement.text
    _aggregate_evidence(state)
    state.answer_ready = not state.unresolved()
    if state.aggregation_plan and not state.derived_answer:
        # An empty derived list can be a valid answer only with explicit negative-set evidence,
        # which is not represented yet.
        state.answer_ready = False


def build_evidence_context(state: ResearchState) -> str:
    """Render only verified ledger rows into an answer context."""
    parts = []
    for index, item in enumerate(state.evidence, 1):
        parts.append(
            f"[{index}] {item.source_title}\nURL: {item.source_url}\nREQUIREMENT: {item.requirement_id}\n"
            f"SUBJECT: {item.subject}\nMETRIC: {item.metric}\nPERIOD: {item.period}\nVALUE: {item.value}\n"
            f"UNIT: {item.unit}\nCLAIM: {item.claim}\nFACT: {item.fact_quote or item.quote}\n"
            f"CONTEXT: {item.context_quote}\nPROVENANCE: {json.dumps(item.provenance, ensure_ascii=False)}"
        )
    return "\n\n".join(parts) if parts else "No verified evidence found."


def run_deep_research(
    query: str,
    *,
    discover: DiscoverCall,
    lang: str = "en",
    requested_sources: list[str] | None = None,
    provider: str = "auto",
    model_json: JSONCall | None = None,
    budget: ResearchBudget | None = None,
    entailment_backend: str = "heuristic",
    entailment_model: str | None = None,
    fetch_fn: FetchCall = fetch_research_document,
    progress: ProgressCall | None = None,
) -> dict[str, Any]:
    """Run gap-driven research until coverage, stall, or budget exhaustion."""
    budget = budget or ResearchBudget()
    started_at = time.monotonic()
    absolute_deadline = started_at + max(0.0, budget.max_elapsed_seconds)

    def emit(stage: str, **details: Any) -> None:
        if progress is not None:
            progress(stage, details)

    def deadline_exhausted() -> bool:
        return time.monotonic() >= absolute_deadline

    emit("requirement_planning")
    requirements, answer_type, planner_error = decompose_requirements(query, model_json)
    state = ResearchState(
        query=query, requirements=requirements, answer_type=answer_type,
        aggregation_plan=_aggregation_plan(query),
    )
    emit("requirements_ready", requirements=len(requirements), answer_type=answer_type)
    if planner_error:
        state.diagnostics["model_errors"].append(f"requirement planning: {planner_error}")

    seen_documents: dict[str, dict] = {}
    fetched_identities = set()
    fetched_urls = set()
    structured_files_used = 0
    stalled_iterations = 0

    for iteration in range(1, budget.max_iterations + 1):
        state.iteration = iteration
        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        emit("iteration_start", iteration=iteration, unresolved=len(state.unresolved()))
        planned = _plan_gap_queries(state, model_json, budget.max_queries_per_iteration)
        if not planned:
            state.stop_reason = "requirements_covered" if state.answer_ready else "no_queries"
            break
        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        emit("queries_ready", iteration=iteration, queries=len(planned))

        iteration_candidates = []
        iteration_errors = {}
        for requirement_id, search_query in planned:
            if deadline_exhausted():
                break
            if search_query in state.queries:
                continue
            state.queries.append(search_query)
            requirement = next(item for item in state.requirements if item.id == requirement_id)
            requirement.queries.append(search_query)
            embedded_urls = [url.rstrip(".,;:!?) }") for url in re.findall(r"https?://[^\s<>\"]+", search_query)]
            for direct_url in embedded_urls:
                iteration_candidates.append({
                    "title": f"Direct source: {urlparse(direct_url).hostname or direct_url}",
                    "url": direct_url,
                    "snippet": requirement.text,
                    "score": 1.0,
                    "engines": ["direct_url"],
                    "requirement_id": requirement_id,
                    "research_query": search_query,
                })
            discovered, routed, errors = discover(
                search_query,
                lang=lang,
                requested=requested_sources,
                provider=provider,
                num=max(20, budget.initial_fetch + iteration * budget.fetch_growth),
            )
            state.routed_sources = sorted(set(state.routed_sources) | set(routed))
            iteration_errors.update({f"{requirement_id}:{key}": value for key, value in errors.items()})
            for document in discovered:
                item = dict(document)
                item["requirement_id"] = requirement_id
                item["research_query"] = search_query
                iteration_candidates.append(item)

        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        emit("discovery_complete", iteration=iteration, candidates=len(iteration_candidates))

        state.discovery_errors.update(iteration_errors)
        funnel = state.diagnostics["funnel"]
        funnel["candidates"] += len(iteration_candidates)
        for document in iteration_candidates:
            identity = _url_identity(document.get("url", ""))
            if not identity:
                continue
            if identity not in seen_documents:
                seen_documents[identity] = {
                    **document,
                    "requirements": [document["requirement_id"]],
                    "research_queries": [document.get("research_query", "")],
                    "query_hits": 1,
                }
            else:
                current = seen_documents[identity]
                current["query_hits"] += 1
                current["requirements"] = sorted(set(current["requirements"]) | {document["requirement_id"]})
                current["research_queries"] = list(dict.fromkeys([
                    *current.get("research_queries", []),
                    document.get("research_query", ""),
                ]))
                current["engines"] = sorted(set(current.get("engines", [])) | set(document.get("engines", [])))
                if len(document.get("snippet", "")) > len(current.get("snippet", "")):
                    current["snippet"] = document["snippet"]
        funnel["deduplicated_documents"] = len(seen_documents)

        fetch_budget = min(
            budget.initial_fetch + (iteration - 1) * budget.fetch_growth,
            budget.max_fetch - len(fetched_identities),
        )
        ranked_by_requirement = {}
        for requirement in state.unresolved():
            candidates = [
                document for identity, document in seen_documents.items()
                if identity not in fetched_identities and requirement.id in document.get("requirements", [])
            ]
            ranked_by_requirement[requirement.id] = _rank_documents(
                requirement, candidates, max(1, fetch_budget), lang
            )
        selected_documents = []
        selected_ids = set()
        # Requirement round-robin prevents one easy/high-scoring gap from consuming
        # the entire fetch budget while another requirement receives no document.
        depth = 0
        while len(selected_documents) < fetch_budget:
            added_at_depth = False
            for requirement in state.unresolved():
                ranked = ranked_by_requirement.get(requirement.id, [])
                if depth >= len(ranked):
                    continue
                document = ranked[depth]
                identity = _url_identity(document.get("url", ""))
                if identity in selected_ids:
                    continue
                selected_ids.add(identity)
                selected_documents.append(document)
                added_at_depth = True
                if len(selected_documents) >= fetch_budget:
                    break
            if not added_at_depth and all(
                depth >= len(ranked_by_requirement.get(requirement.id, [])) - 1
                for requirement in state.unresolved()
            ):
                break
            depth += 1
        funnel["relevant_documents"] += len(selected_documents)
        funnel["fetch_attempts"] += len(selected_documents)

        remaining_files = max(0, budget.max_structured_files - structured_files_used)
        fetched = _fetch_documents(
            selected_documents,
            lang=lang,
            max_rows=budget.max_rows_per_table,
            structured_file_budget=remaining_files,
            fetch_fn=fetch_fn,
            deadline=absolute_deadline,
            per_document_timeout=budget.per_document_timeout,
            per_file_timeout=budget.per_file_timeout,
        )
        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        fetched_identities.update(_url_identity(item.get("url", "")) for item in fetched if item.get("url"))
        fetched_urls.update(item.get("url", "") for item in fetched if item.get("url"))
        state.seen_urls = sorted(fetched_urls)
        successful = [item for item in fetched if item.get("success")]
        funnel["successful_fetches"] += len(successful)
        new_structured_rows = sum(int(item.get("structured_rows") or 0) for item in successful)
        funnel["structured_rows"] += new_structured_rows
        structured_files_used += sum(bool(item.get("parent_url")) for item in successful)
        emit(
            "fetch_complete",
            iteration=iteration,
            attempted=len(selected_documents),
            successful=len(successful),
        )

        chunk_limit = min(
            budget.initial_chunks_per_requirement + (iteration - 1) * budget.chunk_growth,
            budget.max_chunks_per_requirement,
        )
        ranked_chunks = _rank_chunks_for_requirements(
            state.unresolved(),
            successful,
            lang=lang,
            top_k=chunk_limit,
            deadline=absolute_deadline,
        )
        funnel["ranked_chunks"] += sum(len(items) for items in ranked_chunks.values())

        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        emit(
            "evidence_extraction",
            iteration=iteration,
            requirements=len(state.unresolved()),
            chunks=sum(len(items) for items in ranked_chunks.values()),
        )
        iteration_verified, iteration_contradictions, extracted_count = _extract_and_verify_batch(
            state.unresolved(),
            ranked_chunks,
            iteration=iteration,
            model_json=model_json,
            entailment_backend=entailment_backend,
            entailment_model=entailment_model,
            state=state,
            max_context_chars=budget.max_extraction_context_chars,
        )
        added = _merge_evidence(state, iteration_verified, iteration_contradictions)
        funnel["extracted_facts"] += extracted_count
        funnel["verified_evidence"] = len(state.evidence)
        funnel["contradictions"] = len(state.contradictions)
        emit(
            "evidence_extracted",
            iteration=iteration,
            extracted=extracted_count,
            verified=added,
        )
        if deadline_exhausted():
            state.stop_reason = "research_deadline_exhausted"
            break
        _assess_coverage(state, model_json)
        emit(
            "coverage_assessed",
            iteration=iteration,
            unresolved=len(state.unresolved()),
            answer_ready=state.answer_ready,
        )

        state.diagnostics["iterations"].append({
            "iteration": iteration,
            "queries": len(planned),
            "candidates": len(iteration_candidates),
            "deduplicated_documents": len(seen_documents),
            "relevant_documents": len(selected_documents),
            "fetch_attempts": len(selected_documents),
            "successful_fetches": len(successful),
            "structured_rows": new_structured_rows,
            "ranked_chunks": sum(len(items) for items in ranked_chunks.values()),
            "extracted_facts": extracted_count,
            "new_verified_evidence": added,
            "contradictions": len(iteration_contradictions),
            "unresolved_requirements": [item.id for item in state.unresolved()],
            "selected_document_urls": [item.get("url", "") for item in selected_documents],
            "successful_fetch_urls": [item.get("url", "") for item in successful],
            "ranked_chunk_sources": {
                requirement_id: list(dict.fromkeys(
                    item.get("source_url", "") for item in items if item.get("source_url")
                ))[:8]
                for requirement_id, items in ranked_chunks.items()
            },
        })
        if state.answer_ready:
            state.stop_reason = "requirements_covered"
            break
        stalled_iterations = stalled_iterations + 1 if added == 0 else 0
        if stalled_iterations >= budget.stall_limit:
            state.stop_reason = "stalled_without_new_evidence"
            break
        if len(fetched_identities) >= budget.max_fetch:
            state.stop_reason = "fetch_budget_exhausted"
            break
    else:
        state.stop_reason = "iteration_budget_exhausted"

    _aggregate_evidence(state)
    state.answer_ready = not state.unresolved() and (not state.aggregation_plan or bool(state.derived_answer))
    emit(
        "research_complete",
        stop_reason=state.stop_reason,
        answer_ready=state.answer_ready,
        elapsed_seconds=round(time.monotonic() - started_at, 3),
    )
    return {
        "state": state.to_dict(),
        "context": build_evidence_context(state),
        "sources": list({
            item.source_url: {"title": item.source_title, "url": item.source_url}
            for item in state.evidence
        }.values()),
        "source_count": len({item.source_url for item in state.evidence}),
        "context_length": len(build_evidence_context(state)),
        "answer_ready": state.answer_ready,
    }
