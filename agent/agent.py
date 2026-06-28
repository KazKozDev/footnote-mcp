#!/usr/bin/env python3
"""LangGraph agent — plans, executes step-by-step, evaluates, re-plans.

Nodes:
  plan        → break task into steps, pick first action
  execute     → run tool(s), collect results  
  evaluate    → enough info? → answer / replan
  answer      → draft a source-grounded JSON answer
  verify      → validate claims against the same sources
  strategy    → evolve search strategy after insufficient evidence
"""

import asyncio
import ast
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version.*",
    category=Warning,
)

from langgraph.graph import StateGraph, END

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tools_data import build_research_debug_report

# ── config ──────────────────────────────────────────────────────────────
SERVER_CMD = [sys.executable, os.path.join(PROJECT_ROOT, "server.py")]
MAX_PLAN_STEPS = 40
MAX_EVIDENCE_ROUNDS = 5
TOOL_RESULT_MAX_CHARS = 20000
SOURCE_CONTENT_MAX_CHARS = 25000
TOTAL_SOURCES_MAX_CHARS = 100000
INSUFFICIENT_EVIDENCE_MESSAGE = "The found sources do not provide enough information for a reliable answer."

# ── model picker ────────────────────────────────────────────────────────
def pick_model() -> tuple[str, str]:
    try:
        models = ollama.list()
    except Exception as e:
        print(f"[!] ollama not reachable: {e}")
        sys.exit(1)
    if not models.models:
        print("[!] No models. Run: ollama pull qwen2.5:7b")
        sys.exit(1)

    print(f"\nOllama models ({len(models.models)} total):\n")
    for i, m in enumerate(models.models, 1):
        size_gb = (m.size or 0) / (1024**3)
        print(f"  {i:>2}. {m.model:<35} {size_gb:5.1f} GB")

    available = [m.model or "" for m in models.models]

    DEFAULT_MAIN = "gemma4:26b-mlx"

    def _pick(prompt: str, default: str) -> str:
        default_idx = next((i + 1 for i, m in enumerate(available) if m == default), None)
        hint = f" [Enter = {default}]" if default_idx else ""
        while True:
            try:
                choice = input(f"\n{prompt}{hint} > ").strip()
                if not choice and default_idx:
                    return default
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx]
            except (ValueError, IndexError):
                pass
            print(f"  Enter 1–{len(available)}")

    main_model = _pick("Pick model number", DEFAULT_MAIN)
    print(f"\nUsing: {main_model}")
    return main_model, main_model


# ── MCP tools ───────────────────────────────────────────────────────────
def _tool_schema_list(result) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in result.tools
    ]


async def load_mcp_tools() -> list[dict]:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return _tool_schema_list(result)


# ── helpers ─────────────────────────────────────────────────────────────
def _get_dict(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return msg


def _ollama_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    system: str = "",
    temperature: float = 0.3,
    json_mode: bool = False,
    num_predict: int = 32768,
) -> dict:
    """Call ollama, normalize tool_calls to have 'id'."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in messages:
        d = _get_dict(m)
        entry = {}
        role = d.get("role") or d.get("type", "")
        if role in ("human", "user"):
            entry["role"] = "user"
        elif role in ("ai", "assistant"):
            entry["role"] = "assistant"
        elif role == "tool":
            entry["role"] = "tool"
        elif role == "system":
            entry["role"] = "system"
        else:
            continue
        if d.get("content"):
            entry["content"] = d["content"]
        if d.get("name"):
            entry["name"] = d["name"]
        if d.get("tool_call_id"):
            entry["tool_call_id"] = d["tool_call_id"]
        # tool_calls: convert langchain flat back to ollama nested
        if d.get("tool_calls"):
            tcs = []
            for tc in d["tool_calls"]:
                if "function" in tc:
                    tcs.append(tc)
                else:
                    tcs.append({"function": {"name": tc.get("name", ""), "arguments": tc.get("args", {})}})
            entry["tool_calls"] = tcs
        msgs.append(entry)

    kwargs: dict = {"model": model, "messages": msgs, "tools": tools, "options": {"temperature": temperature, "num_predict": num_predict}, "think": False}
    if json_mode and not tools:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    msg = response["message"]

    if msg.get("tool_calls"):
        import uuid as _uuid
        tcs = []
        for tc in msg["tool_calls"]:
            d = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            d["id"] = f"call_{_uuid.uuid4().hex[:8]}"
            tcs.append(d)
        return {"role": "assistant", "content": "", "tool_calls": tcs}
    return {"role": "assistant", "content": msg.get("content", "")}


async def _call_mcp_tool(name: str, args: dict, session: ClientSession | None = None) -> str:
    """Execute one MCP tool, return result text."""
    if session is not None:
        result = await session.call_tool(name, args)
        text = ""
        for c in result.content:
            if hasattr(c, "text") and isinstance(c.text, str):  # type: ignore[union-attr]
                text += c.text  # type: ignore[union-attr]
        return text[:TOOL_RESULT_MAX_CHARS]

    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            text = ""
            for c in result.content:
                if hasattr(c, "text") and isinstance(c.text, str):  # type: ignore[union-attr]
                    text += c.text  # type: ignore[union-attr]
            return text[:TOOL_RESULT_MAX_CHARS]


def _extract_json_text(content: str) -> str:
    """Return the most likely JSON object/array substring from an LLM response."""
    content = content.strip()

    fenced = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    fenced = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(content)):
            char = content[idx]
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
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return content[start : idx + 1]
    return content


def _ollama_chat_json(model: str, messages: list[dict], system: str, temperature: float = 0) -> str:
    """Call ollama expecting JSON; retry once with explicit nudge if result is unparseable."""
    response = _ollama_chat(model, messages, tools=None, system=system, temperature=temperature, json_mode=True)
    content = response.get("content", "")
    try:
        json.loads(_extract_json_text(content))
        return content
    except (json.JSONDecodeError, TypeError):
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Your response was not valid JSON. Return valid JSON only — no explanation, no markdown."},
        ]
        retry = _ollama_chat(model, retry_messages, tools=None, system=system, temperature=0, json_mode=True)
        return retry.get("content", content)


def _json_loads_best_effort(content: str, fallback):
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        pass
    try:
        return json.loads(_extract_json_text(content))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _slug_key(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:80] or "item"


class ResearchMemoryStore:
    """Persistent JSON-backed strategy/skill memory for research tasks."""

    def __init__(self, path: str | None = None):
        default_path = os.getenv("WEBOPERATOR_RESEARCH_MEMORY", "~/.weboperator-mcp/research_memory.json")
        self.path = Path(path or default_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    return {
                        "strategies": data.get("strategies", {}),
                        "skills": data.get("skills", {}),
                        "experiences": data.get("experiences", []),
                        "metrics": data.get("metrics", {}),
                    }
            except (OSError, json.JSONDecodeError):
                pass
        return {"strategies": {}, "skills": {}, "experiences": [], "metrics": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2, default=str))

    def get_strategies(self, limit: int = 20) -> list[dict]:
        strategies = list(self.data.get("strategies", {}).values())
        strategies.sort(key=lambda item: (item.get("success_rate", 0.0), item.get("wins", 0), item.get("plays", 0)), reverse=True)
        return strategies[:limit]

    def best_strategy(self, min_plays: int = 1, min_success_rate: float = 0.6) -> dict | None:
        for strategy in self.get_strategies():
            if strategy.get("plays", 0) >= min_plays and strategy.get("success_rate", 0.0) >= min_success_rate:
                return strategy
        return None

    def record_strategy(self, desc: str, success: bool, won: bool = False, meta: dict | None = None) -> dict:
        if not desc:
            return {}
        key = _slug_key(desc)
        strategies = self.data.setdefault("strategies", {})
        strategy = strategies.get(key) or {
            "key": key,
            "desc": desc,
            "plays": 0,
            "wins": 0,
            "success_rate": 0.5,
        }
        strategy["plays"] = strategy.get("plays", 0) + 1
        if won:
            strategy["wins"] = strategy.get("wins", 0) + 1
        old = float(strategy.get("success_rate", 0.5))
        strategy["success_rate"] = round(old * 0.7 + (1.0 if success else 0.0) * 0.3, 3)
        strategy["last_used"] = datetime.now().isoformat(timespec="seconds")
        if meta:
            strategy["last_meta"] = meta
        strategies[key] = strategy
        self._save()
        return strategy

    def add_experience(self, exp: dict) -> None:
        experiences = self.data.setdefault("experiences", [])
        exp["timestamp"] = datetime.now().isoformat(timespec="seconds")
        experiences.append(exp)
        self.data["experiences"] = experiences[-500:]
        self._save()

    def save_skill(self, skill: dict) -> None:
        name = skill.get("name") or _slug_key(skill.get("trigger", "research-skill"))
        skill["name"] = name
        skill["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.data.setdefault("skills", {})[name] = skill
        self._save()

    def get_skills(self, limit: int = 10) -> list[dict]:
        skills = list(self.data.get("skills", {}).values())
        skills.sort(key=lambda item: (item.get("success_rate", 0.0), item.get("use_count", 0)), reverse=True)
        return skills[:limit]


_research_store: ResearchMemoryStore | None = None


def _get_research_store() -> ResearchMemoryStore:
    global _research_store
    if _research_store is None:
        _research_store = ResearchMemoryStore()
    return _research_store


def _clip_text(text: str, limit: int = SOURCE_CONTENT_MAX_CHARS) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0].strip()


def _normalize_source_url(url: str) -> str:
    return (url or "").split("#", 1)[0].rstrip("/")


def _split_deep_search_context(context: str, source_lookup: dict[int, dict]) -> list[dict]:
    if not context:
        return []

    matches = list(re.finditer(r"(?m)^\[(\d+)\]\s+(.+)$", context))
    sources = []
    for pos, match in enumerate(matches):
        src_num = int(match.group(1))
        title = match.group(2).strip()
        content_start = match.end()
        content_end = matches[pos + 1].start() if pos + 1 < len(matches) else len(context)
        source_meta = source_lookup.get(src_num, {})
        content = _clip_text(context[content_start:content_end])
        url = source_meta.get("url", "")
        if content and url:
            sources.append(
                {
                    "title": source_meta.get("title") or title,
                    "url": url,
                    "published": source_meta.get("pub_date") or source_meta.get("published"),
                    "content": content,
                    "kind": "page",
                }
            )
    return sources


def _sources_from_tool_result(tool_name: str, tool_result: str) -> list[dict]:
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        return []

    if tool_name == "web_read":
        text = _clip_text(payload.get("text", ""))
        url = payload.get("url", "")
        if payload.get("error") or not text or not url:
            return []
        source_quality = payload.get("source_type") if isinstance(payload.get("source_type"), dict) else {}
        return [
            {
                "title": payload.get("title") or url,
                "url": url,
                "published": payload.get("pub_date"),
                "content": text,
                "kind": "page",
                "source_quality": source_quality,
            }
        ]

    if tool_name == "web_deep_search":
        source_lookup = {}
        for source in payload.get("sources", []):
            if isinstance(source, dict) and source.get("num"):
                source_lookup[int(source["num"])] = source
        return _split_deep_search_context(payload.get("context", ""), source_lookup)

    if tool_name == "web_extract_tables":
        url = payload.get("url", "")
        tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
        if payload.get("error") or not url or not tables:
            return []
        content = _clip_text(json.dumps({"tables": tables}, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": f"Structured tables from {url}",
                "url": url,
                "published": payload.get("published"),
                "content": content,
                "kind": "table",
            }
        ]

    if tool_name == "web_parse_file":
        url = payload.get("url", "")
        if payload.get("error") or not url:
            return []
        evidence_payload = {
            "file_type": payload.get("file_type"),
            "tables": payload.get("tables", []),
            "pages": payload.get("pages", []),
            "json": payload.get("json"),
        }
        content = _clip_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), 12000)
        if not content:
            return []
        return [
            {
                "title": f"Parsed file from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "file",
            }
        ]

    if tool_name == "web_fetch_json":
        url = payload.get("url", "")
        if payload.get("error") or not url or "json" not in payload:
            return []
        content = _clip_text(json.dumps({"json": payload.get("json")}, ensure_ascii=False, indent=2), 12000)
        if not content:
            return []
        return [
            {
                "title": f"JSON API response from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "json",
            }
        ]

    if tool_name == "tool_code_run_sandboxed":
        if not payload.get("ok"):
            return []
        _result_raw = payload.get("result")
        result: dict = _result_raw if isinstance(_result_raw, dict) else {}
        _rows_raw = result.get("rows")
        rows: list = _rows_raw if isinstance(_rows_raw, list) else []
        if not rows:
            return []
        url = ""
        for row in rows:
            if isinstance(row, dict) and row.get("source_url"):
                url = row["source_url"]
                break
        url = url or "recipe://sandboxed-extraction"
        content = _clip_text(json.dumps(result, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": f"Sandboxed extraction recipe result from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "recipe_rows",
            }
        ]

    if tool_name == "browser_extract_tables":
        url = payload.get("url", "")
        tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
        if not url or not tables:
            return []
        content = _clip_text(json.dumps({"tables": tables}, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": payload.get("title") or f"Browser tables from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "browser_table",
            }
        ]

    return []


def _merge_sources(existing: list[dict], new_sources: list[dict]) -> list[dict]:
    merged = list(existing or [])
    seen = {_normalize_source_url(src.get("url", "")) for src in merged}
    for source in new_sources:
        key = _normalize_source_url(source.get("url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def _format_sources_for_llm(sources: list[dict]) -> tuple[str, set[int]]:
    parts = []
    valid_ids = set()
    total_chars = 0

    for src_id, source in enumerate(sources or [], 1):
        content = _clip_text(source.get("content", ""))
        if not content:
            continue
        block = (
            f"[{src_id}]\n"
            f"Title: {source.get('title') or 'Untitled'}\n"
            f"URL: {source.get('url') or ''}\n"
            f"Published: {source.get('published') or 'unknown'}\n"
            f"Source type: {(source.get('source_quality') or {}).get('source_type') or source.get('kind') or 'unknown'}\n"
            f"Content:\n{content}\n"
        )
        if total_chars + len(block) > TOTAL_SOURCES_MAX_CHARS and parts:
            break
        total_chars += len(block)
        valid_ids.add(src_id)
        parts.append(block)

    return "\n".join(parts), valid_ids


def _effective_question(task: str) -> str:
    marker = "\n\nNew question:"
    if marker in task:
        return task.rsplit(marker, 1)[1].strip()
    return task.strip()



def _structured_source_count(sources: list[dict]) -> int:
    structured_kinds = {"table", "file", "json", "recipe_rows", "browser_table"}
    return sum(1 for source in sources or [] if isinstance(source, dict) and source.get("kind") in structured_kinds)


def _audit_evidence_state(state: dict[str, Any]) -> dict:
    sources = [source for source in state.get("sources", []) or [] if isinstance(source, dict)]
    gaps = []
    strategy_hints = []

    if not sources:
        gaps.append("No fetched evidence sources are available.")
        strategy_hints.append("Fetch source pages before answering.")

    passed = not gaps
    return {
        "passed": passed,
        "gaps": list(dict.fromkeys(gaps)),
        "strategy_hints": list(dict.fromkeys(strategy_hints)),
        "source_count": len(sources),
        "structured_source_count": _structured_source_count(sources),
    }


def _payload_row_count(payload: dict) -> int:
    if isinstance(payload.get("tables"), list):
        return sum(len(table.get("rows", []) or []) for table in payload["tables"] if isinstance(table, dict))
    if isinstance(payload.get("json"), (dict, list)):
        raw = payload["json"]
        if isinstance(raw, list):
            return len(raw)
        for key in ("rows", "data", "prices", "items", "results"):
            value = raw.get(key) if isinstance(raw, dict) else None
            if isinstance(value, list):
                return len(value)
    _res_raw = payload.get("result")
    _result: dict = _res_raw if isinstance(_res_raw, dict) else {}
    _rows_raw = _result.get("rows")
    _rows: list = _rows_raw if isinstance(_rows_raw, list) else []
    return len(_rows)


ALLOWED_OBSERVATION_ACTION_TAGS = {
    "search_better_sources",
    "search_structured_sources",
    "search_machine_readable",
    "browser_fallback",
    "recipe_candidate",
    "refine_query",
    "stop_and_answer",
}


def _compact_payload_for_observation(payload: dict, max_chars: int = 4000) -> dict:
    compact = {}
    for key in (
        "url",
        "title",
        "pub_date",
        "published",
        "source_type",
        "content_type",
        "status_code",
        "error",
        "count",
        "table_count",
        "file_type",
    ):
        if key in payload:
            compact[key] = payload[key]
    if isinstance(payload.get("text"), str):
        compact["text"] = payload["text"][:max_chars]
    if isinstance(payload.get("tables"), list):
        compact["tables"] = payload["tables"][:2]
    if isinstance(payload.get("downloads"), list):
        compact["downloads"] = payload["downloads"][:10]
    if "json" in payload:
        compact["json_preview"] = json.dumps(payload.get("json"), ensure_ascii=False)[:max_chars]
    if isinstance(payload.get("result"), dict):
        compact["result"] = payload["result"]
    return compact


def _normalize_observation(
    raw: dict | None,
    tool_name: str,
    args: dict,
    payload: dict,
    current_step: str,
) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    source_url = _normalize_source_url(str(args.get("url") or payload.get("url") or ""))
    row_count = _payload_row_count(payload)
    is_structured = tool_name in {"web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"}
    has_rows = row_count > 0 or bool(payload.get("json"))
    source_type = str(raw.get("source_quality") or raw.get("source_type") or "unknown").lower().strip()
    if source_type not in {"primary", "secondary", "aggregator", "blog", "forum", "interactive", "blocked", "unknown"}:
        source_type = "unknown"
    gaps = [str(gap).strip()[:120] for gap in raw.get("gaps", []) if str(gap).strip()]
    next_action_tags = [
        str(tag).strip()
        for tag in raw.get("next_action_tags", [])
        if str(tag).strip() in ALLOWED_OBSERVATION_ACTION_TAGS
    ]
    suggested_queries = []
    for query in raw.get("suggested_queries", []):
        query = re.sub(r"\s+", " ", str(query)).strip()
        if query and query not in suggested_queries:
            suggested_queries.append(query[:300])

    return {
        "tool": tool_name,
        "step": current_step,
        "url": source_url,
        "source_type": source_type,
        "useful": bool(raw.get("useful")),
        "dated": bool(raw.get("dated")),
        "structured": bool(raw.get("structured", is_structured)),
        "has_rows": bool(raw.get("has_rows", has_rows)),
        "row_count": row_count,
        "gaps": list(dict.fromkeys(gaps)),
        "next_action_tags": list(dict.fromkeys(next_action_tags)),
        "suggested_queries": suggested_queries[:4],
        "reason": str(raw.get("reason") or "Observation diagnosis unavailable.")[:500],
    }


def _fallback_observation(tool_name: str, args: dict, payload: dict, current_step: str) -> dict:
    # Used only when the LLM observation diagnosis is unavailable. Records objective
    # facts about the tool result (did content come back) and leaves all strategy
    # decisions — gaps and next actions — to the LLM. No deterministic strategy inference.
    payload = payload if isinstance(payload, dict) else {}
    row_count = _payload_row_count(payload)
    has_rows = row_count > 0 or bool(payload.get("json"))
    if payload.get("error"):
        useful = False
    elif tool_name in {"web_search", "web_deep_search"}:
        useful = bool(payload.get("count") or payload.get("source_count"))
    elif tool_name == "web_read":
        useful = bool(payload.get("text"))
    else:
        useful = has_rows
    raw = {
        "useful": useful,
        "structured": tool_name in {"web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"},
        "has_rows": has_rows,
        "dated": bool(payload.get("pub_date") or payload.get("published")),
        "source_quality": "unknown",
        "gaps": [],
        "next_action_tags": [],
        "suggested_queries": [],
        "reason": "Objective fallback (LLM diagnosis unavailable).",
    }
    return _normalize_observation(raw, tool_name, args, payload, current_step)


def _diagnose_observation_with_model(
    model: str,
    tool_name: str,
    args: dict,
    payload: dict,
    *,
    question: str,
    requirements: dict,
    current_step: str,
) -> dict:
    fallback = _fallback_observation(tool_name, args, payload, current_step)
    prompt = f"""QUESTION:
{question}

TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

CURRENT_STEP:
{current_step}

TOOL_NAME:
{tool_name}

TOOL_ARGUMENTS:
{json.dumps(args, ensure_ascii=False, indent=2)}

TOOL_RESULT_PREVIEW:
{json.dumps(_compact_payload_for_observation(payload), ensure_ascii=False, indent=2)}

Diagnose this observation and choose the next search action tags."""
    try:
        response = _ollama_chat(
            model,
            [{"role": "user", "content": prompt}],
            tools=None, json_mode=True,
            system=OBSERVATION_SYSTEM_PROMPT,
            temperature=0,
        )
        raw = _json_loads_best_effort(response.get("content", ""), {})
        if isinstance(raw, dict):
            observation = _normalize_observation(raw, tool_name, args, payload, current_step)
            if observation["next_action_tags"] or observation["useful"] or observation["gaps"]:
                return observation
    except Exception:
        return fallback
    return fallback


def _record_observation(memory: dict | None, observation: dict, requirements: dict) -> dict:
    memory = _merge_search_memory(memory)
    compact = {
        "tool": observation.get("tool"),
        "url": observation.get("url"),
        "useful": observation.get("useful"),
        "dated": observation.get("dated"),
        "structured": observation.get("structured"),
        "has_rows": observation.get("has_rows"),
        "gaps": observation.get("gaps", []),
        "next_action_tags": observation.get("next_action_tags", []),
        "reason": observation.get("reason", ""),
    }
    memory["observations"].append(compact)
    memory["observations"] = memory["observations"][-50:]
    for gap in observation.get("gaps", []):
        _append_unique(memory["open_gaps"], gap)
    if observation.get("useful"):
        for gap in observation.get("gaps", []):
            if gap in memory["open_gaps"]:
                memory["open_gaps"].remove(gap)
    if observation.get("url") and not observation.get("useful"):
        _append_unique(memory["avoid_urls"], observation["url"])
        m = re.search(r"https?://(?:www\.)?([^/]+)", observation["url"])
        if m:
            _append_unique(memory.setdefault("bad_domains", []), m.group(1).lower())
    for tag in observation.get("next_action_tags", []):
        _append_unique(memory["next_actions"], tag)
    if observation.get("useful") and observation.get("url"):
        criteria = requirements.get("completion_criteria") if isinstance(requirements, dict) else []
        key = _slug_key(" | ".join(criteria[:2]) if criteria else requirements.get("target", "evidence"))
        evidence = memory["evidence_map"].setdefault(key, [])
        entry = {
            "url": observation["url"],
            "tool": observation.get("tool"),
            "dated": observation.get("dated"),
            "structured": observation.get("structured"),
            "row_count": observation.get("row_count", 0),
        }
        if entry not in evidence:
            evidence.append(entry)
    return memory



def _normalize_tool_arguments(name: str, args: dict) -> dict:
    args = dict(args or {})
    if name == "check_date_completeness":
        if args.get("actual_items") is None:
            args["actual_items"] = []
        if args.get("holidays") is None:
            args["holidays"] = []
        args.setdefault("granularity", "day")
        args.setdefault("calendar", "calendar")
    return args


def _tool_call_from_python_like_step(step: str) -> dict | None:
    try:
        parsed = ast.parse(step.strip(), mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    name = call.func.id
    if call.args:
        return None

    allowed = {
        "check_date_completeness",
        "browser_set_date_range",
        "browser_extract_tables_for_date_range",
        "validate_unit_rows",
        "web_fetch_json",
    }
    if name not in allowed:
        return None

    args = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return None
    return {"function": {"name": name, "arguments": _normalize_tool_arguments(name, args)}}


def _tool_call_from_step(step: str) -> dict | None:
    python_like = _tool_call_from_python_like_step(step)
    if python_like:
        return python_like

    if ":" not in step:
        return None
    name, raw_arg = step.split(":", 1)
    name = name.strip()
    raw_arg = raw_arg.strip()
    if not raw_arg:
        return None

    if name == "web_search":
        return {"function": {"name": "web_search", "arguments": {"query": raw_arg, "num": 10}}}
    if name == "web_deep_search":
        return {"function": {"name": "web_deep_search", "arguments": {"query": raw_arg}}}
    if name == "web_read":
        return {"function": {"name": "web_read", "arguments": {"url": raw_arg}}}
    if name == "web_extract_tables":
        return {"function": {"name": "web_extract_tables", "arguments": {"url": raw_arg}}}
    if name == "web_detect_downloads":
        return {"function": {"name": "web_detect_downloads", "arguments": {"url": raw_arg}}}
    if name == "web_parse_file":
        return {"function": {"name": "web_parse_file", "arguments": {"url": raw_arg}}}
    if name == "web_fetch_json":
        return {"function": {"name": "web_fetch_json", "arguments": {"url": raw_arg}}}
    if name == "classify_source":
        return {"function": {"name": "classify_source", "arguments": {"url": raw_arg}}}
    if name == "generate_search_queries":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "generate_search_queries", "arguments": parsed}}
        return {"function": {"name": "generate_search_queries", "arguments": {"task": raw_arg}}}
    if name == "resolve_units":
        return {"function": {"name": "resolve_units", "arguments": {"text": raw_arg}}}
    if name == "validate_unit_rows":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "validate_unit_rows", "arguments": parsed}}
    if name == "source_cache_get":
        return {"function": {"name": "source_cache_get", "arguments": {"url": raw_arg}}}
    if name == "check_date_completeness":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "check_date_completeness", "arguments": _normalize_tool_arguments(name, parsed)}}
    if name == "evidence_entailment":
        if "||" in raw_arg:
            claim, excerpt = raw_arg.split("||", 1)
            return {
                "function": {
                    "name": "evidence_entailment",
                    "arguments": {"claim": claim.strip(), "source_excerpt": excerpt.strip()},
                }
            }
    if name == "tool_spec_propose":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_spec_propose", "arguments": parsed}}
        return {"function": {"name": "tool_spec_propose", "arguments": {"task": raw_arg}}}
    if name == "tool_code_generate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_code_generate", "arguments": parsed}}
    if name == "tool_code_validate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_code_validate", "arguments": parsed}}
        return {"function": {"name": "tool_code_validate", "arguments": {"code": raw_arg}}}
    if name == "tool_code_run_sandboxed":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_code_run_sandboxed", "arguments": parsed}}
    if name == "tool_promote":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_promote", "arguments": parsed}}
    if name == "web_navigate":
        return {"function": {"name": "web_navigate", "arguments": {"url": raw_arg}}}
    if name == "browser_extract_tables":
        return {"function": {"name": "browser_extract_tables", "arguments": {}}}
    if name == "browser_set_date_range":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "browser_set_date_range", "arguments": parsed}}
    if name == "browser_extract_tables_for_date_range":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "browser_extract_tables_for_date_range", "arguments": parsed}}
    return None


def _default_search_memory() -> dict:
    return {
        "attempted_queries": [],
        "read_urls": [],
        "failed_urls": [],
        "search_rounds": [],
        "strategy_notes": [],
        "next_queries": [],
        "current_strategy": "",
        "strategy_candidates": [],
        "empty_structured_attempts": [],
        "observations": [],
        "open_gaps": [],
        "next_actions": [],
        "avoid_urls": [],
        "evidence_map": {},
    }


def _append_unique(items: list, value: str) -> list:
    value = (value or "").strip()
    if value and value not in items:
        items.append(value)
    return items


def _merge_search_memory(memory: dict | None) -> dict:
    merged = _default_search_memory()
    if isinstance(memory, dict):
        for key in merged:
            if isinstance(memory.get(key), list):
                merged[key] = list(memory[key])
            elif key in memory and not isinstance(merged.get(key), list):
                merged[key] = memory[key]
    return merged


def _update_search_memory(memory: dict | None, tool_name: str, args: dict, tool_result: str) -> dict:
    memory = _merge_search_memory(memory)
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        payload = {}

    if tool_name in ("web_search", "web_deep_search"):
        query = str(args.get("query", "")).strip()
        _append_unique(memory["attempted_queries"], query)
        memory["search_rounds"].append(
            {
                "tool": tool_name,
                "query": query,
                "result_count": payload.get("count") or payload.get("source_count") or 0,
                "source_count": payload.get("source_count") or 0,
            }
        )
        for source in payload.get("sources", []):
            if isinstance(source, dict):
                _append_unique(memory["read_urls"], _normalize_source_url(source.get("url", "")))

    if tool_name == "web_read":
        url = _normalize_source_url(args.get("url", ""))
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif payload.get("text"):
            _append_unique(memory["read_urls"], url)
        else:
            _append_unique(memory["failed_urls"], url)

    if tool_name in ("web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"):
        url = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        if tool_name == "tool_code_run_sandboxed":
            _r1 = payload.get("result")
            _result1: dict = _r1 if isinstance(_r1, dict) else {}
            _rr1 = _result1.get("rows")
            _rows1: list = _rr1 if isinstance(_rr1, list) else []
            for row in _rows1:
                if isinstance(row, dict) and row.get("source_url"):
                    url = _normalize_source_url(row["source_url"])
                    break
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif url:
            _append_unique(memory["read_urls"], url)
            table_count = payload.get("table_count") or len(payload.get("tables", []) or [])
            if tool_name == "tool_code_run_sandboxed":
                _r2 = payload.get("result")
                _result2: dict = _r2 if isinstance(_r2, dict) else {}
                _rr2 = _result2.get("rows")
                table_count = len(_rr2 if isinstance(_rr2, list) else [])
            memory["search_rounds"].append(
                {
                    "tool": tool_name,
                    "url": url,
                    "table_count": table_count,
                    "file_type": payload.get("file_type"),
                    "json": "json" in payload,
                }
            )
            if tool_name in {"web_extract_tables", "browser_extract_tables"} and table_count == 0:
                _append_unique(memory["empty_structured_attempts"], url)
            if tool_name == "web_parse_file" and not payload.get("tables") and "json" not in payload:
                _append_unique(memory["empty_structured_attempts"], url)

    if tool_name == "web_detect_downloads":
        url = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif (payload.get("count") or len(payload.get("downloads", []) or [])) == 0:
            _append_unique(memory["empty_structured_attempts"], url)

    return memory


def _strategy_plan_from_memory(question: str, memory: dict | None) -> list[str]:
    memory = _merge_search_memory(memory)
    queries = [query for query in memory.get("next_queries", []) if query]
    if not queries:
        queries = [question]
    # Only search steps. After the search batch, the post-batch LLM decides which
    # result URLs to read (NEXT → web_read: <url>). No placeholder scaffolding.
    return [f"web_search: {query}" for query in queries[:4]]





def _format_search_memory_for_prompt(memory: dict | None) -> str:
    memory = _merge_search_memory(memory)
    store = _get_research_store()
    return json.dumps(
        {
            "attempted_queries": memory["attempted_queries"][-20:],
            "read_urls": memory["read_urls"][-20:],
            "failed_urls": memory["failed_urls"][-20:],
            "unhelpful_urls_this_session": memory.get("avoid_urls", [])[-15:],
            "unhelpful_domains_this_session": memory.get("bad_domains", [])[-15:],
            "empty_structured_attempts": memory["empty_structured_attempts"][-10:],
            "observations": memory["observations"][-10:],
            "open_gaps": memory["open_gaps"][-10:],
            "next_actions": memory["next_actions"][-10:],
            "evidence_map": memory["evidence_map"],
            "recent_search_rounds": memory["search_rounds"][-10:],
            "strategy_notes": memory["strategy_notes"][-5:],
            "proven_strategies": store.get_strategies(limit=5),
            "reusable_skills": store.get_skills(limit=5),
            "barren_domains": list({d for sk in store.get_skills(limit=20) for d in sk.get("barren_domains", [])})[:15],
        },
        ensure_ascii=False,
        indent=2,
    )


def _queries_from_strategy_desc(desc: str, question: str, memory: dict | None, limit: int = 2) -> list[str]:
    if not desc:
        return []
    memory = _merge_search_memory(memory)
    attempted = {query.lower() for query in memory["attempted_queries"]}
    # Use question as-is; desc is a strategy note, not a search term
    candidates = [question]
    fresh = []
    for query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        if query.lower() not in attempted and query not in fresh:
            fresh.append(query)
        if len(fresh) >= limit:
            break
    return fresh


def _assimilate_research(state: dict) -> None:
    """Record research experience regardless of outcome — success and failure both teach."""
    verification = state.get("verification_result", {})
    succeeded = (
        verification.get("task_complete") is True
        and verification.get("insufficient_evidence") is not True
    )

    memory = _merge_search_memory(state.get("search_memory"))
    store = _get_research_store()
    requirements = state.get("requirements_result", {})
    current_strategy = memory.get("current_strategy") or "source-grounded iterative search"

    # Record strategy outcome
    meta = {
        "query_count": len(memory.get("attempted_queries", [])),
        "source_count": len(state.get("sources", [])),
        "requirements": requirements,
        "gaps_resolved": verification.get("gaps", []) if succeeded else [],
    }
    store.record_strategy(current_strategy, success=succeeded, won=succeeded, meta=meta)
    for candidate in memory.get("strategy_candidates", []):
        desc = candidate.get("desc", "")
        if desc and desc != current_strategy:
            store.record_strategy(desc, success=False, won=False, meta={"reason": "not selected"})

    # Collect source domains (useful) and read-but-skipped domains (not useful)
    source_domains: list[str] = []
    for source in state.get("sources", []):
        url = source.get("url", "")
        m = re.search(r"https?://([^/]+)", url)
        if m:
            _append_unique(source_domains, m.group(1).lower())

    read_urls = memory.get("read_urls", [])
    barren_domains: list[str] = []
    for url in read_urls:
        m = re.search(r"https?://([^/]+)", url)
        if m:
            domain = m.group(1).lower()
            if domain not in source_domains:
                _append_unique(barren_domains, domain)

    if succeeded:
        skill = {
            "name": f"research-{_slug_key(str(requirements.get('target') or state.get('task', 'task')))}",
            "trigger": requirements.get("target") or state.get("task", ""),
            "steps": [
                f"Use strategy: {current_strategy}",
                "Search for source pages, then fetch pages before answering.",
                "Verify both source grounding and task completion gaps before finishing.",
            ],
            "source_domains": source_domains[:10],
            "barren_domains": barren_domains[:10],
            "success_rate": 1.0,
            "use_count": 0,
        }
        store.save_skill(skill)

    store.add_experience(
        {
            "task": state.get("task", ""),
            "result": "success" if succeeded else "failure",
            "strategy": current_strategy,
            "requirements": requirements,
            "queries": memory.get("attempted_queries", []),
            "sources": [s.get("url", "") for s in state.get("sources", [])],
            "barren_domains": barren_domains[:10],
        }
    )


def _write_debug_report(state: dict) -> str | None:
    if os.getenv("WEBOPERATOR_DEBUG_REPORTS", "0") != "1":
        return None
    output_dir = Path(os.getenv("WEBOPERATOR_DEBUG_REPORT_DIR", "~/.weboperator-mcp/debug_reports")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_research_debug_report(
        task=state.get("task", ""),
        requirements=state.get("requirements_result", {}),
        search_memory=state.get("search_memory", {}),
        sources=state.get("sources", []),
        verification=state.get("verification_result", {}),
    )
    path = output_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug_key(state.get('task', 'task'))}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return str(path)


def _default_requirements(question: str) -> dict:
    return {
        "target": question,
        "scope": "answer the user question",
        "granularity": None,
        "unit_or_pair": None,
        "required_coverage": "all explicitly requested parts of the question",
        "output_format": "direct answer",
        "completion_criteria": [
            "The answer addresses every explicit requirement in the user question.",
            "Every factual claim is grounded in the provided sources.",
            "The answer uses one consistent unit, currency pair, or measurement basis unless the user explicitly asked for multiple.",
        ],
        "missing_data_policy": "If any required part cannot be sourced, mark task_complete=false and list the gap.",
        "search_hints": [],
    }


def _normalize_requirements(raw: dict | None, question: str) -> dict:
    defaults = _default_requirements(question)
    if not isinstance(raw, dict):
        return defaults
    result = defaults | {key: value for key, value in raw.items() if key in defaults and value not in (None, "")}
    for key in ("completion_criteria", "search_hints"):
        if not isinstance(result.get(key), list):
            result[key] = defaults[key]
        result[key] = [str(item) for item in result[key] if str(item).strip()]
    return result


# ── state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    task: str
    conversation_context: str  # prior Q&A history — injected into plan only
    requirements_result: dict
    plan: list[str]          # remaining plan steps (not yet executed)
    completed_steps: list[dict]  # [{step, result, tools_used}]
    scratchpad: str          # raw collected data
    sources: list[dict]      # normalized page sources used as answer evidence
    draft_result: dict       # structured answer before verification
    verification_result: dict
    evidence_audit: dict
    final_answer: str
    iteration: int
    replan_count: int        # ponytail: hard cap at 3 replans. upgrade: configurable limit when agent gets stuck on legitimate multi-step tasks.
    evidence_round: int      # additional search rounds after insufficient evidence
    search_memory: dict      # live memory for search attempts and strategy evolution


# ── system prompts ──────────────────────────────────────────────────────
TODAY = __import__("datetime").date.today().strftime("%Y-%m-%d")

REQUIREMENTS_SYSTEM_PROMPT = f"""Today is {TODAY}. You extract task completion requirements for a source-grounded research agent.

Convert the user question into explicit requirements that can later be verified against the final answer.
Do not answer the question. Do not use outside knowledge.

Important:
- Extract the required unit, currency pair, measurement basis, denominator, or quote asset into unit_or_pair.
- For exchange-rate tasks, identify both base and quote currency when stated or strongly implied.
- For Russian-language "kurs euro" / "exchange rate of euro" requests with no other quote currency, treat the expected quote currency as RUB unless the user requested another pair.
- If a unit or pair is ambiguous, say so in unit_or_pair and add a completion criterion requiring the final answer to explicitly state the chosen basis.
- Completion criteria must make partial coverage fail when the user requested a full range, list, table, or every item.

Return JSON only:
{{
  "target": "what must be researched",
  "scope": "boundaries of the request",
  "granularity": "requested level of detail or null",
  "unit_or_pair": "required unit/currency pair/measurement basis or null",
  "required_coverage": "what complete coverage means",
  "output_format": "expected answer shape",
  "completion_criteria": [
    "criterion 1",
    "criterion 2"
  ],
  "missing_data_policy": "what to do if a required part is missing",
  "search_hints": ["optional query/source hints"]
}}"""

PLAN_PROMPT = f"""Today is {TODAY}. You are a research planner. Given a task, break it into concrete steps.

If the task includes "Previous conversation" context, use it only to understand follow-up questions.
Do not treat previous answers as evidence; search/read current sources again for the new answer.

Tool roles:
- web_search(query, lang, num): discovery only. It returns URLs and snippets. Search snippets are NOT valid evidence for the final answer.
- web_read(url, lang, use_cache): evidence collection with persistent source cache. Use it after web_search to fetch page text.
- web_extract_tables(url, lang, max_tables, max_rows): structured HTML table extraction with columns, rows, and URL provenance.
- web_detect_downloads(url, lang, max_links): find CSV, TSV, XLS, XLSX, PDF, JSON, and XML files linked from a page.
- web_parse_file(url, lang, max_rows): download and parse CSV, TSV, XLS, XLSX, PDF, or JSON files into structured rows/text with provenance.
- web_fetch_json(url, lang, use_cache, timeout): fetch direct API/JSON endpoints and return parsed JSON with provenance.
- web_deep_search(query, lang): search + fetch + extract + rerank. Use only for web search queries, NOT for analyzing already-fetched content.
- generate_search_queries(task, requirements, max_queries): produce operator-style search queries for difficult data-source discovery.
- classify_source(url, status_code, content_type, text_sample): classify source type and prefer official or primary data sources.
- check_date_completeness(start_date, end_date, actual_items, granularity, calendar, holidays): deterministic coverage validator for required date ranges, including business-day calendars.
- resolve_units(text): normalize currency pairs, currencies, and units before mixing rows from multiple sources.
- validate_unit_rows(rows, expected_unit_or_pair, text_fields): reject structured rows with incompatible units, entities, or currency pairs.
- evidence_entailment(claim, source_excerpt, backend, model): strict support check for a claim against one source excerpt. Use backend="auto" unless a specific judge backend is required.
- tool_spec_propose(task, source_url, observed_failure, desired_output): propose a controlled task-specific extraction recipe when generic tools found a source but failed to return structured rows.
- tool_code_generate(spec): create a small starter recipe function. Generated code must be validated before running.
- tool_code_validate(code): statically validate recipe code against a strict allowlist.
- tool_code_run_sandboxed(code, source_text, input_payload, timeout): run validated recipe code in a limited subprocess. The code must define extract(source_text, input_payload).
- tool_promote(name, spec, code, sample_source_text, input_payload, expected_min_rows): save a validated successful recipe as reusable memory. This does not edit the MCP server.
- source_cache_get(url) and source_cache_put(url, payload): persistent source cache for repeated URLs across retries.
- Browser tools are for interactive pages only; use browser_set_date_range and browser_extract_tables_for_date_range when fetch/table/file tools cannot access date-filtered data.

Rules:
- Each step must be ONE tool call
- Never write Python-like function syntax such as check_date_completeness(...). Use "tool_name: argument" or "tool_name: {{json_object}}" only.
- generate_search_queries is ONLY for structured data/API/CSV/historical dataset discovery. Do NOT use it for news, current events, or general information queries.
- For news or general queries, use web_search directly with ONE short query per step.
- web_search queries must NOT be wrapped in double quotes. Use plain keywords only.
- Each web_search step must contain ONE query only — never a comma-separated list of queries.
- If the task already gives explicit URLs to read, add a web_read step for each of those exact URLs directly — do NOT search for them. Searching for a page you were already handed wastes steps and pulls in unrelated third-party sources.
- Otherwise (when URLs are unknown and must be discovered), do NOT add web_read steps — the system will decide which URLs to read after seeing search results.
- For known API or JSON endpoints, use web_fetch_json directly instead of web_read or web_parse_file.
- For tabular pages, add web_extract_tables after the corresponding web_read step.
- For official pages that may host files, add web_detect_downloads after web_read and web_parse_file for discovered file URLs when available.
- If HTML table extraction and download detection are empty or blocked, switch to API, JSON, and CSV search strategies immediately.
- If a source appears relevant but generic tools cannot extract rows from its text, HTML, JSON, or script blob, use the recipe flow: tool_spec_propose, tool_code_generate, tool_code_validate, then tool_code_run_sandboxed against already fetched source_text. Promote only after a successful smoke run.
- Never ask recipe code to access files, environment variables, subprocesses, or the network. Fetch sources with MCP tools first, then pass source_text/input_payload to the recipe runner.
- For date-range tasks, add check_date_completeness after structured rows are collected. Use calendar="business_day" only when the source or task requires business/trading days.
- For unit, currency, exchange-rate, or measurement tasks, add resolve_units before comparing or merging source rows, then validate_unit_rows when structured rows are available.
- A plan that includes web_search but no evidence-fetching step is invalid. Evidence-fetching steps include web_read, web_deep_search, web_extract_tables, web_parse_file, web_fetch_json, tool_code_run_sandboxed, browser_extract_tables, or browser_extract_tables_for_date_range.
- Never finish research with search results only.
- When the task needs the exact textual structure of a file (headings, formatting, raw markdown, source code), prefer fetching the file's raw/plain form over a rendered web page. Rendered pages flatten headings and lose markup, so verbatim heading or structure extraction fails.
- If this is an additional evidence round, use a different query strategy and fresh sources.
- Prefer primary/official sources when the task asks about a factual current state, historical data, financial data, legal status, official statistics, or exact tables.
- Do not mix incompatible units, entities, base/quote currencies, or denominators.
- Output ONLY a JSON array. No explanations."""

OBSERVATION_SYSTEM_PROMPT = f"""Today is {TODAY}. You are the search controller's observation diagnostician.

Diagnose one tool result against the user question and task requirements.
Use only the provided tool result preview and requirements. Do not use outside knowledge.
Do not decide the final answer. Choose what the controller should try next.

Allowed source_quality values:
primary, secondary, aggregator, blog, forum, interactive, blocked, unknown

Allowed next_action_tags:
search_better_sources, search_structured_sources, search_machine_readable, browser_fallback, recipe_candidate, refine_query, stop_and_answer

Return JSON only:
{{
  "useful": true,
  "structured": false,
  "has_rows": false,
  "dated": false,
  "source_quality": "unknown",
  "gaps": ["short gap"],
  "next_action_tags": ["refine_query"],
  "suggested_queries": ["optional query"],
  "reason": "short reason"
}}"""

EXECUTE_PROMPT = f"""Today is {TODAY}. Execute the next step. You have ONE tool call available.
Call the tool with the best arguments for this step.
Respond with a tool call ONLY — no text before or after."""

EVAL_PROMPT = """You are an evaluator. Given the task, completed steps, search state, and remaining plan, decide the next action.

Decision rules:
- DONE: plan has steps remaining AND sources have been collected AND pending queries duplicate what was already attempted. Also DONE if queries_attempted >= 6 and sources_collected >= 2.
- REPLAN: the plan is clearly wrong, all steps failed, or the pending generated queries suggest a better direction. Include new_plan incorporating pending generated queries. REPLAN must stay anchored to TASK_REQUIREMENTS — do not change the topic or time period.
- CONTINUE: the current plan still has untried steps that differ meaningfully from what was already attempted.

Do NOT continue indefinitely when no new sources are being found despite repeated searches.

Output JSON ONLY: {"decision": "CONTINUE|REPLAN|DONE", "reason": "..."}
Do NOT include new_plan — the system constructs the new plan automatically from pending queries."""

ANSWER_PROSE_SYSTEM_PROMPT = f"""Today is {TODAY}. You are a meticulous research analyst.

Answer exclusively from the SOURCES section. Produce a thorough, well-organized answer — not a terse summary.

GROUNDING RULES (non-negotiable):
1. Use ONLY the SOURCES. Never use your own knowledge, model memory, or assumptions.
2. Cite EVERY factual claim inline as [1], [2], ... [N], matching the source numbers. A sentence with a fact and no citation is a defect.
3. If a source does not contain something, do NOT invent it. Say what is actually present.
4. If sources contradict each other, state the contradiction explicitly with both citations.
5. Never claim information is current unless the source's date supports it.

DEPTH RULES (extract everything — do not summarize):
6. Be exhaustive, not selective. For every entity the task names, go through every part of its source in full and report what each part actually contains. Do not pick only a few parts; do not stop early.
7. When the task asks for specific elements of a source, extract them EXACTLY as written, not paraphrased. Quote the author's actual wording — exact phrases in "quotes" — rather than paraphrasing into generic descriptions.
8. Default to completeness over brevity. A long answer is expected and correct. Never collapse distinct items into "etc." or "and more".
9. Include concrete specifics: names, dates, numbers, commands, code snippets, examples — whatever the sources provide.
10. Structure: first give the full per-entity breakdown covering each entity completely, then, only if the task asks for it, a synthesis at the end. The synthesis never replaces the full breakdown.

Satisfy every item in TASK_REQUIREMENTS in full. Write in plain text / Markdown. Do NOT output JSON."""

VERIFY_PROSE_SYSTEM_PROMPT = f"""Today is {TODAY}. You are a strict grounding verifier.

You receive SOURCES, TASK_REQUIREMENTS, and a DRAFT_ANSWER written by an analyst.
Your job is to return a corrected version of the answer that is fully grounded in SOURCES.

Do this:
1. Check every factual statement in DRAFT_ANSWER against SOURCES.
2. Remove or correct any statement not supported by the SOURCES. Do not keep guesses, outside knowledge, or invented details.
3. Verify each inline [n] citation actually points to a source that supports that statement; fix wrong numbers, remove citations that don't hold.
4. Preserve EVERYTHING that is supported — do NOT shorten, summarize, merge, or drop sections. Keep the full length, every verbatim quote, every item, every example, and the complete per-entity breakdown. Removing supported detail is a defect.
5. Keep the inline [n] citation style and the same overall structure. Your output must be at least as detailed as the draft.

Output ONLY the corrected answer as plain text / Markdown. No preamble, no JSON, no commentary about what you changed. If essentially nothing in the draft is supported by SOURCES, output exactly: {INSUFFICIENT_EVIDENCE_MESSAGE}"""

VERIFY_VERDICT_SYSTEM_PROMPT = f"""Today is {TODAY}. You are a quality auditor.

You receive TASK_REQUIREMENTS, SOURCES, and a FINAL_ANSWER that was already grounded against the sources.
Judge how well FINAL_ANSWER satisfies TASK_REQUIREMENTS using only SOURCES.

Return ONLY this compact JSON (short strings, no long text inside):
{{
  "coverage_complete": true,
  "missing": ["requirement or entity not fully covered"],
  "notes": ["short note, e.g. a requirement only partially met"]
}}

Rules:
- "coverage_complete" is true only if every requirement is fully addressed for every entity the task names.
- "missing" lists short labels of what is not covered. Empty list if nothing is missing.
- Keep every string under ~12 words. Never include the answer text itself."""

STRATEGY_SYSTEM_PROMPT = f"""Today is {TODAY}. You evolve web search strategy after an evidence-grounded answer failed.

You receive the user question, task requirements, previous search memory, current source count, and verifier feedback.
Your job is to propose fresh search queries that target the verifier's gaps and are meaningfully different from previous queries.

Rules:
1. Do not repeat previous queries.
2. Directly target missing requirements or gaps from verifier feedback.
3. Prefer primary, official, data-table, API, CSV, historical-data, documentation, or report sources when relevant.
4. If previous results were broad, make the next queries more specific.
5. If previous results were too specific or empty, broaden the next queries.
6. Include domain/operator style queries only when they are likely to help.
7. Return 2 to 4 queries.

Return JSON only:
{{
  "next_queries": ["query 1", "query 2"],
  "strategy_note": "short explanation of what changed"
}}"""

# ── nodes ───────────────────────────────────────────────────────────────
async def requirements_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Extract task completion requirements before planning."""
    if state.get("requirements_result"):
        return {"iteration": state["iteration"] + 1}

    question = _effective_question(state["task"])
    iteration = state["iteration"] + 1
    print("\n  [REQUIREMENTS] Extracting completion criteria...")
    content = _ollama_chat_json(
        model,
        [{"role": "user", "content": f"QUESTION:\n{question}\n\nExtract task completion requirements."}],
        system=REQUIREMENTS_SYSTEM_PROMPT,
    )
    raw = _json_loads_best_effort(content, {})
    requirements = _normalize_requirements(raw, question)
    criteria = requirements.get("completion_criteria", [])
    print(f"  [REQUIREMENTS] {len(criteria)} criteria")
    for i, criterion in enumerate(criteria[:4], 1):
        print(f"    {i}. {criterion[:100]}")
    return {"requirements_result": requirements, "iteration": iteration}


async def plan_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Break task into steps on first call, or use replanned steps."""
    task = state["task"]
    plan = state["plan"]
    requirements = _normalize_requirements(state.get("requirements_result"), _effective_question(task))
    evidence_round = state.get("evidence_round", 0)
    search_memory = _merge_search_memory(state.get("search_memory"))
    iteration = state["iteration"] + 1

    if plan:
        # already have a plan, use it
        return {"iteration": iteration}

    if evidence_round and search_memory.get("next_queries"):
        question = _effective_question(task)
        steps = _strategy_plan_from_memory(question, search_memory)
        print(f"\n  [PLAN] Using evolved search strategy (retry {evidence_round}/{MAX_EVIDENCE_ROUNDS})")
        print(f"  [PLAN] {len(steps)} steps:")
        for i, s in enumerate(steps, 1):
            print(f"    {i}. {s}")
        search_memory["next_queries"] = []
        return {"plan": steps, "search_memory": search_memory, "iteration": iteration}

    if evidence_round:
        question = _effective_question(task)
        planning_input = f"""Task: {question}

Task requirements:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

The previous answer attempt did not have enough supported evidence.
Additional evidence retry: {evidence_round} of {MAX_EVIDENCE_ROUNDS}.

Search memory:
{_format_search_memory_for_prompt(search_memory)}

Create a new plan with fresh search terms only (web_search steps). Do NOT add web_read steps."""
        print(f"\n  [PLAN] Searching for more evidence (retry {evidence_round}/{MAX_EVIDENCE_ROUNDS})")
    else:
        conv_ctx = state.get("conversation_context", "")
        context_block = f"\nPrior conversation (for context only — do NOT copy into steps):\n{conv_ctx}\n" if conv_ctx else ""
        planning_input = f"""Task: {task}
{context_block}
Task requirements:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

Create a plan (JSON array of step strings)."""
        print(f"\n  [PLAN] Breaking down: {task[:80]}")

    response = _ollama_chat(
        model,
        [{"role": "user", "content": planning_input}],
        tools=None, json_mode=True,
        system=PLAN_PROMPT,
    )
    content = response.get("content", "[]")

    steps = _json_loads_best_effort(content, [f"web_search: {task}"])
    if not isinstance(steps, list):
        steps = [f"web_search: {task}"]

    if not steps:
        steps = [f"web_search: {task}"]

    print(f"  [PLAN] {len(steps)} steps:")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")

    return {"plan": steps, "iteration": iteration}


async def _execute_single_step(
    step: str,
    model: str,
    tools: list[dict],
    mcp_session: ClientSession | None,
    question: str,
    requirements: dict,
) -> tuple[str, str, list, dict]:
    """Execute one resolved step. Returns (step, result_text, sources, search_memory_updates)."""
    deterministic_tool_call = _tool_call_from_step(step)
    if deterministic_tool_call:
        tool_calls = [deterministic_tool_call]
        response: dict = {}
    else:
        response = _ollama_chat(
            model,
            [{"role": "user", "content": f"Execute this step using ONE tool call: {step}"}],
            tools=tools,
            system=EXECUTE_PROMPT,
        )
        tool_calls = response.get("tool_calls", [])

    result_text = ""
    step_sources: list = []
    sm_updates: dict = {}

    if not tool_calls:
        content = response.get("content", "")
        if content:
            result_text = content
    else:
        for tc in tool_calls:
            name = tc.get("function", tc).get("name", "?")
            args = tc.get("function", tc).get("arguments", tc.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"    [{name}]", end=" ", flush=True)
            try:
                tool_result = await _call_mcp_tool(name, args, session=mcp_session)
                result_text += f"\n--- {name} ---\n{tool_result}\n"
                sm_updates = {"name": name, "args": args, "result": tool_result}
                tool_payload = _json_loads_best_effort(tool_result, {})
                observation = _diagnose_observation_with_model(
                    model, name, args,
                    tool_payload if isinstance(tool_payload, dict) else {},
                    question=question, requirements=requirements, current_step=step,
                )
                sm_updates["observation"] = observation
                step_sources = _sources_from_tool_result(name, tool_result)
                print(f"→ {len(tool_result)} chars")
            except Exception as e:
                result_text += f"\n[{name} error: {e}]\n"
                print(f"✗ {str(e)[:60]}")

    return step, result_text, step_sources, sm_updates


async def execute_node(state: AgentState, model: str, tools: list[dict], mcp_session: ClientSession | None = None, fast_model: str = "") -> dict:
    """Execute a batch of same-tool plan steps in parallel."""
    plan = state["plan"]
    completed = list(state["completed_steps"])
    scratchpad = state["scratchpad"]
    sources = list(state.get("sources", []))
    search_memory = _merge_search_memory(state.get("search_memory"))
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    iteration = state["iteration"] + 1

    if iteration > MAX_PLAN_STEPS * 3:
        return {"iteration": iteration}

    if not plan:
        return {"iteration": iteration}

    # Collect batch: all consecutive steps sharing the same tool prefix
    current_tool = plan[0].split(":", 1)[0].strip()
    batch: list[str] = []
    for step in plan:
        if step.split(":", 1)[0].strip() == current_tool:
            batch.append(step)
        else:
            break
    remaining = plan[len(batch):]

    # Batch steps are concrete tool calls (the planner and the post-batch LLM emit real
    # URLs directly). No placeholder resolution, no deterministic URL picking/filtering.
    final_steps = batch

    # Execute all steps in batch in parallel
    # web_search is rate-limited by DDG — cap concurrency at 2
    is_search_batch = current_tool == "web_search"
    concurrency = 2 if is_search_batch else len(final_steps)
    print(f"\n  [EXEC] Batch ({len(final_steps)} steps, concurrency={concurrency}):")
    for s in final_steps:
        print(f"    • {s[:100]}")

    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(step: str):
        async with sem:
            if is_search_batch and concurrency < len(final_steps):
                await asyncio.sleep(0.5)
            return await _execute_single_step(step, model, tools, mcp_session, question, requirements)

    batch_results = await asyncio.gather(*[run_with_sem(s) for s in final_steps])

    # Merge results sequentially into state
    for step, result_text, step_sources, sm_updates in batch_results:
        scratchpad += f"\n## Step: {step}\n{result_text}\n"
        sources = _merge_sources(sources, step_sources)
        if sm_updates:
            name = sm_updates.get("name", "")
            args = sm_updates.get("args", {})
            tool_result = sm_updates.get("result", "")
            search_memory = _update_search_memory(search_memory, name, args, tool_result)
            observation = sm_updates.get("observation")
            if observation:
                search_memory = _record_observation(search_memory, observation, requirements)
            if name == "generate_search_queries":
                generated = _json_loads_best_effort(tool_result, {})
                queries = generated.get("queries", []) if isinstance(generated, dict) else []
                attempted = {q.lower() for q in search_memory.get("attempted_queries", [])}
                for query in queries:
                    query = str(query).strip()
                    if query and query.lower() not in attempted:
                        _append_unique(search_memory.setdefault("next_queries", []), query)
        completed.append({
            "step": step,
            "result": result_text,
            "tools_used": [],
        })

    # Post-batch decision: LLM sees what was found and decides next action
    POST_BATCH_PROMPT = (
        "You are a research agent. After executing a batch of steps, decide what to do next.\n"
        "Return JSON only: "
        '{"decision": "DONE"|"CONTINUE"|"NEXT", '
        '"next_steps": ["step1", ...], '
        '"reason": "one line"}\n'
        "- DONE: all task requirements are fully covered by fetched sources\n"
        "- CONTINUE: execute remaining planned steps as-is\n"
        "- NEXT: after web_search, use NEXT to add web_read steps for the most relevant URLs; "
        "also use NEXT if remaining steps are wrong — provide better next_steps\n"
        "After a web_search batch with no remaining plan: use NEXT with web_read steps for the best URLs.\n"
        "Be thorough — prefer reading more sources if requirements are only partially covered."
    )
    completion_criteria = requirements.get("completion_criteria", [])
    read_urls = [c["step"].split(": ", 1)[1] for c in completed if c["step"].startswith("web_read: ")]
    unhelpful_urls = search_memory.get("avoid_urls", [])[-15:]
    unhelpful_domains = search_memory.get("bad_domains", [])[-15:]
    post_input = (
        f"Task: {question}\n\n"
        f"Completion criteria (ALL must be met before DONE):\n"
        + "\n".join(f"  - {c}" for c in completion_criteria)
        + f"\n\nLast batch executed: {len(final_steps)} {current_tool} steps\n"
        f"Steps completed so far ({len(completed)} total):\n"
        + "\n".join(f"  - {c['step']}" for c in completed[-8:])
        + f"\n\nURLs already read (do NOT revisit these):\n"
        + ("\n".join(f"  - {u}" for u in read_urls) if read_urls else "  (none yet)")
        + (f"\n\nURLs/domains that proved unhelpful this session (do NOT read these again):\n"
           + "\n".join(f"  - {u}" for u in unhelpful_urls + unhelpful_domains)
           if (unhelpful_urls or unhelpful_domains) else "")
        + f"\n\nFetched sources (pages actually read): {len(sources)}\n"
        f"Remaining plan: {[s[:60] for s in remaining[:4]] if remaining else '(empty — no more steps planned)'}\n\n"
        f"Recent findings:\n{scratchpad[-2500:]}"
    )
    post_content = _ollama_chat_json(model, [{"role": "user", "content": post_input}], system=POST_BATCH_PROMPT)
    post = _json_loads_best_effort(post_content, {"decision": "CONTINUE"})
    post_decision = post.get("decision", "CONTINUE").upper()

    # DONE requires at least one fetched source — search snippets are not enough
    if post_decision == "DONE" and not sources:
        post_decision = "NEXT"
    print(f"  [POST-BATCH] {post_decision} — {post.get('reason', '')[:80]}")

    if post_decision == "DONE":
        remaining = []
    elif post_decision == "NEXT":
        next_steps = post.get("next_steps", [])
        if next_steps and isinstance(next_steps, list):
            remaining = [str(s) for s in next_steps] + remaining

    return {
        "plan": remaining,
        "completed_steps": completed,
        "scratchpad": scratchpad,
        "sources": sources,
        "search_memory": search_memory,
        "iteration": iteration,
    }


async def evaluate_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Decide: CONTINUE, REPLAN, or DONE."""
    plan = state["plan"]
    completed = state["completed_steps"]
    scratchpad = state["scratchpad"]
    task = state["task"]
    iteration = state["iteration"] + 1

    if not plan:
        audit = _audit_evidence_state(state)  # type: ignore[arg-type]
        print(f"  [EVAL] sources in state: {len(state.get('sources', []))}")
        if audit.get("passed"):
            print("  [EVAL] Evidence audit passed → ANSWER")
            return {"evidence_audit": audit, "iteration": iteration}
        print("  [EVAL] Evidence audit failed → STRATEGY")
        for gap in audit.get("gaps", [])[:4]:
            print(f"    gap: {gap[:120]}")
        search_memory = _merge_search_memory(state.get("search_memory"))

        # Reflexion: LLM diagnoses why search failed and what to try next
        reflection_input = f"""Task: {_effective_question(state['task'])}

Gaps identified:
{json.dumps(audit.get('gaps', []), ensure_ascii=False)}

Queries already attempted:
{json.dumps(search_memory.get('attempted_queries', []), ensure_ascii=False)}

Sources collected: {len(state.get('sources', []))}
Scratchpad (last 1500 chars):
{state.get('scratchpad', '')[-1500:]}

In 2-3 sentences, diagnose WHY the search failed and what specific approach should fix it next round."""

        reflection_response = _ollama_chat(
            model,
            [{"role": "user", "content": reflection_input}],
            tools=None, system="You are a search strategist. Be concise and specific. Output plain text only.",
        )
        reflection = (reflection_response.get("content") or "").strip()
        if reflection:
            print(f"  [REFLECT] {reflection[:200]}")
            search_memory.setdefault("reflections", []).append(reflection)

        verification = {
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claim_checks": [],
            "task_complete": False,
            "coverage": {
                "requirements_addressed": [],
                "overall_status": "missing",
            },
            "gaps": audit.get("gaps", []),
            "insufficient_evidence": True,
            "audit": audit,
        }
        return {
            "evidence_audit": audit,
            "verification_result": verification,
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "search_memory": search_memory,
            "evidence_round": state.get("evidence_round", 0) + 1,
            "iteration": iteration,
        }

    # if we've done too many steps, force answer
    replan_count = state.get("replan_count", 0)
    if len(completed) >= MAX_PLAN_STEPS or replan_count >= 3:
        reason = "max steps" if len(completed) >= MAX_PLAN_STEPS else "too many replans"
        print(f"  [EVAL] {reason} → DONE")
        return {"plan": [], "iteration": iteration}

    # ask LLM to evaluate
    search_memory = _merge_search_memory(state.get("search_memory"))
    requirements = _normalize_requirements(state.get("requirements_result"), _effective_question(task))
    eval_input = f"""Task: {task}

TASK_REQUIREMENTS (must not be abandoned during replan):
{json.dumps(requirements.get('completion_criteria', []), ensure_ascii=False)}

Completed steps ({len(completed)} total):
{json.dumps([c['step'] for c in completed[-10:]], ensure_ascii=False)}

Sources collected: {len(state.get('sources', []))}
Queries attempted ({len(search_memory.get('attempted_queries', []))} total): {json.dumps(search_memory.get('attempted_queries', [])[-6:], ensure_ascii=False)}
Pending generated queries: {json.dumps(search_memory.get('next_queries', [])[:4], ensure_ascii=False)}
Open gaps: {json.dumps(search_memory.get('open_gaps', [])[:5], ensure_ascii=False)}

Remaining plan ({len(plan)} steps):
{json.dumps(plan[:8], ensure_ascii=False)}

Summary of findings (last 2000 chars):
{scratchpad[-2000:]}"""

    print("  [EVAL] Assessing progress...", end=" ", flush=True)
    response = _ollama_chat(model, [{"role": "user", "content": eval_input}], tools=None, system=EVAL_PROMPT, json_mode=True)
    content = response.get("content", "{}")

    decision = _json_loads_best_effort(content, {"decision": "CONTINUE", "reason": "parse error"})

    d = decision.get("decision", "CONTINUE").upper()
    print(d)

    if d == "REPLAN":
        # Build plan from search_memory instead of LLM new_plan:
        # LLM decides WHETHER to replan; plan construction is deterministic
        # to avoid malformed/duplicate queries from LLM free-form generation.
        pending_queries = search_memory.get("next_queries", [])
        attempted = {q.lower() for q in search_memory.get("attempted_queries", [])}
        fresh_queries = [q for q in pending_queries if q.lower() not in attempted]
        if not fresh_queries:
            fresh_queries = [_effective_question(task)]
        normalized = [f"web_search: {q}" for q in fresh_queries[:4]]
        search_memory["next_queries"] = []
        print(f"  [REPLAN] New plan: {len(normalized)} steps")
        for i, s in enumerate(normalized, 1):
            print(f"    {i}. {s[:100]}")
        return {"plan": normalized, "replan_count": replan_count + 1, "search_memory": search_memory, "iteration": iteration}

    if d == "DONE":
        return {"plan": [], "iteration": iteration}

    return {"iteration": iteration}


def route_after_execute(state: AgentState) -> str:
    """Route after each execute step — evaluate only when plan is exhausted."""
    plan = state["plan"]
    completed = state["completed_steps"]
    if len(completed) >= MAX_PLAN_STEPS:
        return "evaluate"
    if not plan:
        return "evaluate"
    return "execute"


def route_after_evaluate(state: AgentState) -> str:
    plan = state["plan"]
    completed = state["completed_steps"]

    audit = state.get("evidence_audit", {})
    if audit and audit.get("passed") is False and state.get("evidence_round", 0) <= MAX_EVIDENCE_ROUNDS:
        return "strategy"

    if not plan or len(completed) >= MAX_PLAN_STEPS:
        return "answer"

    if not state["plan"]:
        return "answer"
    return "execute"


def route_after_plan(state: AgentState) -> str:
    if state["plan"]:
        return "execute"
    return "answer"


def route_after_verify(state: AgentState) -> str:
    final_answer = state.get("final_answer", "")
    draft = state.get("draft_result", {}) if isinstance(state.get("draft_result"), dict) else {}

    has_real_answer = bool(final_answer) and final_answer != INSUFFICIENT_EVIDENCE_MESSAGE
    draft_answer = draft.get("answer", "") if isinstance(draft, dict) else ""
    has_valid_draft = (
        bool(draft_answer)
        and draft_answer != INSUFFICIENT_EVIDENCE_MESSAGE
        and not draft.get("insufficient_evidence")
        and bool(state.get("sources"))
    )

    if has_real_answer or has_valid_draft:
        return "assimilate"

    if not state.get("sources"):
        if (
            state.get("evidence_round", 0) <= MAX_EVIDENCE_ROUNDS
            and len(state.get("completed_steps", [])) < MAX_PLAN_STEPS
        ):
            return "strategy"
    return "assimilate"


async def assimilate_node(state: AgentState, _model: str, _tools: list[dict]) -> dict:
    """Persist research experience — always, regardless of outcome."""
    iteration = state["iteration"] + 1
    print("  [ASSIMILATE] Recording research experience...")
    _assimilate_research(state)  # type: ignore[arg-type]
    return {"iteration": iteration}


async def strategy_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Evolve search queries from live search memory and verifier feedback."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    memory = _merge_search_memory(state.get("search_memory"))
    verification = state.get("verification_result", {})
    evidence_audit = state.get("evidence_audit", {})
    source_count = len(state.get("sources", []))
    store = _get_research_store()
    champion = store.best_strategy()
    iteration = state["iteration"] + 1

    reflections = memory.get("reflections", [])
    reflection_block = ""
    if reflections:
        reflection_block = f"\nREFLECTION (why previous search failed):\n{chr(10).join(f'- {r}' for r in reflections[-3:])}\n"

    prompt = f"""QUESTION:
{question}

SOURCE_COUNT:
{source_count}

TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

VERIFIER_FEEDBACK:
{json.dumps(verification, ensure_ascii=False, indent=2)[:3000]}

EVIDENCE_AUDIT:
{json.dumps(evidence_audit, ensure_ascii=False, indent=2)[:3000]}
{reflection_block}
SEARCH_MEMORY:
{_format_search_memory_for_prompt(memory)}

Propose the next search strategy."""

    print("  [STRATEGY] Evolving search queries...")
    response = _ollama_chat(
        model,
        [{"role": "user", "content": prompt}],
        tools=None, json_mode=True,
        system=STRATEGY_SYSTEM_PROMPT,
        temperature=0.4,
    )
    result = _json_loads_best_effort(response.get("content", ""), {})
    next_queries = result.get("next_queries") if isinstance(result, dict) else None
    if not isinstance(next_queries, list):
        next_queries = []

    attempted = {query.lower() for query in memory["attempted_queries"]}
    candidates = []
    if champion:
        candidates.append({"origin": "exploit", "desc": champion.get("desc", ""), "success_rate": champion.get("success_rate", 0.0)})
    note = result.get("strategy_note") if isinstance(result, dict) else ""
    if note:
        candidates.append({"origin": "explore", "desc": str(note)[:300]})
    candidates.append({"origin": "fallback", "desc": "Use a different source-discovery angle and fetch evidence before answering."})

    fresh_queries = []
    if champion:
        fresh_queries.extend(_queries_from_strategy_desc(champion.get("desc", ""), question, memory, limit=2))
    for query in next_queries:
        query = str(query).strip()
        if query and query.lower() not in attempted and query not in fresh_queries:
            fresh_queries.append(query)

    for hint in evidence_audit.get("strategy_hints", []):
        for query in _queries_from_strategy_desc(str(hint), question, memory, limit=2):
            if query not in fresh_queries:
                fresh_queries.append(query)
    if not fresh_queries:
        fresh_queries = [question]

    if not note:
        note = "Fallback strategy: vary query intent and seek primary/data-table sources."
    memory["strategy_notes"].append(str(note)[:500])
    memory["next_queries"] = fresh_queries[:4]
    memory["strategy_candidates"] = candidates
    memory["current_strategy"] = candidates[0]["desc"] if candidates else str(note)

    print(f"  [STRATEGY] Next queries: {len(memory['next_queries'])}")
    for i, query in enumerate(memory["next_queries"], 1):
        print(f"    {i}. {query[:100]}")

    return {"search_memory": memory, "iteration": iteration}


# ── answer node ─────────────────────────────────────────────────────────
async def answer_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Draft a source-grounded structured answer."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    sources_text, valid_source_ids = _format_sources_for_llm(state.get("sources", []))
    iteration = state["iteration"] + 1

    print(f"\n  [ANSWER] Drafting from {len(valid_source_ids)} sources...")
    if not valid_source_ids:
        draft = {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claims": [],
            "coverage": {
                "requirements_addressed": [],
                "overall_status": "missing",
            },
            "insufficient_evidence": True,
        }
        return {"draft_result": draft, "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE, "iteration": iteration}

    prompt = f"""QUESTION:
{question}

TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

Answer only from SOURCES and satisfy TASK_REQUIREMENTS."""

    # Prose-first: rich, reliable prose with inline [n] citations. A large structured
    # JSON envelope is fragile on content-heavy answers, so we draft prose and verify
    # the prose against the same bounded source set (see verify_node).
    prose = _ollama_chat(
        model,
        [{"role": "user", "content": prompt}],
        tools=None,
        system=ANSWER_PROSE_SYSTEM_PROMPT,
    ).get("content", "").strip()

    if prose and prose != INSUFFICIENT_EVIDENCE_MESSAGE:
        draft = {
            "answer": prose,
            "claims": [],
            "coverage": {"requirements_addressed": [], "overall_status": "partial"},
            "insufficient_evidence": False,
            "salvaged_prose": True,
        }
    else:
        draft = {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claims": [],
            "coverage": {"requirements_addressed": [], "overall_status": "missing"},
            "insufficient_evidence": True,
        }

    return {"draft_result": draft, "iteration": iteration}


async def verify_node(state: AgentState, model: str, _tools: list[dict]) -> dict:
    """Verify the drafted answer against the same bounded source set."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    sources_text, valid_source_ids = _format_sources_for_llm(state.get("sources", []))
    draft = state.get("draft_result", {})
    iteration = state["iteration"] + 1

    print("  [VERIFY] Checking claims against sources...")
    if draft.get("salvaged_prose") is True and valid_source_ids:
        # Prose answer → verify in prose: model re-checks every statement and inline
        # citation against the bounded source set and returns a corrected prose answer.
        # Prose-in/prose-out avoids the fragile JSON envelope while keeping real grounding.
        print("  [VERIFY] Grounding prose answer against sources...")
        verify_prompt = f"""TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

DRAFT_ANSWER:
{draft.get("answer", "")}

Return the corrected, fully-grounded answer."""
        verified = _ollama_chat(
            model,
            [{"role": "user", "content": verify_prompt}],
            tools=None,
            system=VERIFY_PROSE_SYSTEM_PROMPT,
        ).get("content", "").strip()

        # If verification produced nothing usable, fall back to the original prose draft
        # rather than losing a grounded answer to a flaky verify call.
        if not verified:
            verified = draft.get("answer", "")
        insufficient = (not verified) or verified == INSUFFICIENT_EVIDENCE_MESSAGE

        # Small JSON verdict: a compact, machine-readable quality report about the prose
        # answer. No answer text inside, so it parses reliably (unlike the old big envelope).
        coverage_complete = True
        missing: list[str] = []
        verdict_notes: list[str] = []
        if not insufficient:
            verdict_prompt = f"""TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

FINAL_ANSWER:
{verified}

Return the compact JSON verdict."""
            verdict_raw = _ollama_chat_json(
                model,
                [{"role": "user", "content": verdict_prompt}],
                system=VERIFY_VERDICT_SYSTEM_PROMPT,
            )
            verdict = _json_loads_best_effort(verdict_raw, {})
            if isinstance(verdict, dict):
                coverage_complete = bool(verdict.get("coverage_complete", True))
                missing = [str(m) for m in verdict.get("missing", []) if m]
                verdict_notes = [str(n) for n in verdict.get("notes", []) if n]
            if missing:
                print(f"  [VERIFY] Coverage gaps: {', '.join(missing[:4])}")

        verification = {
            "final_answer": verified,
            "claim_checks": [],
            "task_complete": (not insufficient) and coverage_complete,
            "coverage": {"overall_status": "complete" if coverage_complete else "partial"},
            "gaps": missing,
            "notes": verdict_notes,
            "insufficient_evidence": insufficient,
        }
        update = {
            "verification_result": verification,
            "final_answer": verified,
            "iteration": iteration,
        }
        if insufficient:
            update["evidence_round"] = state.get("evidence_round", 0) + 1
        return update
    if not valid_source_ids or draft.get("insufficient_evidence") is True:
        next_round = state.get("evidence_round", 0) + 1
        verification = {
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claim_checks": [],
            "task_complete": False,
            "coverage": {
                "requirements_addressed": [],
                "overall_status": "missing",
            },
            "gaps": ["No source-backed draft answer is available."],
            "insufficient_evidence": True,
        }
        return {
            "verification_result": verification,
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "evidence_round": next_round,
            "iteration": iteration,
        }

    # Unreachable in normal flow: answer_node always produces either a salvaged_prose
    # draft (handled above) or an insufficient_evidence draft (handled above). This is a
    # safety net for any unexpected draft shape.
    verification = {
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "task_complete": False,
        "gaps": ["No usable answer draft."],
        "insufficient_evidence": True,
    }
    return {
        "verification_result": verification,
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": state.get("evidence_round", 0) + 1,
        "iteration": iteration,
    }


# ── build graph ─────────────────────────────────────────────────────────
def build_graph(model: str, tools: list[dict], mcp_session: ClientSession | None = None, fast_model: str = ""):
    graph = StateGraph(AgentState)

    async def _requirements(state): return await requirements_node(state, model, tools)
    async def _plan(state): return await plan_node(state, model, tools)
    _fast = fast_model or model
    async def _execute(state): return await execute_node(state, model, tools, mcp_session=mcp_session, fast_model=_fast)
    async def _evaluate(state): return await evaluate_node(state, model, tools)
    async def _answer(state): return await answer_node(state, model, tools)
    async def _verify(state): return await verify_node(state, model, tools)
    async def _strategy(state): return await strategy_node(state, model, tools)
    async def _assimilate(state): return await assimilate_node(state, model, tools)

    graph.add_node("requirements", _requirements)
    graph.add_node("plan", _plan)
    graph.add_node("execute", _execute)
    graph.add_node("evaluate", _evaluate)
    graph.add_node("answer", _answer)
    graph.add_node("verify", _verify)
    graph.add_node("strategy", _strategy)
    graph.add_node("assimilate", _assimilate)

    graph.set_entry_point("requirements")

    graph.add_edge("requirements", "plan")
    graph.add_conditional_edges("plan", route_after_plan, {"execute": "execute", "answer": "answer"})
    graph.add_conditional_edges("execute", route_after_execute, {"execute": "execute", "evaluate": "evaluate"})
    graph.add_conditional_edges("evaluate", route_after_evaluate, {"execute": "execute", "answer": "answer", "strategy": "strategy"})

    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"strategy": "strategy", "assimilate": "assimilate"})
    graph.add_edge("strategy", "plan")
    graph.add_edge("assimilate", END)

    return graph.compile()


# ── main ────────────────────────────────────────────────────────────────
async def main():
    model, fast_model = pick_model()
    print(f"\nUsing: {model}")

    print("Connecting to weboperator-mcp...", end=" ", flush=True)
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = _tool_schema_list(await session.list_tools())
                print(f"✓ ({len(tools)} tools)")

                if not tools:
                    print("[!] No tools")
                    sys.exit(1)

                graph = build_graph(model, tools, mcp_session=session, fast_model=fast_model)
                history: list[dict] = []  # conversation memory

                print(f"\n{'='*50}")
                print("  Interactive mode. Type 'exit' to quit.")
                print(f"{'='*50}\n")

                while True:
                    try:
                        task = input(">>> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nBye!")
                        break

                    if not task or task.lower() in ("exit", "quit"):
                        print("Bye!")
                        break

                    # build context from history — kept separate from task
                    history_context = ""
                    if history:
                        recent = history[-3:]  # last 3 exchanges
                        history_context = "\n".join(
                            f"Q: {h['q']}\nA: {h['a'][:200]}" for h in recent
                        )

                    print(f"\n{'─'*50}")
                    state: AgentState = {
                        "task": task,  # always clean — no history embedded
                        "conversation_context": history_context,
                        "requirements_result": {},
                        "plan": [],
                        "completed_steps": [],
                        "scratchpad": "",
                        "sources": [],
                        "draft_result": {},
                        "verification_result": {},
                        "evidence_audit": {},
                        "final_answer": "",
                        "iteration": 0,
                        "replan_count": 0,
                        "evidence_round": 0,
                        "search_memory": _default_search_memory(),
                    }

                    try:
                        final = await graph.ainvoke(state, {"recursion_limit": 200})
                    except Exception as e:
                        print(f"\n[!] Error: {e}")
                        continue

                    answer = final.get("final_answer", "")
                    if hasattr(answer, "content"):
                        answer = answer.content

                    history.append({"q": task, "a": answer})
                    debug_report_path = _write_debug_report(final)

                    print(f"\n{answer}\n")
                    sources = final.get("sources", [])
                    if sources:
                        print("Sources:")
                        for i, src in enumerate(sources, 1):
                            title = src.get("title", "").strip()
                            url = src.get("url", "").strip()
                            print(f"  [{i}] {title} — {url}" if title else f"  [{i}] {url}")
                    verdict = final.get("verification_result", {}) or {}
                    gaps = verdict.get("gaps", []) or []
                    notes = verdict.get("notes", []) or []
                    if gaps or notes:
                        print("\nCoverage note (not fully covered by sources):")
                        for g in gaps[:6]:
                            print(f"  - missing: {g}")
                        for n in notes[:4]:
                            print(f"  - note: {n}")
                    if debug_report_path:
                        print(f"Debug report: {debug_report_path}")
                    print(f"{'─'*50}")
                    print(f"  Steps: {len(final['completed_steps'])} | Type next question or 'exit'")
                    print(f"{'─'*50}")
    except Exception as e:
        print(f"\n[!] Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
