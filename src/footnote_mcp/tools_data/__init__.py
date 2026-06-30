"""Structured data tools — facade re-exporting the domain submodules.

Implementation lives in the sibling submodules. This package re-exports the surface
that external callers actually use — `from .tools_data import X` (server, tools_extra,
tools_search), `from footnote_mcp.tools_data import X` (benchmark, the agent), and the
`tools_data.X` calls in the tests. Internal helpers stay private to their submodule;
tests that need to patch them do so on the submodule (e.g. `cache.CACHE_DIR`).
"""

from .cache import CACHE_DIR, _read_cache, _write_cache, source_cache_get, source_cache_put
from .classify import classify_source, generate_search_queries, resolve_units, validate_unit_rows
from .dates import check_date_completeness
from .entailment import _extract_json_object, evidence_entailment
from .files import (
    _fetch_bytes,
    web_detect_downloads,
    web_extract_tables,
    web_fetch_json,
    web_parse_file,
)
from .health import build_research_debug_report, startup_health_check
from .sandbox import (
    _load_recipe_store,
    _recipe_rows,
    _save_recipe_store,
    tool_code_generate,
    tool_code_run_sandboxed,
    tool_code_validate,
    tool_promote,
    tool_spec_propose,
)

__all__ = [
    "CACHE_DIR",
    "_read_cache",
    "_write_cache",
    "source_cache_get",
    "source_cache_put",
    "classify_source",
    "generate_search_queries",
    "resolve_units",
    "validate_unit_rows",
    "check_date_completeness",
    "_extract_json_object",
    "evidence_entailment",
    "_fetch_bytes",
    "web_detect_downloads",
    "web_extract_tables",
    "web_fetch_json",
    "web_parse_file",
    "build_research_debug_report",
    "startup_health_check",
    "_load_recipe_store",
    "_recipe_rows",
    "_save_recipe_store",
    "tool_code_generate",
    "tool_code_run_sandboxed",
    "tool_code_validate",
    "tool_promote",
    "tool_spec_propose",
]
