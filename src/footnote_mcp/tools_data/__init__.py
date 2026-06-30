"""Structured data tools facade.

Implementation lives in sibling submodules. This package re-exports only the public
tool surface; underscored helpers stay private to their defining submodule.
"""

from .cache import CACHE_DIR, source_cache_get, source_cache_put
from .classify import classify_source, generate_search_queries, resolve_units, validate_unit_rows
from .dates import check_date_completeness
from .entailment import evidence_entailment
from .files import (
    web_detect_downloads,
    web_extract_tables,
    web_fetch_json,
    web_parse_file,
)
from .health import build_research_debug_report, startup_health_check
from .sandbox import (
    tool_code_generate,
    tool_code_run_sandboxed,
    tool_code_validate,
    tool_promote,
    tool_spec_propose,
)

__all__ = [
    "CACHE_DIR",
    "source_cache_get",
    "source_cache_put",
    "classify_source",
    "generate_search_queries",
    "resolve_units",
    "validate_unit_rows",
    "check_date_completeness",
    "evidence_entailment",
    "web_detect_downloads",
    "web_extract_tables",
    "web_fetch_json",
    "web_parse_file",
    "build_research_debug_report",
    "startup_health_check",
    "tool_code_generate",
    "tool_code_run_sandboxed",
    "tool_code_validate",
    "tool_promote",
    "tool_spec_propose",
]
