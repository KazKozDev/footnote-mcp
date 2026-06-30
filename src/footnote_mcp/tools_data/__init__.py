"""Structured data tools — facade re-exporting the domain submodules.

Implementation lives in the sibling submodules; importers keep using
`from .tools_data import X` and `tools_data.X` unchanged.
"""

from .cache import (
    CACHE_DIR,
    _cache_key,
    _cache_path,
    _read_cache,
    _write_cache,
    source_cache_get,
    source_cache_put,
)
from .sandbox import (
    RECIPE_STORE_PATH,
    _recipe_slug,
    tool_spec_propose,
    tool_code_generate,
    _ALLOWED_RECIPE_IMPORTS,
    _FORBIDDEN_RECIPE_CALLS,
    _FORBIDDEN_RECIPE_NODES,
    _import_root,
    tool_code_validate,
    _recipe_preexec_limits,
    tool_code_run_sandboxed,
    _load_recipe_store,
    _save_recipe_store,
    _recipe_rows,
    tool_promote,
)
from .files import (
    _fetch_bytes,
    _table_to_rows,
    web_extract_tables,
    web_detect_downloads,
    _parse_csv_bytes,
    _parse_xlsx_bytes,
    _parse_xls_bytes,
    _parse_pdf_bytes,
    _ocr_pdf_bytes,
    web_parse_file,
    web_fetch_json,
)
from .dates import (
    _add_month,
    _period_key,
    _extract_periods,
    _calendar_holidays,
    check_date_completeness,
)
from .classify import (
    classify_source,
    generate_search_queries,
    resolve_units,
    validate_unit_rows,
)
from .entailment import (
    _extract_json_object,
    _heuristic_entailment,
    _local_nli_entailment,
    _ollama_entailment,
    evidence_entailment,
)
from .health import (
    build_research_debug_report,
    startup_health_check,
)
