from __future__ import annotations

import io
import json
from datetime import date

import pytest

import tools_data


HTML = """<!doctype html>
<html>
  <head><title>Data</title></head>
  <body>
    <table>
      <caption>Rates</caption>
      <tr><th>date</th><th>pair</th><th>rate</th></tr>
      <tr><td>2026-05-01</td><td>EUR/RUB</td><td>90.1</td></tr>
      <tr><td>2026-05-02</td><td>EUR/RUB</td><td>91.2</td></tr>
    </table>
    <a href="/files/rates.csv">CSV</a>
    <a href="https://example.com/report.pdf">Download report</a>
  </body>
</html>"""


def _patch_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(tools_data, "CACHE_DIR", tmp_path)


def test_web_extract_tables_parses_and_caches(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    calls = {"count": 0}

    def fake_fetch_page(url, lang="en"):
        calls["count"] += 1
        return url, HTML, date(2026, 6, 1), None

    monkeypatch.setattr(tools_data, "fetch_page", fake_fetch_page)

    first = tools_data.web_extract_tables("https://example.com/data", use_cache=True)
    second = tools_data.web_extract_tables("https://example.com/data", use_cache=True)

    assert first["cached"] is False
    assert first["table_count"] == 1
    assert first["tables"][0]["caption"] == "Rates"
    assert first["tables"][0]["columns"] == ["date", "pair", "rate"]
    assert first["tables"][0]["rows"][0]["rate"] == "90.1"
    assert second["cached"] is True
    assert calls["count"] == 1


def test_web_extract_tables_records_fetch_error(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(tools_data, "fetch_page", lambda url, lang="en": (url, None, None, "HTTP 500"))

    result = tools_data.web_extract_tables("https://example.com/bad")

    assert result["error"] == "HTTP 500"
    assert result["table_count"] == 0


def test_web_detect_downloads_finds_files(monkeypatch):
    monkeypatch.setattr(tools_data, "fetch_page", lambda url, lang="en": (url, HTML, None, None))

    result = tools_data.web_detect_downloads("https://example.com/page", max_links=10)

    urls = [item["url"] for item in result["downloads"]]
    assert "https://example.com/files/rates.csv" in urls
    assert "https://example.com/report.pdf" in urls
    assert result["count"] == 2


def test_web_parse_file_parses_csv_tsv_xlsx_json_and_cache(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)

    xlsx_buffer = io.BytesIO()
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Rates"
    sheet.append(["date", "rate"])
    sheet.append(["2026-05-01", 90.1])
    workbook.save(xlsx_buffer)

    payloads = {
        "https://example.com/rates.csv": (b"date,rate\n2026-05-01,90.1\n", "text/csv", None),
        "https://example.com/rates.tsv": (b"date\trate\n2026-05-01\t90.1\n", "text/tab-separated-values", None),
        "https://example.com/rates.xlsx": (xlsx_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", None),
        "https://example.com/rates.json": (json.dumps({"rows": [{"date": "2026-05-01"}]}).encode(), "application/json", None),
    }
    calls = {"count": 0}

    def fake_fetch_bytes(url, lang="en", timeout=20):
        calls["count"] += 1
        return payloads[url]

    monkeypatch.setattr(tools_data, "_fetch_bytes", fake_fetch_bytes)

    csv_result = tools_data.web_parse_file("https://example.com/rates.csv", use_cache=True)
    cached_csv = tools_data.web_parse_file("https://example.com/rates.csv", use_cache=True)
    tsv_result = tools_data.web_parse_file("https://example.com/rates.tsv", use_cache=False)
    xlsx_result = tools_data.web_parse_file("https://example.com/rates.xlsx", use_cache=False)
    json_result = tools_data.web_parse_file("https://example.com/rates.json", use_cache=False)

    assert csv_result["file_type"] == "csv"
    assert csv_result["tables"][0]["rows"][0]["rate"] == "90.1"
    assert cached_csv["cached"] is True
    assert tsv_result["tables"][0]["columns"] == ["date", "rate"]
    assert xlsx_result["file_type"] == "xlsx"
    assert xlsx_result["tables"][0]["sheet"] == "Rates"
    assert json_result["json"]["rows"][0]["date"] == "2026-05-01"
    assert calls["count"] == 4


def test_web_parse_file_reports_download_and_unknown_type(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(tools_data, "_fetch_bytes", lambda url, lang="en", timeout=20: (None, "", "boom"))
    assert tools_data.web_parse_file("https://example.com/missing.csv")["error"] == "boom"

    monkeypatch.setattr(tools_data, "_fetch_bytes", lambda url, lang="en", timeout=20: (b"hello", "text/plain", None))
    result = tools_data.web_parse_file("https://example.com/file.bin", use_cache=False)
    assert result["file_type"] == "unknown"
    assert "Unsupported" in result["error"]


def test_web_fetch_json_parses_errors_and_caches(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    calls = {"count": 0}

    def fake_fetch_bytes(url, lang="en", timeout=20):
        calls["count"] += 1
        if url.endswith("/bad"):
            return b"not-json", "application/json", None
        if url.endswith("/missing"):
            return None, "application/json", "HTTP 404"
        return json.dumps({"rows": [{"date": "2026-05-01", "rate": 90.1}]}).encode(), "application/json", None

    monkeypatch.setattr(tools_data, "_fetch_bytes", fake_fetch_bytes)

    first = tools_data.web_fetch_json("https://example.com/api", use_cache=True)
    cached = tools_data.web_fetch_json("https://example.com/api", use_cache=True)
    bad = tools_data.web_fetch_json("https://example.com/bad", use_cache=False)
    missing = tools_data.web_fetch_json("https://example.com/missing", use_cache=False)

    assert first["cached"] is False
    assert first["json"]["rows"][0]["rate"] == 90.1
    assert cached["cached"] is True
    assert calls["count"] == 3
    assert "Invalid JSON" in bad["error"]
    assert missing["error"] == "HTTP 404"


def test_recipe_code_validate_run_and_promote(monkeypatch, tmp_path):
    monkeypatch.setattr(tools_data, "RECIPE_STORE_PATH", tmp_path / "recipes.json")
    spec = tools_data.tool_spec_propose(
        "extract daily rates",
        source_url="https://example.com/rates",
        observed_failure="HTML table parser returned no rows",
    )
    generated = tools_data.tool_code_generate(spec)
    code = generated["code"]
    sample = "2026-05-01 rate 90.1\n2026-05-02 rate 91.2"

    validation = tools_data.tool_code_validate(code)
    result = tools_data.tool_code_run_sandboxed(
        code,
        source_text=sample,
        input_payload={"source_url": "https://example.com/rates", "expected_unit_or_pair": "EUR/USD"},
    )
    promoted = tools_data.tool_promote(
        "daily rates",
        spec,
        code,
        sample_source_text=sample,
        input_payload={"source_url": "https://example.com/rates"},
        expected_min_rows=2,
    )

    assert spec["code_contract"]["entrypoint"] == "extract(source_text, input_payload)"
    assert validation["valid"] is True
    assert result["ok"] is True
    assert result["result"]["row_count"] == 2
    assert result["result"]["rows"][0]["source_url"] == "https://example.com/rates"
    assert promoted["promoted"] is True
    assert promoted["recipe_id"] in json.loads((tmp_path / "recipes.json").read_text())["recipes"]


def test_recipe_code_validate_rejects_unsafe_code():
    unsafe = "import os\n\ndef extract(source_text, input_payload):\n    return os.environ\n"
    file_write = "def extract(source_text, input_payload):\n    return open('/tmp/x', 'w')\n"

    unsafe_validation = tools_data.tool_code_validate(unsafe)
    file_validation = tools_data.tool_code_validate(file_write)

    assert unsafe_validation["valid"] is False
    assert "import not allowed: os" in unsafe_validation["errors"]
    assert file_validation["valid"] is False
    assert "forbidden call: open" in file_validation["errors"]


def test_cache_put_merges_payloads(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    url = "https://example.com/cache"

    tools_data.source_cache_put(url, {"web_read": {"text": "a"}})
    tools_data.source_cache_put(url, {"tables": [{"rows": []}]})
    cached = tools_data.source_cache_get(url)

    assert cached["found"] is True
    assert cached["cache"]["web_read"]["text"] == "a"
    assert cached["cache"]["tables"] == [{"rows": []}]


def test_date_completeness_supports_calendar_business_week_month_and_errors():
    daily = tools_data.check_date_completeness("2026-05-01", "2026-05-03", ["2026-05-01"])
    business = tools_data.check_date_completeness(
        "2026-05-01",
        "2026-05-05",
        ["2026-05-01", "2026-05-05"],
        calendar="business_day",
        holidays=["2026-05-04"],
    )
    week = tools_data.check_date_completeness("2026-05-01", "2026-05-14", ["2026-W18"], granularity="week")
    month = tools_data.check_date_completeness("2026-05-01", "2026-06-30", ["2026-05"], granularity="month")

    assert daily["missing_items"] == ["2026-05-02", "2026-05-03"]
    assert business["complete"] is True
    assert week["missing_items"] == ["2026-W19"]
    assert month["missing_items"] == ["2026-06"]
    assert "error" in tools_data.check_date_completeness("2026-05-02", "2026-05-01", [])
    assert "error" in tools_data.check_date_completeness("2026-05-01", "2026-05-02", [], granularity="hour")
    us = tools_data.check_date_completeness(
        "2026-07-03",
        "2026-07-06",
        ["2026-07-06"],
        calendar="us_business_day",
    )
    assert us["complete"] is True
    assert "2026-07-04" not in us["missing_items"]
    assert "error" in tools_data.check_date_completeness("2026-05-01", "2026-05-02", [], calendar="trading")


def test_classify_source_categories():
    assert tools_data.classify_source("https://data.gov/example")["source_type"] == "official"
    assert tools_data.classify_source("https://reddit.com/r/data")["source_type"] == "forum"
    assert tools_data.classify_source("https://example.substack.com/post")["source_type"] == "blog"
    assert tools_data.classify_source("https://example.com", status_code=403)["source_type"] == "blocked"
    assert tools_data.classify_source("https://example.com", status_code=500)["source_type"] == "error"
    assert tools_data.classify_source("https://example.com", text_sample="Please enable JavaScript")["source_type"] == "interactive"


def test_generate_search_queries_uses_requirements_and_hints():
    result = tools_data.generate_search_queries(
        "exchange rates",
        {"target": "EUR RUB", "unit_or_pair": "EUR/RUB", "search_hints": ["provided source hint"]},
        max_queries=4,
    )

    assert result["queries"][0] == "provided source hint"
    assert any("official data" in query for query in result["queries"])
    assert not any(query.startswith("exchange rates EUR RUB site:") for query in result["queries"][1:])
    assert result["count"] == 4


def test_resolve_units_and_validate_unit_rows():
    units = tools_data.resolve_units("euro to ruble and USDJPY rate")
    validation = tools_data.validate_unit_rows(
        [{"pair": "EUR/RUB", "rate": "90.1"}, {"pair": "EUR/USD", "rate": "1.08"}, {"pair": "EUR RUB", "rate": "91"}],
        "EUR/RUB",
        ["pair"],
    )

    assert "EUR/RUB" in units["currency_pairs"]
    assert "USD/JPY" in units["currency_pairs"]
    assert validation["accepted_count"] == 2
    assert validation["rejected_count"] == 1


def test_entailment_heuristic_auto_and_ollama_fallback(monkeypatch):
    contradicted = tools_data.evidence_entailment("2026-05-01 rate 90.1", "2026-05-01 rate 91.2", backend="heuristic")
    invalid = tools_data.evidence_entailment("a", "a", backend="bad")

    def fake_ollama(*args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(tools_data, "_ollama_entailment", fake_ollama)
    auto = tools_data.evidence_entailment("bank cut rates", "central bank reduced interest rates", backend="auto")
    forced = tools_data.evidence_entailment("bank cut rates", "central bank reduced interest rates", backend="ollama")

    monkeypatch.setattr(
        tools_data,
        "_local_nli_entailment",
        lambda **kwargs: {"status": "supported", "score": 0.9, "reason": "nli", "backend": "local_nli", "model": "m"},
    )
    local_nli = tools_data.evidence_entailment("bank cut rates", "central bank reduced interest rates", backend="local_nli")

    assert contradicted["status"] == "contradicted"
    assert invalid["reason"] == "unknown backend: bad"
    assert auto["backend"] == "heuristic"
    assert "fallback_reason" in auto
    assert forced["backend"] == "ollama"
    assert forced["fallback"]["backend"] == "heuristic"
    assert local_nli["backend"] == "local_nli"
    assert local_nli["heuristic_precheck"]["backend"] == "heuristic"


def test_json_object_extraction_for_llm_responses():
    parsed = tools_data._extract_json_object('prefix {"status":"supported","reason":"ok"} suffix')
    assert parsed == {"status": "supported", "reason": "ok"}
    assert tools_data._extract_json_object("no json") == {}


def test_debug_report_and_startup_health(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    tools_data.source_cache_put("https://example.com/source", {"source_quality": {"source_type": "official"}})

    report = tools_data.build_research_debug_report(
        task="task",
        requirements={"target": "rates"},
        search_memory={"attempted_queries": ["q"], "read_urls": ["u"], "failed_urls": ["f"], "search_rounds": [{"query": "q"}]},
        sources=[{"title": "S", "url": "https://example.com/source", "kind": "page", "content": "abc"}],
        verification={"task_complete": False, "gaps": ["missing date"]},
    )
    health = tools_data.startup_health_check()

    assert report["attempted_queries"] == ["q"]
    assert report["sources"][0]["quality"]["source_type"] == "official"
    assert report["verification"]["gaps"] == ["missing date"]
    assert "checks" in health
    assert "cache_dir" in health["checks"]
