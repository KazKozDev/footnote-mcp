from __future__ import annotations

import json
from datetime import date

import pytest

from footnote_mcp import tools_data
from footnote_mcp import tools_extra
from footnote_mcp.tools_data import sandbox


# ── web_archive_fetch ──

def test_web_archive_fetch_returns_snapshot_and_text(monkeypatch):
    avail = {
        "archived_snapshots": {
            "closest": {"available": True, "url": "http://web.archive.org/web/2020/http://x.com", "timestamp": "20200101", "status": "200"}
        }
    }
    monkeypatch.setattr(tools_extra, "_fetch_bytes", lambda *a, **k: (json.dumps(avail).encode(), "application/json", None))
    monkeypatch.setattr(tools_extra, "fetch_page", lambda url, lang="en": (url, "<html><title>Old</title><body>archived body</body></html>", date(2020, 1, 1), None))

    result = tools_extra.web_archive_fetch("http://x.com")

    assert result["archived"] is True
    assert result["snapshot_url"].startswith("http://web.archive.org/")
    assert "text" in result and result["text_length"] >= 0


def test_web_archive_fetch_handles_no_snapshot(monkeypatch):
    monkeypatch.setattr(tools_extra, "_fetch_bytes", lambda *a, **k: (json.dumps({"archived_snapshots": {}}).encode(), "application/json", None))
    result = tools_extra.web_archive_fetch("http://x.com")
    assert result["archived"] is False
    assert "error" in result


# ── scholarly_search ──

def test_scholarly_search_arxiv_parses_atom(monkeypatch):
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>http://arxiv.org/abs/2401.00001</id>
        <title>Deep Nets</title>
        <summary>A paper about deep nets.</summary>
        <published>2024-01-01T00:00:00Z</published>
        <author><name>Ada Lovelace</name></author>
      </entry>
    </feed>"""
    monkeypatch.setattr(tools_extra, "_fetch_bytes", lambda *a, **k: (atom.encode(), "application/atom+xml", None))

    result = tools_extra.scholarly_search("deep nets", source="arxiv", num=5)

    assert result["source"] == "arxiv"
    assert result["count"] == 1
    assert result["results"][0]["title"] == "Deep Nets"
    assert result["results"][0]["authors"] == ["Ada Lovelace"]


def test_scholarly_search_wikipedia_parses_json(monkeypatch):
    payload = {"query": {"search": [{"title": "Photosynthesis", "snippet": "<span>green</span> plants", "size": 123, "timestamp": "2026-01-01"}]}}
    monkeypatch.setattr(tools_extra, "_fetch_bytes", lambda *a, **k: (json.dumps(payload).encode(), "application/json", None))

    result = tools_extra.scholarly_search("photosynthesis", source="wikipedia")

    assert result["count"] == 1
    assert result["results"][0]["title"] == "Photosynthesis"
    assert result["results"][0]["url"].endswith("/wiki/Photosynthesis")
    assert result["results"][0]["snippet"] == "green plants"


def test_scholarly_search_rejects_unknown_source():
    result = tools_extra.scholarly_search("x", source="pubmed")
    assert "error" in result and result["results"] == []


# ── web_search_recent ──

def test_web_search_recent_maps_freshness_and_passes_df(monkeypatch):
    captured = {}

    def fake_ddg(query, num=None, lang="en", df=""):
        captured["df"] = df
        return [{"title": "T", "url": "http://a.com", "snippet": "s"}]

    monkeypatch.setattr(tools_extra, "search_ddg", fake_ddg)

    result = tools_extra.web_search_recent("news", freshness="week", num=5)

    assert captured["df"] == "w"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "http://a.com"


# ── corroborate_claim ──

def test_corroborate_claim_corroborated_across_domains():
    excerpts = [
        {"source_url": "https://a.gov/p", "text": "The capital of France is Paris."},
        {"source_url": "https://b.org/p", "text": "Paris is the capital of France."},
    ]
    result = tools_extra.corroborate_claim("The capital of France is Paris.", excerpts, backend="heuristic")
    assert result["supporting"] >= 2
    assert result["independent_supporting_domains"] >= 2
    assert result["verdict"] == "corroborated"


def test_corroborate_claim_no_evidence():
    result = tools_extra.corroborate_claim("x", [], backend="heuristic")
    assert result["verdict"] == "no_evidence"
    assert result["agreement"] == 0.0


# ── locate_claim_span ──

def test_locate_claim_span_finds_best_sentence():
    source = "The sky is blue. The capital of France is Paris. Water boils at 100 degrees."
    result = tools_extra.locate_claim_span("capital of France is Paris", source, max_spans=2)
    assert result["spans"]
    top = result["spans"][0]
    assert "Paris" in top["text"]
    assert source[top["start"]:top["end"]].strip().startswith("The capital of France")
    assert result["best_score"] > 0


def test_locate_claim_span_empty_inputs():
    assert tools_extra.locate_claim_span("", "text")["spans"] == []
    assert tools_extra.locate_claim_span("claim", "")["spans"] == []


# ── recipe_registry ──

@pytest.fixture
def recipe_store(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "RECIPE_STORE_PATH", tmp_path / "recipes.json")
    code = "def extract(source_text, input_payload):\n    return {'rows': [{'v': 1}]}\n"
    tools_data._save_recipe_store({"recipes": {"r1": {"id": "r1", "name": "demo", "code": code, "created_at": "2026-01-01", "plays": 0, "wins": 0}}})
    return "r1"


def test_recipe_registry_list_and_get(recipe_store):
    listed = tools_extra.recipe_registry("list")
    assert listed["count"] == 1
    assert listed["recipes"][0]["id"] == "r1"

    got = tools_extra.recipe_registry("get", recipe_id="r1")
    assert got["found"] is True
    assert "code" in got["recipe"]


def test_recipe_registry_run_bumps_stats(recipe_store):
    run = tools_extra.recipe_registry("run", recipe_id="r1", source_text="x")
    assert run["ok"] is True
    assert run["rows"] == [{"v": 1}]
    # plays incremented and persisted
    assert tools_extra.recipe_registry("get", recipe_id="r1")["recipe"]["plays"] == 1


def test_recipe_registry_delete(recipe_store):
    assert tools_extra.recipe_registry("delete", recipe_id="r1")["deleted"] is True
    assert tools_extra.recipe_registry("list")["count"] == 0


# ── web_fetch_authenticated ──

def test_web_fetch_authenticated_sends_cookies_and_headers(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200
        headers = {"content-type": "text/html"}
        text = "<html><title>Secret</title><body>members only content here</body></html>"

    def fake_get(url, lang="en", cookies=None, timeout=20, max_retries=1, extra_headers=None):
        captured["cookies"] = cookies
        captured["headers"] = extra_headers
        return FakeResp()

    monkeypatch.setattr(tools_extra, "_get", fake_get)
    monkeypatch.setattr(tools_extra, "extract_content", lambda html, url="": "members only content here")

    result = tools_extra.web_fetch_authenticated("https://x.com", cookies={"sid": "abc"}, headers={"Authorization": "Bearer t"})

    assert captured["cookies"] == {"sid": "abc"}
    assert captured["headers"] == {"Authorization": "Bearer t"}
    assert result["status_code"] == 200
    assert result["text_length"] > 0


def test_web_fetch_authenticated_reports_http_error(monkeypatch):
    class FakeResp:
        status_code = 403
        headers = {"content-type": "text/html"}
        text = ""

    monkeypatch.setattr(tools_extra, "_get", lambda *a, **k: FakeResp())
    result = tools_extra.web_fetch_authenticated("https://x.com")
    assert result["status_code"] == 403
    assert "error" in result


# ── web_crawl ──

def test_web_crawl_bfs_same_domain(monkeypatch):
    pages = {
        "https://site.com/": "<html><title>Home</title><body>home <a href='/a'>a</a> <a href='https://other.com/x'>ext</a></body></html>",
        "https://site.com/a": "<html><title>A</title><body>page a content</body></html>",
    }

    def fake_fetch_page(url, lang="en"):
        url = url.rstrip("/") or "https://site.com/"
        key = "https://site.com/" if url == "https://site.com" else url
        html = pages.get(key)
        if html is None:
            return url, None, None, "404"
        return key, html, None, None

    monkeypatch.setattr(tools_extra, "fetch_page", fake_fetch_page)
    monkeypatch.setattr(tools_extra, "extract_content", lambda html, url="": "text")

    result = tools_extra.web_crawl("https://site.com/", max_pages=5, same_domain=True)

    crawled = {p["url"] for p in result["pages"]}
    assert "https://site.com/" in crawled
    assert "https://site.com/a" in crawled
    # external domain must not be crawled
    assert not any("other.com" in u for u in crawled)


# ── export_dataset ──

def test_export_dataset_csv(tmp_path):
    rows = [{"date": "2026-05-01", "value": "1.0"}, {"date": "2026-05-02", "value": "2.0"}]
    out = tmp_path / "d.csv"
    result = tools_extra.export_dataset(rows, format="csv", path=str(out))
    assert result["exported"] is True
    content = out.read_text()
    assert "date,value" in content
    assert "2026-05-01,1.0" in content


def test_export_dataset_xlsx_and_json(tmp_path):
    rows = [{"a": 1, "b": 2}]
    xlsx = tools_extra.export_dataset(rows, format="xlsx", path=str(tmp_path / "d.xlsx"))
    assert xlsx["exported"] is True and (tmp_path / "d.xlsx").exists()

    js = tools_extra.export_dataset(rows, format="json", path=str(tmp_path / "d.json"))
    assert js["exported"] is True
    assert json.loads((tmp_path / "d.json").read_text()) == rows


def test_export_dataset_rejects_bad_rows():
    result = tools_extra.export_dataset(["not a dict"], format="csv")
    assert result["exported"] is False


# ── reconcile_time_series ──

def test_reconcile_time_series_aligns_and_deltas():
    series = [
        {"name": "ecb", "rows": [{"date": "2026-05-01", "value": "1.0"}, {"date": "2026-05-02", "value": "2.0"}]},
        {"name": "fed", "rows": [{"date": "2026-05-01", "value": "1.5"}]},
    ]
    result = tools_extra.reconcile_time_series(series, on="date", value_field="value")

    assert result["series_names"] == ["ecb", "fed"]
    assert result["key_count"] == 2
    first = result["aligned"][0]
    assert first["ecb"] == 1.0 and first["fed"] == 1.5
    assert first["delta_fed_vs_ecb"] == 0.5
    # 2026-05-02 missing for fed
    assert "2026-05-02" in result["missing_keys"]["fed"]


def test_reconcile_time_series_flags_outliers():
    rows = [{"date": str(d), "value": "1.0"} for d in range(1, 8)]
    rows.append({"date": "99", "value": "1000.0"})
    result = tools_extra.reconcile_time_series([{"name": "s", "rows": rows}])
    assert "99" in result["outliers"]["s"]
