from __future__ import annotations

import json

from footnote_mcp import sources


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_papers_search_crossref_normalizes_result(monkeypatch):
    payload = {
        "message": {
            "items": [
                {
                    "title": ["Grounded Search"],
                    "DOI": "10.1234/example",
                    "URL": "https://doi.org/10.1234/example",
                    "abstract": "<jats:p>Evidence-first research.</jats:p>",
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "published": {"date-parts": [[2026, 7, 1]]},
                }
            ]
        }
    }
    monkeypatch.setattr(
        sources,
        "_fetch_bytes",
        lambda *args, **kwargs: (json.dumps(payload).encode(), "application/json", None),
    )

    result = sources.papers_search("grounded search", source="crossref")

    assert result["count"] == 1
    assert result["results"][0] == {
        "title": "Grounded Search",
        "url": "https://doi.org/10.1234/example",
        "snippet": "Evidence-first research.",
        "published": "2026-07-01",
        "authors": ["Ada Lovelace"],
        "source": "crossref",
        "source_type": "paper",
        "identifiers": {"doi": "10.1234/example"},
    }


def test_papers_search_arxiv_normalizes_result(monkeypatch):
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>https://arxiv.org/abs/2401.00001</id>
        <title>Deep Nets</title>
        <summary>A paper about deep nets.</summary>
        <published>2024-01-01T00:00:00Z</published>
        <author><name>Ada Lovelace</name></author>
      </entry>
    </feed>"""
    monkeypatch.setattr(
        sources,
        "_fetch_bytes",
        lambda *args, **kwargs: (atom.encode(), "application/atom+xml", None),
    )

    result = sources.papers_search("deep nets", source="arxiv")

    assert result["count"] == 1
    assert result["results"][0]["source_type"] == "paper"
    assert result["results"][0]["authors"] == ["Ada Lovelace"]


def test_encyclopedia_search_wikipedia_and_wikidata(monkeypatch):
    wikipedia = {
        "query": {
            "search": [
                {
                    "title": "Photosynthesis",
                    "snippet": "<span>green</span> plants",
                    "timestamp": "2026-01-01",
                }
            ]
        }
    }
    wikidata = {
        "search": [
            {
                "id": "Q11982",
                "label": "photosynthesis",
                "description": "biological process",
                "concepturi": "https://www.wikidata.org/entity/Q11982",
                "aliases": ["carbon assimilation"],
            }
        ]
    }

    def fake_fetch(url, **kwargs):
        payload = wikidata if "wikidata.org" in url else wikipedia
        return json.dumps(payload).encode(), "application/json", None

    monkeypatch.setattr(sources, "_fetch_bytes", fake_fetch)
    result = sources.encyclopedia_search("photosynthesis", source="auto")

    assert result["count"] == 2
    assert {item["source"] for item in result["results"]} == {"wikipedia", "wikidata"}
    assert {item["source_type"] for item in result["results"]} == {"encyclopedia", "knowledge_graph"}


def test_wikidata_search_prefers_popular_exact_label_match(monkeypatch):
    search_payload = {
        "search": [
            {"id": "Q1", "label": "Douglas Adams", "description": "engineer"},
            {"id": "Q42", "label": "Douglas Adams", "description": "writer"},
        ]
    }
    entities_payload = {
        "entities": {
            "Q1": {"sitelinks": {"enwiki": {}}},
            "Q42": {"sitelinks": {"enwiki": {}, "dewiki": {}, "frwiki": {}}},
        }
    }

    def fake_fetch(url, **kwargs):
        payload = entities_payload if "wbgetentities" in url else search_payload
        return json.dumps(payload).encode(), "application/json", None

    monkeypatch.setattr(sources, "_fetch_bytes", fake_fetch)
    result = sources.encyclopedia_search("Douglas Adams", source="wikidata", num=1)

    assert result["results"][0]["identifiers"]["wikidata"] == "Q42"
    assert result["results"][0]["sitelink_count"] == 3


def test_encyclopedia_search_wikidata_sparql_returns_rows(monkeypatch):
    payload = {
        "head": {"vars": ["item", "itemLabel"]},
        "results": {
            "bindings": [
                {
                    "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q42"},
                    "itemLabel": {"type": "literal", "value": "Douglas Adams"},
                }
            ]
        },
    }
    monkeypatch.setattr(
        sources,
        "_fetch_bytes",
        lambda *args, **kwargs: (json.dumps(payload).encode(), "application/sparql-results+json", None),
    )

    result = sources.encyclopedia_search("Douglas Adams", sparql="SELECT ?item ?itemLabel WHERE {}")

    assert result["source"] == "wikidata_sparql"
    assert result["results"][0]["itemLabel"] == "Douglas Adams"


def test_github_search_works_without_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    payload = {
        "items": [
            {
                "id": 1,
                "full_name": "owner/project",
                "html_url": "https://github.com/owner/project",
                "description": "A project",
                "pushed_at": "2026-07-01T00:00:00Z",
                "owner": {"login": "owner"},
            }
        ]
    }
    monkeypatch.setattr(sources, "_get", lambda *args, **kwargs: FakeResponse(payload))

    result = sources.github_search("project")

    assert result["count"] == 1
    assert result["results"][0]["title"] == "owner/project"
    assert result["results"][0]["source"] == "github"


def test_github_search_reranks_exact_repository_name(monkeypatch):
    payload = {
        "items": [
            {
                "id": 1,
                "full_name": "other/docx-mcp",
                "html_url": "https://github.com/other/docx-mcp",
                "description": "MCP server with footnotes",
                "owner": {"login": "other"},
            },
            {
                "id": 2,
                "full_name": "KazKozDev/footnote-mcp",
                "html_url": "https://github.com/KazKozDev/footnote-mcp",
                "description": "Source-grounded web research",
                "owner": {"login": "KazKozDev"},
            },
        ]
    }
    monkeypatch.setattr(sources, "_get", lambda *args, **kwargs: FakeResponse(payload))

    result = sources.github_search("footnote mcp", num=1)

    assert result["results"][0]["title"] == "KazKozDev/footnote-mcp"
    assert result["results"][0]["relevance_score"] > 0


def test_archive_search_common_crawl_returns_capture(monkeypatch):
    indexes = [{"cdx-api": "https://index.commoncrawl.org/CC-MAIN-test-index"}]
    capture = {
        "url": "https://example.com/",
        "timestamp": "20260701000000",
        "filename": "crawl-data/test.warc.gz",
        "offset": "10",
        "length": "20",
        "digest": "sha1:abc",
        "status": "200",
    }

    def fake_fetch(url, **kwargs):
        if "collinfo.json" in url:
            return json.dumps(indexes).encode(), "application/json", None
        return (json.dumps(capture) + "\n").encode(), "application/x-ndjson", None

    monkeypatch.setattr(sources, "_fetch_bytes", fake_fetch)
    result = sources.archive_search("https://example.com", source="common_crawl")

    assert result["count"] == 1
    assert result["results"][0]["source"] == "common_crawl"
    assert result["results"][0]["archive"]["filename"] == "crawl-data/test.warc.gz"
