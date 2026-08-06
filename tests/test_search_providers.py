from __future__ import annotations

import pytest

from footnote_mcp import search


class FakeResp:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    search._PROVIDER_COOLDOWN_UNTIL.clear()
    for var in (
        "FOOTNOTE_SEARXNG_URL",
        "SEARXNG_URL",
        "TAVILY_API_KEY",
        "BRAVE_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
    ):
        monkeypatch.delenv(var, raising=False)


# ── provider order ──

def test_provider_order_auto_uses_only_keyed(monkeypatch):
    assert search._provider_order("auto") == []
    monkeypatch.setenv("FOOTNOTE_SEARXNG_URL", "http://localhost:8080")
    assert search._provider_order("auto") == ["searxng"]
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    assert search._provider_order("auto") == ["searxng", "brave"]
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert search._provider_order("auto") == ["searxng", "tavily", "brave"]


def test_provider_order_explicit_and_scrape():
    assert search._provider_order("google") == ["google"]
    assert search._provider_order("wiby") == ["wiby"]
    assert search._provider_order("marginalia") == ["marginalia"]
    assert search._provider_order("scrape") == []


def test_provider_order_google_needs_both_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert search._provider_order("auto") == []  # CSE id missing
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    assert search._provider_order("auto") == ["google"]


# ── provider parsing → merged shape ──

def test_search_tavily_parses(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    payload = {"results": [{"title": "A", "url": "https://a.com", "content": "snip a"},
                           {"title": "B", "url": "https://b.com", "content": "snip b"}]}
    monkeypatch.setattr(search.http, "post", lambda *a, **k: FakeResp(200, payload))
    out = search.search_tavily("q", num=5)
    assert [r["url"] for r in out] == ["https://a.com", "https://b.com"]
    assert out[0]["engines"] == ["tavily"]
    assert out[0]["score"] > out[1]["score"]


def test_search_brave_parses(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    payload = {"web": {"results": [{"title": "A", "url": "https://a.com", "description": "d"}]}}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))
    out = search.search_brave("q")
    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "d"
    assert out[0]["engines"] == ["brave"]


def test_search_google_parses(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    payload = {"items": [{"title": "A", "link": "https://a.com", "snippet": "s"}]}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))
    out = search.search_google("q")
    assert out[0]["url"] == "https://a.com"
    assert out[0]["engines"] == ["google"]


def test_search_searxng_parses_zero_key_json(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searx.test/")
    payload = {"results": [{"title": "A", "url": "https://a.com", "content": "result text"}]}
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(200, payload))

    out = search.search_searxng("q")

    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "result text"
    assert out[0]["engines"] == ["searxng"]


def test_provider_without_key_returns_empty():
    assert search.search_searxng("q") == []
    assert search.search_tavily("q") == []
    assert search.search_brave("q") == []
    assert search.search_google("q") == []


def test_provider_http_error_raises(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(search.http, "get", lambda *a, **k: FakeResp(429, {}))
    with pytest.raises(RuntimeError):
        search.search_brave("q")


# ── brave HTML scraping (no key) ──

class FakeHtmlResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


class FakeJsonResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


BRAVE_HTML = """
<div id="results">
  <div class="snippet" data-type="web">
    <a href="https://example.com/a"><div class="title">Title A</div></a>
    <div class="snippet-description">Desc A</div>
  </div>
  <div class="snippet" data-type="web">
    <a href="https://example.org/b"><div class="title">Title B</div></a>
    <div class="snippet-description">Desc B</div>
  </div>
  <div class="snippet" data-type="web">
    <a href="https://example.com/a"><div class="title">Dup</div></a>
  </div>
</div>
"""


BING_HTML = """
<ol id="b_results">
  <li class="b_algo">
    <h2><a href="https://modelcontextprotocol.io/">Model Context Protocol</a></h2>
    <div class="b_caption"><p>Official MCP documentation and architecture.</p></div>
  </li>
  <li class="b_algo">
    <h2><a href="https://github.com/modelcontextprotocol">MCP repositories on GitHub</a></h2>
    <div class="b_caption"><p>Open source Model Context Protocol projects.</p></div>
  </li>
</ol>
"""


BING_POISONED_HTML = """
<ol id="b_results">
  <li class="b_algo">
    <h2><a href="https://example.com/real-estate">Property for sale</a></h2>
    <div class="b_caption"><p>Find houses and apartments.</p></div>
  </li>
  <li class="b_algo">
    <h2><a href="https://example.org/college">College application</a></h2>
    <div class="b_caption"><p>Courses and admissions.</p></div>
  </li>
</ol>
"""


def test_search_bing_accepts_relevant_results_and_uses_valid_country(monkeypatch):
    requested = {}

    def fake_get(url, *args, **kwargs):
        requested["url"] = url
        return FakeHtmlResp(BING_HTML)

    monkeypatch.setattr(search, "_get", fake_get)
    out = search.search_bing("Model Context Protocol GitHub", num=10, lang="en")

    assert len(out) == 2
    assert "cc=US" in requested["url"]
    assert "cc=en" not in requested["url"]


def test_search_bing_rejects_unrelated_http_200_results(monkeypatch):
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp(BING_POISONED_HTML))
    assert search.search_bing("Model Context Protocol GitHub") == []


def test_search_bing_accepts_distinctive_query_term(monkeypatch):
    html = """
    <li class="b_algo">
      <h2><a href="https://developers.openai.com/">OpenAI Developers</a></h2>
      <div class="b_caption"><p>Docs and resources for developers.</p></div>
    </li>
    """
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp(html))
    out = search.search_bing("OpenAI official documentation")
    assert [item["url"] for item in out] == ["https://developers.openai.com/"]


def test_search_bing_rejects_antibot_challenge(monkeypatch):
    challenge = "<html><body>One last step: solve the CAPTCHA challenge</body></html>"
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp(challenge))
    assert search.search_bing("python programming language") == []


def test_search_brave_scrape_parses(monkeypatch):
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp(BRAVE_HTML))
    out = search.search_brave_scrape("q", num=10)
    assert [r["url"] for r in out] == ["https://example.com/a", "https://example.org/b"]
    assert out[0]["title"] == "Title A"
    assert out[0]["snippet"] == "Desc A"


def test_search_brave_scrape_handles_http_error(monkeypatch):
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp("", 429))
    assert search.search_brave_scrape("q") == []


def test_search_brave_scrape_cools_down_after_rate_limit(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return FakeResp(429, headers={"Retry-After": "30"})

    monkeypatch.setenv("FOOTNOTE_BRAVE_COOLDOWN_SECONDS", "10")
    monkeypatch.setattr(search, "_get", fake_get)

    assert search.search_brave_scrape("first") == []
    assert search.search_brave_scrape("second") == []
    assert len(calls) == 1
    assert search._PROVIDER_COOLDOWN_UNTIL["brave"] > search.time.monotonic()


def test_search_ddg_cools_down_after_http_202(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args[0])
        return FakeResp(202)

    monkeypatch.setenv("FOOTNOTE_DDG_COOLDOWN_SECONDS", "10")
    monkeypatch.setattr(search, "_get", fake_get)

    assert search.search_ddg("first") == []
    assert search.search_ddg("second") == []
    assert len(calls) == 1


def test_search_brave_scrape_respects_num(monkeypatch):
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeHtmlResp(BRAVE_HTML))
    assert len(search.search_brave_scrape("q", num=1)) == 1


def test_search_wiby_parses_public_json_and_preserves_attribution(monkeypatch):
    payload = [
        {"URL": "https://example.com/a", "Title": "Title A", "Snippet": "Snippet A", "Description": "Desc A"},
        {"URL": "https://example.org/b", "Title": "Title B", "Description": "Desc B"},
    ]
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeJsonResp(payload))

    out = search.search_wiby("q", num=1)

    assert out == [{
        "title": "Title A",
        "url": "https://example.com/a",
        "snippet": "Snippet A",
        "attribution": "https://wiby.me/",
    }]


def test_search_wiby_rejects_broad_but_unrelated_results(monkeypatch):
    payload = [{
        "URL": "https://example.com/geocoding",
        "Title": "Geocoding API Documentation",
        "Snippet": "Parameters and programming tutorials",
    }]
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeJsonResp(payload))
    assert search.search_wiby("Model Context Protocol official documentation") == []


def test_search_wiby_drops_generic_rows_beside_relevant_rows(monkeypatch):
    payload = [
        {"URL": "https://example.com/geocoding", "Title": "API Documentation", "Snippet": "Reference"},
        {"URL": "https://example.org/mcp", "Title": "Model Context Protocol", "Snippet": "Official docs"},
    ]
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeJsonResp(payload))
    out = search.search_wiby("Model Context Protocol official documentation")
    assert [item["url"] for item in out] == ["https://example.org/mcp"]


def test_search_marginalia_parses_public_json_and_preserves_license(monkeypatch):
    payload = {
        "license": "CC-BY-NC-SA 4.0",
        "results": [{"url": "https://example.com/a", "title": "Title A", "description": "Desc A"}],
    }
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeJsonResp(payload))

    out = search.search_marginalia("q", num=5)

    assert out == [{
        "title": "Title A",
        "url": "https://example.com/a",
        "snippet": "Desc A",
        "license": "CC-BY-NC-SA 4.0",
        "attribution": "https://search.marginalia.nu/",
    }]


def test_zero_key_json_providers_fail_closed(monkeypatch):
    monkeypatch.setattr(search, "_get", lambda *a, **k: FakeJsonResp({}, status_code=503))
    assert search.search_wiby("q") == []
    assert search.search_marginalia("q") == []


def test_merge_preserves_zero_key_attribution_and_license():
    shared = {"title": "Shared", "url": "https://example.com/a", "snippet": ""}
    out = search.merge_results(
        [shared],
        [],
        [],
        [{**shared, "attribution": "https://wiby.me/"}],
        [{**shared, "attribution": "https://search.marginalia.nu/", "license": "CC-BY-NC-SA 4.0"}],
        num=5,
    )
    assert out[0]["engines"] == ["bing", "marginalia", "wiby"]
    assert out[0]["attributions"] == ["https://search.marginalia.nu/", "https://wiby.me/"]
    assert out[0]["licenses"] == ["CC-BY-NC-SA 4.0"]


@pytest.mark.parametrize(
    "engine",
    ["bing", "ddg", "brave", "wiby", "marginalia", "searxng", "tavily", "google"],
)
def test_every_provider_uses_the_same_relevance_contract(engine):
    unrelated = [{"title": "Property for sale", "url": "https://example.com/house", "snippet": "Apartments"}]
    assert search._prepare_source_results("Model Context Protocol", unrelated, engine) == []


def test_source_deduplication_keeps_one_row_and_best_text():
    rows = [
        {"title": "Alpha", "url": "https://example.com/a?ref=one", "snippet": "short"},
        {"title": "Alpha result", "url": "http://www.example.com/a#section", "snippet": "a much longer snippet"},
    ]
    out = search._prepare_source_results("Alpha", rows, "test")
    assert out == [{
        "title": "Alpha result",
        "url": "https://example.com/a?ref=one",
        "snippet": "a much longer snippet",
    }]


def test_duplicate_rows_from_one_engine_do_not_add_rank_votes():
    duplicate = {"title": "Alpha", "url": "https://example.com/a", "snippet": "Alpha result"}
    out = search.merge_results([duplicate, duplicate], [], num=5)
    assert out[0]["score"] == 1.0
    assert out[0]["engines"] == ["bing"]


# ── search() routing + fallback ──

def test_search_auto_merges_configured_and_zero_key_providers(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(search, "search_tavily",
                        lambda q, num=10, lang="en": [{"title": "T", "url": "https://t.com", "snippet": "s", "score": 1.0, "engines": ["tavily"]}])
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [{"title": "D", "url": "https://d.com", "snippet": ""}])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_marginalia", lambda *a, **k: [])
    out = search.search("q", num=5)
    assert {item["url"] for item in out} == {"https://t.com", "https://d.com"}
    assert {engine for item in out for engine in item["engines"]} == {"tavily", "ddg"}


def test_explicit_provider_remains_isolated(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(
        search,
        "search_tavily",
        lambda *a, **k: [{"title": "T", "url": "https://t.com", "snippet": "", "score": 1.0, "engines": ["tavily"]}],
    )
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: pytest.fail("fallback should not run"))
    out = search.search("q", num=5, provider="tavily")
    assert [item["url"] for item in out] == ["https://t.com"]
    assert out[0]["engines"] == ["tavily"]


def test_search_falls_back_to_scrape_when_provider_fails(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(search, "search_brave", lambda q, num=10, lang="en": (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [{"title": "Bg", "url": "https://bing.com/x", "snippet": ""}])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_marginalia", lambda *a, **k: [])
    out = search.search("q", num=5)
    assert any("bing.com" in r["url"] for r in out)


def test_search_scrape_when_no_keys(monkeypatch):
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [{"title": "Bg", "url": "https://bing.com/x", "snippet": ""}])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [{"title": "Dg", "url": "https://ddg.com/y", "snippet": ""}])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: [{"title": "Br", "url": "https://brave-hit.com/z", "snippet": ""}])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: [{"title": "Wi", "url": "https://wiby-hit.com/w", "snippet": "", "attribution": "https://wiby.me/"}])
    monkeypatch.setattr(search, "search_marginalia", lambda *a, **k: pytest.fail("marginalia must be explicit-only"))
    out = search.search("q", num=5)
    urls = {r["url"] for r in out}
    assert {
        "https://bing.com/x", "https://ddg.com/y", "https://brave-hit.com/z",
        "https://wiby-hit.com/w",
    } <= urls


def test_search_auto_marginalia_merges_opt_in_provider(monkeypatch):
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: [])
    monkeypatch.setattr(
        search,
        "search_marginalia",
        lambda *a, **k: [{
            "title": "Marginalia result",
            "url": "https://marginalia-hit.example/q",
            "snippet": "query result",
            "license": "CC-BY-NC-SA 4.0",
            "attribution": "https://search.marginalia.nu/",
        }],
    )

    out = search.search("query", num=5, provider="auto+marginalia")

    assert [item["url"] for item in out] == ["https://marginalia-hit.example/q"]
    assert out[0]["engines"] == ["marginalia"]
    assert out[0]["licenses"] == ["CC-BY-NC-SA 4.0"]


def test_a_rate_limit_never_silences_every_provider(monkeypatch):
    import time as _time

    resting = {name: _time.monotonic() + 300 for name in search._BACKOFF_ENGINES}
    monkeypatch.setattr(search, "_PROVIDER_COOLDOWN_UNTIL", resting)

    # With nothing left to ask, every engine must still be queried.
    for engine in search._BACKOFF_ENGINES:
        assert search._provider_on_cooldown(engine) is False


def test_one_resting_provider_is_skipped_while_others_work(monkeypatch):
    import time as _time

    monkeypatch.setattr(
        search, "_PROVIDER_COOLDOWN_UNTIL", {"marginalia": _time.monotonic() + 300}
    )

    assert search._provider_on_cooldown("marginalia") is True
    assert search._provider_on_cooldown("bing") is False


# ── cost-aware provider rotation ──

def _free_tier(monkeypatch, results):
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: results)
    monkeypatch.setattr(search, "search_ddg", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: [])
    monkeypatch.setattr(search, "search_marginalia", lambda *a, **k: [])


def _rows(n, host="free"):
    return [
        {"title": f"quantum tunneling result {i}", "url": f"https://{host}.example/{i}", "snippet": "quantum tunneling"}
        for i in range(n)
    ]


def test_a_healthy_free_tier_spends_no_metered_credit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "3")
    _free_tier(monkeypatch, _rows(5))
    monkeypatch.setattr(search, "search_tavily", lambda *a, **k: pytest.fail("must not be charged"))

    out = search.search("quantum tunneling", num=5)

    assert out


def test_a_thin_free_tier_escalates_to_one_metered_provider(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "5")
    charged = []
    _free_tier(monkeypatch, _rows(1))

    def fake_tavily(q, num=10, lang="en"):
        charged.append("tavily")
        return _rows(3, host="tavily")

    monkeypatch.setattr(search, "search_tavily", fake_tavily)

    out = search.search("quantum tunneling", num=5)

    assert charged == ["tavily"]
    assert any("tavily.example" in item["url"] for item in out)


def test_metered_providers_take_turns_instead_of_draining_the_first(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "5")
    monkeypatch.setattr(search, "_metered_cursor", 0)
    charged = []
    _free_tier(monkeypatch, _rows(1))
    monkeypatch.setattr(search, "search_tavily",
                        lambda q, num=10, lang="en": charged.append("tavily") or _rows(3, "tavily"))
    monkeypatch.setattr(search, "search_brave",
                        lambda q, num=10, lang="en": charged.append("brave") or _rows(3, "brave"))

    for _ in range(4):
        search.search("quantum tunneling", num=5)

    assert charged == ["tavily", "brave", "tavily", "brave"]


def test_a_resting_metered_provider_is_skipped(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "5")
    monkeypatch.setattr(search, "_metered_cursor", 0)
    charged = []
    _free_tier(monkeypatch, _rows(1))
    monkeypatch.setattr(search, "search_tavily",
                        lambda q, num=10, lang="en": charged.append("tavily") or _rows(3, "tavily"))
    monkeypatch.setattr(search, "search_brave",
                        lambda q, num=10, lang="en": charged.append("brave") or _rows(3, "brave"))
    monkeypatch.setattr(search, "_provider_on_cooldown", lambda name: name == "tavily")

    search.search("quantum tunneling", num=5)

    assert charged == ["brave"]


def test_merge_strategy_still_queries_every_configured_provider(monkeypatch):
    monkeypatch.setenv("FOOTNOTE_PROVIDER_STRATEGY", "merge")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "1")
    charged = []
    _free_tier(monkeypatch, _rows(5))
    monkeypatch.setattr(search, "search_tavily",
                        lambda q, num=10, lang="en": charged.append("tavily") or _rows(2, "tavily"))

    search.search("quantum tunneling", num=5)

    assert charged == ["tavily"]  # merge mode pays on every query, by design


def test_on_topic_noise_does_not_count_as_a_strong_free_result(monkeypatch):
    """"Geography of Spain" survives the permissive per-provider filter but answers
    nothing about August 2026; counting it as a hit kept Tavily from being asked."""
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "3")
    monkeypatch.setattr(search, "_metered_cursor", 0)
    charged = []
    noise = [
        {"title": "Geography of Spain - Wikipedia", "url": "https://w.example/geo", "snippet": "Spain"},
        {"title": "100 landmarks in Spain", "url": "https://w.example/marks", "snippet": "Spain"},
        {"title": "Spain travel guide", "url": "https://w.example/guide", "snippet": "Spain"},
        {"title": "Spain photos", "url": "https://w.example/photos", "snippet": "Spain"},
    ]
    _free_tier(monkeypatch, noise)
    monkeypatch.setattr(
        search, "search_tavily",
        lambda q, num=10, lang="en": charged.append("tavily") or [
            {"title": "Spain Events August 2026 calendar", "url": "https://t.example/e",
             "snippet": "events in Spain in August 2026"}
        ],
    )

    out = search.search("Spain events August 2026", num=5)

    assert charged == ["tavily"]
    assert any("t.example" in item["url"] for item in out)


def test_results_that_cover_the_query_keep_the_metered_provider_unused(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("FOOTNOTE_MIN_FREE_RESULTS", "3")
    real = [
        {"title": f"Spain events August 2026 guide {i}", "url": f"https://f.example/{i}",
         "snippet": "events in Spain in August 2026"}
        for i in range(4)
    ]
    _free_tier(monkeypatch, real)
    monkeypatch.setattr(search, "search_tavily", lambda *a, **k: pytest.fail("must not be charged"))

    assert search.search("Spain events August 2026", num=5)


# ── recency reaches every provider, not just the one engine that had a df param ──

def test_a_recency_window_is_translated_per_provider(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["tavily"] = (json or {}).get("time_range")
        return FakeJsonResp({"results": []})

    def fake_get(url, headers=None, params=None, timeout=None):
        seen["brave"] = (params or {}).get("freshness")
        return FakeJsonResp({"web": {"results": []}})

    monkeypatch.setattr(search.http, "post", fake_post)
    monkeypatch.setattr(search.http, "get", fake_get)

    search.search_tavily("q", num=5, freshness="week")
    search.search_brave("q", num=5, freshness="week")

    assert seen == {"tavily": "week", "brave": "pw"}


def test_no_recency_window_leaves_provider_requests_untouched(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["payload"] = json or {}
        return FakeJsonResp({"results": []})

    monkeypatch.setattr(search.http, "post", fake_post)

    search.search_tavily("q", num=5)

    assert "time_range" not in seen["payload"]


def test_a_time_boxed_free_tier_asks_only_the_engine_that_can_filter(monkeypatch):
    """Merging undated engines into a recency-filtered query is how stale content
    farms outranked the dated results."""
    asked = []
    monkeypatch.setattr(search, "search_ddg",
                        lambda q, num=None, lang="en", debug=False, df="": asked.append(("ddg", df)) or [])
    monkeypatch.setattr(search, "search_bing", lambda *a, **k: asked.append(("bing", None)) or [])
    monkeypatch.setattr(search, "search_brave_scrape", lambda *a, **k: asked.append(("brave", None)) or [])
    monkeypatch.setattr(search, "search_wiby", lambda *a, **k: asked.append(("wiby", None)) or [])

    search._run_free_providers("q", 5, "en", False, freshness="day")

    assert asked == [("ddg", "d")]
