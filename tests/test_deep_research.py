from __future__ import annotations

from footnote_mcp import deep_research


def _single_requirement_model(messages):
    prompt = messages[-1]["content"]
    if "Decompose the research question" in prompt:
        return {
            "answer_type": "single",
            "requirements": [{
                "id": "capital",
                "text": "capital of France",
                "subject": "France",
                "metric": "capital",
                "period": "",
                "unit": "",
                "completion_rule": "single",
            }],
        }
    if "Plan targeted web searches" in prompt:
        return {"queries": [{"requirement_id": "capital", "query": "France capital official"}]}
    if "Extract candidate evidence" in prompt:
        return {"items": [{
            "requirement_id": "capital",
            "source_id": "S1",
            "claim": "France capital Paris",
            "subject": "France",
            "metric": "capital",
            "period": "",
            "value": "Paris",
            "unit": "",
            "quote": "France capital Paris.",
        }]}
    if "Assess completeness" in prompt:
        return {"requirements": [{"id": "capital", "covered": True, "gap": ""}]}
    raise AssertionError(prompt)


def _discover(query, **kwargs):
    return ([{
        "title": "Official France data",
        "url": "https://example.gov/france",
        "snippet": "France capital Paris",
        "score": 1.0,
        "engines": ["test"],
    }], ["web"], {})


def _fetch(document, **kwargs):
    return {
        "url": document["url"],
        "title": document["title"],
        "success": True,
        "text": "France capital Paris.",
        "structured_chunks": [],
        "structured_rows": 0,
        "downloads": [],
    }


def test_research_loop_builds_verified_ledger_and_funnel(monkeypatch):
    result = deep_research.run_deep_research(
        "What is the capital of France?",
        discover=_discover,
        model_json=_single_requirement_model,
        fetch_fn=_fetch,
        budget=deep_research.ResearchBudget(max_iterations=3, initial_fetch=2, max_fetch=4),
    )

    state = result["state"]
    assert result["answer_ready"] is True
    assert state["stop_reason"] == "requirements_covered"
    assert state["coverage"] == 1.0
    assert state["evidence"][0]["value"] == "Paris"
    assert state["evidence"][0]["entity"] == "France"
    assert state["evidence"][0]["fact_quote"] == "France capital Paris."
    assert state["evidence"][0]["provenance"]["segment_id"] == "text-0"
    assert state["evidence"][0]["source_url"] == "https://example.gov/france"
    assert state["diagnostics"]["funnel"] == {
        "candidates": 1,
        "deduplicated_documents": 1,
        "relevant_documents": 1,
        "fetch_attempts": 1,
        "successful_fetches": 1,
        "structured_rows": 0,
        "ranked_chunks": 1,
        "extracted_facts": 1,
        "verified_evidence": 1,
        "contradictions": 0,
    }
    assert "VALUE: Paris" in result["context"]


def test_ungrounded_extracted_value_is_rejected():
    def bad_model(messages):
        prompt = messages[-1]["content"]
        if "Decompose the research question" in prompt:
            return _single_requirement_model(messages)
        if "Plan targeted web searches" in prompt:
            return _single_requirement_model(messages)
        if "Extract candidate evidence" in prompt:
            return {"items": [{
                "requirement_id": "capital",
                "source_id": "S1",
                "claim": "France capital Lyon",
                "subject": "France",
                "metric": "capital",
                "period": "",
                "value": "Lyon",
                "unit": "",
                "quote": "France capital Paris.",
            }]}
        return {"requirements": [{"id": "capital", "covered": False, "gap": "capital"}]}

    result = deep_research.run_deep_research(
        "What is the capital of France?",
        discover=_discover,
        model_json=bad_model,
        fetch_fn=_fetch,
        budget=deep_research.ResearchBudget(max_iterations=1, initial_fetch=2, max_fetch=2),
    )

    assert result["state"]["evidence"] == []
    assert result["answer_ready"] is False
    assert result["state"]["diagnostics"]["funnel"]["extracted_facts"] == 1
    assert result["state"]["diagnostics"]["funnel"]["verified_evidence"] == 0
    assert result["state"]["diagnostics"]["evidence_rejections"] == {"value_not_grounded_in_segment": 1}


def test_fetch_research_document_extracts_html_table_and_download(monkeypatch):
    html = """
    <html><body>
      <table><tr><th>Country</th><th>Value</th></tr><tr><td>France</td><td>42</td></tr></table>
      <a href="/data.csv">Download CSV</a>
    </body></html>
    """
    monkeypatch.setattr(
        deep_research,
        "fetch_page",
        lambda url, lang="en": (url, html, None, None),
    )

    result = deep_research.fetch_research_document(
        {"url": "https://example.gov/stats", "title": "Stats"},
        max_rows=10,
    )

    assert result["success"] is True
    assert result["structured_rows"] == 1
    assert result["structured_chunks"][0]["extraction_type"] == "html_table"
    assert "Country: France" in result["structured_chunks"][0]["text"]
    assert result["downloads"] == ["https://example.gov/data.csv"]


def test_fetch_research_document_detects_extensionless_minutes_link(monkeypatch):
    html = """
    <html><body>
      <a href="/FileStream.ashx?DocumentId=10">Unrelated attachment.pdf</a>
      <a href="/FileStream.ashx?DocumentId=20">Meeting Minutes Link</a>
    </body></html>
    """
    monkeypatch.setattr(
        deep_research,
        "fetch_page",
        lambda url, lang="en": (url, html, None, None),
    )

    result = deep_research.fetch_research_document(
        {
            "url": "https://example.gov/Meeting.aspx?id=1",
            "title": "Executive Committee October 2021",
            "research_query": "Executive Committee October 2021 meeting minutes",
        },
    )

    assert result["downloads"][0] == "https://example.gov/FileStream.ashx?DocumentId=20"
    assert result["download_candidates"][0]["text"] == "Meeting Minutes Link"
    assert result["download_candidates"][0]["score"] > result["download_candidates"][1]["score"]


def test_fetch_research_document_uses_file_parser(monkeypatch):
    monkeypatch.setattr(
        deep_research,
        "web_parse_file",
        lambda url, **kwargs: {
            "file_type": "csv",
            "tables": [{"columns": ["country", "value"], "rows": [{"country": "France", "value": "42"}]}],
        },
    )

    result = deep_research.fetch_research_document(
        {"url": "https://example.gov/data.csv", "title": "CSV"},
    )

    assert result["success"] is True
    assert result["structured_rows"] == 1
    assert result["structured_chunks"][0]["extraction_type"] == "csv"


def test_all_items_requirement_drives_second_gap_iteration():
    def model(messages):
        prompt = messages[-1]["content"]
        if "Decompose the research question" in prompt:
            return {
                "answer_type": "set",
                "requirements": [{
                    "id": "items",
                    "text": "all reported values",
                    "subject": "",
                    "metric": "value",
                    "period": "",
                    "unit": "",
                    "completion_rule": "all_items",
                }],
            }
        if "Plan targeted web searches" in prompt:
            return {"queries": [{"requirement_id": "items", "query": "second official table"}]}
        if "Extract candidate evidence" in prompt:
            if "Alpha value 1" in prompt:
                return {"items": [{
                    "requirement_id": "items",
                    "source_id": "S1", "claim": "Alpha value 1", "subject": "Alpha",
                    "metric": "value", "period": "", "value": "1", "unit": "",
                    "quote": "Alpha value 1.",
                }]}
            return {"items": [{
                "requirement_id": "items",
                "source_id": "S1", "claim": "Beta value 2", "subject": "Beta",
                "metric": "value", "period": "", "value": "2", "unit": "",
                "quote": "Beta value 2.",
            }]}
        if "Assess completeness" in prompt:
            covered = '"value": "1"' in prompt and '"value": "2"' in prompt
            return {"requirements": [{"id": "items", "covered": covered, "gap": "find remaining values"}]}
        raise AssertionError(prompt)

    def discover(query, **kwargs):
        slug = "second" if query.startswith("second") else "first"
        return ([{
            "title": slug,
            "url": f"https://example.gov/{slug}",
            "snippet": f"{slug} table",
            "score": 1.0,
            "engines": ["test"],
        }], ["web"], {})

    def fetch(document, **kwargs):
        text = "Beta value 2." if document["url"].endswith("second") else "Alpha value 1."
        return {
            "url": document["url"], "title": document["title"], "success": True,
            "text": "", "structured_chunks": [{
                "segment_id": "table-1-row-1", "text": text,
                "context_text": "HEADER: entity | value", "extraction_type": "html_table",
                "provenance": {"segment_type": "table_row", "complete_set": document["url"].endswith("second")},
            }], "structured_rows": 1, "downloads": [],
        }

    result = deep_research.run_deep_research(
        "Find all reported values",
        discover=discover,
        model_json=model,
        fetch_fn=fetch,
        budget=deep_research.ResearchBudget(max_iterations=3, initial_fetch=1, fetch_growth=1, max_fetch=3),
    )

    state = result["state"]
    assert state["iteration"] == 2
    assert state["stop_reason"] == "requirements_covered"
    assert [item["value"] for item in state["evidence"]] == ["1", "2"]
    assert len(state["queries"]) == 2
    assert state["diagnostics"]["funnel"]["fetch_attempts"] == 2


def test_multiple_requirements_use_isolated_extraction_calls():
    extraction_calls = 0

    def model(messages):
        nonlocal extraction_calls
        prompt = messages[-1]["content"]
        if "Decompose the research question" in prompt:
            return {
                "answer_type": "set",
                "requirements": [
                    {"id": "fr", "text": "capital of France", "subject": "France", "metric": "capital", "period": "", "unit": "", "completion_rule": "single"},
                    {"id": "de", "text": "capital of Germany", "subject": "Germany", "metric": "capital", "period": "", "unit": "", "completion_rule": "single"},
                ],
            }
        if "Plan targeted web searches" in prompt:
            return {"queries": [
                {"requirement_id": "fr", "query": "France capital official"},
                {"requirement_id": "de", "query": "Germany capital official"},
            ]}
        if "Extract candidate evidence" in prompt:
            extraction_calls += 1
            return {"items": [
                {"requirement_id": "fr", "source_id": "S1", "claim": "France capital Paris", "subject": "France", "metric": "capital", "period": "", "value": "Paris", "unit": "", "quote": "France capital Paris."},
                {"requirement_id": "de", "source_id": "S1", "claim": "Germany capital Berlin", "subject": "Germany", "metric": "capital", "period": "", "value": "Berlin", "unit": "", "quote": "Germany capital Berlin."},
            ]}
        raise AssertionError(prompt)

    def discover(query, **kwargs):
        return ([{
            "title": "Official capitals",
            "url": "https://example.gov/capitals",
            "snippet": "France Paris Germany Berlin",
            "score": 1.0,
            "engines": ["test"],
        }], ["web"], {})

    def fetch(document, **kwargs):
        return {
            "url": document["url"], "title": document["title"], "success": True,
            "text": "France capital Paris. Germany capital Berlin.",
            "structured_chunks": [], "structured_rows": 0, "downloads": [],
        }

    result = deep_research.run_deep_research(
        "What are the capitals of France and Germany?",
        discover=discover,
        model_json=model,
        fetch_fn=fetch,
        budget=deep_research.ResearchBudget(max_iterations=1, initial_fetch=2, max_fetch=2),
    )

    assert extraction_calls == 2
    assert result["answer_ready"] is True
    assert {item["value"] for item in result["state"]["evidence"]} == {"Paris", "Berlin"}


def test_batched_extraction_context_is_bounded():
    requirement = deep_research.ResearchRequirement(id="r1", text="target fact")
    captured = {}

    def model(messages):
        captured["prompt"] = messages[-1]["content"]
        return {"items": []}

    state = deep_research.ResearchState(query="target fact", requirements=[requirement])
    deep_research._extract_and_verify_batch(
        [requirement],
        {"r1": [{
            "text": "target fact " + "x" * 10000,
            "source_url": "https://example.com",
            "source_title": "Example",
            "extraction_type": "text",
        }]},
        iteration=1,
        model_json=model,
        entailment_backend="heuristic",
        entailment_model=None,
        state=state,
        max_context_chars=1000,
    )

    assert len(captured["prompt"]) < 4000
    assert "x" * 1500 not in captured["prompt"]


def test_research_deadline_stops_before_discovery_and_reports_progress():
    stages = []
    result = deep_research.run_deep_research(
        "What is the capital of France?",
        discover=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not discover")),
        model_json=_single_requirement_model,
        fetch_fn=_fetch,
        budget=deep_research.ResearchBudget(max_elapsed_seconds=0),
        progress=lambda stage, details: stages.append(stage),
    )

    assert result["state"]["stop_reason"] == "research_deadline_exhausted"
    assert stages[:2] == ["requirement_planning", "requirements_ready"]
    assert stages[-1] == "research_complete"


def test_direct_url_is_fetched_even_when_search_returns_nothing():
    fetched_urls = []

    def model(messages):
        prompt = messages[-1]["content"]
        if "Decompose the research question" in prompt:
            return {"answer_type": "single", "requirements": [{"id": "r1", "text": "capital", "completion_rule": "single"}]}
        if "Extract candidate evidence" in prompt:
            return {"items": [{"requirement_id": "r1", "source_id": "S1", "entity": "France", "predicate": "capital", "value": "Paris", "qualifiers": {}}]}
        return {"queries": []}

    def fetch(document, **kwargs):
        fetched_urls.append(document["url"])
        return {"url": document["url"], "title": "Direct", "success": True, "text": "France capital Paris", "structured_chunks": [], "structured_rows": 0}

    result = deep_research.run_deep_research(
        "Use https://example.test/fact to find the capital",
        discover=lambda *args, **kwargs: ([], [], {}), model_json=model, fetch_fn=fetch,
        budget=deep_research.ResearchBudget(max_iterations=1, initial_fetch=2, max_fetch=2),
    )
    assert fetched_urls == ["https://example.test/fact"]
    assert result["answer_ready"] is True


def test_cross_month_attendance_question_expands_to_atomic_requirements():
    def model(messages):
        return {
            "answer_type": "set",
            "requirements": [{
                "id": "broad",
                "text": "members absent more than once across the meetings",
                "subject": "board members",
                "metric": "absent",
                "period": "May, June, September, October, and November 2021",
                "unit": "meetings",
                "completion_rule": "all_items",
            }],
        }

    requirements, answer_type, error = deep_research.decompose_requirements(
        "According to the publicly available meeting minutes for the Executive Committee of the Toronto and "
        "Region Conservation Authority, which board members were absent more than once from meetings in May, "
        "June, September, October, and November 2021?",
        model,
    )

    assert error == ""
    assert answer_type == "set"
    assert [item.period for item in requirements] == [
        "May 2021", "June 2021", "September 2021", "October 2021", "November 2021",
    ]
    assert all(item.completion_rule == "all_items" for item in requirements)
    assert all(item.metric == "absent" for item in requirements)
    assert all(item.unit == "" for item in requirements)
    assert all("Toronto and Region Conservation Authority" in item.subject for item in requirements)
    assert all(item.scope["organization"].endswith("Toronto and Region Conservation Authority") for item in requirements)


def test_requirement_planner_accepts_string_scope_and_qualifiers():
    requirements, _, error = deep_research.decompose_requirements("Question", lambda _messages: {
        "answer_type": "single",
        "requirements": [{
            "id": "r1", "text": "fact", "subject": "", "metric": "population",
            "period": "2025", "unit": "people", "scope": "Example City",
            "predicate": "population", "qualifiers": "official estimate",
            "completion_rule": "single",
        }],
    })
    assert error == ""
    assert requirements[0].scope == {"scope": "Example City"}
    assert requirements[0].qualifiers == {
        "context": "official estimate", "period": "2025", "unit": "people",
    }


def test_planner_normalizes_na_and_unit_aliases_are_deterministic():
    requirements, _, _ = deep_research.decompose_requirements("Question", lambda _messages: {
        "answer_type": "single",
        "requirements": [{
            "id": "r1", "text": "fact", "period": "N/A", "unit": "not applicable",
            "completion_rule": "single",
        }],
    })
    assert requirements[0].period == ""
    assert requirements[0].unit == ""


def test_unit_verdict_separates_a_stated_conflict_from_a_missing_label():
    grounded = deep_research._QUALIFIER_GROUNDED
    absent = deep_research._QUALIFIER_ABSENT
    conflict = deep_research._QUALIFIER_CONFLICT

    assert deep_research._unit_verdict("USD billions", "", "$42 billion") == grounded
    assert deep_research._unit_verdict("percentage", "", "Value: 4.2%") == grounded
    # Entity-shaped labels describe the row type, so no document must repeat them.
    assert deep_research._unit_verdict("people", "", "Ada Lovelace") == grounded
    assert deep_research._unit_verdict("person names", "", "Ada Lovelace") == grounded
    # A unit the document never states is missing context, not a mismatch.
    assert deep_research._unit_verdict("kilograms", "", "Value: 42") == absent
    # A rival unit of the same dimension is a genuine defect.
    assert deep_research._unit_verdict("kilograms", "", "Weight: 42 pounds") == conflict
    assert deep_research._unit_verdict("USD", "", "Revenue: 42 million euros") == conflict


def test_unit_lives_in_the_column_header_not_the_row():
    chunk = {
        "text": "Norway | 42",
        "context_text": "HEADER: Country | Revenue",
        "provenance": {"columns": ["Country", "Revenue (USD billions)"]},
    }
    grounding = " ".join([chunk["text"], chunk["context_text"], deep_research._provenance_header_text(chunk)])
    assert deep_research._unit_verdict("USD billions", "", grounding) == deep_research._QUALIFIER_GROUNDED


def test_period_verdict_rejects_a_different_month_but_tolerates_silence():
    assert deep_research._period_verdict("September 2021", "", "Minutes of the September 2021 meeting") == deep_research._QUALIFIER_GROUNDED
    assert deep_research._period_verdict("September 2021", "", "Minutes of the October 2021 meeting") == deep_research._QUALIFIER_CONFLICT
    assert deep_research._period_verdict("September 2021", "", "Minutes of the September 2019 meeting") == deep_research._QUALIFIER_CONFLICT
    assert deep_research._period_verdict("September 2021", "", "Paula Fletcher was absent") == deep_research._QUALIFIER_ABSENT


def test_scope_accepts_an_organization_stated_by_acronym():
    assert deep_research._scope_is_grounded(
        "Toronto and Region Conservation Authority", "TRCA Executive Committee minutes"
    )
    assert not deep_research._scope_is_grounded(
        "Toronto and Region Conservation Authority", "Credit Valley Conservation minutes"
    )


def test_scope_accepts_a_publisher_that_shortens_its_own_name():
    assert deep_research._scope_is_grounded(
        "New Zealand Electoral Commission", "Electoral Commission official results"
    )
    # A different body sharing one generic word is still rejected.
    assert not deep_research._scope_is_grounded(
        "New Zealand Electoral Commission", "Australian Bureau of Statistics commission report"
    )


def test_single_answer_cannot_leave_all_items_completion_rule():
    requirements, answer_type, _ = deep_research.decompose_requirements("Where is it?", lambda _messages: {
        "answer_type": "single",
        "requirements": [{"id": "r1", "text": "location", "completion_rule": "all_items"}],
    })
    assert answer_type == "single"
    assert requirements[0].completion_rule == "single"


def test_chunks_are_never_mixed_across_requirements():
    requirements = [
        deep_research.ResearchRequirement(id="fr", text="France capital"),
        deep_research.ResearchRequirement(id="de", text="Germany capital"),
    ]
    fetched = [
        {"success": True, "url": "https://fr.example", "title": "France", "text": "France Paris", "requirements": ["fr"]},
        {"success": True, "url": "https://de.example", "title": "Germany", "text": "Germany Berlin", "requirements": ["de"]},
    ]
    ranked = deep_research._rank_chunks_for_requirements(requirements, fetched, lang="en", top_k=10)
    assert {item["source_url"] for item in ranked["fr"]} == {"https://fr.example"}
    assert {item["source_url"] for item in ranked["de"]} == {"https://de.example"}


def test_large_chunk_pool_uses_bounded_lexical_requirement_prefilter(monkeypatch):
    seen = []

    def fake_rerank(query, chunks, top_k, lang):
        seen.append(len(chunks))
        return chunks[:top_k]

    monkeypatch.setattr(deep_research, "rerank_chunks", fake_rerank)
    structured = [
        {"segment_id": f"row-{index}", "text": f"country {index}", "context_text": "table", "provenance": {}}
        for index in range(1000)
    ]
    deep_research._rank_chunks_for_requirements(
        [deep_research.ResearchRequirement(id="r1", text="target country")],
        [{"success": True, "url": "u", "title": "t", "text": "", "requirements": ["r1"], "structured_chunks": structured}],
        lang="en", top_k=24,
    )
    assert seen == []


def test_structured_segments_preserve_header_row_cell_and_pdf_lines():
    chunks, count = deep_research._parsed_file_chunks({
        "file_type": "csv",
        "tables": [{"table_index": 2, "columns": ["Name", "Status"], "rows": [{"Name": "Ada", "Status": "Absent"}]}],
        "pages": [{"page": 4, "text": "Committee Minutes\nNovember 2021\nABSENT\nAda Lovelace"}],
    }, "https://example.test/minutes.pdf")
    assert count == 1
    table = next(item for item in chunks if item["provenance"]["segment_type"] == "table_row")
    assert table["context_text"] == "HEADER: Name | Status"
    assert table["provenance"]["cells"] == {"Name": "Ada", "Status": "Absent"}
    page = next(item for item in chunks if item["provenance"]["segment_type"] == "pdf_lines")
    assert page["provenance"]["page"] == 4
    assert page["provenance"]["line_start"] == 1
    assert page["provenance"]["line_end"] == 4


def test_wrong_scope_is_rejected_but_period_may_come_from_context():
    requirement = deep_research.ResearchRequirement(
        id="may", text="absent members", subject="Target Authority", metric="absent",
        period="May 2021", scope={"organization": "Target Authority"},
        predicate="absent", qualifiers={"period": "May 2021"}, completion_rule="all_items",
    )
    state = deep_research.ResearchState(query="q", requirements=[requirement])

    def model(_messages):
        return {"items": [{"requirement_id": "may", "source_id": "S1", "entity": "Ada", "predicate": "absent", "value": "Absent", "qualifiers": {"period": "May 2021"}}]}

    common = {"segment_id": "row-1", "text": "Ada | Absent", "source_url": "https://example.test", "extraction_type": "table", "provenance": {"complete_set": True}}
    verified, _, _ = deep_research._extract_and_verify_batch(
        [requirement], {"may": [{**common, "source_title": "Other Authority", "context_text": "May 2021 attendance"}]},
        iteration=1, model_json=model, entailment_backend="heuristic", entailment_model=None,
        state=state, max_context_chars=4000,
    )
    assert verified == []
    assert state.diagnostics["evidence_rejections"] == {"requirement_scope_not_grounded": 1}

    state.diagnostics["evidence_rejections"] = {}
    verified, _, _ = deep_research._extract_and_verify_batch(
        [requirement], {"may": [{**common, "source_title": "Target Authority", "context_text": "May 2021 attendance"}]},
        iteration=1, model_json=model, entailment_backend="heuristic", entailment_model=None,
        state=state, max_context_chars=4000,
    )
    assert len(verified) == 1
    assert verified[0].fact_quote == "Ada | Absent"
    assert verified[0].context_quote == "May 2021 attendance"


def test_ledger_group_count_filter_derives_list_in_code():
    requirements = [deep_research.ResearchRequirement(id=f"r{i}", text="attendance", status="covered") for i in range(3)]
    state = deep_research.ResearchState(
        query="Who was absent more than once?", requirements=requirements,
        aggregation_plan={"operation": "group_count_filter", "group_by": "entity", "operator": ">", "threshold": 1},
    )
    for requirement_id, entity, period in [("r0", "Ada", "May"), ("r1", "Ada", "June"), ("r2", "Grace", "July")]:
        state.evidence.append(deep_research.EvidenceItem(
            id=f"{requirement_id}-{entity}", requirement_id=requirement_id, claim="", subject=entity,
            metric="absent", period=period, value="Absent", unit="", quote="", source_url="u",
            source_title="t", status="supported", score=1, verification="test", iteration=1,
            entity=entity, predicate="absent", qualifiers={"period": period},
        ))
    deep_research._aggregate_evidence(state)
    assert state.derived_answer == ["Ada"]


def _evidence(requirement_id, entity, *, source_url="u", period="", provenance=None):
    return deep_research.EvidenceItem(
        id=f"{requirement_id}-{entity}-{source_url}", requirement_id=requirement_id, claim="",
        subject=entity, metric="m", period=period, value="v", unit="", quote="",
        source_url=source_url, source_title="t", status="supported", score=1,
        verification="test", iteration=1, entity=entity, predicate="m",
        qualifiers={"period": period} if period else {},
        provenance=dict(provenance or {}),
    )


def test_filter_steps_are_supporting_and_do_not_block_the_answer():
    requirements, _, _ = deep_research.decompose_requirements("Question", lambda _messages: {
        "answer_type": "set",
        "requirements": [
            {"id": "r1", "text": "Identify the set of individuals who signed the Declaration.", "completion_rule": "all_items"},
            {"id": "r2", "text": "Filter the identified signers to exclude any who served as President.", "completion_rule": "all_items"},
        ],
    })
    assert [item.necessity for item in requirements] == ["required", "supporting"]

    state = deep_research.ResearchState(query="q", requirements=requirements)
    state.evidence.append(_evidence("r1", "Button Gwinnett", provenance={"complete_set": True}))
    deep_research._assess_coverage(state)
    assert [item.id for item in state.unresolved()] == ["r2"]
    assert state.blocking() == []
    assert state.answer_ready is True


def test_answer_is_never_ready_without_ledger_evidence():
    requirement = deep_research.ResearchRequirement(id="r1", text="fact", status="covered")
    state = deep_research.ResearchState(query="q", requirements=[requirement])
    deep_research._assess_coverage(state)
    assert state.answer_ready is False


def test_exclusion_question_is_answered_by_set_difference_over_the_entity_table():
    requirements = [
        deep_research.ResearchRequirement(id="r1", text="Identify the signers", completion_rule="all_items"),
        deep_research.ResearchRequirement(
            id="r2", text="Exclude any individual who served as President", necessity="supporting",
        ),
    ]
    state = deep_research.ResearchState(
        query="Which signers of the Declaration never served as President?",
        requirements=requirements,
        aggregation_plan=deep_research._aggregation_plan(
            "Which signers of the Declaration never served as President?"
        ),
    )
    assert state.aggregation_plan["operation"] == "set_difference"
    state.evidence.extend([
        _evidence("r1", "John Adams"),
        _evidence("r1", "Button Gwinnett"),
        _evidence("r2", "John Adams"),
    ])
    deep_research._aggregate_evidence(state)
    assert state.derived_answer == ["Button Gwinnett"]
    assert [row["entity"] for row in state.candidate_entities] == ["Button Gwinnett", "John Adams"]


def test_intersection_question_keeps_only_entities_meeting_every_requirement():
    requirements = [
        deep_research.ResearchRequirement(id="r1", text="Cities hosting the summit"),
        deep_research.ResearchRequirement(id="r2", text="Cities on the coast"),
    ]
    state = deep_research.ResearchState(
        query="Which cities hosted the summit as well as sitting on the coast?",
        requirements=requirements,
        aggregation_plan={"operation": "set_intersection", "group_by": "entity"},
    )
    state.evidence.extend([
        _evidence("r1", "Lisbon"), _evidence("r2", "Lisbon"), _evidence("r1", "Vienna"),
    ])
    deep_research._aggregate_evidence(state)
    assert state.derived_answer == ["Lisbon"]


def test_exhaustive_set_closes_on_corroborating_documents_without_a_container():
    requirement = deep_research.ResearchRequirement(id="r1", text="all members", completion_rule="all_items")
    partial = [_evidence("r1", "Ada", source_url="https://one.test", provenance={"enumeration": True})]
    assert deep_research._set_is_closed(requirement, partial) is False

    corroborated = partial + [
        _evidence("r1", "Ada", source_url="https://two.test", provenance={"enumeration": True})
    ]
    assert deep_research._set_is_closed(requirement, corroborated) is True

    counted = deep_research.ResearchRequirement(
        id="r1", text="all members", completion_rule="all_items", expected_count=1,
    )
    assert deep_research._set_is_closed(counted, partial) is True


def test_unit_absent_from_the_document_no_longer_rejects_the_fact():
    requirement = deep_research.ResearchRequirement(
        id="r1", text="population", subject="Norway", metric="population", unit="people",
        predicate="population", qualifiers={"unit": "people"},
    )
    state = deep_research.ResearchState(query="q", requirements=[requirement])

    def model(_messages):
        return {"items": [{
            "requirement_id": "r1", "source_id": "S1", "entity": "Norway",
            "predicate": "population", "value": "5425000", "qualifiers": {},
        }]}

    chunk = {
        "segment_id": "row-1", "text": "Norway | 5425000", "context_text": "HEADER: Country | Population",
        "source_url": "https://example.test", "source_title": "Population table",
        "extraction_type": "table", "provenance": {"columns": ["Country", "Population"]},
    }
    verified, _, _ = deep_research._extract_and_verify_batch(
        [requirement], {"r1": [chunk]}, iteration=1, model_json=model,
        entailment_backend="heuristic", entailment_model=None, state=state, max_context_chars=4000,
    )
    assert len(verified) == 1
    assert state.diagnostics["evidence_rejections"] == {}


def test_conflicting_period_is_still_rejected():
    requirement = deep_research.ResearchRequirement(
        id="r1", text="absent members", subject="Committee", metric="absent", period="September 2021",
        predicate="absent", qualifiers={"period": "September 2021"},
    )
    state = deep_research.ResearchState(query="q", requirements=[requirement])

    def model(_messages):
        return {"items": [{
            "requirement_id": "r1", "source_id": "S1", "entity": "Paula Fletcher",
            "predicate": "absent", "value": "Absent", "qualifiers": {},
        }]}

    chunk = {
        "segment_id": "row-1", "text": "Paula Fletcher | Absent",
        "context_text": "Minutes of the October 2021 meeting", "source_url": "https://example.test",
        "source_title": "Committee minutes", "extraction_type": "table", "provenance": {},
    }
    verified, _, _ = deep_research._extract_and_verify_batch(
        [requirement], {"r1": [chunk]}, iteration=1, model_json=model,
        entailment_backend="heuristic", entailment_model=None, state=state, max_context_chars=4000,
    )
    assert verified == []
    assert state.diagnostics["evidence_rejections"] == {"required_period_conflicts_with_document": 1}


def test_fetch_pool_returns_at_deadline_without_waiting_for_worker():
    import time

    def slow_fetch(document, **kwargs):
        time.sleep(0.5)
        return {"url": document["url"], "success": True, "text": "late"}

    started = time.monotonic()
    result = deep_research._fetch_documents(
        [{"url": "https://slow.test", "requirements": ["r1"]}], lang="en", max_rows=10,
        structured_file_budget=0, fetch_fn=slow_fetch, deadline=time.monotonic() + 0.05,
    )
    assert result == []
    assert time.monotonic() - started < 0.2


def test_single_answer_question_never_gets_a_set_operation_plan(monkeypatch):
    question = "Which treaty did the state sign that did not include a defence clause?"
    assert deep_research._aggregation_plan(question)["operation"] == "set_difference"

    def fake_decompose(query, model_json=None):
        return [deep_research.ResearchRequirement(id="r1", text="treaty")], "single", ""

    def fake_discover(*args, **kwargs):
        return [], [], {}

    monkeypatch.setattr(deep_research, "decompose_requirements", fake_decompose)

    result = deep_research.run_deep_research(
        question, discover=fake_discover, model_json=None,
        budget=deep_research.ResearchBudget(max_iterations=1),
    )
    assert result["state"]["aggregation_plan"] == {}
