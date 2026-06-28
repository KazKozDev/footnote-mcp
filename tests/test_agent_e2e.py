from __future__ import annotations

import asyncio
import json

import agent.agent as agent_module


def test_agent_graph_completes_with_fake_model_and_fake_tool(monkeypatch):
    def fake_chat(model, messages, tools=None, system="", temperature=0.3, json_mode=False, num_predict=32768):
        if system == agent_module.REQUIREMENTS_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "target": "test rate",
                        "scope": "single fact",
                        "granularity": None,
                        "unit_or_pair": "EUR/RUB",
                        "required_coverage": "rate value",
                        "output_format": "sentence",
                        "completion_criteria": ["Provide the sourced rate."],
                        "missing_data_policy": "fail if missing",
                        "search_hints": [],
                    }
                )
            }
        if system == agent_module.PLAN_PROMPT:
            return {"content": json.dumps(["web_read: https://example.com/source"])}
        # Prose-first answer/verify flow: draft prose, re-ground it, then a compact
        # JSON verdict (see answer_node / verify_node).
        if system == agent_module.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The sourced EUR/RUB rate is 90.1 [1]."}
        if system == agent_module.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The sourced EUR/RUB rate is 90.1 [1]."}
        if system == agent_module.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {"content": json.dumps({"coverage_complete": True, "missing": [], "notes": []})}
        # execute_node's post-batch controller decides DONE/CONTINUE/NEXT. Its prompt is
        # a local string, so match on stable wording. The web_read source is fetched, so
        # DONE is valid and ends the loop.
        if "After executing a batch of steps" in system:
            return {"content": json.dumps({"decision": "DONE", "reason": "rate is covered"})}
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_read"
        return json.dumps(
            {
                "url": args["url"],
                "title": "Source",
                "text": "Official source says EUR/RUB rate is 90.1.",
                "pub_date": "2026-05-01",
                "text_length": 41,
            }
        )

    class FakeStore:
        def get_strategies(self, limit=5):
            return []

        def get_skills(self, limit=5):
            return []

        def best_strategy(self):
            return None

        def record_strategy(self, *args, **kwargs):
            return None

        def save_skill(self, *args, **kwargs):
            return None

        def add_experience(self, *args, **kwargs):
            return None

    monkeypatch.setattr(agent_module, "_ollama_chat", fake_chat)
    monkeypatch.setattr(agent_module, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(agent_module, "_get_research_store", lambda: FakeStore())

    graph = agent_module.build_graph("fake-model", tools=[])
    state = {
        "task": "Find the test rate",
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
        "search_memory": agent_module._default_search_memory(),
    }

    final = asyncio.run(graph.ainvoke(state, {"recursion_limit": 80}))

    assert final["final_answer"] == "The sourced EUR/RUB rate is 90.1 [1]."
    assert final["verification_result"]["task_complete"] is True
    assert len(final["completed_steps"]) == 1
    assert final["sources"][0]["url"] == "https://example.com/source"


def test_agent_writes_debug_report_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBOPERATOR_DEBUG_REPORTS", "1")
    monkeypatch.setenv("WEBOPERATOR_DEBUG_REPORT_DIR", str(tmp_path))
    state = {
        "task": "Debug task",
        "requirements_result": {"target": "debug"},
        "search_memory": {"attempted_queries": ["q"]},
        "sources": [{"url": "https://example.com", "title": "S", "content": "abc"}],
        "verification_result": {"task_complete": False, "gaps": ["gap"]},
    }

    path = agent_module._write_debug_report(state)

    assert path is not None
    data = json.loads((tmp_path / path.split("/")[-1]).read_text())
    assert data["task"] == "Debug task"
    assert data["attempted_queries"] == ["q"]


def test_agent_normalizes_python_like_tool_steps():
    call = agent_module._tool_call_from_step(
        "check_date_completeness(start_date='2026-05-01', end_date='2026-05-31', "
        "actual_items=None, granularity='day', calendar='calendar', holidays=None)"
    )

    assert call == {
        "function": {
            "name": "check_date_completeness",
            "arguments": {
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
                "actual_items": [],
                "granularity": "day",
                "calendar": "calendar",
                "holidays": [],
            },
        }
    }


def test_agent_accepts_direct_json_fetch_steps():
    call = agent_module._tool_call_from_step("web_fetch_json: https://example.com/api.json")

    assert call == {"function": {"name": "web_fetch_json", "arguments": {"url": "https://example.com/api.json"}}}


def test_strategy_plan_uses_memory_queries_as_search_steps():
    memory = agent_module._default_search_memory()
    memory["next_queries"] = ["specific source query", "another query"]

    steps = agent_module._strategy_plan_from_memory("daily rate for requested range", memory)

    # The refactored strategy emits search-only steps; the post-batch LLM decides
    # which result URLs to read. No deterministic placeholder scaffolding remains.
    assert steps == ["web_search: specific source query", "web_search: another query"]
    assert all(step.startswith("web_search: ") for step in steps)


def test_strategy_plan_falls_back_to_question_when_no_memory_queries():
    steps = agent_module._strategy_plan_from_memory("daily rate", agent_module._default_search_memory())

    assert steps == ["web_search: daily rate"]


def test_generate_search_queries_step_accepts_json_arguments():
    call = agent_module._tool_call_from_step(
        'generate_search_queries: {"task":"rates","requirements":{"granularity":"day"},"max_queries":6}'
    )

    assert call == {
        "function": {
            "name": "generate_search_queries",
            "arguments": {"task": "rates", "requirements": {"granularity": "day"}, "max_queries": 6},
        }
    }


def test_recipe_run_step_and_result_become_source():
    code = "def extract(source_text, input_payload):\n    return {'rows': []}"
    step = agent_module._tool_call_from_step(
        'tool_code_run_sandboxed: {"code": '
        + json.dumps(code)
        + ', "source_text": "2026-05-01 90.1", "input_payload": {"source_url": "https://example.com/rates"}}'
    )
    sources = agent_module._sources_from_tool_result(
        "tool_code_run_sandboxed",
        json.dumps(
            {
                "ok": True,
                "result": {
                    "rows": [
                        {
                            "date": "2026-05-01",
                            "value": "90.1",
                            "unit": "EUR/USD",
                            "source_url": "https://example.com/rates",
                        }
                    ],
                    "row_count": 1,
                },
            }
        ),
    )

    assert step["function"]["name"] == "tool_code_run_sandboxed"
    assert step["function"]["arguments"]["input_payload"]["source_url"] == "https://example.com/rates"
    assert sources[0]["kind"] == "recipe_rows"
    assert sources[0]["url"] == "https://example.com/rates"


def test_observation_normalization_accepts_only_controller_tags():
    payload = {"url": "https://example.org/source", "text": "Fetched text."}
    observation = agent_module._normalize_observation(
        {
            "useful": True,
            "structured": False,
            "has_rows": False,
            "dated": False,
            "source_quality": "secondary",
            "gaps": ["needs another source"],
            "next_action_tags": ["search_better_sources", "unsupported_custom_tag"],
            "suggested_queries": ["specific follow-up query", "specific follow-up query"],
            "reason": "The source only covers part of the requirement.",
        },
        "web_read",
        {"url": payload["url"]},
        payload,
        "web_read: https://example.org/source",
    )

    assert observation["useful"] is True
    assert observation["source_type"] == "secondary"
    assert observation["next_action_tags"] == ["search_better_sources"]
    assert observation["suggested_queries"] == ["specific follow-up query"]


def test_evidence_audit_blocks_plan_exhaustion_without_sources(monkeypatch):
    # When the audit fails, evaluate_node asks the model to reflect; stub it out.
    monkeypatch.setattr(agent_module, "_ollama_chat", lambda *a, **k: {"content": ""})
    state = {
        "task": "Find the requested fact",
        "requirements_result": {
            "target": "requested fact",
            "required_coverage": "Provide the requested fact from fetched evidence.",
            "completion_criteria": ["Provide one sourced answer."],
        },
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
        "search_memory": agent_module._default_search_memory(),
    }

    update = asyncio.run(agent_module.evaluate_node(state, model="fake", _tools=[]))
    routed_state = state | update

    assert update["evidence_audit"]["passed"] is False
    assert update["verification_result"]["insufficient_evidence"] is True
    assert agent_module.route_after_evaluate(routed_state) == "strategy"


def test_evidence_audit_passes_once_any_source_is_fetched():
    # The refactored audit only requires that some evidence was fetched; it no
    # longer gates on structured sources (that deterministic heuristic was removed).
    state = {
        "task": "euro rate for everyday may 2026",
        "requirements_result": {
            "target": "daily exchange rate",
            "granularity": "day",
            "required_coverage": "Provide a rate for each day in the requested date range.",
            "completion_criteria": ["Provide a rate for each of the 31 days."],
        },
        "sources": [
            {
                "title": "Exchange-rate article",
                "url": "https://example.org/rates-note",
                "published": "2026-05-01",
                "content": "2026-05-01 rate was 90.1.",
                "kind": "page",
            }
        ],
    }

    audit = agent_module._audit_evidence_state(state)

    assert audit["passed"] is True
    assert audit["gaps"] == []
    assert audit["source_count"] == 1


def test_record_observation_tracks_unhelpful_source_in_memory():
    requirements = {
        "target": "requested fact",
        "required_coverage": "Answer all requested parts.",
        "completion_criteria": ["Cover the requested fact."],
    }
    observation = agent_module._normalize_observation(
        {
            "useful": False,
            "structured": False,
            "has_rows": False,
            "dated": False,
            "source_quality": "unknown",
            "gaps": ["source does not cover required detail"],
            "next_action_tags": ["search_better_sources"],
            "suggested_queries": ["specific required detail source"],
            "reason": "Needs a better-matching source.",
        },
        "web_read",
        {"url": "https://example.org/source"},
        {"url": "https://example.org/source", "text": "Partial source."},
        "web_read: https://example.org/source",
    )
    memory = agent_module._record_observation(agent_module._default_search_memory(), observation, requirements)

    # Next-step planning is now an LLM decision; the deterministic memory bookkeeping
    # is what we assert: controller tags recorded, unhelpful source quarantined.
    assert observation["useful"] is False
    assert "search_better_sources" in memory["next_actions"]
    assert "source does not cover required detail" in memory["open_gaps"]
    assert "https://example.org/source" in memory["avoid_urls"]
    assert "example.org" in memory["bad_domains"]


def test_fallback_observation_records_objective_facts_for_empty_tables():
    requirements = {
        "target": "daily exchange rate",
        "granularity": "day",
        "required_coverage": "Provide a rate for each day in the requested date range.",
        "completion_criteria": ["Provide a rate for each of the 31 days."],
    }
    payload = {"url": "https://example.org/rates", "tables": [], "table_count": 0}

    observation = agent_module._fallback_observation(
        "web_extract_tables",
        {"url": payload["url"]},
        payload,
        f"web_extract_tables: {payload['url']}",
    )
    memory = agent_module._record_observation(agent_module._default_search_memory(), observation, requirements)

    # The fallback now records only objective facts and leaves strategy (gaps,
    # next actions, escalation) to the LLM — no deterministic inference.
    assert observation["structured"] is True
    assert observation["has_rows"] is False
    assert observation["useful"] is False
    assert observation["gaps"] == []
    # An unhelpful structured fetch is quarantined so the agent stops retrying it.
    assert "https://example.org/rates" in memory["avoid_urls"]
