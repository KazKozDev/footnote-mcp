#!/usr/bin/env python3
"""Smoke test for the planning agent."""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.agent import load_mcp_tools, build_graph, _default_search_memory

async def main():
    tools = await load_mcp_tools()
    print(f"Tools: {len(tools)}")

    graph = build_graph("llama3.2:3b", tools)

    state = {
        "task": "find today's USD exchange rate",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "sources": [],
        "draft_result": {},
        "verification_result": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": _default_search_memory(),
    }

    result = await graph.ainvoke(state, {"recursion_limit": 60})

    print(f"\nSteps: {len(result['completed_steps'])}")
    for c in result["completed_steps"]:
        print(f"  {c['step'][:80]}")

    answer = result.get("final_answer", "")
    if hasattr(answer, "content"):
        answer = answer.content
    print(f"\nAnswer ({len(answer)} chars):\n{answer[:500]}")
    print("\nOK")

asyncio.run(main())
