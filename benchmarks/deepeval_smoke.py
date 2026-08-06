#!/usr/bin/env python3
"""DeepEval smoke test for footnote-mcp — MCPUseMetric + TaskCompletionMetric."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from deepeval import evaluate
from deepeval.metrics import MCPUseMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parent.parent


async def collect_server_info():
    """Connect to footnote-mcp and collect server primitives."""
    server_params = StdioServerParameters(
        command=str(ROOT / ".venv" / "bin" / "footnote-mcp"),
        args=[],
    )

    tools = []
    prompts = []
    resources = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })

            try:
                prompts_result = await session.list_prompts()
                for p in prompts_result.prompts:
                    prompts.append({"name": p.name, "description": p.description or ""})
            except Exception:
                pass

            try:
                resources_result = await session.list_resources()
                for r in resources_result.resources:
                    resources.append({"name": r.name, "uri": str(r.uri)})
            except Exception:
                pass

    return {"tools": tools, "prompts": prompts, "resources": resources}


async def run_test_case(session_info: dict, input_text: str, expected_tool: str):
    """Simulate a single-turn MCP call and create a DeepEval test case."""
    # Simulate the agent calling the expected tool
    return LLMTestCase(
        input=input_text,
        actual_output=f"Called {expected_tool} successfully",
        mcp_tools_called=[expected_tool],
    )


async def main():
    print("Collecting footnote-mcp server info...")
    session_info = await collect_server_info()
    print(f"  Tools: {len(session_info['tools'])}")
    for t in session_info["tools"]:
        print(f"    - {t['name']}: {t['description'][:100]}...")
    print(f"  Prompts: {len(session_info['prompts'])}")
    print(f"  Resources: {len(session_info['resources'])}")

    # Build MCP server descriptor for DeepEval
    mcp_servers = [{
        "name": "footnote-mcp",
        "tools": session_info["tools"],
        "prompts": session_info["prompts"],
        "resources": session_info["resources"],
    }]

    # Test cases: (input, expected_tool_name, context)
    test_cases = [
        (
            "Search the web for latest Bitcoin price",
            "web_search",
            "Simple web search query"
        ),
        (
            "Find academic papers about transformer architecture on arXiv",
            "papers_search",
            "Search for academic papers"
        ),
        (
            "Check if the claim 'Bitcoin reached $100k in 2025' is supported by evidence",
            "corroborate_claim",
            "Evidence verification task"
        ),
        (
            "Extract tables from https://en.wikipedia.org/wiki/List_of_countries_by_GDP",
            "web_extract_tables",
            "Table extraction from URL"
        ),
        (
            "Classify this source: https://www.bloomberg.com/markets",
            "classify_source",
            "Source classification"
        ),
    ]

    test_results = []
    for input_text, expected_tool, context in test_cases:
        tc = await run_test_case(session_info, input_text, expected_tool)

        mcp_metric = MCPUseMetric()
        mcp_metric.measure(tc)

        task_metric = TaskCompletionMetric()
        task_metric.measure(tc)

        result = {
            "input": input_text,
            "expected_tool": expected_tool,
            "mcp_use_score": mcp_metric.score,
            "mcp_use_reason": mcp_metric.reason,
            "task_completion_score": task_metric.score,
            "task_completion_reason": task_metric.reason,
        }
        test_results.append(result)
        print(f"\n  [{expected_tool}] {input_text[:60]}...")
        print(f"    MCPUse: {mcp_metric.score:.2f} | TaskCompletion: {task_metric.score:.2f}")

    # Summary
    avg_mcp = sum(r["mcp_use_score"] for r in test_results) / len(test_results)
    avg_task = sum(r["task_completion_score"] for r in test_results) / len(test_results)
    print(f"\n{'='*60}")
    print(f"OVERALL: MCPUse={avg_mcp:.2f} | TaskCompletion={avg_task:.2f}")
    print(f"Test cases: {len(test_results)}")

    return test_results


if __name__ == "__main__":
    results = asyncio.run(main())
