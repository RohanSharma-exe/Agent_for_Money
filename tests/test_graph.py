from __future__ import annotations

import unittest
from unittest.mock import Mock

from local_opportunity_agent.agents.supervisor import (
    Supervisor,
)
from local_opportunity_agent.llm import ChatResult
from local_opportunity_agent.runtime.graph import (
    build_graph,
)
from local_opportunity_agent.runtime.state import AgentState
from local_opportunity_agent.tools.base import ToolResult


class GraphTests(unittest.TestCase):
    def make_supervisor(
        self,
        response: str,
    ) -> Supervisor:
        llm = Mock()

        llm.chat.return_value = ChatResult(
            content=response,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        return Supervisor(llm)

    def make_memory_tool(
        self,
        result: ToolResult | None = None,
    ) -> Mock:
        tool = Mock()

        if result is None:
            result = ToolResult(
                tool_name="memory_search",
                success=True,
                data={
                    "query": "previous opportunities",
                    "count": 2,
                    "results": [
                        {
                            "point_id": "opportunity-1",
                            "score": 0.95,
                            "payload": {
                                "title": "AI lead follow-up",
                            },
                        },
                        {
                            "point_id": "opportunity-2",
                            "score": 0.88,
                            "payload": {
                                "title": "AI appointment booking",
                            },
                        },
                    ],
                },
            )

        tool.execute.return_value = result

        return tool

    def make_research_tool(
        self,
        result: ToolResult | None = None,
    ) -> Mock:
        tool = Mock()

        if result is None:
            result = ToolResult(
                tool_name="research",
                success=True,
                data={
                    "query": "AI opportunities for dentists",
                    "count": 2,
                    "results": [
                        {
                            "title": "AI appointment scheduling trends",
                            "url": "https://example.com/appointment-scheduling",
                            "snippet": "Dental practices are adopting automated scheduling.",
                        },
                        {
                            "title": "AI lead follow-up for local businesses",
                            "url": "https://example.com/lead-follow-up",
                            "snippet": "Automated lead follow-up can reduce missed opportunities.",
                        },
                    ],
                },
            )

        tool.execute.return_value = result

        return tool

    def test_graph_builds(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"answer","query":"test","reason":"direct answer"}'
        )
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        self.assertIsNotNone(graph)

    def test_graph_processes_direct_answer(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"answer","query":"Python income","reason":"Can answer directly"}'
        )
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="How can I make money with Python?"))

        self.assertEqual(
            result["next_action"],
            "answer",
        )

        self.assertEqual(
            result["tool_calls"][0]["action"],
            "answer",
        )

        self.assertEqual(
            result["final_answer"],
            "The supervisor selected a direct answer.",
        )

        memory_tool.execute.assert_not_called()

    def test_graph_routes_research(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"research","query":"AI opportunities for dentists","reason":"Requires current information"}'
        )
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="Find AI opportunities for dentists."))

        self.assertEqual(
            result["next_action"],
            "research",
        )

        self.assertEqual(
            result["tool_results"][0]["tool"],
            "research",
        )

        self.assertTrue(
            result["tool_results"][0]["success"],
        )

        self.assertEqual(
            len(result["evidence"]),
            2,
        )

        self.assertEqual(
            result["evidence"][0]["title"],
            "AI appointment scheduling trends",
        )

        research_tool.execute.assert_called_once_with(
            {
                "query": "AI opportunities for dentists",
                "max_results": 5,
            }
        )

        memory_tool.execute.assert_not_called()

        self.assertEqual(
            result["final_answer"],
            "Research completed and found 2 result(s).",
        )

    def test_graph_routes_memory_search(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"memory_search","query":"previous opportunities","reason":"Information may already exist"}'
        )
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="What opportunities have we already found?"))

        self.assertEqual(
            result["next_action"],
            "memory_search",
        )

        self.assertEqual(
            result["tool_results"][0]["tool"],
            "memory_search",
        )

        self.assertTrue(result["tool_results"][0]["success"])

        self.assertEqual(
            len(result["memories"]),
            2,
        )

        self.assertEqual(
            result["memories"][0]["point_id"],
            "opportunity-1",
        )

        memory_tool.execute.assert_called_once_with(
            {
                "query": "previous opportunities",
                "limit": 5,
            }
        )

        self.assertEqual(
            result["final_answer"],
            "Memory search found 2 relevant stored result(s).",
        )

    def test_memory_search_failure_becomes_graph_error(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"memory_search","query":"previous opportunities","reason":"Information may already exist"}'
        )

        memory_tool = self.make_memory_tool(
            ToolResult(
                tool_name="memory_search",
                success=False,
                data={},
                error="Memory search failed: Qdrant unavailable",
            )
        )
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="What opportunities have we already found?"))

        self.assertEqual(
            result["error"],
            "Memory search failed: Qdrant unavailable",
        )

        self.assertEqual(
            result["final_answer"],
            "Unable to process request: Memory search failed: Qdrant unavailable",
        )

    def test_empty_request_does_not_call_llm(self) -> None:
        supervisor = self.make_supervisor('{"action":"answer","query":"unused","reason":"unused"}')
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="   "))

        self.assertEqual(
            result["error"],
            "User request cannot be empty.",
        )

    def test_invalid_supervisor_json_becomes_graph_error(
        self,
    ) -> None:
        supervisor = self.make_supervisor("not valid json")
        memory_tool = self.make_memory_tool()
        research_tool = self.make_research_tool()

        graph = build_graph(
            supervisor,
            memory_tool,
            research_tool,
        )

        result = graph.invoke(AgentState(user_request="Find opportunities."))

        self.assertEqual(
            result["error"],
            "Supervisor returned invalid JSON.",
        )


if __name__ == "__main__":
    unittest.main()
