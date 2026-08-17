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

    def test_graph_builds(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"answer","query":"test","reason":"direct answer"}'
        )

        graph = build_graph(supervisor)

        self.assertIsNotNone(graph)

    def test_graph_processes_direct_answer(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"answer","query":"Python income","reason":"Can answer directly"}'
        )

        graph = build_graph(supervisor)

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

    def test_graph_routes_research(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"research","query":"AI opportunities for dentists","reason":"Requires current information"}'
        )

        graph = build_graph(supervisor)

        result = graph.invoke(AgentState(user_request="Find AI opportunities for dentists."))

        self.assertEqual(
            result["next_action"],
            "research",
        )

        self.assertEqual(
            result["tool_results"][0]["tool"],
            "research",
        )

        self.assertEqual(
            result["tool_results"][0]["status"],
            "not_implemented",
        )

    def test_graph_routes_memory_search(self) -> None:
        supervisor = self.make_supervisor(
            '{"action":"memory_search","query":"previous opportunities","reason":"Information may already exist"}'
        )

        graph = build_graph(supervisor)

        result = graph.invoke(AgentState(user_request="What opportunities have we already found?"))

        self.assertEqual(
            result["next_action"],
            "memory_search",
        )

        self.assertEqual(
            result["tool_results"][0]["tool"],
            "memory_search",
        )

    def test_empty_request_does_not_call_llm(self) -> None:
        supervisor = self.make_supervisor('{"action":"answer","query":"unused","reason":"unused"}')

        graph = build_graph(supervisor)

        result = graph.invoke(AgentState(user_request="   "))

        self.assertEqual(
            result["error"],
            "User request cannot be empty.",
        )

    def test_invalid_supervisor_json_becomes_graph_error(
        self,
    ) -> None:
        supervisor = self.make_supervisor("not valid json")

        graph = build_graph(supervisor)

        result = graph.invoke(AgentState(user_request="Find opportunities."))

        self.assertEqual(
            result["error"],
            "Supervisor returned invalid JSON.",
        )


if __name__ == "__main__":
    unittest.main()
