from __future__ import annotations

import unittest

from local_opportunity_agent.runtime.graph import build_graph
from local_opportunity_agent.runtime.state import AgentState


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph()

    def test_graph_builds(self) -> None:
        self.assertIsNotNone(self.graph)

    def test_graph_processes_request(self) -> None:
        state = AgentState(user_request="Find profitable AI opportunities.")

        result = self.graph.invoke(state)

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertEqual(
            result["next_action"],
            "finish",
        )

        self.assertIsNotNone(
            result["final_answer"],
        )

    def test_empty_request_returns_error(self) -> None:
        state = AgentState(user_request="   ")

        result = self.graph.invoke(state)

        self.assertIsNotNone(
            result["error"],
        )

        self.assertIsNotNone(
            result["final_answer"],
        )


if __name__ == "__main__":
    unittest.main()
