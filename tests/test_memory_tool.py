from __future__ import annotations

import unittest
from unittest.mock import Mock

from local_opportunity_agent.llm import (
    EmbeddingResult,
    LLMError,
)
from local_opportunity_agent.memory.vector_store import SearchResult
from local_opportunity_agent.tools.memory import MemorySearchTool


class MemorySearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = Mock()
        self.vector_store = Mock()

        self.llm.embed.return_value = EmbeddingResult(
            embedding=[1.0, 0.0, 0.0],
            dimensions=3,
        )

        self.vector_store.search.return_value = [
            SearchResult(
                point_id="opportunity-1",
                score=0.95,
                payload={
                    "type": "opportunity",
                    "title": "AI lead follow-up",
                },
            ),
            SearchResult(
                point_id="opportunity-2",
                score=0.82,
                payload={
                    "type": "opportunity",
                    "title": "AI appointment booking",
                },
            ),
        ]

        self.tool = MemorySearchTool(
            llm=self.llm,
            vector_store=self.vector_store,
        )

    def test_search_returns_results(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI automation opportunities",
                "limit": 5,
            }
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.error)

        self.assertEqual(
            result.data["count"],
            2,
        )

        self.assertEqual(
            result.data["results"][0]["point_id"],
            "opportunity-1",
        )

        self.llm.embed.assert_called_once_with("AI automation opportunities")

        self.vector_store.search.assert_called_once_with(
            vector=[1.0, 0.0, 0.0],
            limit=5,
        )

    def test_default_limit_is_five(self) -> None:
        self.tool.execute(
            {
                "query": "AI opportunities",
            }
        )

        self.vector_store.search.assert_called_once_with(
            vector=[1.0, 0.0, 0.0],
            limit=5,
        )

    def test_empty_query_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "   ",
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "query must be a non-empty string",
        )

        self.llm.embed.assert_not_called()

    def test_non_integer_limit_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI opportunities",
                "limit": "5",
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "limit must be an integer",
        )

    def test_limit_out_of_range_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI opportunities",
                "limit": 21,
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "limit must be between 1 and 20",
        )

    def test_embedding_failure_is_returned_as_tool_error(self) -> None:
        self.llm.embed.side_effect = LLMError("Ollama unavailable")

        result = self.tool.execute(
            {
                "query": "AI opportunities",
            }
        )

        self.assertFalse(result.success)
        self.assertIn(
            "Memory search failed",
            result.error or "",
        )


if __name__ == "__main__":
    unittest.main()
