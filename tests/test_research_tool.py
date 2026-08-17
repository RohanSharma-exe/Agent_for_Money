from __future__ import annotations

import unittest
from unittest.mock import patch

from ddgs.exceptions import DDGSException

from local_opportunity_agent.tools.web.research import ResearchTool


class ResearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ResearchTool()

    def test_empty_query_is_rejected(self) -> None:
        result = self.tool.execute({"query": "   "})

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "query must be a non-empty string",
        )

    def test_non_string_query_is_rejected(self) -> None:
        result = self.tool.execute({"query": 123})

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "query must be a non-empty string",
        )

    @patch("local_opportunity_agent.tools.web.research.DDGS")
    def test_default_max_results_is_five(self, mock_ddgs) -> None:
        mock_ddgs.return_value.text.return_value = []

        result = self.tool.execute({"query": "AI automation"})

        self.assertTrue(result.success)
        mock_ddgs.return_value.text.assert_called_once_with(
            "AI automation",
            max_results=5,
        )

    def test_non_integer_max_results_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI automation",
                "max_results": "5",
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "max_results must be an integer",
        )

    def test_bool_max_results_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI automation",
                "max_results": True,
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "max_results must be an integer",
        )

    def test_max_results_below_one_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI automation",
                "max_results": 0,
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "max_results must be between 1 and 10",
        )

    def test_max_results_above_ten_is_rejected(self) -> None:
        result = self.tool.execute(
            {
                "query": "AI automation",
                "max_results": 11,
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.error,
            "max_results must be between 1 and 10",
        )

    @patch("local_opportunity_agent.tools.web.research.DDGS")
    def test_successful_search_returns_normalized_results(self, mock_ddgs) -> None:
        mock_ddgs.return_value.text.return_value = [
            {
                "title": "AI Automation Guide",
                "href": "https://example.com/ai",
                "body": "A guide to AI automation.",
            },
            {
                "title": "Local Business AI",
                "href": "https://example.com/business",
                "body": "AI opportunities for local businesses.",
            },
        ]

        result = self.tool.execute(
            {
                "query": "AI automation for local businesses",
                "max_results": 2,
            }
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.error)

        self.assertEqual(
            result.data["query"],
            "AI automation for local businesses",
        )
        self.assertEqual(
            result.data["count"],
            2,
        )
        self.assertEqual(
            result.data["results"],
            [
                {
                    "title": "AI Automation Guide",
                    "url": "https://example.com/ai",
                    "snippet": "A guide to AI automation.",
                },
                {
                    "title": "Local Business AI",
                    "url": "https://example.com/business",
                    "snippet": "AI opportunities for local businesses.",
                },
            ],
        )

        mock_ddgs.return_value.text.assert_called_once_with(
            "AI automation for local businesses",
            max_results=2,
        )

    @patch("local_opportunity_agent.tools.web.research.DDGS")
    def test_search_failure_returns_tool_error(self, mock_ddgs) -> None:
        mock_ddgs.return_value.text.side_effect = DDGSException("search failed")

        result = self.tool.execute(
            {
                "query": "AI automation",
            }
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.tool_name,
            "research",
        )
        self.assertIn(
            "Research failed: search failed",
            result.error or "",
        )


if __name__ == "__main__":
    unittest.main()
