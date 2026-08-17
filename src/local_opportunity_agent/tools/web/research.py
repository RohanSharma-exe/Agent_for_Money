from __future__ import annotations

from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from local_opportunity_agent.tools.base import Tool, ToolResult


class ResearchTool(Tool):
    """Perform current web research and return sourced search results."""

    name = "research"

    description = "Search the current web for information and return sourced search results."

    def __init__(self, max_results: int = 5) -> None:
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError("max_results must be an integer")

        if max_results < 1 or max_results > 10:
            raise ValueError("max_results must be between 1 and 10")

        self.max_results = max_results

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="query must be a non-empty string",
            )

        max_results = arguments.get("max_results", self.max_results)

        if isinstance(max_results, bool) or not isinstance(max_results, int):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="max_results must be an integer",
            )

        if max_results < 1 or max_results > 10:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="max_results must be between 1 and 10",
            )

        try:
            results = DDGS().text(
                query,
                max_results=max_results,
            )
        except (DDGSException, RatelimitException, TimeoutException) as error:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Research failed: {error}",
            )

        normalized_results = [
            {
                "title": str(result.get("title", "")),
                "url": str(result.get("href", "")),
                "snippet": str(result.get("body", "")),
            }
            for result in results
        ]

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "query": query,
                "count": len(normalized_results),
                "results": normalized_results,
            },
        )
