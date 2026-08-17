from __future__ import annotations

from typing import Any

from local_opportunity_agent.llm import (
    LLMError,
    OllamaService,
)
from local_opportunity_agent.memory.vector_store import (
    VectorStore,
    VectorStoreError,
)
from local_opportunity_agent.tools.base import Tool, ToolResult


class MemorySearchTool(Tool):
    """Search persistent semantic memory using Qdrant."""

    name = "memory_search"

    description = (
        "Search previously stored opportunity research and memories using semantic similarity."
    )

    def __init__(
        self,
        *,
        llm: OllamaService,
        vector_store: VectorStore,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store

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

        limit = arguments.get("limit", 5)

        if isinstance(limit, bool) or not isinstance(limit, int):
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="limit must be an integer",
            )

        if limit < 1 or limit > 20:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error="limit must be between 1 and 20",
            )

        try:
            embedding = self.llm.embed(query)

            results = self.vector_store.search(
                vector=embedding.embedding,
                limit=limit,
            )
        except (LLMError, VectorStoreError) as error:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error=f"Memory search failed: {error}",
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "point_id": result.point_id,
                        "score": result.score,
                        "payload": result.payload,
                    }
                    for result in results
                ],
            },
        )
