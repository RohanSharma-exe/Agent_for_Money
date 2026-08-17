from __future__ import annotations

from local_opportunity_agent.tools.base import (
    Tool,
    ToolResult,
)


class ToolRegistryError(RuntimeError):
    """Raised when a tool registry operation fails."""


class ToolRegistry:
    """Registry for reusable agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name.strip():
            raise ToolRegistryError("Tool name cannot be empty.")

        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolRegistryError(f"Unknown tool: {name}") from error

    def execute(
        self,
        name: str,
        arguments: dict,
    ) -> ToolResult:
        return self.get(name).execute(arguments)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptions(self) -> dict[str, str]:
        return {name: tool.description for name, tool in sorted(self._tools.items())}
