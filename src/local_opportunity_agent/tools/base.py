from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by every agent tool."""

    tool_name: str
    success: bool
    data: dict[str, Any]
    error: str | None = None


class Tool(ABC):
    """Base interface for reusable agent tools."""

    name: str
    description: str

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute the tool with validated arguments."""
        raise NotImplementedError
