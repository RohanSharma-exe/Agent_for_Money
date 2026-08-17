from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """State shared by nodes in the opportunity-agent graph."""

    user_request: str

    messages: list[dict[str, Any]] = field(default_factory=list)

    next_action: str | None = None

    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    tool_results: list[dict[str, Any]] = field(default_factory=list)

    memories: list[dict[str, Any]] = field(default_factory=list)

    evidence: list[dict[str, Any]] = field(default_factory=list)

    final_answer: str | None = None

    error: str | None = None
