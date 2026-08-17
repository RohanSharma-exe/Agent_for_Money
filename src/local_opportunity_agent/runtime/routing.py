from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RouteAction = Literal[
    "research",
    "memory_search",
    "answer",
]


@dataclass(frozen=True)
class RouteDecision:
    """Validated decision produced by the supervisor."""

    action: RouteAction
    query: str
    reason: str
