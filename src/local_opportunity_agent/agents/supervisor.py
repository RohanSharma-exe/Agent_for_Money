from __future__ import annotations

import json

from local_opportunity_agent.llm import OllamaService
from local_opportunity_agent.runtime.routing import (
    RouteDecision,
)


class SupervisorError(RuntimeError):
    """Raised when the supervisor cannot produce a valid decision."""


SYSTEM_PROMPT = """
You are the supervisor of a local AI opportunity research agent.

The user's goal is to discover realistic opportunities to make money.

Choose exactly one action:

research
- Use when the request requires new external information.

memory_search
- Use when the answer may already exist in stored research or memory.

answer
- Use when the request can be answered directly without research or memory.

Return ONLY valid JSON:

{
  "action": "research | memory_search | answer",
  "query": "short useful query",
  "reason": "brief explanation"
}

Rules:
- Never invent another action.
- Keep query concise.
- Do not answer the user yourself.
- Do not include markdown.
"""


class Supervisor:
    """LLM-powered router for the opportunity agent."""

    def __init__(self, llm: OllamaService) -> None:
        self.llm = llm

    def decide(self, user_request: str) -> RouteDecision:
        if not user_request.strip():
            raise SupervisorError("User request cannot be empty.")

        response = self.llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_request,
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )

        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as error:
            raise SupervisorError("Supervisor returned invalid JSON.") from error

        action = data.get("action")
        query = data.get("query")
        reason = data.get("reason")

        if action not in {
            "research",
            "memory_search",
            "answer",
        }:
            raise SupervisorError(f"Invalid supervisor action: {action!r}")

        if not isinstance(query, str) or not query.strip():
            raise SupervisorError("Supervisor query must be a non-empty string.")

        if not isinstance(reason, str) or not reason.strip():
            raise SupervisorError("Supervisor reason must be a non-empty string.")

        return RouteDecision(
            action=action,
            query=query.strip(),
            reason=reason.strip(),
        )
