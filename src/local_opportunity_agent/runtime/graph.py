from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from local_opportunity_agent.agents.supervisor import (
    Supervisor,
    SupervisorError,
)
from local_opportunity_agent.runtime.state import AgentState


def supervisor_node(
    state: AgentState,
    supervisor: Supervisor,
) -> AgentState:
    """Ask the supervisor to choose the next action."""
    if not state.user_request.strip():
        state.error = "User request cannot be empty."
        return state

    try:
        decision = supervisor.decide(state.user_request)
    except SupervisorError as error:
        state.error = str(error)
        return state

    state.next_action = decision.action

    state.tool_calls.append(
        {
            "action": decision.action,
            "query": decision.query,
            "reason": decision.reason,
        }
    )

    return state


def route_after_supervisor(
    state: AgentState,
) -> str:
    """Choose the next graph node from the supervisor decision."""
    if state.error:
        return "finalize"

    if state.next_action == "research":
        return "research"

    if state.next_action == "memory_search":
        return "memory_search"

    return "finalize"


def research_placeholder(
    state: AgentState,
) -> AgentState:
    """Placeholder for the future web-research tool."""
    state.tool_results.append(
        {
            "tool": "research",
            "status": "not_implemented",
        }
    )

    return state


def memory_search_placeholder(
    state: AgentState,
) -> AgentState:
    """Placeholder for the future memory-search tool."""
    state.tool_results.append(
        {
            "tool": "memory_search",
            "status": "not_implemented",
        }
    )

    return state


def finalize(
    state: AgentState,
) -> AgentState:
    """Finalize graph execution."""
    if state.error:
        state.final_answer = f"Unable to process request: {state.error}"
        return state

    if state.next_action == "research":
        state.final_answer = (
            "The supervisor selected research. The research tool will be connected next."
        )
    elif state.next_action == "memory_search":
        state.final_answer = (
            "The supervisor selected memory search. The memory tool will be connected next."
        )
    else:
        state.final_answer = "The supervisor selected a direct answer."

    return state


def build_graph(
    supervisor: Supervisor,
):
    """Build the single-agent opportunity research graph."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "supervisor",
        lambda state: supervisor_node(
            state,
            supervisor,
        ),
    )

    graph.add_node(
        "research",
        research_placeholder,
    )

    graph.add_node(
        "memory_search",
        memory_search_placeholder,
    )

    graph.add_node(
        "finalize",
        finalize,
    )

    graph.add_edge(
        START,
        "supervisor",
    )

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research": "research",
            "memory_search": "memory_search",
            "finalize": "finalize",
        },
    )

    graph.add_edge(
        "research",
        "finalize",
    )

    graph.add_edge(
        "memory_search",
        "finalize",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()
