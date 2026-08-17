from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from local_opportunity_agent.runtime.state import AgentState


def supervisor(
    state: AgentState,
) -> AgentState:
    """Initial supervisor node.

    Phase 7 starts with a deterministic supervisor.
    LLM-driven routing will be added after the graph
    lifecycle is verified.
    """
    if not state.user_request.strip():
        state.error = "User request cannot be empty."
        return state

    state.next_action = "finish"

    return state


def finalize(
    state: AgentState,
) -> AgentState:
    """Finalize the current graph execution."""
    if state.error:
        state.final_answer = f"Unable to process request: {state.error}"
    else:
        state.final_answer = (
            "The opportunity research runtime is ready. "
            "Research tools will be connected in later phases."
        )

    return state


def build_graph():
    """Build the initial single-agent LangGraph."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "supervisor",
        supervisor,
    )

    graph.add_node(
        "finalize",
        finalize,
    )

    graph.add_edge(
        START,
        "supervisor",
    )

    graph.add_edge(
        "supervisor",
        "finalize",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    return graph.compile()
