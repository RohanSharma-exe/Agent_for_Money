from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from local_opportunity_agent.agents.supervisor import (
    Supervisor,
    SupervisorError,
)
from local_opportunity_agent.runtime.state import AgentState
from local_opportunity_agent.tools.memory import MemorySearchTool
from local_opportunity_agent.tools.web.research import ResearchTool


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


def research_node(
    state: AgentState,
    research_tool: ResearchTool,
) -> AgentState:
    """Execute the web-research tool."""
    if not state.tool_calls:
        state.error = "Research was requested without a tool call."
        return state

    tool_call = state.tool_calls[-1]
    query = tool_call.get("query")

    if not isinstance(query, str) or not query.strip():
        state.error = "Research requires a non-empty query."
        return state

    result = research_tool.execute(
        {
            "query": query,
            "max_results": 5,
        }
    )

    state.tool_results.append(
        {
            "tool": result.tool_name,
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }
    )

    if not result.success:
        state.error = result.error or "Research failed."
        return state

    state.evidence = result.data.get("results", [])

    return state


def memory_search_node(
    state: AgentState,
    memory_search_tool: MemorySearchTool,
) -> AgentState:
    """Execute the semantic memory-search tool."""
    if not state.tool_calls:
        state.error = "Memory search was requested without a tool call."
        return state

    tool_call = state.tool_calls[-1]
    query = tool_call.get("query")

    if not isinstance(query, str) or not query.strip():
        state.error = "Memory search requires a non-empty query."
        return state

    result = memory_search_tool.execute(
        {
            "query": query,
            "limit": 5,
        }
    )

    state.tool_results.append(
        {
            "tool": result.tool_name,
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }
    )

    if not result.success:
        state.error = result.error or "Memory search failed."
        return state

    state.memories = result.data.get("results", [])

    return state


def finalize(
    state: AgentState,
) -> AgentState:
    """Finalize graph execution."""
    if state.error:
        state.final_answer = f"Unable to process request: {state.error}"
        return state

    if state.next_action == "research":
        if state.evidence:
            state.final_answer = f"Research completed and found {len(state.evidence)} result(s)."
        else:
            state.final_answer = "Research completed but found no results."
    elif state.next_action == "memory_search":
        if state.memories:
            state.final_answer = (
                f"Memory search found {len(state.memories)} relevant stored result(s)."
            )
        else:
            state.final_answer = "Memory search completed but found no relevant stored results."
    else:
        state.final_answer = "The supervisor selected a direct answer."

    return state


def build_graph(
    supervisor: Supervisor,
    memory_search_tool: MemorySearchTool,
    research_tool: ResearchTool,
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
        lambda state: research_node(
            state,
            research_tool,
        ),
    )

    graph.add_node(
        "memory_search",
        lambda state: memory_search_node(
            state,
            memory_search_tool,
        ),
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
