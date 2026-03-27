"""Graph assembly for the multi-department support system.

This module wires together all nodes, edges, and routing logic into a compiled
LangGraph StateGraph. The graph models a tiered support workflow:

  START -> supervisor_router
               |
               +-- needs clarification -> ask_clarification --[interrupt]-> supervisor_router
               |
               +-- single department  -> <dept>_agent -> supervisor_aggregator
               |
               +-- multi department   -> [Send(...) x N in parallel] -> supervisor_aggregator
                                                    |
               +--(has escalations)---+             |
               |                     |             v
               |            re-route via Send -> <dept>_agent -> supervisor_aggregator
               |
               +--(all resolved)--> compose_response -> END

Key LangGraph concepts demonstrated here:
- StateGraph with a TypedDict state schema
- Conditional edges with a routing function
- Send() for parallel fan-out to multiple department nodes
- interrupt/Command-based human-in-the-loop clarification
- Optional checkpointer injection for stateful (persistent) execution
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models import SupportState
from nodes import (
    supervisor_router,
    supervisor_aggregator,
    compose_response,
    ask_clarification,
    billing_agent,
    tech_support_agent,
    scheduling_agent,
    content_agent,
)

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_from_supervisor(state: SupportState):
    """Decide where to go after supervisor_router has classified the request.

    Three possible outcomes:
    1. Needs clarification → route to ask_clarification (which will interrupt
       and resume back to supervisor_router via Command).
    2. Single department → route directly to the one relevant agent node.
    3. Multiple departments → fan-out via Send() so all agents run in parallel.

    Returns either:
    - a string node name, OR
    - a list of Send objects for parallel dispatch
    """
    classification = state.get("classification", {})

    # --- Case 1: clarification needed ---
    if classification.get("needs_clarification"):
        return "ask_clarification"

    departments = classification.get("departments", [])

    # --- Case 2: single department ---
    # Map department name -> node name (e.g. "tech_support" -> "tech_support_agent")
    if len(departments) == 1:
        dept = departments[0]
        return f"{dept}_agent"

    # --- Case 3: multiple departments (or empty fallback) ---
    # Use Send to dispatch each department concurrently. Send copies the full
    # current state into each sub-invocation, so all agents get the same input.
    if departments:
        return [Send(f"{dept}_agent", state) for dept in departments]

    # Fallback if classification is empty/unexpected — ask for clarification
    return "ask_clarification"


def route_from_aggregator(state: SupportState):
    """Decide where to go after supervisor_aggregator has scanned the results.

    Two possible outcomes:
    1. Escalations pending → fan-out via Send() to re-route each escalation
       to the appropriate department agent for a second pass.
    2. No escalations → proceed to compose_response to build the final reply.

    Returns either:
    - the string "compose_response", OR
    - a list of Send objects targeting the escalation destinations
    """
    escalation_queue = state.get("escalation_queue", [])

    # --- Case 1: escalations to handle ---
    if escalation_queue:
        # Each escalation entry has a "target" field (department name). We Send
        # the full state so the agent can see the escalation_queue for context.
        return [Send(f"{e['target']}_agent", state) for e in escalation_queue]

    # --- Case 2: all resolved ---
    return "compose_response"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Assemble and compile the multi-department support StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver or
                      SqliteSaver). When provided, the graph gains persistence
                      and human-in-the-loop interrupt/resume support across
                      invocations (thread_id required in config).

    Returns:
        A compiled LangGraph (CompiledStateGraph) ready to invoke or stream.
    """
    # -------------------------------------------------------------------------
    # 1. Initialise the StateGraph with our shared state schema
    # -------------------------------------------------------------------------
    graph = StateGraph(SupportState)

    # -------------------------------------------------------------------------
    # 2. Register all nodes
    # -------------------------------------------------------------------------

    # Supervisor nodes — orchestration layer
    graph.add_node("supervisor_router", supervisor_router)
    graph.add_node("supervisor_aggregator", supervisor_aggregator)

    # Clarification node — uses interrupt/Command for human-in-the-loop.
    # The `ends` kwarg tells LangGraph that this node may dynamically route
    # to "supervisor_router" via a Command return value. Without this, the
    # graph builder wouldn't know about that edge and would raise a validation
    # error at compile time.
    graph.add_node("ask_clarification", ask_clarification, ends=["supervisor_router"])

    # Department sub-agent nodes — each handles one business domain
    graph.add_node("billing_agent", billing_agent)
    graph.add_node("tech_support_agent", tech_support_agent)
    graph.add_node("scheduling_agent", scheduling_agent)
    graph.add_node("content_agent", content_agent)

    # Final synthesis node — merges all department responses
    graph.add_node("compose_response", compose_response)

    # -------------------------------------------------------------------------
    # 3. Static edges
    # -------------------------------------------------------------------------

    # Entry point: always start with classification
    graph.add_edge(START, "supervisor_router")

    # Every department agent, after completing its work, feeds into the
    # aggregator. This is the same for both first-pass and escalation-pass runs.
    graph.add_edge("billing_agent", "supervisor_aggregator")
    graph.add_edge("tech_support_agent", "supervisor_aggregator")
    graph.add_edge("scheduling_agent", "supervisor_aggregator")
    graph.add_edge("content_agent", "supervisor_aggregator")

    # After composing the final response, the workflow is complete
    graph.add_edge("compose_response", END)

    # -------------------------------------------------------------------------
    # 4. Conditional edges — dynamic routing via routing functions
    # -------------------------------------------------------------------------

    # From supervisor_router: clarification, single dept, or parallel multi-dept
    # The path_map explicitly lists all possible string destinations so that
    # the graph builder can validate the topology at compile time.
    graph.add_conditional_edges(
        "supervisor_router",
        route_from_supervisor,
        path_map={
            "ask_clarification": "ask_clarification",
            "billing_agent": "billing_agent",
            "tech_support_agent": "tech_support_agent",
            "scheduling_agent": "scheduling_agent",
            "content_agent": "content_agent",
        },
    )

    # From supervisor_aggregator: escalation re-dispatch or final composition
    graph.add_conditional_edges(
        "supervisor_aggregator",
        route_from_aggregator,
        path_map={
            "compose_response": "compose_response",
            "billing_agent": "billing_agent",
            "tech_support_agent": "tech_support_agent",
            "scheduling_agent": "scheduling_agent",
            "content_agent": "content_agent",
        },
    )

    # -------------------------------------------------------------------------
    # 5. Compile — optionally inject a checkpointer for persistence & HITL
    # -------------------------------------------------------------------------
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick sanity check: compile the graph and print a summary.

    Run with:
        python graph.py

    For a real invocation (requires ANTHROPIC_API_KEY in .env):
        python -c "
        from graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver
        g = build_graph(checkpointer=MemorySaver())
        result = g.invoke(
            {
                'request': 'I cannot login and my invoice is wrong',
                'request_metadata': {'sender_type': 'student', 'student_id': 'S001', 'priority': 'high'},
            },
            config={'configurable': {'thread_id': 'test-1'}},
        )
        print(result.get('final_response'))
        "
    """
    from langgraph.checkpoint.memory import MemorySaver

    # Compile without a checkpointer first (stateless mode)
    stateless_graph = build_graph()
    print("Stateless graph compiled:", stateless_graph)

    # Compile with an in-memory checkpointer (stateful + HITL mode)
    stateful_graph = build_graph(checkpointer=MemorySaver())
    print("Stateful graph compiled: ", stateful_graph)

    # Print the graph's Mermaid diagram for visual inspection
    try:
        print("\nGraph structure (Mermaid):")
        print(stateless_graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"(Mermaid rendering skipped: {e})")
