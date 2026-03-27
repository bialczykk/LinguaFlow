"""Graph assembly for the autonomous operations orchestrator — Project 08.

This module wires together all nodes, edges, and routing logic into a compiled
LangGraph StateGraph. The graph models a fully autonomous multi-department
orchestration workflow with risk assessment, optional human approval, parallel
department dispatch, and autonomous follow-up cascading.

Graph flow:
  START -> request_classifier -> risk_assessor
                                      |
                    (low risk) -------+------- (high risk)
                         |                          |
                 dispatch_departments         approval_gate [interrupt]
                         |                     /          \\
                  [Send x N]          approved             rejected
                         |               |                    |
               department_executor  dispatch_departments   compose_output
               (parallel branches)      |
                         |         [Send x N]
                         |      department_executor
                         |         (parallel branches)
                    result_aggregator
                         |
                   check_task_queue
                    /            \\
          (tasks pending)     (queue empty)
                 |                  |
       request_classifier       compose_output
                                    |
                           reporting_snapshot -> END

Key LangGraph concepts demonstrated here:
- StateGraph with a TypedDict state schema using operator.add for parallel merging
- Conditional edges with a routing function returning either a string or Send list
- Send() for parallel fan-out to multiple department_executor nodes
- interrupt/Command-based human-in-the-loop approval gate
- Command with `goto` for dynamic routing from nodes (approval_gate, check_task_queue)
- Optional checkpointer injection for stateful execution with SqliteSaver or MemorySaver
- `ends=` parameter to declare Command-based dynamic routing targets at node registration
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models import OrchestratorState
from nodes import (
    request_classifier,
    risk_assessor,
    approval_gate,
    department_executor,
    result_aggregator,
    check_task_queue,
    compose_output,
    reporting_snapshot,
)

# ---------------------------------------------------------------------------
# Pass-through node for dispatch
# ---------------------------------------------------------------------------

def _dispatch_pass_through(state: OrchestratorState) -> dict:
    """Pass-through node — routing to parallel department_executor happens via
    the fan_out_to_departments conditional edge defined below.

    This node exists because approval_gate routes here via Command(goto=...).
    LangGraph requires Command targets to be registered nodes; we cannot route
    a Command directly to a conditional edge function. So we introduce this
    no-op node as the stable landing target, then use a conditional edge off
    it to fan out to parallel department_executor branches via Send.
    """
    return {}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_from_risk(state: OrchestratorState):
    """Decide where to go after risk_assessor has evaluated the classification.

    Two possible outcomes:
    1. High risk → route to approval_gate so a human operator can approve or
       reject the action before any department agent is invoked.
    2. Low risk  → route to dispatch_departments (pass-through node) which will
       fan out to parallel department_executor nodes via Send.

    Returns either:
    - the string "approval_gate", OR
    - the string "dispatch_departments"
    """
    risk_level = state.get("risk_level", "low")

    if risk_level == "high":
        return "approval_gate"

    # Low risk — proceed directly to the dispatch pass-through node,
    # which will immediately fan out to department_executor nodes via
    # the fan_out_to_departments conditional edge.
    return "dispatch_departments"


def fan_out_to_departments(state: OrchestratorState):
    """Fan out to parallel department_executor nodes via Send.

    Called as a conditional edge function from dispatch_departments.
    Reads the classification's department list and creates one Send per
    department, injecting `_target_dept` into each branch's state so that
    department_executor knows which agent to instantiate.

    Returns:
    - A list of Send objects — one per department in the classification.
    - Falls back to an empty list (unreachable in practice) if no departments.
    """
    classification = state.get("classification", {})
    departments = classification.get("departments", [])

    if not departments:
        # Fallback: should not happen in a well-classified request, but we
        # guard against it to avoid a crash. Route to compose_output via a
        # string so the graph can still terminate gracefully.
        return "compose_output"

    # Each Send copies the full current state into a fresh department branch,
    # adding _target_dept so department_executor knows which agent to run.
    return [
        Send("department_executor", {**state, "_target_dept": dept})
        for dept in departments
    ]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Assemble and compile the autonomous operations orchestrator StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. InMemorySaver or
                      SqliteSaver). When provided, the graph gains:
                      - State persistence across invocations (thread_id required).
                      - Human-in-the-loop interrupt/resume for approval_gate.
                      - Metrics accumulation across requests (metrics_store).

    Returns:
        A compiled LangGraph (CompiledStateGraph) ready to invoke or stream.
    """
    # -------------------------------------------------------------------------
    # 1. Initialise the StateGraph with our shared state schema
    # -------------------------------------------------------------------------
    graph = StateGraph(OrchestratorState)

    # -------------------------------------------------------------------------
    # 2. Register all nodes
    # -------------------------------------------------------------------------

    # Entry point — classifies the request and selects target departments
    graph.add_node("request_classifier", request_classifier)

    # Risk evaluation — pure logic, determines if human approval is needed
    graph.add_node("risk_assessor", risk_assessor)

    # Human-in-the-loop interrupt — uses Command to route after operator response.
    # `ends` tells LangGraph that this node can dynamically route to either
    # "dispatch_departments" or "compose_output" via Command(goto=...).
    # Without `ends`, the graph builder would raise a validation error because
    # those edges don't appear in any static add_edge() call.
    graph.add_node(
        "approval_gate",
        approval_gate,
        ends=["dispatch_departments", "compose_output"],
    )

    # Pass-through node that acts as a stable Command target for approval_gate.
    # The real fan-out to department_executor happens via the conditional edge
    # fan_out_to_departments defined below.
    graph.add_node("dispatch_departments", _dispatch_pass_through)

    # Parallel executor — one instance runs per department, dispatched by Send
    graph.add_node("department_executor", department_executor)

    # Aggregator — collects follow-up tasks from department results
    graph.add_node("result_aggregator", result_aggregator)

    # Queue checker — autonomous follow-up loop or final composition.
    # `ends` declares the two possible Command(goto=...) destinations.
    graph.add_node(
        "check_task_queue",
        check_task_queue,
        ends=["request_classifier", "compose_output"],
    )

    # Final response composition — merges all department results into one reply
    graph.add_node("compose_output", compose_output)

    # Reporting — updates cumulative platform metrics after each request
    graph.add_node("reporting_snapshot", reporting_snapshot)

    # -------------------------------------------------------------------------
    # 3. Static edges — deterministic transitions
    # -------------------------------------------------------------------------

    # Entry point: every invocation starts with classification
    graph.add_edge(START, "request_classifier")

    # Classification always leads to risk assessment
    graph.add_edge("request_classifier", "risk_assessor")

    # Every parallel department branch feeds into the aggregator.
    # LangGraph's operator.add reducer on department_results ensures that
    # results from all parallel branches are merged before aggregation runs.
    graph.add_edge("department_executor", "result_aggregator")

    # After aggregating results, check whether autonomous follow-ups remain
    graph.add_edge("result_aggregator", "check_task_queue")

    # After composing the final response, capture a reporting snapshot
    graph.add_edge("compose_output", "reporting_snapshot")

    # Reporting is the terminal node — graph ends here
    graph.add_edge("reporting_snapshot", END)

    # -------------------------------------------------------------------------
    # 4. Conditional edges — dynamic routing via routing functions
    # -------------------------------------------------------------------------

    # From risk_assessor: low risk → dispatch_departments, high risk → approval_gate
    # The path_map enumerates all possible string destinations so the graph
    # builder can validate the topology without executing the function.
    graph.add_conditional_edges(
        "risk_assessor",
        route_from_risk,
        path_map={
            "approval_gate": "approval_gate",
            "dispatch_departments": "dispatch_departments",
        },
    )

    # From dispatch_departments (pass-through): fan out to department_executor
    # via Send. The path_map must include "department_executor" to validate the
    # Send targets, and also "compose_output" for the empty-departments fallback.
    graph.add_conditional_edges(
        "dispatch_departments",
        fan_out_to_departments,
        path_map={
            "department_executor": "department_executor",
            "compose_output": "compose_output",
        },
    )

    # Note: approval_gate and check_task_queue use Command(goto=...) for routing,
    # so their edges are declared via `ends=` at node registration — no
    # add_conditional_edges() calls needed for them.

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

    For a real invocation with approval (requires ANTHROPIC_API_KEY in .env):
        python -c "
        from graph import build_graph
        from langgraph.checkpoint.memory import InMemorySaver
        g = build_graph(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'test-1'}}
        # First invoke — will interrupt at approval_gate for high-risk actions
        result = g.invoke(
            {
                'request': 'Onboard 50 new students and assign tutors',
                'request_metadata': {'user_id': 'admin', 'priority': 'high', 'source': 'api'},
                'department_results': [],
                'task_queue': [],
                'completed_tasks': [],
                'metrics_store': {},
            },
            config=config,
        )
        print(result)
        "
    """
    from langgraph.checkpoint.memory import InMemorySaver

    # Compile without a checkpointer first (stateless mode — no HITL support)
    stateless_graph = build_graph()
    print("Stateless graph compiled:", stateless_graph)

    # Compile with an in-memory checkpointer (stateful + HITL mode)
    stateful_graph = build_graph(checkpointer=InMemorySaver())
    print("Stateful graph compiled: ", stateful_graph)

    # Print the graph's Mermaid diagram for visual inspection
    try:
        print("\nGraph structure (Mermaid):")
        print(stateless_graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"(Mermaid rendering skipped: {e})")
