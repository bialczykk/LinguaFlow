"""StateGraph assembly for the Tutor Matching Agent.

Wires together the agent loop: agent_node ↔ tool_node with conditional
routing. Accepts an optional checkpointer for persistence.

LangGraph concepts demonstrated:
- StateGraph with MessagesState-based schema
- Agent loop pattern: LLM node → conditional edge → tool node → back
- ToolNode from langgraph.prebuilt for automatic tool dispatch
- Checkpointer injection at compile time (pluggable persistence)
- Conditional edges with routing functions
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END

from models import TutorMatchingState
from nodes import agent_node, should_continue, get_tool_node


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the Tutor Matching Agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    tool_node = get_tool_node()

    graph = (
        StateGraph(TutorMatchingState)
        .add_node("agent_node", agent_node)
        .add_node("tool_node", tool_node)
        .add_edge(START, "agent_node")
        .add_conditional_edges("agent_node", should_continue, ["tool_node", "__end__"])
        .add_edge("tool_node", "agent_node")
        .compile(checkpointer=checkpointer)
    )

    return graph
