# graph.py
"""StateGraph assembly for the Content Moderation & QA System.

Wires together 6 nodes with conditional routing and a revision loop.
Two interrupt points pause for human moderator review.

LangGraph concepts demonstrated:
- StateGraph with interrupt() for human-in-the-loop
- Conditional edges for approve/edit/reject routing
- Revision loop (revise → draft_review cycle)
- RetryPolicy on LLM nodes for transient error handling
- Checkpointer injection (required for interrupts)

4-tier error handling:
- Tier 1: RetryPolicy on generate, revise, polish nodes
- Tier 3: interrupt() for user-fixable issues (moderator review)
- Tier 4: unexpected errors bubble up
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from models import ContentModerationState
from nodes import (
    generate_node,
    draft_review_node,
    revise_node,
    polish_node,
    final_review_node,
    publish_node,
    route_after_draft_review,
    route_after_final_review,
)

# RetryPolicy for LLM nodes — handles transient API errors
_llm_retry = RetryPolicy(max_attempts=3)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the content moderation StateGraph.

    Args:
        checkpointer: Required for interrupt/resume. Defaults to InMemorySaver
            if None is provided, since interrupts need a checkpointer.

    Returns:
        Compiled LangGraph graph ready for .invoke().
    """
    # Interrupts require a checkpointer — default to InMemorySaver
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = (
        StateGraph(ContentModerationState)
        # -- Nodes --
        # LLM nodes get RetryPolicy for transient error handling (Tier 1)
        .add_node("generate", generate_node, retry_policy=_llm_retry)
        .add_node("draft_review", draft_review_node)
        .add_node("revise", revise_node, retry_policy=_llm_retry)
        .add_node("polish", polish_node, retry_policy=_llm_retry)
        .add_node("final_review", final_review_node)
        .add_node("publish", publish_node)
        # -- Edges --
        # Linear flow: generate → draft_review
        .add_edge(START, "generate")
        .add_edge("generate", "draft_review")
        # Conditional: draft_review → polish | revise | END
        .add_conditional_edges(
            "draft_review",
            route_after_draft_review,
            ["polish", "revise", "__end__"],
        )
        # Revision loop: revise → back to draft_review
        .add_edge("revise", "draft_review")
        # Linear: polish → final_review
        .add_edge("polish", "final_review")
        # Conditional: final_review → publish | END
        .add_conditional_edges(
            "final_review",
            route_after_final_review,
            ["publish", "__end__"],
        )
        # Terminal
        .add_edge("publish", END)
        # Compile with checkpointer (required for interrupts)
        .compile(checkpointer=checkpointer)
    )

    return graph
