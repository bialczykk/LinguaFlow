"""LangGraph StateGraph definition for the Lesson Plan Generator.

This module wires together the node functions into a directed graph with
conditional routing and a review loop.

LangGraph concepts demonstrated:
- StateGraph: creating a graph with a typed state schema
- add_node(): registering node functions
- add_edge(): static edges (always go to the next node)
- add_conditional_edges(): dynamic routing based on state
- Graph cycles: the review → draft loop (with a max revision guard)
- compile(): producing an executable graph
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from models import LessonPlanState
from nodes import (
    research_node,
    draft_conversation_node,
    draft_grammar_node,
    draft_exam_prep_node,
    review_node,
    finalize_node,
)


def route_by_lesson_type(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep"
]:
    """Conditional edge: route to the correct drafting node based on lesson type.

    LangGraph concept: conditional routing
    This function is passed to add_conditional_edges(). LangGraph calls it
    after the research node completes, and uses the return value to decide
    which drafting node to execute next.
    """
    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"


def route_after_review(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep", "finalize"
]:
    """Conditional edge: decide whether to finalize or loop back for revision.

    LangGraph concept: graph cycles (loops)
    If the review approved the draft, we proceed to finalize.
    If not, and we haven't hit the max revision count (2), we loop back
    to the same drafting node so it can incorporate the feedback.
    If we've exhausted revisions, we finalize the best-effort draft.
    """
    if state["is_approved"]:
        return "finalize"

    if state["revision_count"] >= 2:
        return "finalize"

    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"


def build_graph():
    """Build and compile the Lesson Plan Generator StateGraph.

    Graph topology:
        START → research → route_by_type → draft_* → review → finalize → END
                                              ↑                  │
                                              └──────────────────┘
                                           (if not approved & revisions < 2)

    Returns:
        A compiled LangGraph graph ready for .invoke() or .stream().
    """
    workflow = StateGraph(LessonPlanState)

    # Add nodes — each node is a Python function that receives state and returns
    # a dict of state updates. LangGraph merges those updates into the shared state.
    workflow.add_node("research", research_node)
    workflow.add_node("draft_conversation", draft_conversation_node)
    workflow.add_node("draft_grammar", draft_grammar_node)
    workflow.add_node("draft_exam_prep", draft_exam_prep_node)
    workflow.add_node("review", review_node)
    workflow.add_node("finalize", finalize_node)

    # START always goes to research — every lesson plan starts with research
    workflow.add_edge(START, "research")

    # After research, route to the correct drafting node based on lesson type.
    # add_conditional_edges() takes: source node, routing function, and a mapping
    # of routing function return values → destination node names.
    workflow.add_conditional_edges(
        "research",
        route_by_lesson_type,
        {
            "draft_conversation": "draft_conversation",
            "draft_grammar": "draft_grammar",
            "draft_exam_prep": "draft_exam_prep",
        },
    )

    # All drafting nodes converge to review — static edges, no branching here
    workflow.add_edge("draft_conversation", "review")
    workflow.add_edge("draft_grammar", "review")
    workflow.add_edge("draft_exam_prep", "review")

    # After review, either finalize or loop back for revision.
    # This is the graph cycle: review can send execution back to a drafting node,
    # creating a feedback loop until the plan is approved or revisions are exhausted.
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "draft_conversation": "draft_conversation",
            "draft_grammar": "draft_grammar",
            "draft_exam_prep": "draft_exam_prep",
            "finalize": "finalize",
        },
    )

    # Finalize goes to END — the graph terminates after producing the final plan
    workflow.add_edge("finalize", END)

    # compile() validates the graph topology (no orphan nodes, all edges connected)
    # and returns an executable CompiledGraph object.
    return workflow.compile()
