# graph.py
"""StateGraph assembly for the Intelligent Curriculum Engine.

Wires together 9 nodes into a linear workflow with 4 HITL interrupt
points and conditional routing for revision loops.

Workflow:
  plan_curriculum → review_plan → [approve/revise]
    → generate_lesson → review_lesson → [approve/revise/reject]
    → generate_exercises → review_exercises → [approve/revise/reject]
    → generate_assessment → review_assessment → [approve/revise/reject]
    → assemble_module → END

LangGraph concepts demonstrated:
- StateGraph with multiple interrupt() points
- Conditional edges for approve/revise routing at each stage
- Revision loops (re-generate with feedback)
- Checkpointer injection (required for interrupts)

DeepAgents concepts demonstrated:
- LangGraph as the outer orchestrator wrapping DeepAgents sub-agents
- Each generation node internally creates and invokes a DeepAgent
- Clean separation: LangGraph owns workflow, DeepAgents own content generation
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from models import CurriculumEngineState
from nodes import (
    plan_curriculum_node,
    review_plan_node,
    generate_lesson_node,
    review_lesson_node,
    generate_exercises_node,
    review_exercises_node,
    generate_assessment_node,
    review_assessment_node,
    assemble_module_node,
    route_after_plan_review,
    route_after_lesson_review,
    route_after_exercises_review,
    route_after_assessment_review,
)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the curriculum engine StateGraph.

    Args:
        checkpointer: Required for interrupt/resume. Defaults to InMemorySaver.

    Returns:
        Compiled LangGraph graph ready for .invoke().
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = (
        StateGraph(CurriculumEngineState)
        .add_node("plan_curriculum", plan_curriculum_node)
        .add_node("review_plan", review_plan_node)
        .add_node("generate_lesson", generate_lesson_node)
        .add_node("review_lesson", review_lesson_node)
        .add_node("generate_exercises", generate_exercises_node)
        .add_node("review_exercises", review_exercises_node)
        .add_node("generate_assessment", generate_assessment_node)
        .add_node("review_assessment", review_assessment_node)
        .add_node("assemble_module", assemble_module_node)
        .add_edge(START, "plan_curriculum")
        .add_edge("plan_curriculum", "review_plan")
        .add_conditional_edges(
            "review_plan",
            route_after_plan_review,
            ["generate_lesson", "plan_curriculum"],
        )
        .add_edge("generate_lesson", "review_lesson")
        .add_conditional_edges(
            "review_lesson",
            route_after_lesson_review,
            ["generate_exercises", "generate_lesson"],
        )
        .add_edge("generate_exercises", "review_exercises")
        .add_conditional_edges(
            "review_exercises",
            route_after_exercises_review,
            ["generate_assessment", "generate_exercises"],
        )
        .add_edge("generate_assessment", "review_assessment")
        .add_conditional_edges(
            "review_assessment",
            route_after_assessment_review,
            ["assemble_module", "generate_assessment"],
        )
        .add_edge("assemble_module", END)
        .compile(checkpointer=checkpointer)
    )

    return graph
