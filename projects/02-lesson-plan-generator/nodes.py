"""Node functions for the Lesson Plan Generator StateGraph.

Each function represents a node in the graph. It receives the full
LessonPlanState and returns a partial dict with only the fields it updates.

LangGraph concept demonstrated:
- Node functions as the building blocks of a StateGraph
- Each node performs one focused task (research, draft, review, finalize)
- Nodes return partial state updates — the graph engine merges them

LangChain concepts demonstrated:
- Prompt | Model pipeline for LLM calls
- .with_structured_output() for the finalize node
- @traceable for LangSmith observability
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langsmith import traceable

from models import LessonPlanState, LessonPlan
from prompts import (
    RESEARCH_PROMPT,
    DRAFT_CONVERSATION_PROMPT,
    DRAFT_GRAMMAR_PROMPT,
    DRAFT_EXAM_PREP_PROMPT,
    REVIEW_PROMPT,
    FINALIZE_PROMPT,
)

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
# temperature=0 for structured/deterministic output in most nodes
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
# Slightly higher temperature for drafting — allows more creative lesson plans
_creative_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)


def _format_list(items: list[str]) -> str:
    """Format a list of strings as a comma-separated string for prompts."""
    return ", ".join(items)


def _build_revision_context(state: LessonPlanState) -> str:
    """Build revision context string if this is a revision cycle."""
    if state["revision_count"] > 0 and state["review_feedback"]:
        return (
            f"REVISION REQUESTED (attempt {state['revision_count']}):\n"
            f"Reviewer feedback: {state['review_feedback']}\n\n"
            "Please address the feedback and improve the lesson plan.\n\n"
        )
    return ""


@traceable(name="research", run_type="chain", tags=["p2-lesson-plan-generator"])
def research_node(state: LessonPlanState) -> dict:
    """Research node: suggests materials and activities based on the student profile."""
    profile = state["student_profile"]
    chain = RESEARCH_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "lesson_type": profile.lesson_type,
    })
    return {"research_notes": result.content}


@traceable(name="draft_conversation", run_type="chain", tags=["p2-lesson-plan-generator"])
def draft_conversation_node(state: LessonPlanState) -> dict:
    """Draft node for conversation-focused lessons."""
    profile = state["student_profile"]
    chain = DRAFT_CONVERSATION_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="draft_grammar", run_type="chain", tags=["p2-lesson-plan-generator"])
def draft_grammar_node(state: LessonPlanState) -> dict:
    """Draft node for grammar-focused lessons."""
    profile = state["student_profile"]
    chain = DRAFT_GRAMMAR_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="draft_exam_prep", run_type="chain", tags=["p2-lesson-plan-generator"])
def draft_exam_prep_node(state: LessonPlanState) -> dict:
    """Draft node for exam preparation lessons."""
    profile = state["student_profile"]
    chain = DRAFT_EXAM_PREP_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="review", run_type="chain", tags=["p2-lesson-plan-generator"])
def review_node(state: LessonPlanState) -> dict:
    """Review node: critiques the draft and decides whether to approve."""
    profile = state["student_profile"]
    chain = REVIEW_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })

    response_text = result.content
    is_approved = response_text.strip().startswith("APPROVED")

    return {
        "is_approved": is_approved,
        "review_feedback": response_text,
        "revision_count": state["revision_count"] + 1,
    }


@traceable(name="finalize", run_type="chain", tags=["p2-lesson-plan-generator"])
def finalize_node(state: LessonPlanState) -> dict:
    """Finalize node: parses the draft into a structured LessonPlan."""
    profile = state["student_profile"]
    structured_model = _model.with_structured_output(
        LessonPlan, method="json_schema"
    )
    chain = FINALIZE_PROMPT | structured_model
    plan = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })
    return {"final_plan": plan}
