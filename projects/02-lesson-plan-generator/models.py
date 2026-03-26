"""Pydantic models and LangGraph state schema for the Lesson Plan Generator.

This module defines all data structures used in the lesson plan pipeline:
- StudentProfile: input from the intake conversation
- Activity: a single lesson activity
- LessonPlan: the structured final output
- LessonPlanState: the TypedDict that flows through the LangGraph StateGraph

Key LangGraph concept demonstrated:
- TypedDict as graph state schema — every node reads from and writes to this shared state.
  Fields without reducers use "last write wins" semantics, which is fine here since
  each field is written by exactly one node at a time.
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class StudentProfile(BaseModel):
    """Student information gathered during the intake conversation."""

    name: str = Field(description="Student's name")
    proficiency_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="CEFR proficiency level assessed or self-reported"
    )
    learning_goals: list[str] = Field(
        description="What the student wants to achieve"
    )
    preferred_topics: list[str] = Field(
        description="Topics the student is interested in"
    )
    lesson_type: Literal["conversation", "grammar", "exam_prep"] = Field(
        description="Type of lesson to generate, inferred from the student's goals"
    )


class Activity(BaseModel):
    """A single activity within a lesson plan."""

    name: str = Field(description="Short name for the activity")
    description: str = Field(description="What the activity involves")
    duration_minutes: int = Field(description="Estimated duration in minutes")
    materials: list[str] = Field(description="Materials needed")


class LessonPlan(BaseModel):
    """Complete structured lesson plan — the final output of the graph."""

    title: str = Field(description="Descriptive lesson title")
    level: str = Field(description="CEFR level this lesson targets")
    lesson_type: str = Field(description="conversation, grammar, or exam_prep")
    objectives: list[str] = Field(description="Learning objectives")
    warm_up: str = Field(description="Warm-up activity description")
    main_activities: list[Activity] = Field(description="Core lesson activities")
    wrap_up: str = Field(description="Wrap-up / review activity")
    homework: str = Field(description="Homework assignment")
    estimated_duration_minutes: int = Field(description="Total estimated duration in minutes")


class LessonPlanState(TypedDict):
    """State schema for the LangGraph StateGraph.

    This TypedDict defines the shared state that flows through every node.
    Each node reads what it needs and returns a partial dict updating only
    the fields it's responsible for.

    LangGraph concept: TypedDict state schemas
    - Every field uses "last write wins" (no reducers needed here)
    - Nodes return partial dicts like {"research_notes": "..."}
    - The graph engine merges these updates into the full state
    """

    student_profile: StudentProfile
    research_notes: str
    draft_plan: str
    review_feedback: str
    revision_count: int
    is_approved: bool
    final_plan: LessonPlan | None
