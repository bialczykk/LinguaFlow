# models.py
"""Pydantic models and LangGraph state schema for the Content Moderation system.

This module defines:
- ContentRequest: what kind of content to generate
- PublishMetadata: metadata attached to published content
- ContentModerationState: the TypedDict flowing through the graph

Key concepts demonstrated:
- Plain TypedDict state (not MessagesState) for non-conversational pipelines
- State fields map to specific nodes in the graph
- No reducers needed — each field is written by exactly one node
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Valid CEFR levels
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# Valid content types the system can generate
CONTENT_TYPES = (
    "grammar_explanation",
    "vocabulary_exercise",
    "reading_passage",
)


class ContentRequest(BaseModel):
    """A request to generate lesson content."""

    topic: str = Field(description="The topic to cover (e.g., 'Present Perfect Tense')")
    content_type: Literal[
        "grammar_explanation", "vocabulary_exercise", "reading_passage"
    ] = Field(description="Type of content to generate")
    difficulty: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Target CEFR difficulty level"
    )


class PublishMetadata(BaseModel):
    """Metadata attached to published content."""

    moderator_notes: str = Field(default="", description="Notes from the moderator")
    review_rounds: int = Field(default=0, description="Number of review rounds")


class ContentModerationState(TypedDict):
    """State schema for the content moderation StateGraph.

    This is a pipeline state (not conversational), so we use a plain
    TypedDict instead of MessagesState. Each field is written by exactly
    one node — no reducers needed.

    LangGraph concept: TypedDict state for non-conversational workflows.
    Human-in-the-loop concept: draft_decision and final_decision hold
    the values returned from Command(resume=...) at each interrupt point.
    """

    # -- Input (set at invocation) --
    content_request: dict              # ContentRequest as dict

    # -- After generate/revise --
    draft_content: str                 # The generated lesson snippet
    generation_confidence: float       # LLM's self-assessed confidence (0-1)

    # -- After draft_review --
    draft_decision: dict               # {"action": "approve"|"edit"|"reject", ...}
    revision_count: int                # Tracks revision rounds (max 2)

    # -- After polish --
    polished_content: str              # Cleaned-up content ready for final review

    # -- After final_review --
    final_decision: dict               # {"action": "approve"|"reject", ...}

    # -- After publish --
    published: bool                    # Whether content was published
    publish_metadata: dict | None      # Timestamp, moderator notes, etc.
