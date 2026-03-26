"""Pydantic models and LangGraph state schema for the Tutor Matching Agent.

This module defines:
- Tutor: a tutor profile from the mock database
- TimeSlot: a single availability window
- BookingConfirmation: result of a successful booking
- TutorMatchingState: the TypedDict state that flows through the graph,
  extending MessagesState with phase tracking and booking fields

Key concepts demonstrated:
- Extending MessagesState with custom fields for domain-specific state
- MessagesState provides messages: Annotated[list, operator.add] automatically
- Custom fields use "last write wins" (no reducers) — each is updated by one node
"""

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class Tutor(BaseModel):
    """A tutor profile from the LinguaFlow tutor database."""
    tutor_id: str = Field(description="Unique identifier for the tutor")
    name: str = Field(description="Tutor's full name")
    specializations: list[str] = Field(description="Areas of expertise: grammar, conversation, business_english, exam_prep")
    timezone: str = Field(description="Tutor's timezone (e.g., 'Europe/London')")
    rating: float = Field(ge=0.0, le=5.0, description="Average student rating (0-5)")
    bio: str = Field(description="Short biography / teaching philosophy")
    hourly_rate: float = Field(description="Rate in USD per hour")


class TimeSlot(BaseModel):
    """A single availability window for a tutor."""
    date: str = Field(description="Date in YYYY-MM-DD format")
    start_time: str = Field(description="Start time in HH:MM format")
    end_time: str = Field(description="End time in HH:MM format")


class BookingConfirmation(BaseModel):
    """Confirmation returned after a successful session booking."""
    confirmation_id: str = Field(description="Unique booking reference")
    tutor_name: str = Field(description="Name of the booked tutor")
    student_name: str = Field(description="Name of the student")
    date: str = Field(description="Session date in YYYY-MM-DD format")
    time: str = Field(description="Session start time in HH:MM format")
    duration_minutes: int = Field(description="Session length in minutes")


class TutorMatchingState(MessagesState):
    """State schema for the LangGraph StateGraph.

    Extends MessagesState which provides:
        messages: Annotated[list[AnyMessage], operator.add]

    Custom fields track conversation phase, student preferences,
    search results, and booking outcome. All use "last write wins"
    since each is written by the agent_node only.

    LangGraph concept: extending MessagesState with domain-specific fields.
    """
    phase: str                          # "gather", "search", "present", "book", "done"
    preferences: dict                   # {"specialization": ..., "timezone": ..., "availability": ...}
    search_results: list[dict]          # List of matching tutor records
    selected_tutor: dict | None         # The tutor the student chose
    booking_confirmation: dict | None   # Final booking details
