# Tutor Matching & Scheduling Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conversational LangGraph agent that helps students find and book tutors through tool calling, with checkpointer-based persistence for multi-session conversations.

**Architecture:** MessagesState extended with phase-tracking fields. An agent loop (LLM node → conditional edge → tool node → back to LLM) drives the conversation. Three `@tool` functions (search_tutors, check_availability, book_session) simulate external APIs. Checkpointer swappable between InMemorySaver and SqliteSaver via CLI flag.

**Tech Stack:** LangGraph (StateGraph, MessagesState, ToolNode), LangChain (ChatAnthropic, @tool, bind_tools), langgraph-checkpoint-sqlite, LangSmith (@traceable), Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-tutor-matching-agent-design.md`

---

## File Structure

```
projects/04-tutor-matching-agent/
├── models.py              # TutorMatchingState, Tutor, TimeSlot, BookingConfirmation
├── prompts.py             # Phase-aware system prompts
├── tools.py               # @tool: search_tutors, check_availability, book_session
├── nodes.py               # agent_node, should_continue, get_tool_node
├── graph.py               # build_graph() → compiled StateGraph
├── main.py                # CLI with --persist, --thread flags
├── data/
│   ├── __init__.py
│   ├── tutors.py          # TUTORS list (~10-12 tutor dicts)
│   └── calendar.py        # SCHEDULES dict (tutor_id → available slots)
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Shared fixtures
│   ├── test_models.py     # State schema and model validation
│   ├── test_tools.py      # Tool logic against mock data (no LLM)
│   ├── test_nodes.py      # Node + LLM integration tests
│   └── test_graph.py      # End-to-end conversation + persistence tests
├── README.md
└── requirements.txt
```

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `projects/04-tutor-matching-agent/requirements.txt`
- Create: `projects/04-tutor-matching-agent/data/__init__.py`
- Create: `projects/04-tutor-matching-agent/tests/__init__.py`

- [ ] **Step 1: Create project directory and requirements.txt**

```
projects/04-tutor-matching-agent/requirements.txt
```

```
langchain-core
langchain-anthropic
langgraph
langgraph-checkpoint-sqlite
langsmith
python-dotenv
pytest
```

- [ ] **Step 2: Create empty `__init__.py` files**

Create empty files:
- `projects/04-tutor-matching-agent/data/__init__.py`
- `projects/04-tutor-matching-agent/tests/__init__.py`

- [ ] **Step 3: Install dependencies**

Run: `cd projects/04-tutor-matching-agent && pip install -r requirements.txt`

- [ ] **Step 4: Verify imports work**

Run: `cd projects/04-tutor-matching-agent && python -c "from langgraph.graph import StateGraph, MessagesState; from langgraph.prebuilt import ToolNode; from langgraph.checkpoint.memory import InMemorySaver; from langgraph.checkpoint.sqlite import SqliteSaver; print('All imports OK')"`

- [ ] **Step 5: Commit**

```bash
git add projects/04-tutor-matching-agent/
git commit -m "feat(p4): scaffold project skeleton and dependencies"
```

---

### Task 2: Pydantic Models and State Schema (TDD)

**Files:**
- Create: `projects/04-tutor-matching-agent/tests/test_models.py`
- Create: `projects/04-tutor-matching-agent/models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# tests/test_models.py
"""Tests for Pydantic models and LangGraph state schema.

Validates that all models enforce constraints and that the state schema
extends MessagesState correctly.
"""

from models import Tutor, TimeSlot, BookingConfirmation, TutorMatchingState


class TestTutor:
    """Tutor model validation."""

    def test_valid_tutor(self):
        tutor = Tutor(
            tutor_id="t1",
            name="Alice Smith",
            specializations=["grammar", "exam_prep"],
            timezone="Europe/London",
            rating=4.8,
            bio="10 years teaching experience.",
            hourly_rate=35.0,
        )
        assert tutor.tutor_id == "t1"
        assert "grammar" in tutor.specializations
        assert tutor.rating == 4.8

    def test_rating_bounds(self):
        """Rating must be between 0.0 and 5.0."""
        import pytest

        with pytest.raises(Exception):
            Tutor(
                tutor_id="t1", name="X", specializations=["grammar"],
                timezone="UTC", rating=5.5, bio="x", hourly_rate=10.0,
            )


class TestTimeSlot:
    """TimeSlot model validation."""

    def test_valid_time_slot(self):
        slot = TimeSlot(date="2026-04-01", start_time="09:00", end_time="10:00")
        assert slot.date == "2026-04-01"
        assert slot.start_time == "09:00"


class TestBookingConfirmation:
    """BookingConfirmation model validation."""

    def test_valid_booking(self):
        booking = BookingConfirmation(
            confirmation_id="BK-001",
            tutor_name="Alice Smith",
            student_name="Bob",
            date="2026-04-01",
            time="09:00",
            duration_minutes=60,
        )
        assert booking.confirmation_id == "BK-001"
        assert booking.duration_minutes == 60


class TestTutorMatchingState:
    """State schema structure checks."""

    def test_state_has_required_fields(self):
        """TutorMatchingState must have all custom fields plus messages from MessagesState."""
        import typing
        hints = typing.get_type_hints(TutorMatchingState)
        assert "messages" in hints, "Must inherit messages from MessagesState"
        assert "phase" in hints
        assert "preferences" in hints
        assert "search_results" in hints
        assert "selected_tutor" in hints
        assert "booking_confirmation" in hints
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

```python
# models.py
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
    specializations: list[str] = Field(
        description="Areas of expertise: grammar, conversation, business_english, exam_prep"
    )
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

    # Phase tracking — guides the system prompt behavior
    phase: str                          # "gather", "search", "present", "book", "done"

    # Gathered student preferences (accumulated across conversation turns)
    preferences: dict                   # {"specialization": ..., "timezone": ..., "availability": ...}

    # Search results from tutor database tool
    search_results: list[dict]          # List of matching tutor records

    # Booking outcome
    selected_tutor: dict | None         # The tutor the student chose
    booking_confirmation: dict | None   # Final booking details
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/04-tutor-matching-agent/models.py projects/04-tutor-matching-agent/tests/test_models.py
git commit -m "feat(p4): add Pydantic models and state schema with tests"
```

---

### Task 3: Mock Data — Tutor Database and Calendar

**Files:**
- Create: `projects/04-tutor-matching-agent/data/tutors.py`
- Create: `projects/04-tutor-matching-agent/data/calendar.py`

- [ ] **Step 1: Create the mock tutor database**

```python
# data/tutors.py
"""Mock tutor database for the LinguaFlow operations department.

Contains ~10-12 tutor profiles with varied specializations, timezones,
ratings, and hourly rates. This data simulates what a real tutor database
API would return.

This is scaffolding — it exists only to give the tools realistic data
to filter and return. The LangGraph concepts are in tools.py and nodes.py.
"""

TUTORS = [
    {
        "tutor_id": "t1",
        "name": "Alice Smith",
        "specializations": ["grammar", "exam_prep"],
        "timezone": "Europe/London",
        "rating": 4.9,
        "bio": "Cambridge-certified with 10 years of exam prep experience. Specializes in IELTS and FCE.",
        "hourly_rate": 45.0,
    },
    {
        "tutor_id": "t2",
        "name": "Carlos Rivera",
        "specializations": ["conversation", "business_english"],
        "timezone": "America/New_York",
        "rating": 4.7,
        "bio": "Former corporate trainer. Makes business English practical and engaging.",
        "hourly_rate": 40.0,
    },
    {
        "tutor_id": "t3",
        "name": "Yuki Tanaka",
        "specializations": ["grammar", "conversation"],
        "timezone": "Asia/Tokyo",
        "rating": 4.8,
        "bio": "Patient and methodical. Great at building confidence in beginners.",
        "hourly_rate": 35.0,
    },
    {
        "tutor_id": "t4",
        "name": "Priya Patel",
        "specializations": ["exam_prep", "grammar"],
        "timezone": "Asia/Kolkata",
        "rating": 4.6,
        "bio": "IELTS examiner with insider knowledge of scoring criteria.",
        "hourly_rate": 38.0,
    },
    {
        "tutor_id": "t5",
        "name": "Emma Johansson",
        "specializations": ["conversation", "grammar"],
        "timezone": "Europe/Stockholm",
        "rating": 4.5,
        "bio": "Focuses on natural speech patterns and everyday fluency.",
        "hourly_rate": 32.0,
    },
    {
        "tutor_id": "t6",
        "name": "David Chen",
        "specializations": ["business_english", "exam_prep"],
        "timezone": "Asia/Shanghai",
        "rating": 4.9,
        "bio": "MBA graduate who teaches professional communication and presentation skills.",
        "hourly_rate": 50.0,
    },
    {
        "tutor_id": "t7",
        "name": "Sarah O'Brien",
        "specializations": ["conversation", "grammar"],
        "timezone": "Europe/Dublin",
        "rating": 4.3,
        "bio": "Friendly and approachable. Loves helping students overcome speaking anxiety.",
        "hourly_rate": 30.0,
    },
    {
        "tutor_id": "t8",
        "name": "Ahmed Hassan",
        "specializations": ["grammar", "business_english"],
        "timezone": "Africa/Cairo",
        "rating": 4.7,
        "bio": "Structured approach to grammar with real-world business applications.",
        "hourly_rate": 28.0,
    },
    {
        "tutor_id": "t9",
        "name": "Maria Garcia",
        "specializations": ["exam_prep", "conversation"],
        "timezone": "Europe/Madrid",
        "rating": 4.4,
        "bio": "Bilingual examiner who understands the challenges of learning English as a second language.",
        "hourly_rate": 36.0,
    },
    {
        "tutor_id": "t10",
        "name": "James Wilson",
        "specializations": ["business_english", "conversation"],
        "timezone": "America/Chicago",
        "rating": 4.6,
        "bio": "Former journalist. Teaches clear, concise professional writing and speaking.",
        "hourly_rate": 42.0,
    },
    {
        "tutor_id": "t11",
        "name": "Lena Müller",
        "specializations": ["grammar", "exam_prep", "conversation"],
        "timezone": "Europe/Berlin",
        "rating": 4.8,
        "bio": "Polyglot and linguistics PhD. Makes grammar intuitive through pattern recognition.",
        "hourly_rate": 48.0,
    },
    {
        "tutor_id": "t12",
        "name": "Kenji Nakamura",
        "specializations": ["conversation", "business_english"],
        "timezone": "Asia/Tokyo",
        "rating": 4.2,
        "bio": "Easygoing style focused on building vocabulary through conversation practice.",
        "hourly_rate": 25.0,
    },
]
```

- [ ] **Step 2: Create the mock calendar data**

```python
# data/calendar.py
"""Mock calendar/scheduling data for the LinguaFlow tutor system.

Contains pre-built availability schedules for each tutor. The book_session
tool updates this data in-memory when a session is booked (marks slots
as taken).

This is scaffolding — it simulates what a real calendar API would provide.
"""

# Each tutor has a list of available slots for the next few days.
# Slots are dicts with: date, start_time, end_time, booked (bool).
SCHEDULES = {
    "t1": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "14:00", "end_time": "15:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "15:00", "end_time": "16:00", "booked": False},
    ],
    "t2": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-03", "start_time": "09:00", "end_time": "10:00", "booked": False},
    ],
    "t3": [
        {"date": "2026-04-01", "start_time": "07:00", "end_time": "08:00", "booked": False},
        {"date": "2026-04-01", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-02", "start_time": "07:00", "end_time": "08:00", "booked": False},
    ],
    "t4": [
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "16:00", "end_time": "17:00", "booked": False},
        {"date": "2026-04-02", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-03", "start_time": "14:00", "end_time": "15:00", "booked": False},
    ],
    "t5": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-02", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-02", "start_time": "14:00", "end_time": "15:00", "booked": False},
    ],
    "t6": [
        {"date": "2026-04-01", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-01", "start_time": "17:00", "end_time": "18:00", "booked": False},
        {"date": "2026-04-02", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-03", "start_time": "08:00", "end_time": "09:00", "booked": False},
    ],
    "t7": [
        {"date": "2026-04-01", "start_time": "12:00", "end_time": "13:00", "booked": False},
        {"date": "2026-04-01", "start_time": "15:00", "end_time": "16:00", "booked": False},
        {"date": "2026-04-02", "start_time": "12:00", "end_time": "13:00", "booked": False},
    ],
    "t8": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "16:00", "end_time": "17:00", "booked": False},
    ],
    "t9": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-01", "start_time": "16:00", "end_time": "17:00", "booked": False},
        {"date": "2026-04-03", "start_time": "10:00", "end_time": "11:00", "booked": False},
    ],
    "t10": [
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "14:00", "end_time": "15:00", "booked": False},
        {"date": "2026-04-02", "start_time": "11:00", "end_time": "12:00", "booked": False},
    ],
    "t11": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-03", "start_time": "09:00", "end_time": "10:00", "booked": False},
    ],
    "t12": [
        {"date": "2026-04-01", "start_time": "07:00", "end_time": "08:00", "booked": False},
        {"date": "2026-04-01", "start_time": "18:00", "end_time": "19:00", "booked": False},
        {"date": "2026-04-02", "start_time": "07:00", "end_time": "08:00", "booked": False},
    ],
}
```

- [ ] **Step 3: Verify data loads**

Run: `cd projects/04-tutor-matching-agent && python -c "from data.tutors import TUTORS; from data.calendar import SCHEDULES; print(f'{len(TUTORS)} tutors, {len(SCHEDULES)} schedules'); assert len(TUTORS) == 12; assert len(SCHEDULES) == 12; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add projects/04-tutor-matching-agent/data/
git commit -m "feat(p4): add mock tutor database and calendar data"
```

---

### Task 4: Tool Functions (TDD)

**Files:**
- Create: `projects/04-tutor-matching-agent/tests/test_tools.py`
- Create: `projects/04-tutor-matching-agent/tools.py`

- [ ] **Step 1: Write failing tests for tools**

```python
# tests/test_tools.py
"""Tests for tool functions against mock data.

These tests verify the filtering, availability, and booking logic
WITHOUT any LLM calls. The tools are plain functions that operate
on the mock data in data/.
"""

import pytest
from tools import search_tutors, check_availability, book_session
from data.calendar import SCHEDULES


class TestSearchTutors:
    """search_tutors tool filtering logic."""

    def test_search_by_specialization(self):
        """Should return only tutors with the requested specialization."""
        result = search_tutors.invoke({"specialization": "exam_prep"})
        assert len(result) > 0
        for tutor in result:
            assert "exam_prep" in tutor["specializations"]

    def test_search_by_specialization_and_timezone(self):
        """Should narrow results by timezone when provided."""
        result = search_tutors.invoke({
            "specialization": "grammar",
            "timezone": "Europe/London",
        })
        assert len(result) > 0
        for tutor in result:
            assert "grammar" in tutor["specializations"]
            assert tutor["timezone"] == "Europe/London"

    def test_search_no_matches(self):
        """Should return empty list for impossible criteria."""
        result = search_tutors.invoke({
            "specialization": "grammar",
            "timezone": "Antarctica/South_Pole",
        })
        assert result == []

    def test_search_returns_all_fields(self):
        """Each result should have all expected tutor fields."""
        result = search_tutors.invoke({"specialization": "conversation"})
        assert len(result) > 0
        tutor = result[0]
        for field in ["tutor_id", "name", "specializations", "timezone", "rating", "bio", "hourly_rate"]:
            assert field in tutor


class TestCheckAvailability:
    """check_availability tool slot lookup logic."""

    def test_available_slots_exist(self):
        """Should return slots for a tutor on a date with availability."""
        result = check_availability.invoke({
            "tutor_id": "t1",
            "date": "2026-04-01",
        })
        assert len(result) > 0
        for slot in result:
            assert slot["date"] == "2026-04-01"

    def test_no_slots_on_missing_date(self):
        """Should return empty list for a date with no availability."""
        result = check_availability.invoke({
            "tutor_id": "t1",
            "date": "2026-12-25",
        })
        assert result == []

    def test_invalid_tutor_id(self):
        """Should return error message for unknown tutor."""
        result = check_availability.invoke({
            "tutor_id": "t999",
            "date": "2026-04-01",
        })
        assert "not found" in result.lower() or result == []


class TestBookSession:
    """book_session tool booking logic."""

    def test_successful_booking(self):
        """Should return a confirmation dict with all required fields."""
        # Reset the slot to unbooked before test
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "09:00":
                slot["booked"] = False

        result = book_session.invoke({
            "tutor_id": "t1",
            "date": "2026-04-01",
            "time": "09:00",
            "student_name": "Test Student",
        })
        assert "confirmation_id" in result
        assert result["tutor_name"] == "Alice Smith"
        assert result["student_name"] == "Test Student"

    def test_booking_marks_slot_as_taken(self):
        """After booking, the slot should be marked as booked."""
        # Reset first
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "11:00":
                slot["booked"] = False

        book_session.invoke({
            "tutor_id": "t1",
            "date": "2026-04-01",
            "time": "11:00",
            "student_name": "Test Student",
        })

        # Verify slot is now booked
        for slot in SCHEDULES["t1"]:
            if slot["date"] == "2026-04-01" and slot["start_time"] == "11:00":
                assert slot["booked"] is True

    def test_booking_unavailable_slot(self):
        """Should return an error message if the slot is already booked."""
        # Force the slot to be booked
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "14:00":
                slot["booked"] = True

        result = book_session.invoke({
            "tutor_id": "t1",
            "date": "2026-04-01",
            "time": "14:00",
            "student_name": "Test Student",
        })
        assert "not available" in str(result).lower() or "already booked" in str(result).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Implement tools.py**

```python
# tools.py
"""LangGraph tool functions for the Tutor Matching Agent.

Defines three tools that simulate external API calls:
- search_tutors: query the tutor database by criteria
- check_availability: look up a tutor's calendar slots
- book_session: reserve a session and get a confirmation

LangGraph concepts demonstrated:
- @tool decorator: turns a plain function into a LangGraph-compatible tool
- Tool functions are bound to the model via model.bind_tools()
- The prebuilt ToolNode executes these when the LLM requests them

These tools operate on the mock data in data/tutors.py and data/calendar.py.
In a real application, they would call REST APIs.
"""

import uuid

from langchain_core.tools import tool
from langsmith import traceable

from data.tutors import TUTORS
from data.calendar import SCHEDULES

# LangSmith tags for all tool traces
_TAGS = ["p4-tutor-matching"]


@tool
@traceable(name="search_tutors", run_type="tool", tags=_TAGS)
def search_tutors(
    specialization: str,
    timezone: str | None = None,
    availability: str | None = None,
) -> list[dict]:
    """Search the tutor database by specialization, timezone, and availability.

    Args:
        specialization: Required. One of: grammar, conversation, business_english, exam_prep.
        timezone: Optional. Filter by tutor's timezone (e.g., 'Europe/London').
        availability: Optional. Preferred date in YYYY-MM-DD format to check availability.

    Returns:
        List of matching tutor profiles with all fields.
    """
    results = []
    for tutor in TUTORS:
        # Filter by specialization (required)
        if specialization not in tutor["specializations"]:
            continue

        # Filter by timezone (optional)
        if timezone and tutor["timezone"] != timezone:
            continue

        # Filter by availability on a specific date (optional)
        if availability:
            tutor_slots = SCHEDULES.get(tutor["tutor_id"], [])
            has_availability = any(
                s["date"] == availability and not s["booked"]
                for s in tutor_slots
            )
            if not has_availability:
                continue

        results.append(tutor)

    return results


@tool
@traceable(name="check_availability", run_type="tool", tags=_TAGS)
def check_availability(tutor_id: str, date: str) -> list[dict] | str:
    """Check a tutor's available time slots for a specific date.

    Args:
        tutor_id: The tutor's unique ID (e.g., 't1').
        date: The date to check in YYYY-MM-DD format.

    Returns:
        List of available time slots, or an error message if tutor not found.
    """
    if tutor_id not in SCHEDULES:
        return f"Tutor {tutor_id} not found in the system."

    slots = SCHEDULES[tutor_id]
    available = [
        {"date": s["date"], "start_time": s["start_time"], "end_time": s["end_time"]}
        for s in slots
        if s["date"] == date and not s["booked"]
    ]
    return available


@tool
@traceable(name="book_session", run_type="tool", tags=_TAGS)
def book_session(
    tutor_id: str, date: str, time: str, student_name: str
) -> dict | str:
    """Book a tutoring session with a specific tutor.

    Args:
        tutor_id: The tutor's unique ID (e.g., 't1').
        date: Session date in YYYY-MM-DD format.
        time: Session start time in HH:MM format.
        student_name: The student's name for the booking.

    Returns:
        Booking confirmation dict, or an error message if the slot is not available.
    """
    if tutor_id not in SCHEDULES:
        return f"Tutor {tutor_id} not found in the system."

    # Find the matching slot
    for slot in SCHEDULES[tutor_id]:
        if slot["date"] == date and slot["start_time"] == time:
            if slot["booked"]:
                return f"Slot {time} on {date} is not available — already booked."
            # Mark as booked
            slot["booked"] = True

            # Look up the tutor name
            tutor_name = tutor_id
            for t in TUTORS:
                if t["tutor_id"] == tutor_id:
                    tutor_name = t["name"]
                    break

            return {
                "confirmation_id": f"BK-{uuid.uuid4().hex[:8].upper()}",
                "tutor_name": tutor_name,
                "student_name": student_name,
                "date": date,
                "time": time,
                "duration_minutes": 60,
            }

    return f"No slot found for tutor {tutor_id} at {time} on {date}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_tools.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/04-tutor-matching-agent/tools.py projects/04-tutor-matching-agent/tests/test_tools.py
git commit -m "feat(p4): add tool functions with tests for tutor search, availability, and booking"
```

---

### Task 5: Prompt Templates

**Files:**
- Create: `projects/04-tutor-matching-agent/prompts.py`

- [ ] **Step 1: Create phase-aware system prompts**

```python
# prompts.py
"""Phase-aware prompt templates for the Tutor Matching Agent.

The agent uses a single system prompt that adapts based on the current
conversation phase. Each phase guides the LLM's behavior — what to ask,
when to call tools, and how to present results.

LangChain concept demonstrated:
- System prompts as behavioral guides for tool-calling agents
- Phase-based prompt switching within a single conversational node
"""

# Base identity shared across all phases
_BASE_IDENTITY = (
    "You are a friendly scheduling assistant for LinguaFlow, an English tutoring platform. "
    "You help students find the right tutor and book sessions. "
    "Be conversational, helpful, and concise."
)

# Phase-specific instructions
PHASE_PROMPTS = {
    "gather": (
        f"{_BASE_IDENTITY}\n\n"
        "Your current task: gather the student's preferences for a tutor.\n\n"
        "You need to find out:\n"
        "1. What they want to focus on (grammar, conversation, business English, or exam prep)\n"
        "2. Their timezone or preferred time zone for sessions\n"
        "3. Any preferred dates or times\n\n"
        "Ask these questions naturally in conversation — don't dump all questions at once. "
        "Once you have at least the specialization, you can use the search_tutors tool to "
        "find matching tutors. Don't call search_tutors until you have the specialization."
    ),
    "search": (
        f"{_BASE_IDENTITY}\n\n"
        "Your current task: you've gathered enough preferences. Use the search_tutors tool "
        "to find matching tutors. Pass the specialization (required) and any timezone or "
        "availability preferences the student mentioned."
    ),
    "present": (
        f"{_BASE_IDENTITY}\n\n"
        "Your current task: present the search results to the student.\n\n"
        "Show each tutor's name, specializations, timezone, rating, and hourly rate. "
        "Help the student compare options. If they want to refine their search, use "
        "search_tutors again with updated criteria.\n\n"
        "When the student picks a tutor, use check_availability to show available slots."
    ),
    "book": (
        f"{_BASE_IDENTITY}\n\n"
        "Your current task: the student has selected a tutor and is ready to book.\n\n"
        "Use check_availability to show open slots if you haven't already. "
        "Once the student picks a time, use book_session to finalize the booking. "
        "You'll need: tutor_id, date, time, and the student's name."
    ),
    "done": (
        f"{_BASE_IDENTITY}\n\n"
        "The booking is confirmed! Summarize the booking details and wish the student well. "
        "Let them know they can start a new conversation anytime to book another session."
    ),
}


def get_system_prompt(phase: str) -> str:
    """Return the system prompt for the given conversation phase.

    Args:
        phase: One of 'gather', 'search', 'present', 'book', 'done'.

    Returns:
        The phase-appropriate system prompt string.
    """
    return PHASE_PROMPTS.get(phase, PHASE_PROMPTS["gather"])
```

- [ ] **Step 2: Verify prompts load correctly**

Run: `cd projects/04-tutor-matching-agent && python -c "from prompts import get_system_prompt, PHASE_PROMPTS; assert len(PHASE_PROMPTS) == 5; print(get_system_prompt('gather')[:80]); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add projects/04-tutor-matching-agent/prompts.py
git commit -m "feat(p4): add phase-aware prompt templates"
```

---

### Task 6: Node Functions (TDD)

**Files:**
- Create: `projects/04-tutor-matching-agent/tests/test_nodes.py`
- Create: `projects/04-tutor-matching-agent/nodes.py`

- [ ] **Step 1: Write failing tests for nodes**

```python
# tests/test_nodes.py
"""Integration tests for node functions.

These tests hit the LLM (Anthropic API) to verify that:
- agent_node produces valid AIMessage responses
- agent_node calls tools when appropriate
- should_continue routes correctly based on state

Marked with @pytest.mark.integration for selective running.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from models import TutorMatchingState
from nodes import agent_node, should_continue


@pytest.mark.integration
class TestAgentNode:
    """agent_node invokes the LLM with phase-aware prompts."""

    def test_gather_phase_responds_conversationally(self):
        """In gather phase, agent should ask about preferences."""
        state: TutorMatchingState = {
            "messages": [HumanMessage(content="Hi, I need help finding a tutor")],
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        }
        result = agent_node(state)
        assert "messages" in result
        # Should have responded with an AIMessage
        ai_msg = result["messages"][-1]
        assert isinstance(ai_msg, AIMessage)
        # Should be asking about preferences, not calling tools yet
        assert not ai_msg.tool_calls

    def test_gather_phase_calls_search_when_ready(self):
        """When student provides specialization, agent should call search_tutors."""
        state: TutorMatchingState = {
            "messages": [
                HumanMessage(content="I want to improve my grammar skills"),
                AIMessage(content="I'd love to help you find a grammar tutor! What timezone are you in?"),
                HumanMessage(content="I'm in London timezone. Can you find me someone?"),
            ],
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        }
        result = agent_node(state)
        ai_msg = result["messages"][-1]
        assert isinstance(ai_msg, AIMessage)
        # The LLM should decide to call search_tutors
        assert len(ai_msg.tool_calls) > 0
        assert ai_msg.tool_calls[0]["name"] == "search_tutors"


class TestShouldContinue:
    """should_continue routing logic."""

    def test_routes_to_tools_when_tool_calls_present(self):
        """If last message has tool_calls, route to tool_node."""
        ai_msg = AIMessage(content="", tool_calls=[{
            "id": "call_1", "name": "search_tutors",
            "args": {"specialization": "grammar"},
        }])
        state = {"messages": [ai_msg], "phase": "gather"}
        assert should_continue(state) == "tool_node"

    def test_routes_to_end_when_done(self):
        """If phase is done, route to END."""
        state = {
            "messages": [AIMessage(content="Booking confirmed!")],
            "phase": "done",
        }
        assert should_continue(state) == "__end__"

    def test_routes_to_end_when_waiting_for_user(self):
        """If no tool calls and not done, route to END (wait for user)."""
        state = {
            "messages": [AIMessage(content="What specialization are you looking for?")],
            "phase": "gather",
        }
        assert should_continue(state) == "__end__"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement nodes.py**

```python
# nodes.py
"""Node functions for the Tutor Matching Agent StateGraph.

Contains:
- agent_node: the conversational LLM node that drives the agent loop
- should_continue: routing function for conditional edges
- get_tool_node: factory for the prebuilt ToolNode

LangGraph concepts demonstrated:
- Agent loop pattern: LLM node → conditional edge → tool node → back to LLM
- bind_tools(): attaching tool definitions to the model so it can request calls
- ToolNode: prebuilt node that executes tool calls from the LLM response
- Phase-based state updates within a single node function

LangChain concepts demonstrated:
- ChatAnthropic with bound tools
- SystemMessage for phase-aware behavior
- @traceable for LangSmith observability
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from models import TutorMatchingState
from prompts import get_system_prompt
from tools import search_tutors, check_availability, book_session

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
# Using Claude Haiku for fast, cost-effective tool calling
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# Bind all three tools to the model so it can request tool calls
_tools = [search_tutors, check_availability, book_session]
_model_with_tools = _model.bind_tools(_tools)

# -- LangSmith Tags --
_TAGS = ["p4-tutor-matching"]


@traceable(name="agent_node", run_type="chain", tags=_TAGS)
def agent_node(state: TutorMatchingState) -> dict:
    """Conversational agent node — the brain of the matching workflow.

    Reads the current phase from state, constructs a phase-aware system
    prompt, and invokes the model with bound tools. After the LLM responds,
    infers the next phase from deterministic heuristics.

    LangGraph concept: a single agentic node that drives multi-turn
    conversation through tool calling, contrasting with P2/P3's
    deterministic one-node-per-step approach.

    Args:
        state: Current graph state with messages, phase, and domain fields.

    Returns:
        Partial state update with the new AI message and updated phase/preferences.
    """
    phase = state.get("phase", "gather")

    # Build message list with phase-aware system prompt
    system_msg = SystemMessage(content=get_system_prompt(phase))
    messages = [system_msg] + state["messages"]

    # Invoke the LLM with bound tools
    response = _model_with_tools.invoke(messages, config={"tags": _TAGS})

    # -- Deterministic phase inference --
    # Phase transitions are based on what just happened, not LLM output.
    new_phase = phase
    updates: dict = {"messages": [response]}

    if phase == "gather" and response.tool_calls:
        # If the LLM is calling search_tutors, we're moving to search/present
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "search_tutors" in tool_names:
            new_phase = "present"

    elif phase == "present" and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "check_availability" in tool_names:
            new_phase = "book"
        elif "search_tutors" in tool_names:
            new_phase = "present"  # Refining search, stay in present

    elif phase == "book" and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "book_session" in tool_names:
            new_phase = "done"

    updates["phase"] = new_phase
    return updates


def should_continue(state: TutorMatchingState) -> Literal["tool_node", "__end__"]:
    """Routing function for the conditional edge after agent_node.

    Determines what happens next:
    - If the LLM made tool calls → route to tool_node for execution
    - If the conversation is done → route to END
    - Otherwise → route to END (graph yields, waits for next user message)

    LangGraph concept: conditional edges route dynamically based on state.
    """
    last_message = state["messages"][-1]

    # If the LLM requested tool calls, execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"

    # Otherwise, end this turn (either done, or waiting for user input)
    return "__end__"


def get_tool_node() -> ToolNode:
    """Create the prebuilt ToolNode for executing tool calls.

    LangGraph concept: ToolNode from langgraph.prebuilt automatically
    dispatches tool calls from the LLM response to the matching tool
    functions. handle_tool_errors=True means errors are returned as
    ToolMessages so the LLM can recover gracefully.

    Returns:
        Configured ToolNode instance.
    """
    return ToolNode(_tools, handle_tool_errors=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_nodes.py -v`
Expected: `TestShouldContinue` tests PASS (3/3). `TestAgentNode` tests PASS if API key is set (2/2 integration tests).

- [ ] **Step 5: Commit**

```bash
git add projects/04-tutor-matching-agent/nodes.py projects/04-tutor-matching-agent/tests/test_nodes.py
git commit -m "feat(p4): add agent node, routing function, and ToolNode with tests"
```

---

### Task 7: Graph Assembly and End-to-End Tests (TDD)

**Files:**
- Create: `projects/04-tutor-matching-agent/tests/test_graph.py`
- Create: `projects/04-tutor-matching-agent/tests/conftest.py`
- Create: `projects/04-tutor-matching-agent/graph.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
"""Shared pytest fixtures for the Tutor Matching Agent tests.

Provides pre-configured graph instances for integration testing.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph


@pytest.fixture
def graph_with_memory():
    """Compiled graph with InMemorySaver checkpointer for persistence tests."""
    checkpointer = InMemorySaver()
    return build_graph(checkpointer=checkpointer)
```

- [ ] **Step 2: Write failing tests for graph**

```python
# tests/test_graph.py
"""End-to-end integration tests for the Tutor Matching Agent graph.

Tests the full conversation flow: user messages in → tool calls happen →
booking confirmed. Also tests checkpointer-based persistence.

These tests hit the LLM (Anthropic API).
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from graph import build_graph


@pytest.mark.integration
class TestGraphFlow:
    """Full conversation flow through the graph."""

    def test_initial_message_gets_response(self, graph_with_memory):
        """First message should get a conversational response."""
        config = {"configurable": {"thread_id": "test-1"}, "tags": ["p4-tutor-matching"]}
        result = graph_with_memory.invoke(
            {
                "messages": [HumanMessage(content="Hi, I need a grammar tutor")],
                "phase": "gather",
                "preferences": {},
                "search_results": [],
                "selected_tutor": None,
                "booking_confirmation": None,
            },
            config=config,
        )
        # Should have at least the user message + one AI response
        assert len(result["messages"]) >= 2
        # Last human-visible message should be from AI
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

    def test_multi_turn_conversation_persists(self, graph_with_memory):
        """Messages should accumulate across invocations with same thread_id."""
        config = {"configurable": {"thread_id": "test-persist"}, "tags": ["p4-tutor-matching"]}

        # Turn 1
        result1 = graph_with_memory.invoke(
            {
                "messages": [HumanMessage(content="Hello!")],
                "phase": "gather",
                "preferences": {},
                "search_results": [],
                "selected_tutor": None,
                "booking_confirmation": None,
            },
            config=config,
        )
        turn1_count = len(result1["messages"])

        # Turn 2 — same thread
        result2 = graph_with_memory.invoke(
            {"messages": [HumanMessage(content="I want to focus on grammar")]},
            config=config,
        )
        # Should have more messages than turn 1 (accumulated)
        assert len(result2["messages"]) > turn1_count

    def test_separate_threads_are_isolated(self, graph_with_memory):
        """Different thread_ids should maintain independent state."""
        initial_state = {
            "messages": [HumanMessage(content="Hi")],
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        }

        config_a = {"configurable": {"thread_id": "thread-A"}, "tags": ["p4-tutor-matching"]}
        config_b = {"configurable": {"thread_id": "thread-B"}, "tags": ["p4-tutor-matching"]}

        result_a = graph_with_memory.invoke(initial_state, config=config_a)
        result_b = graph_with_memory.invoke(initial_state, config=config_b)

        # Both should work independently
        assert len(result_a["messages"]) >= 2
        assert len(result_b["messages"]) >= 2


@pytest.mark.integration
class TestGraphStructure:
    """Verify the graph is wired correctly."""

    def test_graph_compiles(self):
        """build_graph should return a compiled, invokable graph."""
        graph = build_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_graph_has_expected_nodes(self):
        """Graph should contain agent_node and tool_node."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "agent_node" in node_names
        assert "tool_node" in node_names
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 4: Implement graph.py**

```python
# graph.py
"""StateGraph assembly for the Tutor Matching Agent.

Wires together the agent loop: agent_node ↔ tool_node with conditional
routing. Accepts an optional checkpointer for persistence.

LangGraph concepts demonstrated:
- StateGraph with MessagesState-based schema
- Agent loop pattern: LLM node → conditional edge → tool node → back
- ToolNode from langgraph.prebuilt for automatic tool dispatch
- Checkpointer injection at compile time (pluggable persistence)
- Conditional edges with routing functions

Architecture note: this graph has only two nodes (agent + tools) unlike
P2/P3's multi-node deterministic pipelines. The LLM drives the flow
through tool calls, and the graph provides the execution loop. Phase
transitions happen inside agent_node via deterministic heuristics.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from models import TutorMatchingState
from nodes import agent_node, should_continue, get_tool_node


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the Tutor Matching Agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.
            Pass InMemorySaver() for in-process persistence,
            SqliteSaver for durable persistence, or None for no persistence.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    # Create the tool node from prebuilt ToolNode
    tool_node = get_tool_node()

    # Build the agent loop graph
    graph = (
        StateGraph(TutorMatchingState)
        # Two nodes: the LLM agent and the tool executor
        .add_node("agent_node", agent_node)
        .add_node("tool_node", tool_node)
        # Entry point: always start with the agent
        .add_edge(START, "agent_node")
        # After agent: route based on whether it made tool calls
        .add_conditional_edges("agent_node", should_continue, ["tool_node", "__end__"])
        # After tools execute: always go back to the agent
        .add_edge("tool_node", "agent_node")
        # Compile with optional checkpointer
        .compile(checkpointer=checkpointer)
    )

    return graph
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/test_graph.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add projects/04-tutor-matching-agent/graph.py projects/04-tutor-matching-agent/tests/test_graph.py projects/04-tutor-matching-agent/tests/conftest.py
git commit -m "feat(p4): add graph assembly with agent loop and end-to-end tests"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `projects/04-tutor-matching-agent/main.py`

- [ ] **Step 1: Implement main.py**

```python
# main.py
"""CLI entry point for the Tutor Matching & Scheduling Agent.

Provides an interactive conversation loop where students can find and
book tutors. Supports two persistence modes:
- Default (InMemorySaver): state persists within the session only
- --persist (SqliteSaver): state survives process restarts

Usage:
    python main.py                          # New conversation, in-memory
    python main.py --persist                # New conversation, durable
    python main.py --persist --thread ID    # Resume a durable conversation

LangGraph concepts demonstrated:
- Checkpointer swapping: one-line change between InMemorySaver and SqliteSaver
- Thread management: thread_id in config for conversation isolation
- graph.invoke() with message appending for multi-turn conversation
"""

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from repo root .env
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph


def main():
    parser = argparse.ArgumentParser(
        description="LinguaFlow Tutor Matching Agent — find and book the right tutor"
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Use SQLite for durable persistence (state survives restarts)",
    )
    parser.add_argument(
        "--thread", type=str, default=None,
        help="Resume an existing conversation by thread ID (requires --persist)",
    )
    args = parser.parse_args()

    # -- Select checkpointer --
    if args.persist:
        db_path = Path(__file__).parent / "tutor_matching.db"
        checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        persistence_label = f"SQLite ({db_path.name})"
    else:
        checkpointer = InMemorySaver()
        persistence_label = "in-memory (lost on exit)"

    # -- Set up thread --
    thread_id = args.thread or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p4-tutor-matching"],
    }

    # -- Build graph --
    graph = build_graph(checkpointer=checkpointer)

    # -- Print session info --
    print("=" * 60)
    print("LinguaFlow Tutor Matching Agent")
    print("=" * 60)
    print(f"Thread ID:   {thread_id}")
    print(f"Persistence: {persistence_label}")
    if args.thread:
        print("(Resuming existing conversation)")
    print("-" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    # -- Check if resuming an existing conversation --
    is_first_turn = args.thread is None

    # -- Conversation loop --
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Your thread ID for resuming: " + thread_id)
            break

        # Build the input for this turn
        turn_input = {"messages": [HumanMessage(content=user_input)]}

        # On the very first turn of a new conversation, include initial state
        if is_first_turn:
            turn_input.update({
                "phase": "gather",
                "preferences": {},
                "search_results": [],
                "selected_tutor": None,
                "booking_confirmation": None,
            })
            is_first_turn = False

        # Stream the graph execution to show progress
        for chunk in graph.stream(turn_input, config=config, stream_mode="updates"):
            # Print agent responses as they come
            for node_name, updates in chunk.items():
                if node_name == "agent_node" and "messages" in updates:
                    for msg in updates["messages"]:
                        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                            print(f"\nAgent: {msg.content}\n")
                        elif hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  [Calling tool: {tc['name']}...]")

        # Check if conversation is done
        state = graph.get_state(config)
        if state.values.get("phase") == "done":
            print("\n" + "=" * 60)
            print("Session complete! Thank you for using LinguaFlow.")
            print(f"Thread ID for reference: {thread_id}")
            print("=" * 60)
            break


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI starts and responds**

Run: `cd projects/04-tutor-matching-agent && echo "Hi, I want a grammar tutor\nquit" | python main.py`
Expected: Agent responds conversationally, then exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add projects/04-tutor-matching-agent/main.py
git commit -m "feat(p4): add CLI entry point with --persist and --thread flags"
```

---

### Task 9: Project README

**Files:**
- Create: `projects/04-tutor-matching-agent/README.md`

- [ ] **Step 1: Write the README**

```markdown
# Project 4: Tutor Matching & Scheduling Agent

A conversational LangGraph agent that helps students find and book the right English tutor on the LinguaFlow platform.

## What This Teaches

- **Tool calling**: `@tool`, `bind_tools()`, `ToolNode` from `langgraph.prebuilt`
- **MessagesState**: LangGraph's built-in message-based state, extended with custom fields
- **Checkpointers**: `InMemorySaver` vs `SqliteSaver` — pluggable persistence
- **Thread management**: `thread_id` for isolated conversations, resume from checkpoint
- **Agent loop pattern**: LLM → conditional edge → tool execution → back to LLM

## How It Works

The agent guides students through four phases:
1. **Gather** — asks about specialization, timezone, availability
2. **Present** — searches tutors and presents matches
3. **Book** — checks tutor availability and confirms a booking
4. **Done** — summarizes the booking

Unlike Projects 2-3 (deterministic node-per-step), this graph has just two nodes (agent + tools) with the LLM driving the flow through tool calls.

## Running

```bash
# New conversation (in-memory, lost on exit)
python main.py

# New conversation with durable persistence
python main.py --persist

# Resume a previous conversation
python main.py --persist --thread <thread-id>
```

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Just tool logic (no LLM)
python -m pytest tests/test_tools.py -v

# Integration tests (requires API key)
python -m pytest tests/ -v -m integration
```

## Project Structure

```
models.py    — State schema (extends MessagesState) and Pydantic models
prompts.py   — Phase-aware system prompts
tools.py     — @tool functions: search_tutors, check_availability, book_session
nodes.py     — agent_node, should_continue router, ToolNode factory
graph.py     — StateGraph wiring and compilation
main.py      — Interactive CLI with persistence options
data/        — Mock tutor database and calendar (scaffolding)
tests/       — Unit tests (tools) + integration tests (nodes, graph)
```
```

- [ ] **Step 2: Commit**

```bash
git add projects/04-tutor-matching-agent/README.md
git commit -m "docs(p4): add project README"
```

---

### Task 10: Educational Documentation

**Files:**
- Create: `docs/04-tutor-matching-agent.md`

- [ ] **Step 1: Write the educational doc**

Create `docs/04-tutor-matching-agent.md` — a comprehensive educational document that explains:

1. **Tool calling in LangGraph** — what `@tool`, `bind_tools()`, and `ToolNode` do, how the LLM decides to call tools, and how tool results flow back
2. **MessagesState** — why it exists, what `operator.add` does for messages, and how to extend it with custom fields
3. **The agent loop pattern** — how the LLM → tool → LLM loop works, contrasted with P2/P3's deterministic graphs
4. **Checkpointers** — what they do, how `InMemorySaver` vs `SqliteSaver` differ, thread_id semantics, and the pluggable pattern
5. **Phase-based prompting** — how a single agent node can drive a multi-phase conversation through prompt engineering

Reference specific code from the project to illustrate each concept. The doc should teach someone the concepts even if they haven't run the code.

- [ ] **Step 2: Commit**

```bash
git add docs/04-tutor-matching-agent.md
git commit -m "docs(p4): add educational guide for tool calling and persistence concepts"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run the full test suite**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Verify CLI end-to-end**

Run the CLI interactively and complete a full booking flow:
1. Start with `python main.py`
2. Ask for a grammar tutor in the London timezone
3. Pick a tutor from the results
4. Check availability and book a slot
5. Verify the booking confirmation is shown

- [ ] **Step 3: Verify persistence**

1. Run `python main.py --persist`, send one message, note the thread ID, then quit
2. Run `python main.py --persist --thread <id>`, verify the conversation resumes

- [ ] **Step 4: Run all tests one final time and commit any fixes**

Run: `cd projects/04-tutor-matching-agent && python -m pytest tests/ -v`
Expected: All tests pass with no warnings.
