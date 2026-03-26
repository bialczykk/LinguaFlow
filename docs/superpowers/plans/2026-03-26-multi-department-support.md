# Multi-Department Support System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a supervisor-based multi-agent support system that classifies incoming requests, routes to department-specific sub-agents (billing, tech support, scheduling, content), supports parallel dispatch via `Send`, supervisor-mediated escalation, and hybrid clarification via `interrupt()`.

**Architecture:** Flat `StateGraph` with a supervisor router node that classifies requests and dispatches to 4 specialized sub-agent nodes. Each sub-agent runs an internal tool-calling loop. Results converge at a supervisor aggregator that handles escalations, then a compose node synthesizes the final response. Clarification uses `interrupt()` + `Command(resume=...)`.

**Tech Stack:** LangGraph (StateGraph, Send, interrupt, Command), LangChain (ChatAnthropic, @tool, ChatPromptTemplate), LangSmith (tracing, evaluation), Streamlit (app UI)

**Spec:** `docs/superpowers/specs/2026-03-26-multi-department-support-design.md`

---

## File Structure

```
projects/06-multi-department-support/
├── models.py                    # SupportState, DepartmentResult TypedDicts
├── graph.py                     # StateGraph assembly with Send routing
├── nodes.py                     # supervisor_router, 4 sub-agents, aggregator, compose, ask_clarification
├── prompts.py                   # All prompt templates (supervisor classification, 4 dept prompts, compose)
├── tools.py                     # 8 @tool functions (2 per department)
├── evaluation.py                # LangSmith evaluators (routing accuracy, response quality)
├── data/
│   ├── __init__.py
│   ├── invoices.py              # Mock invoice records
│   ├── system_status.py         # Service health map
│   ├── accounts.py              # Student account records
│   ├── lessons.py               # Upcoming lessons (dynamic dates)
│   ├── content_library.py       # Course catalog + enrollment records
│   └── support_requests.py      # 8 sample requests covering all routing patterns
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures (graph_with_memory, sample states)
│   ├── test_models.py           # State schema, DepartmentResult validation
│   ├── test_tools.py            # Tool functions against mock data
│   ├── test_nodes.py            # Sub-agent nodes with mocked LLM
│   ├── test_supervisor.py       # Router + aggregator logic
│   └── test_graph.py            # Integration: single/parallel/escalation/clarification flows
└── README.md                    # Project overview and how to run

app/
├── adapters/support_system.py   # Adapter wrapping graph for Streamlit
└── pages/p6_support.py          # Support system page UI
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `projects/06-multi-department-support/` (directory + `__init__` files)
- Create: `projects/06-multi-department-support/requirements.txt`
- Create: `projects/06-multi-department-support/data/__init__.py`
- Create: `projects/06-multi-department-support/tests/__init__.py`

- [ ] **Step 1: Create project directory structure**

```bash
mkdir -p "projects/06-multi-department-support/data"
mkdir -p "projects/06-multi-department-support/tests"
```

- [ ] **Step 2: Create requirements.txt**

```
langchain-core>=0.3.0
langchain-anthropic>=0.3.0
langgraph>=0.4.0
langsmith>=0.3.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 3: Create package init files**

`projects/06-multi-department-support/data/__init__.py`:
```python
"""Mock data modules for the multi-department support system."""
```

`projects/06-multi-department-support/tests/__init__.py`:
```python
"""Tests for the multi-department support system."""
```

- [ ] **Step 4: Verify dependencies are available**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -c "import langgraph; from langgraph.types import Send; print('Send available:', Send)"
```

Expected: prints Send class reference without error.

- [ ] **Step 5: Commit**

```bash
git add projects/06-multi-department-support/
git commit -m "feat(p6): scaffold project directory structure"
```

---

## Task 2: Mock Data Modules

**Files:**
- Create: `projects/06-multi-department-support/data/invoices.py`
- Create: `projects/06-multi-department-support/data/system_status.py`
- Create: `projects/06-multi-department-support/data/accounts.py`
- Create: `projects/06-multi-department-support/data/lessons.py`
- Create: `projects/06-multi-department-support/data/content_library.py`
- Create: `projects/06-multi-department-support/data/support_requests.py`

- [ ] **Step 1: Create invoices.py**

`projects/06-multi-department-support/data/invoices.py`:
```python
"""Mock invoice data for the billing department.

Provides sample invoices with various statuses. Student IDs and lesson IDs
are consistent with other data modules to enable cross-department scenarios.
"""

INVOICES = [
    {
        "invoice_id": "INV-001",
        "student_id": "S001",
        "amount": 45.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L001",
        "date": "2026-03-20",
        "description": "1-hour grammar lesson with Alice Smith",
    },
    {
        "invoice_id": "INV-002",
        "student_id": "S001",
        "amount": 45.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L002",
        "date": "2026-03-22",
        "description": "1-hour conversation practice with Bob Chen",
    },
    {
        "invoice_id": "INV-003",
        "student_id": "S002",
        "amount": 60.00,
        "currency": "USD",
        "status": "pending",
        "lesson_id": "L003",
        "date": "2026-03-25",
        "description": "1-hour business English with Carol Davis",
    },
    {
        "invoice_id": "INV-004",
        "student_id": "S002",
        "amount": 45.00,
        "currency": "USD",
        "status": "refunded",
        "lesson_id": "L004",
        "date": "2026-03-18",
        "description": "Cancelled grammar lesson — full refund issued",
    },
    {
        "invoice_id": "INV-005",
        "student_id": "S003",
        "amount": 45.00,
        "currency": "USD",
        "status": "disputed",
        "lesson_id": "L005",
        "date": "2026-03-23",
        "description": "1-hour exam prep — student reports duplicate charge",
    },
    {
        "invoice_id": "INV-006",
        "student_id": "S003",
        "amount": 45.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L005",
        "date": "2026-03-23",
        "description": "1-hour exam prep — duplicate charge (same lesson)",
    },
    {
        "invoice_id": "INV-007",
        "student_id": "S004",
        "amount": 60.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L006",
        "date": "2026-03-21",
        "description": "1-hour IELTS prep with Eve Foster",
    },
    {
        "invoice_id": "INV-008",
        "student_id": "S005",
        "amount": 45.00,
        "currency": "USD",
        "status": "pending",
        "lesson_id": "L007",
        "date": "2026-03-26",
        "description": "1-hour vocabulary building with Frank Garcia",
    },
]
```

- [ ] **Step 2: Create system_status.py**

`projects/06-multi-department-support/data/system_status.py`:
```python
"""Mock system status data for the tech support department.

Service health map — most services healthy, one degraded for demo purposes.
"""

SERVICES = {
    "video_platform": {
        "name": "Video Lesson Platform",
        "status": "operational",
        "uptime_percent": 99.9,
        "last_incident": "2026-03-10",
        "notes": "All video conferencing services running normally.",
    },
    "chat_system": {
        "name": "In-App Chat System",
        "status": "degraded",
        "uptime_percent": 95.2,
        "last_incident": "2026-03-26",
        "notes": "Intermittent message delivery delays. Engineering investigating.",
    },
    "payment_gateway": {
        "name": "Payment Gateway",
        "status": "operational",
        "uptime_percent": 99.95,
        "last_incident": "2026-03-05",
        "notes": "All payment processing running normally.",
    },
    "content_cdn": {
        "name": "Content Delivery Network",
        "status": "operational",
        "uptime_percent": 99.8,
        "last_incident": "2026-03-15",
        "notes": "All learning materials loading normally.",
    },
    "auth_service": {
        "name": "Authentication Service",
        "status": "operational",
        "uptime_percent": 99.99,
        "last_incident": "2026-02-20",
        "notes": "Login and SSO services running normally.",
    },
}
```

- [ ] **Step 3: Create accounts.py**

`projects/06-multi-department-support/data/accounts.py`:
```python
"""Mock student account data for tech support lookups.

Student IDs match across all data modules (invoices, lessons, enrollments).
"""

ACCOUNTS = {
    "S001": {
        "student_id": "S001",
        "name": "Maria Garcia",
        "email": "maria.garcia@email.com",
        "plan": "premium",
        "timezone": "Europe/Madrid",
        "joined": "2025-09-15",
        "last_login": "2026-03-26",
        "known_issues": [],
    },
    "S002": {
        "student_id": "S002",
        "name": "Kenji Tanaka",
        "email": "kenji.tanaka@email.com",
        "plan": "standard",
        "timezone": "Asia/Tokyo",
        "joined": "2025-11-01",
        "last_login": "2026-03-25",
        "known_issues": ["password_reset_pending"],
    },
    "S003": {
        "student_id": "S003",
        "name": "Olga Petrov",
        "email": "olga.petrov@email.com",
        "plan": "premium",
        "timezone": "Europe/Moscow",
        "joined": "2026-01-10",
        "last_login": "2026-03-24",
        "known_issues": ["duplicate_charge_reported"],
    },
    "S004": {
        "student_id": "S004",
        "name": "Carlos Silva",
        "email": "carlos.silva@email.com",
        "plan": "basic",
        "timezone": "America/Sao_Paulo",
        "joined": "2026-02-01",
        "last_login": "2026-03-20",
        "known_issues": ["browser_compatibility_chrome"],
    },
    "S005": {
        "student_id": "S005",
        "name": "Aisha Bello",
        "email": "aisha.bello@email.com",
        "plan": "standard",
        "timezone": "Africa/Lagos",
        "joined": "2026-03-01",
        "last_login": "2026-03-26",
        "known_issues": [],
    },
}
```

- [ ] **Step 4: Create lessons.py**

`projects/06-multi-department-support/data/lessons.py`:
```python
"""Mock lesson schedule data for the scheduling department.

Dates are generated dynamically relative to today so demos always show
upcoming lessons. Student and tutor IDs are consistent with other modules.
"""

from datetime import date, timedelta

_today = date.today()


def _day(offset: int) -> str:
    """Return ISO date string offset from today."""
    return (_today + timedelta(days=offset)).strftime("%Y-%m-%d")


# Upcoming and recent lessons
LESSONS = [
    {
        "lesson_id": "L001",
        "student_id": "S001",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Fundamentals",
        "date": _day(-6),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L002",
        "student_id": "S001",
        "tutor_name": "Bob Chen",
        "subject": "Conversation Practice",
        "date": _day(-4),
        "time": "14:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L003",
        "student_id": "S002",
        "tutor_name": "Carol Davis",
        "subject": "Business English",
        "date": _day(1),
        "time": "09:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L004",
        "student_id": "S002",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Review",
        "date": _day(-8),
        "time": "11:00",
        "duration_minutes": 60,
        "status": "cancelled",
    },
    {
        "lesson_id": "L005",
        "student_id": "S003",
        "tutor_name": "Diana Evans",
        "subject": "IELTS Exam Prep",
        "date": _day(-3),
        "time": "15:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L006",
        "student_id": "S004",
        "tutor_name": "Eve Foster",
        "subject": "IELTS Writing",
        "date": _day(2),
        "time": "16:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L007",
        "student_id": "S005",
        "tutor_name": "Frank Garcia",
        "subject": "Vocabulary Building",
        "date": _day(3),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L008",
        "student_id": "S001",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Advanced",
        "date": _day(4),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
]
```

- [ ] **Step 5: Create content_library.py**

`projects/06-multi-department-support/data/content_library.py`:
```python
"""Mock content library and enrollment data for the content department.

Course catalog with topic, level, and type. Plus enrollment records
linking students to courses.
"""

COURSES = [
    {"course_id": "C001", "title": "Grammar Essentials", "level": "A2", "type": "grammar", "modules": 12, "description": "Core grammar rules for elementary learners."},
    {"course_id": "C002", "title": "Everyday Conversations", "level": "B1", "type": "conversation", "modules": 10, "description": "Practical dialogue skills for intermediate speakers."},
    {"course_id": "C003", "title": "Business English Fundamentals", "level": "B2", "type": "business", "modules": 8, "description": "Professional communication and workplace vocabulary."},
    {"course_id": "C004", "title": "Academic Writing", "level": "C1", "type": "writing", "modules": 6, "description": "Essay structure, argumentation, and formal register."},
    {"course_id": "C005", "title": "IELTS Preparation", "level": "B2", "type": "exam_prep", "modules": 15, "description": "Comprehensive IELTS preparation across all four skills."},
    {"course_id": "C006", "title": "Pronunciation Workshop", "level": "A2", "type": "pronunciation", "modules": 8, "description": "Sound patterns, stress, and intonation practice."},
    {"course_id": "C007", "title": "Idioms & Expressions", "level": "B2", "type": "vocabulary", "modules": 10, "description": "Common English idioms, phrasal verbs, and collocations."},
    {"course_id": "C008", "title": "Travel English", "level": "A2", "type": "conversation", "modules": 6, "description": "Essential phrases and situations for travelling abroad."},
    {"course_id": "C009", "title": "Advanced Grammar", "level": "C1", "type": "grammar", "modules": 10, "description": "Complex structures, conditionals, and reported speech."},
    {"course_id": "C010", "title": "News & Current Affairs", "level": "B2", "type": "reading", "modules": 12, "description": "Reading comprehension through real news articles."},
    {"course_id": "C011", "title": "TOEFL Preparation", "level": "B2", "type": "exam_prep", "modules": 14, "description": "Targeted practice for all TOEFL sections."},
    {"course_id": "C012", "title": "Creative Writing", "level": "C1", "type": "writing", "modules": 8, "description": "Fiction, poetry, and narrative techniques in English."},
]

# Student enrollments — links students to courses they're taking
ENROLLMENTS = [
    {"student_id": "S001", "course_id": "C001", "progress_percent": 75, "status": "active"},
    {"student_id": "S001", "course_id": "C002", "progress_percent": 30, "status": "active"},
    {"student_id": "S002", "course_id": "C003", "progress_percent": 50, "status": "active"},
    {"student_id": "S002", "course_id": "C005", "progress_percent": 10, "status": "paused"},
    {"student_id": "S003", "course_id": "C005", "progress_percent": 60, "status": "active"},
    {"student_id": "S004", "course_id": "C005", "progress_percent": 20, "status": "active"},
    {"student_id": "S004", "course_id": "C010", "progress_percent": 45, "status": "active"},
    {"student_id": "S005", "course_id": "C006", "progress_percent": 15, "status": "active"},
    {"student_id": "S005", "course_id": "C008", "progress_percent": 0, "status": "enrolled"},
]
```

- [ ] **Step 6: Create support_requests.py**

`projects/06-multi-department-support/data/support_requests.py`:
```python
"""Sample support requests covering all routing patterns.

Each request includes the text, sender metadata, and expected routing
for use in tests and the Streamlit demo.
"""

SAMPLE_REQUESTS = [
    {
        "text": "I can't log in to my account. I've tried resetting my password but the reset email never arrives.",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "high"},
        "expected_departments": ["tech_support"],
        "pattern": "single",
    },
    {
        "text": "Can I get a refund for the grammar lesson I had last Tuesday? The tutor didn't show up.",
        "metadata": {"sender_type": "student", "student_id": "S001", "priority": "medium"},
        "expected_departments": ["billing"],
        "pattern": "single",
    },
    {
        "text": "I need to reschedule my Business English lesson that's coming up tomorrow. Can we move it to next week?",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "medium"},
        "expected_departments": ["scheduling"],
        "pattern": "single",
    },
    {
        "text": "What B2 materials do you have for business English? I'm looking for something focused on presentations and meetings.",
        "metadata": {"sender_type": "student", "student_id": "S003", "priority": "low"},
        "expected_departments": ["content"],
        "pattern": "single",
    },
    {
        "text": "I want to cancel my Friday vocabulary lesson and get a refund for it.",
        "metadata": {"sender_type": "student", "student_id": "S005", "priority": "medium"},
        "expected_departments": ["billing", "scheduling"],
        "pattern": "parallel",
    },
    {
        "text": "I was charged twice for my IELTS prep lesson and now I can't access my lesson recordings from that session.",
        "metadata": {"sender_type": "student", "student_id": "S003", "priority": "high"},
        "expected_departments": ["billing", "tech_support"],
        "pattern": "parallel",
    },
    {
        "text": "I want to change tutors.",
        "metadata": {"sender_type": "student", "student_id": "S004", "priority": "low"},
        "expected_departments": [],
        "pattern": "clarification",
    },
    {
        "text": "My lesson was cancelled but I still got charged, and now I can't book a new one because the system shows an error.",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "high"},
        "expected_departments": ["billing", "scheduling", "tech_support"],
        "pattern": "parallel",
    },
]
```

- [ ] **Step 7: Verify data modules load correctly**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -c "
from data.invoices import INVOICES
from data.system_status import SERVICES
from data.accounts import ACCOUNTS
from data.lessons import LESSONS
from data.content_library import COURSES, ENROLLMENTS
from data.support_requests import SAMPLE_REQUESTS
print(f'Invoices: {len(INVOICES)}, Services: {len(SERVICES)}, Accounts: {len(ACCOUNTS)}')
print(f'Lessons: {len(LESSONS)}, Courses: {len(COURSES)}, Enrollments: {len(ENROLLMENTS)}')
print(f'Sample requests: {len(SAMPLE_REQUESTS)}')
print('All data modules loaded successfully.')
"
```

Expected: All counts print correctly, no import errors.

- [ ] **Step 8: Commit**

```bash
git add projects/06-multi-department-support/data/
git commit -m "feat(p6): add mock data modules for all four departments"
```

---

## Task 3: State Schema & Models

**Files:**
- Create: `projects/06-multi-department-support/models.py`
- Create: `projects/06-multi-department-support/tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

`projects/06-multi-department-support/tests/test_models.py`:
```python
"""Tests for state schema and model definitions."""

import operator
import pytest

from models import DepartmentResult, SupportState


class TestDepartmentResult:
    """Verify DepartmentResult TypedDict construction."""

    def test_basic_result(self):
        result = DepartmentResult(
            department="billing",
            response="Your invoice INV-001 shows a payment of $45.",
            resolved=True,
            escalation=None,
        )
        assert result["department"] == "billing"
        assert result["resolved"] is True
        assert result["escalation"] is None

    def test_result_with_escalation(self):
        result = DepartmentResult(
            department="billing",
            response="I need scheduling info to process this refund.",
            resolved=False,
            escalation={"target": "scheduling", "context": "Need lesson L004 cancellation details"},
        )
        assert result["resolved"] is False
        assert result["escalation"]["target"] == "scheduling"


class TestSupportState:
    """Verify SupportState schema and reducer behavior."""

    def test_initial_state_construction(self):
        state = SupportState(
            request="I need help",
            request_metadata={"sender_type": "student", "student_id": "S001"},
            classification={},
            department_results=[],
            escalation_queue=[],
            clarification_needed=None,
            user_clarification=None,
            final_response="",
            resolution_status="",
        )
        assert state["request"] == "I need help"
        assert state["department_results"] == []

    def test_department_results_has_add_reducer(self):
        """The department_results field must use operator.add for Send to work."""
        # Check that the annotation metadata includes the add reducer
        hints = SupportState.__annotations__
        # Annotated types store metadata — verify it's set up for accumulation
        from typing import get_type_hints, get_args
        full_hints = get_type_hints(SupportState, include_extras=True)
        dept_type = full_hints["department_results"]
        args = get_args(dept_type)
        # args[1] should be operator.add
        assert args[1] is operator.add, "department_results must use operator.add reducer"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

`projects/06-multi-department-support/models.py`:
```python
"""State schema and model definitions for the multi-department support system.

SupportState is the shared state for the entire graph. All nodes read from
and write to this state. The department_results field uses an operator.add
reducer so that parallel sub-agents (dispatched via Send) can each append
their result independently.

DepartmentResult is the structured output from each sub-agent node.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class DepartmentResult(TypedDict):
    """Structured result from a department sub-agent.

    Each sub-agent returns one of these, appended to department_results.
    If the sub-agent can't fully resolve the request, it sets resolved=False
    and populates escalation with the target department and context.
    """

    department: str           # "billing" | "tech_support" | "scheduling" | "content"
    response: str             # The sub-agent's response text
    resolved: bool            # Whether the sub-agent fully handled its part
    escalation: dict | None   # If not resolved: {"target": "<dept>", "context": "..."}


class SupportState(TypedDict):
    """Shared state for the multi-department support graph.

    Fields are grouped by lifecycle stage:
    - Input: set at invocation
    - Supervisor analysis: set by supervisor_router
    - Sub-agent results: appended by each sub-agent (reducer: operator.add)
    - Escalation: managed by supervisor_aggregator
    - Conversation: used for hybrid clarification flow
    - Output: set by compose_response
    """

    # --- Input ---
    request: str                                            # User's support request text
    request_metadata: dict                                  # sender_type, student_id, priority

    # --- Supervisor analysis ---
    classification: dict                                    # departments, complexity, summary

    # --- Sub-agent results (reducer: append for parallel Send) ---
    department_results: Annotated[list[DepartmentResult], operator.add]

    # --- Escalation tracking ---
    escalation_queue: list[dict]                            # Pending cross-dept escalations

    # --- Conversation (hybrid clarification) ---
    clarification_needed: str | None                        # Question to ask the user
    user_clarification: str | None                          # User's response

    # --- Final output ---
    final_response: str                                     # Unified response to the user
    resolution_status: str                                  # "resolved" | "partial" | "escalated_to_human"


# Valid department names — used for validation in routing
DEPARTMENTS = {"billing", "tech_support", "scheduling", "content"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_models.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/06-multi-department-support/models.py projects/06-multi-department-support/tests/test_models.py
git commit -m "feat(p6): add state schema with DepartmentResult and SupportState"
```

---

## Task 4: Tool Functions

**Files:**
- Create: `projects/06-multi-department-support/tools.py`
- Create: `projects/06-multi-department-support/tests/test_tools.py`

- [ ] **Step 1: Write failing tests for tools**

`projects/06-multi-department-support/tests/test_tools.py`:
```python
"""Tests for department tool functions against mock data."""

import pytest

from tools import (
    lookup_invoice,
    check_refund_status,
    check_system_status,
    lookup_user_account,
    check_lesson_schedule,
    reschedule_lesson,
    search_content_library,
    check_enrollment,
)


class TestBillingTools:
    def test_lookup_invoice_found(self):
        result = lookup_invoice.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(inv["student_id"] == "S001" for inv in result)

    def test_lookup_invoice_not_found(self):
        result = lookup_invoice.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_check_refund_status_found(self):
        result = check_refund_status.invoke({"invoice_id": "INV-004"})
        assert isinstance(result, dict)
        assert result["invoice_id"] == "INV-004"
        assert result["status"] == "refunded"

    def test_check_refund_status_not_found(self):
        result = check_refund_status.invoke({"invoice_id": "INV-999"})
        assert isinstance(result, str)
        assert "not found" in result.lower()


class TestTechSupportTools:
    def test_check_system_status_known(self):
        result = check_system_status.invoke({"service": "chat_system"})
        assert isinstance(result, dict)
        assert result["status"] == "degraded"

    def test_check_system_status_unknown(self):
        result = check_system_status.invoke({"service": "nonexistent"})
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_lookup_user_account_found(self):
        result = lookup_user_account.invoke({"email": "maria.garcia@email.com"})
        assert isinstance(result, dict)
        assert result["student_id"] == "S001"

    def test_lookup_user_account_not_found(self):
        result = lookup_user_account.invoke({"email": "nobody@email.com"})
        assert isinstance(result, str)
        assert "not found" in result.lower()


class TestSchedulingTools:
    def test_check_lesson_schedule(self):
        result = check_lesson_schedule.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(l["student_id"] == "S001" for l in result)

    def test_check_lesson_schedule_no_lessons(self):
        result = check_lesson_schedule.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_reschedule_lesson_success(self):
        result = reschedule_lesson.invoke({"lesson_id": "L003", "new_date": "2026-04-05"})
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["new_date"] == "2026-04-05"

    def test_reschedule_lesson_not_found(self):
        result = reschedule_lesson.invoke({"lesson_id": "L999", "new_date": "2026-04-05"})
        assert isinstance(result, dict)
        assert result["success"] is False


class TestContentTools:
    def test_search_content_library_with_level(self):
        result = search_content_library.invoke({"query": "business", "level": "B2"})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_search_content_library_no_level(self):
        result = search_content_library.invoke({"query": "grammar"})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_search_content_library_no_results(self):
        result = search_content_library.invoke({"query": "quantum physics", "level": "C2"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_check_enrollment_found(self):
        result = check_enrollment.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(e["student_id"] == "S001" for e in result)

    def test_check_enrollment_not_found(self):
        result = check_enrollment.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Implement tools.py**

`projects/06-multi-department-support/tools.py`:
```python
"""Department tool functions for the multi-department support system.

Each department has 2 tools that sub-agents can call to look up information
or perform actions. Tools use @tool for LangChain integration and @traceable
for LangSmith observability.

Tools are stateless functions that operate on mock data. They return structured
data on success or a descriptive error string on failure.
"""

from langchain_core.tools import tool
from langsmith import traceable

from data.invoices import INVOICES
from data.system_status import SERVICES
from data.accounts import ACCOUNTS
from data.lessons import LESSONS
from data.content_library import COURSES, ENROLLMENTS

_TAGS = ["p6-multi-department-support"]


# --- Billing tools ---

@tool
@traceable(name="lookup_invoice", run_type="tool", tags=_TAGS)
def lookup_invoice(student_id: str) -> list[dict]:
    """Look up all invoices for a student by their student ID.

    Returns a list of invoice records with id, amount, status, date, and description.
    Returns an empty list if no invoices are found.
    """
    return [inv for inv in INVOICES if inv["student_id"] == student_id]


@tool
@traceable(name="check_refund_status", run_type="tool", tags=_TAGS)
def check_refund_status(invoice_id: str) -> dict | str:
    """Check the refund status of a specific invoice.

    Returns the invoice record if found, or an error message if not found.
    """
    for inv in INVOICES:
        if inv["invoice_id"] == invoice_id:
            return inv
    return f"Invoice {invoice_id} not found."


# --- Tech support tools ---

@tool
@traceable(name="check_system_status", run_type="tool", tags=_TAGS)
def check_system_status(service: str) -> dict | str:
    """Check the current status of a platform service.

    Valid services: video_platform, chat_system, payment_gateway, content_cdn, auth_service.
    Returns service health info or an error if the service name is not found.
    """
    if service in SERVICES:
        return SERVICES[service]
    return f"Service '{service}' not found. Valid services: {', '.join(SERVICES.keys())}"


@tool
@traceable(name="lookup_user_account", run_type="tool", tags=_TAGS)
def lookup_user_account(email: str) -> dict | str:
    """Look up a student account by email address.

    Returns account details including plan, timezone, and known issues.
    Returns an error message if the account is not found.
    """
    for account in ACCOUNTS.values():
        if account["email"] == email:
            return account
    return f"No account found for email '{email}'."


# --- Scheduling tools ---

@tool
@traceable(name="check_lesson_schedule", run_type="tool", tags=_TAGS)
def check_lesson_schedule(student_id: str) -> list[dict]:
    """Get all lessons (past and upcoming) for a student.

    Returns a list of lesson records with id, tutor, subject, date, time, and status.
    Returns an empty list if no lessons are found.
    """
    return [lesson for lesson in LESSONS if lesson["student_id"] == student_id]


@tool
@traceable(name="reschedule_lesson", run_type="tool", tags=_TAGS)
def reschedule_lesson(lesson_id: str, new_date: str) -> dict:
    """Reschedule a lesson to a new date.

    Returns a dict with success status and details. Only scheduled (not completed
    or cancelled) lessons can be rescheduled.
    """
    for lesson in LESSONS:
        if lesson["lesson_id"] == lesson_id:
            if lesson["status"] != "scheduled":
                return {
                    "success": False,
                    "reason": f"Cannot reschedule — lesson status is '{lesson['status']}'.",
                }
            return {
                "success": True,
                "lesson_id": lesson_id,
                "old_date": lesson["date"],
                "new_date": new_date,
                "message": f"Lesson {lesson_id} rescheduled to {new_date}.",
            }
    return {"success": False, "reason": f"Lesson {lesson_id} not found."}


# --- Content tools ---

@tool
@traceable(name="search_content_library", run_type="tool", tags=_TAGS)
def search_content_library(query: str, level: str | None = None) -> list[dict]:
    """Search the course catalog by keyword and optional CEFR level.

    The query is matched against course title, type, and description (case-insensitive).
    Level filters by exact CEFR level (A2, B1, B2, C1).
    Returns matching courses or an empty list.
    """
    query_lower = query.lower()
    results = []
    for course in COURSES:
        text = f"{course['title']} {course['type']} {course['description']}".lower()
        if query_lower in text:
            if level and course["level"] != level:
                continue
            results.append(course)
    return results


@tool
@traceable(name="check_enrollment", run_type="tool", tags=_TAGS)
def check_enrollment(student_id: str) -> list[dict]:
    """Check which courses a student is enrolled in.

    Returns enrollment records with course_id, progress, and status.
    Returns an empty list if the student has no enrollments.
    """
    return [e for e in ENROLLMENTS if e["student_id"] == student_id]


# --- Tool groups for binding to sub-agent LLMs ---

BILLING_TOOLS = [lookup_invoice, check_refund_status]
TECH_SUPPORT_TOOLS = [check_system_status, lookup_user_account]
SCHEDULING_TOOLS = [check_lesson_schedule, reschedule_lesson]
CONTENT_TOOLS = [search_content_library, check_enrollment]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_tools.py -v
```

Expected: All 18 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/06-multi-department-support/tools.py projects/06-multi-department-support/tests/test_tools.py
git commit -m "feat(p6): add 8 department tool functions with tests"
```

---

## Task 5: Prompt Templates

**Files:**
- Create: `projects/06-multi-department-support/prompts.py`

- [ ] **Step 1: Create prompts.py**

`projects/06-multi-department-support/prompts.py`:
```python
"""Prompt templates for the multi-department support system.

Each prompt serves a specific node in the graph:
- SUPERVISOR_CLASSIFICATION_PROMPT: Classifies requests and decides routing
- Department prompts (BILLING/TECH_SUPPORT/SCHEDULING/CONTENT): Guide sub-agents
- COMPOSE_RESPONSE_PROMPT: Merges multi-department results into a unified reply
"""

from langchain_core.prompts import ChatPromptTemplate

# --- Supervisor classification ---

SUPERVISOR_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a support request classifier for LinguaFlow, an English tutoring platform.

Analyze the user's support request and classify it into one or more departments.

Available departments:
- billing: Payment issues, refunds, invoices, charges, pricing
- tech_support: Login problems, platform bugs, system errors, account access issues
- scheduling: Lesson scheduling, rescheduling, cancellations, availability
- content: Course materials, curriculum questions, content recommendations, enrollments

Rules:
1. If the request clearly maps to one department, return just that department.
2. If the request spans multiple departments, return all relevant departments.
3. If the request is too vague to classify (e.g., "I want to change tutors" — change scheduling? change preferences?), set needs_clarification to true and write a specific clarification question.
4. Err on the side of including a department rather than missing one.

Respond with ONLY valid JSON (no markdown, no explanation):
{{
    "departments": ["billing", "scheduling"],
    "needs_clarification": false,
    "clarification_question": null,
    "summary": "Brief summary of what the user needs",
    "complexity": "single" or "multi"
}}"""),
    ("human", """Support request: {request}

Sender: {sender_type} (Student ID: {student_id})
Priority: {priority}
{clarification_context}"""),
])


# --- Department sub-agent prompts ---

BILLING_PROMPT = """You are the billing support agent for LinguaFlow, an English tutoring platform.

You handle: payment issues, refunds, invoice inquiries, charge disputes, and pricing questions.

You have access to these tools:
- lookup_invoice(student_id): Find all invoices for a student
- check_refund_status(invoice_id): Check status of a specific invoice

Instructions:
1. Use your tools to look up relevant billing information.
2. Provide a clear, helpful response based on what you find.
3. If you need information from another department (e.g., lesson cancellation details from scheduling), set escalation in your response — do NOT try to guess.
4. Be empathetic and professional.

Student ID: {student_id}
Support request: {request}
{escalation_context}"""


TECH_SUPPORT_PROMPT = """You are the tech support agent for LinguaFlow, an English tutoring platform.

You handle: login issues, platform bugs, browser compatibility, account access, and system errors.

You have access to these tools:
- check_system_status(service): Check health of a platform service (video_platform, chat_system, payment_gateway, content_cdn, auth_service)
- lookup_user_account(email): Look up student account details by email

Instructions:
1. Use your tools to diagnose the issue.
2. Check system status for relevant services.
3. Look up the user's account if their email is available.
4. Provide clear troubleshooting steps or status updates.
5. If the issue requires another department, set escalation — do NOT try to resolve it yourself.

Student ID: {student_id}
Support request: {request}
{escalation_context}"""


SCHEDULING_PROMPT = """You are the scheduling support agent for LinguaFlow, an English tutoring platform.

You handle: lesson scheduling, rescheduling, cancellations, and availability checks.

You have access to these tools:
- check_lesson_schedule(student_id): Get all lessons (past and upcoming) for a student
- reschedule_lesson(lesson_id, new_date): Reschedule a lesson to a new date

Instructions:
1. Use your tools to look up the student's lesson schedule.
2. For rescheduling, find the specific lesson and use the reschedule tool.
3. For cancellations, note the lesson details and confirm the action.
4. If the request also involves billing (e.g., refund for cancelled lesson), set escalation to billing.

Student ID: {student_id}
Support request: {request}
{escalation_context}"""


CONTENT_PROMPT = """You are the content support agent for LinguaFlow, an English tutoring platform.

You handle: course recommendations, material access, curriculum questions, and enrollment inquiries.

You have access to these tools:
- search_content_library(query, level): Search course catalog by keyword and CEFR level
- check_enrollment(student_id): Check which courses a student is enrolled in

Instructions:
1. Use your tools to find relevant courses or check enrollments.
2. Make personalized recommendations based on the student's level and interests.
3. If the student needs scheduling or billing help, set escalation to the right department.

Student ID: {student_id}
Support request: {request}
{escalation_context}"""


# --- Response composition ---

COMPOSE_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a support coordinator for LinguaFlow. Your job is to take responses
from multiple department agents and compose a single, coherent reply for the user.

Rules:
1. Merge all department responses into one natural, conversational message.
2. Do NOT show which department handled which part — the user should see one unified response.
3. If any department couldn't fully resolve their part, mention what's still pending.
4. Be professional, empathetic, and concise.
5. End with a summary of actions taken and any next steps."""),
    ("human", """Original request: {request}

Department responses:
{department_responses}

Compose a single unified response for the user."""),
])
```

- [ ] **Step 2: Verify prompts load**

```bash
cd "projects/06-multi-department-support"
python -c "
from prompts import (
    SUPERVISOR_CLASSIFICATION_PROMPT,
    BILLING_PROMPT, TECH_SUPPORT_PROMPT,
    SCHEDULING_PROMPT, CONTENT_PROMPT,
    COMPOSE_RESPONSE_PROMPT,
)
print('All 6 prompts loaded successfully.')
# Test classification prompt renders
msg = SUPERVISOR_CLASSIFICATION_PROMPT.invoke({
    'request': 'test', 'sender_type': 'student',
    'student_id': 'S001', 'priority': 'low',
    'clarification_context': '',
})
print(f'Classification prompt renders: {len(msg.messages)} messages')
"
```

Expected: Prints success messages, no errors.

- [ ] **Step 3: Commit**

```bash
git add projects/06-multi-department-support/prompts.py
git commit -m "feat(p6): add prompt templates for supervisor and all departments"
```

---

## Task 6: Node Functions

**Files:**
- Create: `projects/06-multi-department-support/nodes.py`
- Create: `projects/06-multi-department-support/tests/test_nodes.py`

- [ ] **Step 1: Write failing tests for nodes**

`projects/06-multi-department-support/tests/test_nodes.py`:
```python
"""Tests for node functions.

Tests use mocked LLM responses to verify node logic without API calls.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from models import SupportState, DepartmentResult


def _make_state(**overrides) -> SupportState:
    """Helper to create a SupportState with sensible defaults."""
    defaults = {
        "request": "I need help with billing",
        "request_metadata": {"sender_type": "student", "student_id": "S001", "priority": "medium"},
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }
    defaults.update(overrides)
    return defaults


class TestSupervisorRouter:
    """Test the supervisor_router node."""

    @patch("nodes._classification_model")
    def test_single_department_classification(self, mock_model):
        from nodes import supervisor_router

        # Mock LLM returning a valid classification
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "departments": ["billing"],
            "needs_clarification": False,
            "clarification_question": None,
            "summary": "Billing inquiry",
            "complexity": "single",
        })
        mock_model.invoke.return_value = mock_response

        state = _make_state(request="What's the status of my refund?")
        result = supervisor_router(state)

        assert "classification" in result
        assert result["classification"]["departments"] == ["billing"]
        assert result["classification"]["needs_clarification"] is False

    @patch("nodes._classification_model")
    def test_multi_department_classification(self, mock_model):
        from nodes import supervisor_router

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "departments": ["billing", "scheduling"],
            "needs_clarification": False,
            "clarification_question": None,
            "summary": "Cancel and refund",
            "complexity": "multi",
        })
        mock_model.invoke.return_value = mock_response

        state = _make_state(request="Cancel my lesson and refund me")
        result = supervisor_router(state)

        assert len(result["classification"]["departments"]) == 2

    @patch("nodes._classification_model")
    def test_clarification_needed(self, mock_model):
        from nodes import supervisor_router

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "departments": [],
            "needs_clarification": True,
            "clarification_question": "Could you tell me more about what you'd like to change?",
            "summary": "Ambiguous request",
            "complexity": "single",
        })
        mock_model.invoke.return_value = mock_response

        state = _make_state(request="I want to change tutors")
        result = supervisor_router(state)

        assert result["classification"]["needs_clarification"] is True
        assert result["clarification_needed"] == "Could you tell me more about what you'd like to change?"


class TestSupervisorAggregator:
    """Test the supervisor_aggregator node."""

    def test_all_resolved(self):
        from nodes import supervisor_aggregator

        state = _make_state(
            department_results=[
                DepartmentResult(department="billing", response="Refund processed.", resolved=True, escalation=None),
            ],
            escalation_queue=[],
        )
        result = supervisor_aggregator(state)

        assert result["escalation_queue"] == []

    def test_escalation_detected(self):
        from nodes import supervisor_aggregator

        state = _make_state(
            department_results=[
                DepartmentResult(
                    department="billing",
                    response="Need lesson details.",
                    resolved=False,
                    escalation={"target": "scheduling", "context": "Need L004 cancellation info"},
                ),
            ],
            escalation_queue=[],
        )
        result = supervisor_aggregator(state)

        assert len(result["escalation_queue"]) == 1
        assert result["escalation_queue"][0]["target"] == "scheduling"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_nodes.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement nodes.py**

`projects/06-multi-department-support/nodes.py`:
```python
"""Node functions for the multi-department support graph.

Nodes:
- supervisor_router: Classifies requests, decides routing
- billing_agent, tech_support_agent, scheduling_agent, content_agent: Sub-agents
- supervisor_aggregator: Collects results, processes escalations
- compose_response: Merges department responses into unified reply
- ask_clarification: Interrupts for user clarification

Each sub-agent runs an internal tool-calling loop (max 3 rounds) to gather
information before producing a DepartmentResult.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langgraph.types import interrupt

from models import DepartmentResult, SupportState, DEPARTMENTS
from prompts import (
    SUPERVISOR_CLASSIFICATION_PROMPT,
    BILLING_PROMPT,
    TECH_SUPPORT_PROMPT,
    SCHEDULING_PROMPT,
    CONTENT_PROMPT,
    COMPOSE_RESPONSE_PROMPT,
)
from tools import (
    BILLING_TOOLS,
    TECH_SUPPORT_TOOLS,
    SCHEDULING_TOOLS,
    CONTENT_TOOLS,
)

load_dotenv()

_TAGS = ["p6-multi-department-support"]
_MAX_TOOL_ROUNDS = 3

# --- LLM instances ---
# Classification model (no tools)
_classification_model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Department models (with tools bound)
_billing_model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(BILLING_TOOLS)
_tech_support_model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(TECH_SUPPORT_TOOLS)
_scheduling_model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(SCHEDULING_TOOLS)
_content_model = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(CONTENT_TOOLS)

# Compose model (no tools)
_compose_model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# Tool lookup for executing tool calls
_ALL_TOOLS = {t.name: t for t in BILLING_TOOLS + TECH_SUPPORT_TOOLS + SCHEDULING_TOOLS + CONTENT_TOOLS}


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present."""
    cleaned = text.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _run_agent_loop(
    model, tools_map: dict, system_prompt: str, state: SupportState, escalation_context: str = ""
) -> tuple[str, bool, dict | None]:
    """Run an internal tool-calling loop for a sub-agent.

    Returns (response_text, resolved, escalation_or_none).
    The loop calls the LLM, executes any tool calls, feeds results back,
    and repeats until the LLM responds without tool calls or max rounds hit.
    """
    student_id = state["request_metadata"].get("student_id", "unknown")
    formatted_prompt = system_prompt.format(
        student_id=student_id,
        request=state["request"],
        escalation_context=escalation_context,
    )
    messages = [SystemMessage(content=formatted_prompt)]

    for _ in range(_MAX_TOOL_ROUNDS):
        response: AIMessage = model.invoke(messages, config={"tags": _TAGS})
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute each tool call and append results
        for tool_call in response.tool_calls:
            tool_fn = tools_map.get(tool_call["name"])
            if tool_fn:
                tool_result = tool_fn.invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
                    tool_call_id=tool_call["id"],
                ))
            else:
                messages.append(ToolMessage(
                    content=f"Tool '{tool_call['name']}' not available.",
                    tool_call_id=tool_call["id"],
                ))

    # Extract final response text
    final_text = messages[-1].content if isinstance(messages[-1], AIMessage) else str(messages[-1].content)

    # Check if the agent indicated an escalation is needed
    # Convention: if the response contains "ESCALATE:" followed by JSON, parse it
    escalation = None
    resolved = True
    if "ESCALATE:" in final_text:
        try:
            esc_start = final_text.index("ESCALATE:") + len("ESCALATE:")
            esc_json = final_text[esc_start:].strip()
            escalation = _parse_json_response(esc_json)
            resolved = False
            # Remove escalation marker from response text
            final_text = final_text[:final_text.index("ESCALATE:")].strip()
        except (json.JSONDecodeError, ValueError):
            pass  # If parsing fails, treat as resolved

    return final_text, resolved, escalation


# --- Supervisor nodes ---

@traceable(name="supervisor_router", run_type="chain", tags=_TAGS)
def supervisor_router(state: SupportState) -> dict:
    """Classify the support request and decide routing.

    If the user previously provided clarification, include it in the context
    so the LLM can re-classify with the additional information.
    """
    metadata = state["request_metadata"]
    clarification_ctx = ""
    if state.get("user_clarification"):
        clarification_ctx = f"\nUser clarification: {state['user_clarification']}"

    chain = SUPERVISOR_CLASSIFICATION_PROMPT | _classification_model
    response = chain.invoke({
        "request": state["request"],
        "sender_type": metadata.get("sender_type", "unknown"),
        "student_id": metadata.get("student_id", "unknown"),
        "priority": metadata.get("priority", "medium"),
        "clarification_context": clarification_ctx,
    }, config={"tags": _TAGS})

    classification = _parse_json_response(response.content)

    updates: dict = {"classification": classification}

    # If clarification is needed, store the question
    if classification.get("needs_clarification"):
        updates["clarification_needed"] = classification.get("clarification_question", "Could you provide more details?")

    return updates


@traceable(name="supervisor_aggregator", run_type="chain", tags=_TAGS)
def supervisor_aggregator(state: SupportState) -> dict:
    """Collect department results and process any escalations.

    Scans department_results for unresolved items with escalation targets.
    Populates escalation_queue for the routing function to dispatch.
    """
    escalations = []
    for result in state["department_results"]:
        if not result["resolved"] and result.get("escalation"):
            esc = result["escalation"]
            # Only escalate to valid departments that haven't already responded
            already_handled = {r["department"] for r in state["department_results"]}
            if esc["target"] in DEPARTMENTS and esc["target"] not in already_handled:
                escalations.append(esc)

    return {"escalation_queue": escalations}


@traceable(name="compose_response", run_type="chain", tags=_TAGS)
def compose_response(state: SupportState) -> dict:
    """Synthesize all department results into a single coherent response."""
    # Format department responses for the prompt
    dept_responses = "\n\n".join(
        f"[{r['department'].replace('_', ' ').title()}]: {r['response']}"
        for r in state["department_results"]
    )

    chain = COMPOSE_RESPONSE_PROMPT | _compose_model
    response = chain.invoke({
        "request": state["request"],
        "department_responses": dept_responses,
    }, config={"tags": _TAGS})

    # Determine overall resolution status
    all_resolved = all(r["resolved"] for r in state["department_results"])
    status = "resolved" if all_resolved else "partial"

    return {
        "final_response": response.content,
        "resolution_status": status,
    }


@traceable(name="ask_clarification", run_type="chain", tags=_TAGS)
def ask_clarification(state: SupportState) -> dict:
    """Pause execution and ask the user for clarification.

    Uses interrupt() to pause the graph. The Streamlit adapter resumes
    with Command(resume=<user_response>) when the user provides an answer.
    """
    question = state.get("clarification_needed", "Could you provide more details about your request?")

    # interrupt() pauses here — returns the user's response when resumed
    user_response = interrupt({
        "type": "clarification",
        "question": question,
        "original_request": state["request"],
    })

    return {
        "user_clarification": user_response,
        "clarification_needed": None,  # Clear the flag
    }


# --- Department sub-agent nodes ---

@traceable(name="billing_agent", run_type="chain", tags=_TAGS)
def billing_agent(state: SupportState) -> dict:
    """Billing department sub-agent. Handles refunds, invoices, charges."""
    tools_map = {t.name: t for t in BILLING_TOOLS}
    escalation_ctx = ""
    for esc in state.get("escalation_queue", []):
        if esc.get("target") == "billing":
            escalation_ctx = f"\nEscalation context: {esc['context']}"

    response_text, resolved, escalation = _run_agent_loop(
        _billing_model, tools_map, BILLING_PROMPT, state, escalation_ctx
    )
    return {"department_results": [DepartmentResult(
        department="billing", response=response_text, resolved=resolved, escalation=escalation,
    )]}


@traceable(name="tech_support_agent", run_type="chain", tags=_TAGS)
def tech_support_agent(state: SupportState) -> dict:
    """Tech support sub-agent. Handles login, platform, and account issues."""
    tools_map = {t.name: t for t in TECH_SUPPORT_TOOLS}
    escalation_ctx = ""
    for esc in state.get("escalation_queue", []):
        if esc.get("target") == "tech_support":
            escalation_ctx = f"\nEscalation context: {esc['context']}"

    response_text, resolved, escalation = _run_agent_loop(
        _tech_support_model, tools_map, TECH_SUPPORT_PROMPT, state, escalation_ctx
    )
    return {"department_results": [DepartmentResult(
        department="tech_support", response=response_text, resolved=resolved, escalation=escalation,
    )]}


@traceable(name="scheduling_agent", run_type="chain", tags=_TAGS)
def scheduling_agent(state: SupportState) -> dict:
    """Scheduling sub-agent. Handles lesson scheduling and rescheduling."""
    tools_map = {t.name: t for t in SCHEDULING_TOOLS}
    escalation_ctx = ""
    for esc in state.get("escalation_queue", []):
        if esc.get("target") == "scheduling":
            escalation_ctx = f"\nEscalation context: {esc['context']}"

    response_text, resolved, escalation = _run_agent_loop(
        _scheduling_model, tools_map, SCHEDULING_PROMPT, state, escalation_ctx
    )
    return {"department_results": [DepartmentResult(
        department="scheduling", response=response_text, resolved=resolved, escalation=escalation,
    )]}


@traceable(name="content_agent", run_type="chain", tags=_TAGS)
def content_agent(state: SupportState) -> dict:
    """Content sub-agent. Handles course recommendations and enrollments."""
    tools_map = {t.name: t for t in CONTENT_TOOLS}
    escalation_ctx = ""
    for esc in state.get("escalation_queue", []):
        if esc.get("target") == "content":
            escalation_ctx = f"\nEscalation context: {esc['context']}"

    response_text, resolved, escalation = _run_agent_loop(
        _content_model, tools_map, CONTENT_PROMPT, state, escalation_ctx
    )
    return {"department_results": [DepartmentResult(
        department="content", response=response_text, resolved=resolved, escalation=escalation,
    )]}


# --- Routing functions ---

def route_from_supervisor(state: SupportState):
    """Route from supervisor_router to sub-agents, clarification, or end.

    Returns:
    - "ask_clarification" if clarification is needed
    - A single node name for single-department requests
    - A list of Send objects for parallel multi-department requests
    """
    from langgraph.types import Send

    classification = state["classification"]

    if classification.get("needs_clarification"):
        return "ask_clarification"

    departments = classification.get("departments", [])
    if not departments:
        return "ask_clarification"

    if len(departments) == 1:
        return departments[0] + "_agent"

    # Parallel dispatch via Send
    return [Send(dept + "_agent", state) for dept in departments]


def route_from_aggregator(state: SupportState):
    """Route from supervisor_aggregator based on escalation state.

    If there are pending escalations, dispatch to target departments via Send.
    Otherwise, proceed to compose the final response.
    """
    from langgraph.types import Send

    escalations = state.get("escalation_queue", [])
    if escalations:
        # Fan out to escalation targets — results will flow back to aggregator
        return [Send(esc["target"] + "_agent", state) for esc in escalations]
    return "compose_response"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_nodes.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/06-multi-department-support/nodes.py projects/06-multi-department-support/tests/test_nodes.py
git commit -m "feat(p6): add node functions with supervisor, sub-agents, and routing"
```

---

## Task 7: Supervisor Tests

**Files:**
- Create: `projects/06-multi-department-support/tests/test_supervisor.py`

- [ ] **Step 1: Write supervisor-specific tests**

`projects/06-multi-department-support/tests/test_supervisor.py`:
```python
"""Tests for supervisor routing and aggregation logic."""

import pytest
from langgraph.types import Send

from models import DepartmentResult, SupportState
from nodes import route_from_supervisor, route_from_aggregator


def _make_state(**overrides) -> dict:
    defaults = {
        "request": "test request",
        "request_metadata": {"sender_type": "student", "student_id": "S001", "priority": "medium"},
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }
    defaults.update(overrides)
    return defaults


class TestRouteFromSupervisor:
    """Test the conditional routing function after supervisor_router."""

    def test_single_department(self):
        state = _make_state(classification={
            "departments": ["billing"],
            "needs_clarification": False,
        })
        result = route_from_supervisor(state)
        assert result == "billing_agent"

    def test_multi_department_returns_send_list(self):
        state = _make_state(classification={
            "departments": ["billing", "scheduling"],
            "needs_clarification": False,
        })
        result = route_from_supervisor(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)

    def test_clarification_needed(self):
        state = _make_state(classification={
            "departments": [],
            "needs_clarification": True,
            "clarification_question": "What do you mean?",
        })
        result = route_from_supervisor(state)
        assert result == "ask_clarification"

    def test_empty_departments_triggers_clarification(self):
        state = _make_state(classification={
            "departments": [],
            "needs_clarification": False,
        })
        result = route_from_supervisor(state)
        assert result == "ask_clarification"

    def test_three_departments(self):
        state = _make_state(classification={
            "departments": ["billing", "scheduling", "tech_support"],
            "needs_clarification": False,
        })
        result = route_from_supervisor(state)
        assert isinstance(result, list)
        assert len(result) == 3


class TestRouteFromAggregator:
    """Test the conditional routing function after supervisor_aggregator."""

    def test_no_escalations_goes_to_compose(self):
        state = _make_state(escalation_queue=[])
        result = route_from_aggregator(state)
        assert result == "compose_response"

    def test_has_escalations_returns_send_list(self):
        state = _make_state(escalation_queue=[
            {"target": "scheduling", "context": "Need lesson info"},
        ])
        result = route_from_aggregator(state)
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(s, Send) for s in result)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_supervisor.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add projects/06-multi-department-support/tests/test_supervisor.py
git commit -m "test(p6): add supervisor routing and aggregation unit tests"
```

---

## Task 8: Graph Assembly

**Files:**
- Create: `projects/06-multi-department-support/graph.py`
- Create: `projects/06-multi-department-support/tests/conftest.py`

- [ ] **Step 1: Create conftest.py with fixtures**

`projects/06-multi-department-support/tests/conftest.py`:
```python
"""Shared test fixtures for the multi-department support system."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph
from data.support_requests import SAMPLE_REQUESTS


@pytest.fixture
def graph_with_memory():
    """Compiled graph with InMemorySaver for interrupt/persistence tests."""
    return build_graph(checkpointer=InMemorySaver())


@pytest.fixture
def sample_billing_state():
    """Initial state for a single-department billing request."""
    req = SAMPLE_REQUESTS[1]  # "Can I get a refund..."
    return {
        "request": req["text"],
        "request_metadata": req["metadata"],
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }


@pytest.fixture
def sample_parallel_state():
    """Initial state for a multi-department parallel request."""
    req = SAMPLE_REQUESTS[4]  # "Cancel my Friday lesson and get a refund"
    return {
        "request": req["text"],
        "request_metadata": req["metadata"],
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }


@pytest.fixture
def sample_clarification_state():
    """Initial state for an ambiguous request requiring clarification."""
    req = SAMPLE_REQUESTS[6]  # "I want to change tutors"
    return {
        "request": req["text"],
        "request_metadata": req["metadata"],
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }
```

- [ ] **Step 2: Implement graph.py**

`projects/06-multi-department-support/graph.py`:
```python
"""Graph assembly for the multi-department support system.

Builds a StateGraph with:
- supervisor_router: Classifies and routes requests
- 4 sub-agent nodes: billing, tech_support, scheduling, content
- supervisor_aggregator: Collects results and handles escalations
- handle_escalations: Re-dispatches via Send for cross-department escalation
- compose_response: Synthesizes unified reply
- ask_clarification: Interrupts for user input

The graph supports:
- Single-department routing (direct edge)
- Multi-department parallel dispatch (Send)
- Supervisor-mediated escalation (aggregator -> handle_escalations -> aggregator)
- Hybrid clarification (interrupt/resume)
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, Send

from models import SupportState
from nodes import (
    supervisor_router,
    billing_agent,
    tech_support_agent,
    scheduling_agent,
    content_agent,
    supervisor_aggregator,
    compose_response,
    ask_clarification,
    route_from_supervisor,
    route_from_aggregator,
)

# Retry policy for LLM nodes — handles transient API errors
_llm_retry = RetryPolicy(max_attempts=3)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the multi-department support graph.

    Args:
        checkpointer: State persistence backend. Defaults to InMemorySaver
                      (required for interrupt/resume to work).
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = StateGraph(SupportState)

    # --- Add nodes ---
    graph.add_node("supervisor_router", supervisor_router, retry_policy=_llm_retry)
    graph.add_node("billing_agent", billing_agent, retry_policy=_llm_retry)
    graph.add_node("tech_support_agent", tech_support_agent, retry_policy=_llm_retry)
    graph.add_node("scheduling_agent", scheduling_agent, retry_policy=_llm_retry)
    graph.add_node("content_agent", content_agent, retry_policy=_llm_retry)
    graph.add_node("supervisor_aggregator", supervisor_aggregator)
    graph.add_node("compose_response", compose_response, retry_policy=_llm_retry)
    graph.add_node("ask_clarification", ask_clarification)

    # --- Entry ---
    graph.add_edge(START, "supervisor_router")

    # --- Supervisor router → conditional dispatch ---
    # route_from_supervisor returns a node name (single dept), list of Send (multi),
    # or "ask_clarification" (ambiguous request)
    graph.add_conditional_edges(
        "supervisor_router",
        route_from_supervisor,
        ["billing_agent", "tech_support_agent", "scheduling_agent", "content_agent", "ask_clarification"],
    )

    # --- All sub-agents converge at aggregator ---
    graph.add_edge("billing_agent", "supervisor_aggregator")
    graph.add_edge("tech_support_agent", "supervisor_aggregator")
    graph.add_edge("scheduling_agent", "supervisor_aggregator")
    graph.add_edge("content_agent", "supervisor_aggregator")

    # --- Aggregator → conditional: escalate or compose ---
    # route_from_aggregator returns list of Send (escalations) or "compose_response"
    graph.add_conditional_edges(
        "supervisor_aggregator",
        route_from_aggregator,
        ["billing_agent", "tech_support_agent", "scheduling_agent", "content_agent", "compose_response"],
    )

    # --- Clarification → back to supervisor for re-classification ---
    graph.add_edge("ask_clarification", "supervisor_router")

    # --- Final response → end ---
    graph.add_edge("compose_response", END)

    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 3: Verify graph compiles**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -c "
from graph import build_graph
g = build_graph()
print('Graph compiled successfully.')
print('Nodes:', list(g.get_graph().nodes.keys()))
"
```

Expected: Graph compiles, prints node names.

- [ ] **Step 4: Commit**

```bash
git add projects/06-multi-department-support/graph.py projects/06-multi-department-support/tests/conftest.py
git commit -m "feat(p6): assemble graph with Send routing and escalation loop"
```

---

## Task 9: Integration Tests

**Files:**
- Create: `projects/06-multi-department-support/tests/test_graph.py`

- [ ] **Step 1: Write integration tests**

`projects/06-multi-department-support/tests/test_graph.py`:
```python
"""Integration tests for the multi-department support graph.

These tests hit the real LLM API and verify end-to-end flows.
"""

import pytest
from langgraph.types import Command

from data.support_requests import SAMPLE_REQUESTS


@pytest.mark.integration
class TestSingleDepartmentFlow:
    """Verify single-department requests route correctly and resolve."""

    def test_billing_request(self, graph_with_memory, sample_billing_state):
        config = {"configurable": {"thread_id": "test-billing-1"}, "tags": ["p6-multi-department-support"]}
        result = graph_with_memory.invoke(sample_billing_state, config=config)

        assert result["final_response"], "Should have a final response"
        assert result["resolution_status"] in ("resolved", "partial")
        assert len(result["department_results"]) >= 1
        assert any(r["department"] == "billing" for r in result["department_results"])

    def test_tech_support_request(self, graph_with_memory):
        req = SAMPLE_REQUESTS[0]  # "I can't log in..."
        state = {
            "request": req["text"],
            "request_metadata": req["metadata"],
            "classification": {},
            "department_results": [],
            "escalation_queue": [],
            "clarification_needed": None,
            "user_clarification": None,
            "final_response": "",
            "resolution_status": "",
        }
        config = {"configurable": {"thread_id": "test-tech-1"}, "tags": ["p6-multi-department-support"]}
        result = graph_with_memory.invoke(state, config=config)

        assert result["final_response"]
        assert any(r["department"] == "tech_support" for r in result["department_results"])


@pytest.mark.integration
class TestParallelDispatch:
    """Verify multi-department requests fan out via Send."""

    def test_billing_and_scheduling(self, graph_with_memory, sample_parallel_state):
        config = {"configurable": {"thread_id": "test-parallel-1"}, "tags": ["p6-multi-department-support"]}
        result = graph_with_memory.invoke(sample_parallel_state, config=config)

        assert result["final_response"]
        departments_seen = {r["department"] for r in result["department_results"]}
        # Should have at least 2 departments (billing and scheduling expected)
        assert len(departments_seen) >= 2, f"Expected >=2 departments, got {departments_seen}"

    def test_three_way_parallel(self, graph_with_memory):
        req = SAMPLE_REQUESTS[7]  # billing + scheduling + tech_support
        state = {
            "request": req["text"],
            "request_metadata": req["metadata"],
            "classification": {},
            "department_results": [],
            "escalation_queue": [],
            "clarification_needed": None,
            "user_clarification": None,
            "final_response": "",
            "resolution_status": "",
        }
        config = {"configurable": {"thread_id": "test-three-way-1"}, "tags": ["p6-multi-department-support"]}
        result = graph_with_memory.invoke(state, config=config)

        assert result["final_response"]
        departments_seen = {r["department"] for r in result["department_results"]}
        assert len(departments_seen) >= 2, f"Expected >=2 departments, got {departments_seen}"


@pytest.mark.integration
class TestClarificationFlow:
    """Verify ambiguous requests trigger interrupt and resume correctly."""

    def test_clarification_interrupt_and_resume(self, graph_with_memory, sample_clarification_state):
        config = {"configurable": {"thread_id": "test-clarify-1"}, "tags": ["p6-multi-department-support"]}

        # Step 1: Submit ambiguous request — should hit interrupt
        result = graph_with_memory.invoke(sample_clarification_state, config=config)

        # Check that we're paused at ask_clarification
        snapshot = graph_with_memory.get_state(config)
        assert snapshot.next, "Graph should be paused at a node"

        # Step 2: Resume with clarification
        result = graph_with_memory.invoke(
            Command(resume="I'd like to switch from my current grammar tutor to someone who specializes in conversation practice. Can you help me reschedule?"),
            config=config,
        )

        # After clarification, should route to appropriate department(s) and resolve
        assert result["final_response"], "Should produce a final response after clarification"
        assert result["resolution_status"] in ("resolved", "partial")
```

- [ ] **Step 2: Run integration tests**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -m pytest tests/test_graph.py -v -m integration
```

Expected: All 5 tests PASS (may take 30-60s due to LLM calls).

- [ ] **Step 3: Run full test suite**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/ -v
```

Expected: All tests PASS (unit + integration).

- [ ] **Step 4: Commit**

```bash
git add projects/06-multi-department-support/tests/test_graph.py
git commit -m "test(p6): add integration tests for single, parallel, and clarification flows"
```

---

## Task 10: LangSmith Evaluation

**Files:**
- Create: `projects/06-multi-department-support/evaluation.py`
- Create: `projects/06-multi-department-support/tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for evaluators**

`projects/06-multi-department-support/tests/test_evaluation.py`:
```python
"""Tests for LangSmith evaluator functions."""

import pytest
from unittest.mock import MagicMock

from evaluation import routing_accuracy_evaluator, response_quality_evaluator


class TestRoutingAccuracyEvaluator:
    def test_perfect_routing(self):
        run = MagicMock()
        run.outputs = {
            "classification": {"departments": ["billing"]},
        }
        example = MagicMock()
        example.inputs = {
            "expected_departments": ["billing"],
        }
        result = routing_accuracy_evaluator(run, example)
        assert result["key"] == "routing_accuracy"
        assert result["score"] == 1.0

    def test_partial_routing(self):
        run = MagicMock()
        run.outputs = {
            "classification": {"departments": ["billing"]},
        }
        example = MagicMock()
        example.inputs = {
            "expected_departments": ["billing", "scheduling"],
        }
        result = routing_accuracy_evaluator(run, example)
        assert result["key"] == "routing_accuracy"
        assert 0.0 < result["score"] < 1.0

    def test_wrong_routing(self):
        run = MagicMock()
        run.outputs = {
            "classification": {"departments": ["content"]},
        }
        example = MagicMock()
        example.inputs = {
            "expected_departments": ["billing"],
        }
        result = routing_accuracy_evaluator(run, example)
        assert result["key"] == "routing_accuracy"
        assert result["score"] == 0.0


class TestResponseQualityEvaluator:
    @pytest.mark.integration
    def test_returns_valid_score(self):
        run = MagicMock()
        run.outputs = {
            "final_response": "Your refund for invoice INV-004 has been processed.",
            "request": "Can I get a refund for my cancelled lesson?",
        }
        example = MagicMock()
        example.inputs = {
            "request": "Can I get a refund for my cancelled lesson?",
        }
        result = response_quality_evaluator(run, example)
        assert result["key"] == "response_quality"
        assert 0.0 <= result["score"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_evaluation.py -v -k "not integration"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Implement evaluation.py**

`projects/06-multi-department-support/evaluation.py`:
```python
"""LangSmith evaluation pipeline for the multi-department support system.

Two evaluators:
1. routing_accuracy: Deterministic — compares predicted departments to expected
2. response_quality: LLM-as-judge — evaluates coherence and completeness

Usage:
    python evaluation.py
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langsmith import traceable

load_dotenv()

_TAGS = ["p6-multi-department-support", "evaluation"]

# Evaluation model — used for LLM-as-judge
_eval_model = ChatAnthropic(model="claude-haiku-4-5-20251001")


def _parse_json_from_llm(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def routing_accuracy_evaluator(run, example) -> dict:
    """Deterministic evaluator: compare predicted departments to expected.

    Uses set intersection / union (Jaccard similarity) for scoring.
    Score = |predicted ∩ expected| / |predicted ∪ expected|
    """
    predicted = set(run.outputs.get("classification", {}).get("departments", []))
    expected = set(example.inputs.get("expected_departments", []))

    if not expected and not predicted:
        score = 1.0
    elif not expected or not predicted:
        score = 0.0
    else:
        intersection = predicted & expected
        union = predicted | expected
        score = len(intersection) / len(union)

    return {
        "key": "routing_accuracy",
        "score": score,
        "comment": f"Predicted: {sorted(predicted)}, Expected: {sorted(expected)}",
    }


def response_quality_evaluator(run, example) -> dict:
    """LLM-as-judge evaluator: assess response coherence and completeness.

    Asks the evaluation model to rate the response on a 0-1 scale.
    """
    request = run.outputs.get("request", example.inputs.get("request", ""))
    response = run.outputs.get("final_response", "")

    if not response:
        return {"key": "response_quality", "score": 0.0, "comment": "No response generated."}

    prompt = f"""Rate this support response on a scale of 0.0 to 1.0.

Original request: {request}

Response: {response}

Criteria:
- Does the response address the user's request?
- Is it coherent and well-structured?
- Is it professional and empathetic?
- Does it include specific actions taken or next steps?

Respond with ONLY valid JSON:
{{"score": 0.8, "reasoning": "Brief explanation"}}"""

    try:
        result = _eval_model.invoke(prompt, config={"tags": _TAGS})
        parsed = _parse_json_from_llm(result.content)
        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        return {
            "key": "response_quality",
            "score": score,
            "comment": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {"key": "response_quality", "score": 0.5, "comment": "Failed to parse evaluator response."}


if __name__ == "__main__":
    from langsmith import evaluate

    from graph import build_graph
    from data.support_requests import SAMPLE_REQUESTS

    DATASET_NAME = "p6-multi-department-support-routing"

    # Build the target function
    graph = build_graph()

    def target(inputs: dict) -> dict:
        """Run the graph and return outputs for evaluation."""
        state = {
            "request": inputs["text"],
            "request_metadata": inputs["metadata"],
            "classification": {},
            "department_results": [],
            "escalation_queue": [],
            "clarification_needed": None,
            "user_clarification": None,
            "final_response": "",
            "resolution_status": "",
        }
        config = {"tags": ["p6-multi-department-support", "evaluation"]}
        result = graph.invoke(state, config=config)
        return {
            "classification": result.get("classification", {}),
            "final_response": result.get("final_response", ""),
            "request": inputs["text"],
            "resolution_status": result.get("resolution_status", ""),
        }

    # Filter to non-clarification requests (they'd need interrupt handling)
    eval_requests = [r for r in SAMPLE_REQUESTS if r["pattern"] != "clarification"]

    print(f"Running evaluation on {len(eval_requests)} requests...")
    results = evaluate(
        target,
        data=eval_requests,
        evaluators=[routing_accuracy_evaluator, response_quality_evaluator],
        experiment_prefix="p6-multi-dept-support",
        metadata={"model": "claude-haiku-4-5-20251001", "version": "v1"},
    )
    print("Evaluation complete. Check LangSmith for results.")
```

- [ ] **Step 4: Run unit tests (non-integration)**

```bash
cd "projects/06-multi-department-support"
python -m pytest tests/test_evaluation.py -v -k "not integration"
```

Expected: 3 routing accuracy tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/06-multi-department-support/evaluation.py projects/06-multi-department-support/tests/test_evaluation.py
git commit -m "feat(p6): add LangSmith evaluation pipeline with routing and quality evaluators"
```

---

## Task 11: Streamlit Adapter

**Files:**
- Create: `app/adapters/support_system.py`
- Modify: `app/adapters/_importer.py` (if new module names need adding)

- [ ] **Step 1: Check if _importer.py needs updating**

Review the module names in Project 6. The existing `_CONFLICTING` set already includes `models`, `graph`, `nodes`, `prompts`, `tools`, and `data`. Project 6 adds `evaluation` which is also used by Project 5. Add it if not present.

Check and update `app/adapters/_importer.py`:
```python
_CONFLICTING = {
    "models", "graph", "nodes", "prompts", "chains",
    "conversation", "intake", "ingestion", "tools", "evaluation",
}
```

- [ ] **Step 2: Create the adapter**

`app/adapters/support_system.py`:
```python
"""Adapter for Project 06 — Multi-Department Support System.

Handles sys.path setup, environment loading, and wraps graph functions
for use in the Streamlit app.
"""

import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "06-multi-department-support"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env")

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.support_requests import SAMPLE_REQUESTS  # noqa: E402

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

# Build graph with persistence for interrupt/resume
_graph = build_graph(checkpointer=InMemorySaver())


def create_thread_id() -> str:
    """Generate a unique thread ID for a support session."""
    return str(uuid.uuid4())


def get_sample_requests() -> list[dict]:
    """Return sample support requests for the demo UI."""
    return SAMPLE_REQUESTS


def start_request(thread_id: str, request_text: str, metadata: dict) -> dict | None:
    """Submit a support request. Returns interrupt payload or None if completed.

    If the graph completes without interrupting, returns None and the final
    response can be retrieved via get_state(). If it interrupts (clarification
    needed), returns the interrupt payload.
    """
    initial_state = {
        "request": request_text,
        "request_metadata": metadata,
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }
    config = {"configurable": {"thread_id": thread_id}, "tags": ["p6-multi-department-support"]}

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Support pipeline failed: {e}") from e

    return _get_interrupt_value(thread_id)


def resume_with_clarification(thread_id: str, user_response: str) -> dict | None:
    """Resume after user provides clarification.

    Returns the next interrupt payload, or None if the graph completed.
    """
    config = {"configurable": {"thread_id": thread_id}, "tags": ["p6-multi-department-support"]}

    try:
        _graph.invoke(Command(resume=user_response), config=config)
    except Exception as e:
        raise RuntimeError(f"Resume failed: {e}") from e

    return _get_interrupt_value(thread_id)


def get_state(thread_id: str) -> dict:
    """Get the current graph state for display."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _graph.get_state(config)
    return dict(snapshot.values) if snapshot.values else {}


def _get_interrupt_value(thread_id: str) -> dict | None:
    """Extract interrupt payload from graph state, if paused."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = _graph.get_state(config)
    if snapshot.next:
        for task in snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    return None
```

- [ ] **Step 3: Verify adapter imports work**

```bash
cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning/app"
source ../.venv/bin/activate
python -c "
from adapters import support_system
print('Adapter loaded.')
print(f'Sample requests: {len(support_system.get_sample_requests())}')
print(f'Thread ID: {support_system.create_thread_id()}')
"
```

Expected: Prints success, no import errors.

- [ ] **Step 4: Commit**

```bash
git add app/adapters/support_system.py app/adapters/_importer.py
git commit -m "feat(p6): add Streamlit adapter for support system"
```

---

## Task 12: Streamlit Page

**Files:**
- Create: `app/pages/p6_support.py`
- Modify: `app/app.py` (add tab)

- [ ] **Step 1: Create the page**

`app/pages/p6_support.py`:
```python
"""Streamlit page for Project 06 — Multi-Department Support System.

Hybrid UI: chat-like for conversation flow (clarification), structured
display for results (department responses, routing info).
"""

import streamlit as st

from adapters import support_system
from components import doc_viewer

_DOC_PATH = "docs/06-multi-department-support.md"

_STAGE_LABELS = {
    "idle": "Ready — submit a support request",
    "processing": "Processing request...",
    "clarification": "Waiting for your clarification",
    "done": "Request resolved",
    "error": "An error occurred",
}


def _init_state() -> None:
    """Initialize session state keys for this page."""
    if "p6_stage" not in st.session_state:
        st.session_state["p6_stage"] = "idle"
        st.session_state["p6_thread_id"] = None
        st.session_state["p6_log"] = []
        st.session_state["p6_interrupt"] = None


def _reset_state() -> None:
    """Clear all p6_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p6_"):
            del st.session_state[key]


def _render_request_form() -> None:
    """Render the initial request form with sample selector."""
    samples = support_system.get_sample_requests()
    sample_labels = ["(Write your own)"] + [r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"] for r in samples]

    selected_idx = st.selectbox(
        "Quick-start with a sample request:",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        key="p6_sample_select",
    )

    # Pre-fill from sample or let user type
    if selected_idx and selected_idx > 0:
        sample = samples[selected_idx - 1]
        default_text = sample["text"]
        default_student_id = sample["metadata"].get("student_id", "S001")
    else:
        default_text = ""
        default_student_id = "S001"

    request_text = st.text_area("Support request:", value=default_text, height=100, key="p6_request_text")

    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID:", value=default_student_id, key="p6_student_id")
    with col2:
        priority = st.selectbox("Priority:", ["low", "medium", "high"], index=1, key="p6_priority")

    if st.button("Submit Request", disabled=not request_text, key="p6_submit"):
        thread_id = support_system.create_thread_id()
        st.session_state["p6_thread_id"] = thread_id
        metadata = {"sender_type": "student", "student_id": student_id, "priority": priority}

        try:
            interrupt_val = support_system.start_request(thread_id, request_text, metadata)

            if interrupt_val:
                # Graph paused for clarification
                st.session_state["p6_stage"] = "clarification"
                st.session_state["p6_interrupt"] = interrupt_val
                st.session_state["p6_log"].append(("Supervisor", "Request needs clarification"))
            else:
                # Graph completed
                st.session_state["p6_stage"] = "done"
                st.session_state["p6_log"].append(("Supervisor", "Request processed successfully"))
        except RuntimeError as e:
            st.session_state["p6_stage"] = "error"
            st.session_state["p6_log"].append(("Error", str(e)))

        st.rerun()


def _render_clarification() -> None:
    """Render clarification request from the system."""
    interrupt_val = st.session_state.get("p6_interrupt", {})
    question = interrupt_val.get("question", "Could you provide more details?")

    st.info(f"The system needs more information: **{question}**")

    clarification = st.text_area("Your response:", key="p6_clarification_text", height=80)

    if st.button("Send Clarification", disabled=not clarification, key="p6_clarify_btn"):
        thread_id = st.session_state["p6_thread_id"]

        try:
            interrupt_val = support_system.resume_with_clarification(thread_id, clarification)
            st.session_state["p6_log"].append(("You", clarification))

            if interrupt_val:
                st.session_state["p6_interrupt"] = interrupt_val
                st.session_state["p6_log"].append(("Supervisor", "Still needs more information"))
            else:
                st.session_state["p6_stage"] = "done"
                st.session_state["p6_log"].append(("Supervisor", "Request processed after clarification"))
        except RuntimeError as e:
            st.session_state["p6_stage"] = "error"
            st.session_state["p6_log"].append(("Error", str(e)))

        st.rerun()


def _render_done() -> None:
    """Render the final results."""
    thread_id = st.session_state.get("p6_thread_id")
    if not thread_id:
        return

    state = support_system.get_state(thread_id)

    # Final response
    st.success("Request resolved!")
    st.markdown("### Response")
    st.markdown(state.get("final_response", "No response generated."))

    # Behind the scenes
    with st.expander("Behind the Scenes", expanded=False):
        # Classification
        classification = state.get("classification", {})
        if classification:
            st.markdown("**Routing Decision:**")
            departments = classification.get("departments", [])
            complexity = classification.get("complexity", "unknown")
            st.markdown(f"- Departments: {', '.join(departments)}")
            st.markdown(f"- Complexity: {complexity}")
            st.markdown(f"- Summary: {classification.get('summary', 'N/A')}")

        # Department responses
        dept_results = state.get("department_results", [])
        if dept_results:
            st.markdown("**Department Responses:**")
            for result in dept_results:
                dept_name = result["department"].replace("_", " ").title()
                resolved_badge = "Resolved" if result["resolved"] else "Escalated"
                st.markdown(f"**{dept_name}** ({resolved_badge}):")
                st.markdown(f"> {result['response'][:300]}{'...' if len(result['response']) > 300 else ''}")

        # Resolution status
        st.markdown(f"**Status:** {state.get('resolution_status', 'unknown')}")


def _render_error() -> None:
    """Render error state."""
    st.error("An error occurred while processing your request. Please try again.")


def render() -> None:
    """Main render function called by app.py."""
    st.header("Multi-Department Support System")

    _init_state()

    # Top bar: stage indicator + reset
    col1, col2 = st.columns([3, 1])
    with col1:
        stage = st.session_state.get("p6_stage", "idle")
        st.caption(f"Stage: **{_STAGE_LABELS.get(stage, stage)}**")
    with col2:
        if st.button("Reset", key="p6_reset"):
            _reset_state()
            st.rerun()

    # Pipeline log
    log = st.session_state.get("p6_log", [])
    if log:
        with st.expander("Activity Log", expanded=False):
            for label, content in log:
                st.markdown(f"**{label}:** {content}")

    # Render current stage
    if stage == "idle":
        _render_request_form()
    elif stage == "clarification":
        _render_clarification()
    elif stage == "done":
        _render_done()
    elif stage == "error":
        _render_error()

    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: Multi-Agent Orchestration")
```

- [ ] **Step 2: Register tab in app.py**

Read `app/app.py`, find the tab registration, and add the new tab. The modification should:
1. Add `from pages import p6_support` import
2. Add `"🎯 Support System"` to the `st.tabs()` list
3. Add `with tab6: p6_support.render()`

- [ ] **Step 3: Commit**

```bash
git add app/pages/p6_support.py app/app.py
git commit -m "feat(p6): add Streamlit page and register Support System tab"
```

---

## Task 13: README & Documentation

**Files:**
- Create: `projects/06-multi-department-support/README.md`
- Create: `docs/06-multi-department-support.md`

- [ ] **Step 1: Create project README**

`projects/06-multi-department-support/README.md`:
```markdown
# Project 06: Multi-Department Support System

A supervisor-based multi-agent support system for LinguaFlow that classifies incoming requests, routes them to specialized department agents, and orchestrates responses across billing, tech support, scheduling, and content departments.

## Key Concepts

- **Supervisor agent pattern**: Central router classifies and dispatches requests
- **Parallel execution with Send**: Multi-department requests fan out to multiple agents simultaneously
- **Supervisor-mediated escalation**: Agents escalate through the supervisor, never directly
- **Hybrid clarification**: System can ask users for more info via interrupt/resume
- **Shared state**: All agents read from and write to a common state schema

## Running

```bash
cd projects/06-multi-department-support
source ../../.venv/bin/activate
python -c "from graph import build_graph; print('Graph ready!')"
```

## Testing

```bash
# Unit tests only
python -m pytest tests/ -v -k "not integration"

# All tests (requires API key)
python -m pytest tests/ -v

# Run evaluation
python evaluation.py
```

## Project Structure

```
├── models.py          # State schema (SupportState, DepartmentResult)
├── graph.py           # StateGraph assembly with Send routing
├── nodes.py           # Supervisor + 4 department sub-agents
├── prompts.py         # All prompt templates
├── tools.py           # 8 department tools (2 per dept)
├── evaluation.py      # LangSmith evaluation pipeline
├── data/              # Mock data for all departments
└── tests/             # Unit + integration tests
```
```

- [ ] **Step 2: Create educational documentation**

Create `docs/06-multi-department-support.md` with educational content explaining:
- Supervisor agent pattern and why it matters
- How `Send` enables parallel execution
- Supervisor-mediated escalation vs direct handoff
- The hybrid clarification pattern using `interrupt()`
- How shared state with reducers enables multi-agent coordination
- LangSmith cross-agent tracing

This should be a comprehensive educational document (similar in depth to existing docs in the `docs/` directory).

- [ ] **Step 3: Commit**

```bash
git add projects/06-multi-department-support/README.md docs/06-multi-department-support.md
git commit -m "docs(p6): add README and educational documentation"
```

---

## Task 14: Playwright End-to-End Testing

**Files:** None created — this is a verification task.

- [ ] **Step 1: Launch the app in background**

```bash
cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning/app"
source ../.venv/bin/activate
streamlit run app.py --server.headless true --server.port 8503
```

Run in background.

- [ ] **Step 2: Navigate to app and verify all tabs load**

Use Playwright to:
1. Navigate to `http://localhost:8503`
2. Take a snapshot to verify the app loads
3. Click on each tab (Grammar, Lesson Planner, Assessment, Tutor Matching, Content Moderation, Support System)
4. Verify each tab renders without errors

- [ ] **Step 3: Test the Support System tab**

1. Click the "Support System" tab
2. Select a sample request from the dropdown
3. Click "Submit Request"
4. Verify the response displays correctly
5. Check the "Behind the Scenes" expander
6. Test the Reset button

- [ ] **Step 4: Check for console errors**

Use `browser_console_messages` to check for any runtime errors across all tabs.

- [ ] **Step 5: Fix any issues found and re-test**

If any errors are found, fix them and repeat the verification steps.

- [ ] **Step 6: Stop the app**

Kill the background Streamlit process.

---

## Task 15: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cd "projects/06-multi-department-support"
source ../../.venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests PASS.

- [ ] **Step 2: Verify git status is clean**

```bash
git status
```

Expected: No uncommitted changes related to Project 6.

- [ ] **Step 3: Verify project structure matches spec**

```bash
find projects/06-multi-department-support -type f | sort
```

Expected: All files from the spec are present.
