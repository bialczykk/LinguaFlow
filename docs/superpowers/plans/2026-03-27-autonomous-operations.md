# Autonomous Operations (Capstone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a capstone project that integrates every concept from P1–P7 into a unified autonomous operations system with 6 department agents, task queue cascading, tiered HITL approval, and metrics reporting.

**Architecture:** Two-layer hybrid — LangGraph StateGraph (master orchestrator) wrapping 6 DeepAgent department agents. SqliteSaver for orchestrator persistence, InMemorySaver for department agents. Autonomous task cascading via a follow-up task queue that loops through the orchestrator.

**Tech Stack:** LangGraph, LangChain, DeepAgents, LangSmith, Streamlit, Anthropic (claude-haiku-4-5-20251001), SQLite.

**Spec:** `docs/superpowers/specs/2026-03-27-autonomous-operations-design.md`

---

## File Structure

```
projects/08-autonomous-operations/
├── models.py                    # OrchestratorState, DepartmentResult, MetricsStore, risk types
├── graph.py                     # Master orchestrator StateGraph + build_graph()
├── nodes.py                     # Orchestrator node functions
├── prompts.py                   # Classification, composition, department system prompts
├── tools.py                     # All 17 tools organized by department + tool group exports
├── risk.py                      # HIGH_RISK_ACTIONS map + risk_check() function
├── departments.py               # DeepAgent factory for each of 6 departments
├── evaluation.py                # LangSmith evaluators + dataset creation
├── requirements.txt             # Project dependencies
├── data/
│   ├── __init__.py
│   ├── students.py
│   ├── tutors.py
│   ├── study_plans.py
│   ├── invoices.py
│   ├── accounts.py
│   ├── lessons.py
│   ├── content_library.py
│   ├── content_drafts.py
│   ├── system_status.py
│   ├── qa_records.py
│   ├── metrics_seed.py
│   └── sample_requests.py
├── skills/
│   ├── student-onboarding/SKILL.md
│   ├── tutor-management/SKILL.md
│   ├── content-pipeline/SKILL.md
│   ├── quality-assurance/SKILL.md
│   ├── support/SKILL.md
│   └── reporting/SKILL.md
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    ├── test_tools.py
    ├── test_risk.py
    ├── test_nodes.py
    └── test_graph.py
```

Plus:
- `app/adapters/autonomous_ops.py`
- `app/pages/p8_operations.py`
- `app/app.py` (modify — add tab 8)
- `docs/08-autonomous-operations.md`
- `projects/08-autonomous-operations/README.md`

---

## Phase 1: Foundation (Data Layer + Models)

### Task 1: Scaffold project directory and requirements

**Files:**
- Create: `projects/08-autonomous-operations/requirements.txt`
- Create: `projects/08-autonomous-operations/data/__init__.py`
- Create: `projects/08-autonomous-operations/tests/__init__.py`
- Create: `projects/08-autonomous-operations/tests/conftest.py`

- [ ] **Step 1: Create project directory structure**

```bash
mkdir -p "projects/08-autonomous-operations/"{data,tests,skills/{student-onboarding,tutor-management,content-pipeline,quality-assurance,support,reporting}}
```

- [ ] **Step 2: Create requirements.txt**

```
langchain-core
langchain-anthropic
langgraph
langsmith
deepagents
python-dotenv
streamlit
```

- [ ] **Step 3: Create data/__init__.py**

```python
"""Mock data modules for LinguaFlow Autonomous Operations."""
```

- [ ] **Step 4: Create tests/__init__.py**

```python
"""Tests for the autonomous operations capstone."""
```

- [ ] **Step 5: Create tests/conftest.py**

```python
"""Shared test fixtures for Project 08."""

import os
import sys

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

- [ ] **Step 6: Commit**

```bash
git add projects/08-autonomous-operations/
git commit -m "feat(p8): scaffold project directory structure"
```

---

### Task 2: Pydantic models and state schema

**Files:**
- Create: `projects/08-autonomous-operations/models.py`
- Create: `projects/08-autonomous-operations/tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

```python
"""Tests for state schema and model definitions."""

import operator
import pytest

from models import DepartmentResult, MetricsStore, OrchestratorState, DEPARTMENTS


class TestDepartmentResult:
    """Verify DepartmentResult TypedDict construction."""

    def test_basic_result(self):
        result = DepartmentResult(
            department="student_onboarding",
            response="Student assessed at B1 level.",
            resolved=True,
            follow_up_tasks=[],
            metrics={"actions_taken": 1, "tools_called": ["assess_student"]},
        )
        assert result["department"] == "student_onboarding"
        assert result["resolved"] is True
        assert result["follow_up_tasks"] == []

    def test_result_with_follow_ups(self):
        result = DepartmentResult(
            department="student_onboarding",
            response="Student onboarded, needs tutor.",
            resolved=True,
            follow_up_tasks=[
                {"target_dept": "tutor_management", "action": "match_tutor",
                 "context": {"student_id": "S010", "level": "B1"}}
            ],
            metrics={"actions_taken": 2, "tools_called": ["assess_student", "create_study_plan"]},
        )
        assert len(result["follow_up_tasks"]) == 1
        assert result["follow_up_tasks"][0]["target_dept"] == "tutor_management"


class TestMetricsStore:
    """Verify MetricsStore construction and defaults."""

    def test_empty_metrics(self):
        metrics = MetricsStore(
            students_onboarded=0, tutors_assigned=0,
            content_generated=0, content_published=0,
            qa_reviews=0, qa_flags=0,
            support_requests=0, support_resolved=0,
            total_requests=0, department_invocations={},
        )
        assert metrics["total_requests"] == 0
        assert metrics["department_invocations"] == {}

    def test_metrics_with_data(self):
        metrics = MetricsStore(
            students_onboarded=3, tutors_assigned=2,
            content_generated=5, content_published=3,
            qa_reviews=4, qa_flags=1,
            support_requests=10, support_resolved=8,
            total_requests=15,
            department_invocations={"support": 10, "content_pipeline": 5},
        )
        assert metrics["students_onboarded"] == 3
        assert metrics["department_invocations"]["support"] == 10


class TestOrchestratorState:
    """Verify OrchestratorState schema and reducer."""

    def test_initial_state(self):
        state = OrchestratorState(
            request="Onboard student Maria",
            request_metadata={"user_id": "admin", "priority": "medium", "source": "ui"},
            classification={},
            risk_level="",
            approval_status="",
            department_results=[],
            task_queue=[],
            current_task=None,
            completed_tasks=[],
            metrics_store=MetricsStore(
                students_onboarded=0, tutors_assigned=0,
                content_generated=0, content_published=0,
                qa_reviews=0, qa_flags=0,
                support_requests=0, support_resolved=0,
                total_requests=0, department_invocations={},
            ),
            final_response="",
            resolution_status="",
        )
        assert state["request"] == "Onboard student Maria"
        assert state["department_results"] == []

    def test_department_results_has_add_reducer(self):
        """department_results must use operator.add for Send to work."""
        from typing import get_type_hints, get_args
        full_hints = get_type_hints(OrchestratorState, include_extras=True)
        dept_type = full_hints["department_results"]
        args = get_args(dept_type)
        assert args[1] is operator.add


class TestDepartments:
    """Verify DEPARTMENTS constant."""

    def test_all_six_departments(self):
        assert DEPARTMENTS == {
            "student_onboarding", "tutor_management", "content_pipeline",
            "quality_assurance", "support", "reporting",
        }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

```python
"""State schema and model definitions for the autonomous operations system.

OrchestratorState is the shared state for the master orchestrator graph.
DepartmentResult is the structured output from each department agent.
MetricsStore tracks cumulative platform metrics.

The department_results field uses an operator.add reducer so that parallel
sub-agents (dispatched via Send) can each append their result independently.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class DepartmentResult(TypedDict):
    """Structured result from a department agent.

    Each department returns one of these, appended to department_results.
    follow_up_tasks enables autonomous cascading — one department's output
    can trigger work in another department.
    """

    department: str              # One of DEPARTMENTS
    response: str                # The agent's response text
    resolved: bool               # Whether the agent fully handled its part
    follow_up_tasks: list[dict]  # [{"target_dept": str, "action": str, "context": dict}]
    metrics: dict                # {"actions_taken": int, "tools_called": list[str]}


class MetricsStore(TypedDict):
    """Cumulative platform metrics, updated after each completed request.

    Persisted via SqliteSaver as part of orchestrator state.
    """

    students_onboarded: int
    tutors_assigned: int
    content_generated: int
    content_published: int
    qa_reviews: int
    qa_flags: int
    support_requests: int
    support_resolved: int
    total_requests: int
    department_invocations: dict[str, int]


class OrchestratorState(TypedDict):
    """Shared state for the master orchestrator graph.

    Fields grouped by lifecycle stage:
    - Input: set at invocation
    - Classification: set by request_classifier
    - Approval: managed by risk_assessor and approval_gate
    - Department results: appended by each department (reducer: operator.add)
    - Task queue: autonomous follow-up management
    - Metrics: cumulative platform metrics
    - Output: set by compose_output
    """

    # --- Input ---
    request: str
    request_metadata: dict                                    # user_id, priority, source

    # --- Classification ---
    classification: dict                                      # departments, action_type, complexity
    risk_level: str                                           # "low" | "high"

    # --- Approval ---
    approval_status: str                                      # "approved" | "rejected" | "not_required"

    # --- Department results (reducer: append for parallel Send) ---
    department_results: Annotated[list[DepartmentResult], operator.add]

    # --- Task queue (autonomous follow-ups) ---
    task_queue: list[dict]                                    # pending follow-up tasks
    current_task: dict | None                                 # task being processed
    completed_tasks: list[dict]                               # finished tasks (audit trail)

    # --- Metrics ---
    metrics_store: MetricsStore

    # --- Output ---
    final_response: str
    resolution_status: str                                    # "resolved" | "partial" | "pending_approval"


# Valid department names
DEPARTMENTS = {
    "student_onboarding", "tutor_management", "content_pipeline",
    "quality_assurance", "support", "reporting",
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_models.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add projects/08-autonomous-operations/models.py projects/08-autonomous-operations/tests/test_models.py
git commit -m "feat(p8): add state schema with DepartmentResult, MetricsStore, and OrchestratorState"
```

---

### Task 3: Mock data modules

**Files:**
- Create: `projects/08-autonomous-operations/data/students.py`
- Create: `projects/08-autonomous-operations/data/tutors.py`
- Create: `projects/08-autonomous-operations/data/study_plans.py`
- Create: `projects/08-autonomous-operations/data/invoices.py`
- Create: `projects/08-autonomous-operations/data/accounts.py`
- Create: `projects/08-autonomous-operations/data/lessons.py`
- Create: `projects/08-autonomous-operations/data/content_library.py`
- Create: `projects/08-autonomous-operations/data/content_drafts.py`
- Create: `projects/08-autonomous-operations/data/system_status.py`
- Create: `projects/08-autonomous-operations/data/qa_records.py`
- Create: `projects/08-autonomous-operations/data/metrics_seed.py`
- Create: `projects/08-autonomous-operations/data/sample_requests.py`

This is a large data creation task. Each module follows the same pattern: a module-level constant (list or dict) with realistic mock data. Student IDs must be consistent across all modules.

**Consistent IDs across modules:**
- Students: S001–S006
- Tutors: T001–T008
- Invoices: INV-001–INV-008
- Lessons: L001–L010
- Content: C001–C012
- Study plans: SP-001–SP-004

- [ ] **Step 1: Create students.py**

```python
"""Student profiles for onboarding and cross-department operations.

Mix of new students (no study plan yet — for onboarding demos) and
existing students (enrolled, with history — for support/scheduling).
"""

STUDENTS = [
    {
        "student_id": "S001",
        "name": "Alice Chen",
        "email": "alice.chen@email.com",
        "cefr_level": "B1",
        "goals": ["business English", "presentation skills"],
        "enrollment_date": "2025-09-15",
        "status": "active",
    },
    {
        "student_id": "S002",
        "name": "Marco Rossi",
        "email": "marco.rossi@email.com",
        "cefr_level": "A2",
        "goals": ["general English", "grammar improvement"],
        "enrollment_date": "2025-11-01",
        "status": "active",
    },
    {
        "student_id": "S003",
        "name": "Yuki Tanaka",
        "email": "yuki.tanaka@email.com",
        "cefr_level": "B2",
        "goals": ["IELTS preparation", "academic writing"],
        "enrollment_date": "2026-01-10",
        "status": "active",
    },
    {
        "student_id": "S004",
        "name": "Priya Sharma",
        "email": "priya.sharma@email.com",
        "cefr_level": "C1",
        "goals": ["advanced conversation", "idioms"],
        "enrollment_date": "2025-06-20",
        "status": "active",
    },
    {
        "student_id": "S005",
        "name": "Lars Eriksson",
        "email": "lars.eriksson@email.com",
        "cefr_level": None,
        "goals": ["travel English"],
        "enrollment_date": None,
        "status": "new",  # Not yet onboarded — for onboarding demo
    },
    {
        "student_id": "S006",
        "name": "Maria Silva",
        "email": "maria.silva@email.com",
        "cefr_level": None,
        "goals": ["business English", "email writing"],
        "enrollment_date": None,
        "status": "new",  # Not yet onboarded — for onboarding demo
    },
]
```

- [ ] **Step 2: Create tutors.py**

```python
"""Tutor profiles with specialties, availability, and capacity.

Reuses concepts from P4's tutor matching but with availability slots
and max student capacity for the assignment workflow.
"""

TUTORS = [
    {
        "tutor_id": "T001",
        "name": "Sarah Johnson",
        "specialties": ["business English", "presentation skills", "email writing"],
        "cefr_levels": ["B1", "B2", "C1"],
        "availability": ["Monday 09:00", "Monday 14:00", "Wednesday 10:00", "Friday 09:00"],
        "rating": 4.8,
        "max_students": 8,
        "current_students": 5,
    },
    {
        "tutor_id": "T002",
        "name": "James Wilson",
        "specialties": ["general English", "grammar", "pronunciation"],
        "cefr_levels": ["A2", "B1"],
        "availability": ["Tuesday 10:00", "Tuesday 15:00", "Thursday 10:00"],
        "rating": 4.6,
        "max_students": 10,
        "current_students": 7,
    },
    {
        "tutor_id": "T003",
        "name": "Emma Davis",
        "specialties": ["IELTS preparation", "academic writing", "reading comprehension"],
        "cefr_levels": ["B2", "C1"],
        "availability": ["Monday 11:00", "Wednesday 14:00", "Friday 11:00"],
        "rating": 4.9,
        "max_students": 6,
        "current_students": 4,
    },
    {
        "tutor_id": "T004",
        "name": "David Brown",
        "specialties": ["conversation practice", "idioms", "cultural context"],
        "cefr_levels": ["B2", "C1"],
        "availability": ["Tuesday 09:00", "Thursday 14:00", "Friday 15:00"],
        "rating": 4.7,
        "max_students": 8,
        "current_students": 6,
    },
    {
        "tutor_id": "T005",
        "name": "Lisa Chen",
        "specialties": ["business English", "vocabulary", "meeting skills"],
        "cefr_levels": ["B1", "B2"],
        "availability": ["Monday 15:00", "Wednesday 09:00", "Thursday 11:00"],
        "rating": 4.5,
        "max_students": 10,
        "current_students": 8,
    },
    {
        "tutor_id": "T006",
        "name": "Michael Park",
        "specialties": ["general English", "travel English", "beginner support"],
        "cefr_levels": ["A2", "B1"],
        "availability": ["Monday 10:00", "Tuesday 14:00", "Wednesday 15:00", "Friday 10:00"],
        "rating": 4.4,
        "max_students": 12,
        "current_students": 9,
    },
    {
        "tutor_id": "T007",
        "name": "Anna Kowalski",
        "specialties": ["grammar improvement", "writing skills", "IELTS preparation"],
        "cefr_levels": ["A2", "B1", "B2"],
        "availability": ["Tuesday 11:00", "Thursday 09:00", "Friday 14:00"],
        "rating": 4.8,
        "max_students": 8,
        "current_students": 3,
    },
    {
        "tutor_id": "T008",
        "name": "Carlos Mendez",
        "specialties": ["advanced conversation", "debate", "public speaking"],
        "cefr_levels": ["C1"],
        "availability": ["Wednesday 11:00", "Thursday 15:00"],
        "rating": 4.9,
        "max_students": 5,
        "current_students": 4,
    },
]
```

- [ ] **Step 3: Create study_plans.py**

```python
"""Pre-built study plans linked to existing students.

New students (S005, S006) have no study plans yet — those are
created during onboarding.
"""

STUDY_PLANS = [
    {
        "plan_id": "SP-001",
        "student_id": "S001",
        "tutor_id": "T001",
        "level": "B1",
        "focus_areas": ["business English", "presentation skills"],
        "weekly_hours": 3,
        "created_date": "2025-09-20",
        "status": "active",
    },
    {
        "plan_id": "SP-002",
        "student_id": "S002",
        "tutor_id": "T002",
        "level": "A2",
        "focus_areas": ["general English", "grammar improvement"],
        "weekly_hours": 4,
        "created_date": "2025-11-05",
        "status": "active",
    },
    {
        "plan_id": "SP-003",
        "student_id": "S003",
        "tutor_id": "T003",
        "level": "B2",
        "focus_areas": ["IELTS preparation", "academic writing"],
        "weekly_hours": 5,
        "created_date": "2026-01-15",
        "status": "active",
    },
    {
        "plan_id": "SP-004",
        "student_id": "S004",
        "tutor_id": "T004",
        "level": "C1",
        "focus_areas": ["advanced conversation", "idioms"],
        "weekly_hours": 2,
        "created_date": "2025-07-01",
        "status": "active",
    },
]
```

- [ ] **Step 4: Create data modules adapted from P6**

Copy and adapt from `projects/06-multi-department-support/data/`. Update student IDs to match P8's STUDENTS list. Files to create:
- `invoices.py` — 6-8 invoices for S001–S004 (existing students)
- `accounts.py` — accounts for S001–S006 (all students)
- `lessons.py` — lessons with dynamic dates (relative to today)
- `content_library.py` — course catalog + enrollments
- `system_status.py` — platform service health map

Read P6's data files and adapt them. Keep the same structure but ensure student IDs S001–S006 are consistent.

- [ ] **Step 5: Create content_drafts.py**

```python
"""Content items in various pipeline stages.

Used by the content pipeline and QA departments. Items progress
through: draft → in_review → published (or flagged).
"""

CONTENT_DRAFTS = [
    {
        "content_id": "CD-001",
        "title": "Present Perfect Tense — When to Use It",
        "type": "grammar_explanation",
        "level": "B1",
        "status": "published",
        "author": "system",
        "created_date": "2026-03-10",
        "qa_status": "passed",
    },
    {
        "content_id": "CD-002",
        "title": "Business Email Vocabulary Builder",
        "type": "vocabulary_exercise",
        "level": "B2",
        "status": "published",
        "author": "system",
        "created_date": "2026-03-12",
        "qa_status": "passed",
    },
    {
        "content_id": "CD-003",
        "title": "IELTS Reading: Climate Change Passage",
        "type": "reading_passage",
        "level": "B2",
        "status": "in_review",
        "author": "system",
        "created_date": "2026-03-20",
        "qa_status": "pending",
    },
    {
        "content_id": "CD-004",
        "title": "Phrasal Verbs for Travel",
        "type": "vocabulary_exercise",
        "level": "A2",
        "status": "draft",
        "author": "system",
        "created_date": "2026-03-25",
        "qa_status": None,
    },
    {
        "content_id": "CD-005",
        "title": "Conditionals Deep Dive",
        "type": "grammar_explanation",
        "level": "B2",
        "status": "flagged",
        "author": "system",
        "created_date": "2026-03-15",
        "qa_status": "failed",
        "qa_notes": "Incorrect example in third conditional section",
    },
]
```

- [ ] **Step 6: Create qa_records.py**

```python
"""QA review history for content items.

Links to content_drafts via content_id. Each review records
the outcome, reviewer notes, and any flags raised.
"""

QA_RECORDS = [
    {
        "review_id": "QA-001",
        "content_id": "CD-001",
        "reviewer": "qa_agent",
        "date": "2026-03-11",
        "result": "pass",
        "score": 0.92,
        "notes": "Clear explanations, good examples. Minor formatting suggestion.",
    },
    {
        "review_id": "QA-002",
        "content_id": "CD-002",
        "reviewer": "qa_agent",
        "date": "2026-03-13",
        "result": "pass",
        "score": 0.88,
        "notes": "Appropriate vocabulary level. Could use more context sentences.",
    },
    {
        "review_id": "QA-003",
        "content_id": "CD-005",
        "reviewer": "qa_agent",
        "date": "2026-03-16",
        "result": "fail",
        "score": 0.45,
        "notes": "Incorrect example in third conditional section. Mixed tenses in explanation.",
        "flags": ["accuracy_error", "needs_revision"],
    },
]
```

- [ ] **Step 7: Create metrics_seed.py**

```python
"""Seed data for the metrics store.

Provides initial values so the reporting agent has historical data
to work with even on first run.
"""

METRICS_SEED = {
    "students_onboarded": 4,
    "tutors_assigned": 4,
    "content_generated": 5,
    "content_published": 2,
    "qa_reviews": 3,
    "qa_flags": 1,
    "support_requests": 0,
    "support_resolved": 0,
    "total_requests": 0,
    "department_invocations": {},
}
```

- [ ] **Step 8: Create sample_requests.py**

```python
"""Sample requests covering all routing patterns.

Each request includes text, metadata, expected routing, and pattern
for use in tests, evaluation, and the Streamlit demo.
"""

SAMPLE_REQUESTS = [
    {
        "text": "Onboard new student Maria Silva, she's interested in business English and email writing.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["student_onboarding"],
        "expected_follow_ups": ["tutor_management"],
        "pattern": "cascade",
        "expected_risk": "low",
    },
    {
        "text": "Generate a B2 reading passage about climate change and publish it.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["content_pipeline"],
        "expected_follow_ups": ["quality_assurance"],
        "pattern": "cascade",
        "expected_risk": "high",  # publish is high risk
    },
    {
        "text": "I was charged twice for my IELTS prep lesson and now I can't access my lesson recordings.",
        "metadata": {"user_id": "student", "priority": "high", "source": "support_form"},
        "expected_departments": ["support"],
        "expected_follow_ups": [],
        "pattern": "parallel",  # billing + tech support within support dept
        "expected_risk": "low",
    },
    {
        "text": "How is the platform performing this week?",
        "metadata": {"user_id": "admin", "priority": "low", "source": "ui"},
        "expected_departments": ["reporting"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
    {
        "text": "Review all recently published content for quality.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["quality_assurance"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
    {
        "text": "Find a tutor for Lars Eriksson — he needs help with travel English at beginner level.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["tutor_management"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "high",  # assign_tutor is high risk
    },
    {
        "text": "Cancel my Thursday lesson and refund me.",
        "metadata": {"user_id": "student", "priority": "medium", "source": "support_form"},
        "expected_departments": ["support"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "high",  # refund is high risk
    },
    {
        "text": "Check if student S001 is happy with their progress and review her tutor's performance.",
        "metadata": {"user_id": "admin", "priority": "low", "source": "ui"},
        "expected_departments": ["quality_assurance"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
]
```

- [ ] **Step 9: Commit**

```bash
git add projects/08-autonomous-operations/data/
git commit -m "feat(p8): add mock data modules for all six departments"
```

---

### Task 4: Risk assessment rules

**Files:**
- Create: `projects/08-autonomous-operations/risk.py`
- Create: `projects/08-autonomous-operations/tests/test_risk.py`

- [ ] **Step 1: Write failing tests for risk assessment**

```python
"""Tests for risk assessment rules."""

import pytest

from risk import assess_risk, HIGH_RISK_ACTIONS


class TestHighRiskActions:
    """Verify the risk map covers all expected actions."""

    def test_publish_is_high_risk(self):
        assert "publish_content" in HIGH_RISK_ACTIONS["content_pipeline"]

    def test_assign_tutor_is_high_risk(self):
        assert "assign_tutor" in HIGH_RISK_ACTIONS["tutor_management"]

    def test_create_study_plan_is_high_risk(self):
        assert "create_study_plan" in HIGH_RISK_ACTIONS["student_onboarding"]

    def test_flag_issue_is_high_risk(self):
        assert "flag_issue" in HIGH_RISK_ACTIONS["quality_assurance"]


class TestAssessRisk:
    """Verify risk assessment function."""

    def test_low_risk_lookup(self):
        classification = {
            "departments": ["support"],
            "action_type": "lookup",
        }
        assert assess_risk(classification) == "low"

    def test_high_risk_publish(self):
        classification = {
            "departments": ["content_pipeline"],
            "action_type": "publish_content",
        }
        assert assess_risk(classification) == "high"

    def test_high_risk_assign_tutor(self):
        classification = {
            "departments": ["tutor_management"],
            "action_type": "assign_tutor",
        }
        assert assess_risk(classification) == "high"

    def test_multi_dept_high_if_any_high(self):
        classification = {
            "departments": ["support", "content_pipeline"],
            "action_type": "publish_content",
        }
        assert assess_risk(classification) == "high"

    def test_unknown_action_is_low(self):
        classification = {
            "departments": ["reporting"],
            "action_type": "aggregate_metrics",
        }
        assert assess_risk(classification) == "low"

    def test_follow_up_task_risk(self):
        """Follow-up tasks also get risk-assessed."""
        classification = {
            "departments": ["tutor_management"],
            "action_type": "assign_tutor",
            "is_follow_up": True,
        }
        assert assess_risk(classification) == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_risk.py -v
```

- [ ] **Step 3: Implement risk.py**

```python
"""Risk assessment rules for the tiered approval system.

Two tiers:
- Low risk: auto-execute (data lookups, draft generation, assessments, metrics)
- High risk: requires human approval (refunds, publication, tutor assignment, flagging)

The assess_risk() function is called by the risk_assessor node in the
orchestrator graph. It's pure logic — no LLM involved.
"""

# Actions that require human approval, organized by department.
# If the classification's action_type matches any of these, it's high risk.
HIGH_RISK_ACTIONS: dict[str, set[str]] = {
    "content_pipeline": {"publish_content"},
    "support": {"process_refund"},
    "tutor_management": {"assign_tutor"},
    "quality_assurance": {"flag_issue"},
    "student_onboarding": {"create_study_plan"},
}


def assess_risk(classification: dict) -> str:
    """Determine risk level from a classification dict.

    Args:
        classification: Must contain 'departments' (list[str]) and
                       'action_type' (str). May contain 'is_follow_up' (bool).

    Returns:
        "high" if the action matches any HIGH_RISK_ACTIONS entry,
        "low" otherwise.
    """
    action_type = classification.get("action_type", "")
    departments = classification.get("departments", [])

    for dept in departments:
        high_risk_set = HIGH_RISK_ACTIONS.get(dept, set())
        if action_type in high_risk_set:
            return "high"

    return "low"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_risk.py -v
```

- [ ] **Step 5: Commit**

```bash
git add projects/08-autonomous-operations/risk.py projects/08-autonomous-operations/tests/test_risk.py
git commit -m "feat(p8): add risk assessment rules with tiered approval map"
```

---

### Task 5: Department tools (all 17)

**Files:**
- Create: `projects/08-autonomous-operations/tools.py`
- Create: `projects/08-autonomous-operations/tests/test_tools.py`

- [ ] **Step 1: Write failing tests for all 17 tools**

Write tests covering each tool in isolation against mock data. Group by department. Test both success and not-found cases. Each tool test should verify the return structure and key values.

Pattern per tool:
```python
class TestStudentOnboardingTools:
    def test_assess_student_found(self):
        result = assess_student.invoke({"student_id": "S006"})
        assert "level" in result or "assessment" in result

    def test_assess_student_not_found(self):
        result = assess_student.invoke({"student_id": "S999"})
        assert "not found" in str(result).lower()
```

Cover all 17 tools: `assess_student`, `create_study_plan`, `search_tutors`, `check_availability`, `assign_tutor`, `generate_content`, `submit_for_review`, `publish_content`, `review_content`, `flag_issue`, `check_satisfaction`, `lookup_invoice`, `check_schedule`, `check_system_status`, `check_enrollment`, `aggregate_metrics`, `get_department_state`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_tools.py -v
```

- [ ] **Step 3: Implement tools.py**

All tools use `@tool` + `@traceable(tags=["p8-autonomous-operations"])`. Tools are pure functions operating on mock data. Group by department with clear section comments. Export tool groups at the bottom:

```python
STUDENT_ONBOARDING_TOOLS = [assess_student, create_study_plan]
TUTOR_MANAGEMENT_TOOLS = [search_tutors, check_availability, assign_tutor]
CONTENT_PIPELINE_TOOLS = [generate_content, submit_for_review, publish_content]
QA_TOOLS = [review_content, flag_issue, check_satisfaction]
SUPPORT_TOOLS = [lookup_invoice, check_schedule, check_system_status, check_enrollment]
REPORTING_TOOLS = [aggregate_metrics, get_department_state]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_tools.py -v
```

- [ ] **Step 5: Commit**

```bash
git add projects/08-autonomous-operations/tools.py projects/08-autonomous-operations/tests/test_tools.py
git commit -m "feat(p8): add 17 department tool functions with tests"
```

---

## Phase 2: Department Agents

### Task 6: System prompts for all departments

**Files:**
- Create: `projects/08-autonomous-operations/prompts.py`

- [ ] **Step 1: Create prompts.py**

Contains:
- `CLASSIFIER_PROMPT` — ChatPromptTemplate for request classification. Returns JSON with `departments`, `action_type`, `complexity`, `summary`. Must handle both user requests and follow-up tasks.
- `COMPOSE_OUTPUT_PROMPT` — ChatPromptTemplate for merging department results into unified response.
- 6 department system prompts (plain strings): `STUDENT_ONBOARDING_PROMPT`, `TUTOR_MANAGEMENT_PROMPT`, `CONTENT_PIPELINE_PROMPT`, `QA_PROMPT`, `SUPPORT_PROMPT`, `REPORTING_PROMPT`.

Each department prompt lists the agent's tools, instructions for when to set follow_up_tasks, and the department's scope. Follow P6's prompt pattern.

- [ ] **Step 2: Verify imports**

```bash
cd projects/08-autonomous-operations && python -c "from prompts import CLASSIFIER_PROMPT, COMPOSE_OUTPUT_PROMPT, STUDENT_ONBOARDING_PROMPT; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/prompts.py
git commit -m "feat(p8): add classification, composition, and department system prompts"
```

---

### Task 7: SKILL.md files for all 6 departments

**Files:**
- Create: `projects/08-autonomous-operations/skills/student-onboarding/SKILL.md`
- Create: `projects/08-autonomous-operations/skills/tutor-management/SKILL.md`
- Create: `projects/08-autonomous-operations/skills/content-pipeline/SKILL.md`
- Create: `projects/08-autonomous-operations/skills/quality-assurance/SKILL.md`
- Create: `projects/08-autonomous-operations/skills/support/SKILL.md`
- Create: `projects/08-autonomous-operations/skills/reporting/SKILL.md`

Each SKILL.md follows the P7 pattern — YAML frontmatter with `name` and `description`, then markdown sections defining domain knowledge and output format guidance.

- [ ] **Step 1: Create all 6 SKILL.md files**

Each should be 30-60 lines covering:
- Domain scope and responsibilities
- Key rules and decision criteria
- Output format expectations
- When to generate follow-up tasks vs handle in-department

Example structure for student-onboarding:
```markdown
---
name: student-onboarding
description: Domain knowledge for assessing new students and creating personalized study plans
---

# Student Onboarding

## Responsibilities
- Assess incoming student profiles for CEFR level
- Create personalized study plans based on level, goals, and availability

## CEFR Assessment Criteria
[Brief rubric for A2, B1, B2, C1]

## Study Plan Template
[Structure for a study plan: focus areas, weekly hours, milestones]

## Follow-Up Rules
- After creating a study plan, ALWAYS generate a follow-up task for tutor_management to match a tutor
- Include the student's level and focus areas in the follow-up context
```

- [ ] **Step 2: Commit**

```bash
git add projects/08-autonomous-operations/skills/
git commit -m "feat(p8): add SKILL.md files for all six departments"
```

---

### Task 8: Department agent factories

**Files:**
- Create: `projects/08-autonomous-operations/departments.py`

- [ ] **Step 1: Implement departments.py**

Follow P7's `agents.py` pattern exactly. Create 6 factory functions:
- `create_onboarding_agent()`
- `create_tutor_agent()`
- `create_content_agent()`
- `create_qa_agent()`
- `create_support_agent()`
- `create_reporting_agent()`

Each uses:
```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend

def create_composite_backend():
    def factory(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={"/persistent/": StoreBackend(runtime)},
        )
    return factory
```

All share one `InMemoryStore` instance. All use `ChatAnthropic(model="claude-haiku-4-5-20251001")`. Each binds its department tools via the `tools=` parameter. Each loads skills from `skills/` directory.

Also create a dispatcher dict:
```python
DEPARTMENT_AGENTS = {
    "student_onboarding": create_onboarding_agent,
    "tutor_management": create_tutor_agent,
    "content_pipeline": create_content_agent,
    "quality_assurance": create_qa_agent,
    "support": create_support_agent,
    "reporting": create_reporting_agent,
}
```

- [ ] **Step 2: Verify imports**

```bash
cd projects/08-autonomous-operations && python -c "from departments import DEPARTMENT_AGENTS; print('Departments:', list(DEPARTMENT_AGENTS.keys()))"
```

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/departments.py
git commit -m "feat(p8): add DeepAgent factory functions for all six departments"
```

---

## Phase 3: Orchestrator Graph

### Task 9: Orchestrator node functions

**Files:**
- Create: `projects/08-autonomous-operations/nodes.py`
- Create: `projects/08-autonomous-operations/tests/test_nodes.py`

- [ ] **Step 1: Write failing tests for orchestrator nodes**

Test with mocked LLMs (same pattern as P6). Cover:

```python
class TestRequestClassifier:
    """Test request_classifier node."""

    @patch("nodes._classifier_model")
    def test_classifies_single_department(self, mock_model):
        # Mock LLM returns JSON with departments: ["student_onboarding"]
        ...

    @patch("nodes._classifier_model")
    def test_classifies_follow_up_task(self, mock_model):
        # When current_task is set, classifier reads from it
        ...

class TestRiskAssessor:
    """Test risk_assessor node."""

    def test_low_risk_action(self):
        # Pure logic, no mock needed
        ...

    def test_high_risk_action(self):
        ...

class TestResultAggregator:
    """Test result_aggregator node."""

    def test_extracts_follow_up_tasks(self):
        ...

    def test_no_follow_ups(self):
        ...

class TestCheckTaskQueue:
    """Test check_task_queue node."""

    def test_pops_next_task(self):
        ...

    def test_empty_queue(self):
        ...

class TestReportingSnapshot:
    """Test reporting_snapshot node."""

    def test_increments_metrics(self):
        ...
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_nodes.py -v
```

- [ ] **Step 3: Implement nodes.py**

Nodes to implement:

1. `request_classifier(state)` — Uses `_classifier_model` (module-level ChatAnthropic) with `CLASSIFIER_PROMPT`. If `current_task` is set, classifies from the follow-up task instead of the request. Parses JSON with markdown-fence stripping (same robust parsing as P6's fix). Returns `{"classification": dict}`.

2. `risk_assessor(state)` — Pure logic. Calls `assess_risk(state["classification"])` from `risk.py`. Returns `{"risk_level": str, "approval_status": "not_required" if low else ""}`.

3. `approval_gate(state)` — Calls `interrupt()` with approval payload. On resume, returns `{"approval_status": decision}`. Uses `Command` to route to `dispatch_departments` if approved or to `compose_output` if rejected.

4. `department_executor(state)` — Receives department name from Send payload. Creates the DeepAgent via `DEPARTMENT_AGENTS[dept]()`. Invokes it with the request. Extracts response and follow-up tasks. Returns `{"department_results": [DepartmentResult]}`.

5. `result_aggregator(state)` — Scans department_results for follow_up_tasks. Moves them into task_queue. Returns `{"task_queue": updated_queue, "completed_tasks": updated_completed}`.

6. `check_task_queue(state)` — If task_queue not empty, pops first task into current_task and returns `Command(goto="request_classifier")`. Otherwise returns `Command(goto="compose_output")`.

7. `compose_output(state)` — Uses LLM with `COMPOSE_OUTPUT_PROMPT` to merge all department_results into final_response. Returns `{"final_response": str, "resolution_status": str}`.

8. `reporting_snapshot(state)` — Pure logic. Increments metrics_store counters based on department_results. Returns `{"metrics_store": updated_metrics}`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_nodes.py -v
```

- [ ] **Step 5: Commit**

```bash
git add projects/08-autonomous-operations/nodes.py projects/08-autonomous-operations/tests/test_nodes.py
git commit -m "feat(p8): add orchestrator node functions with tests"
```

---

### Task 10: Graph assembly

**Files:**
- Create: `projects/08-autonomous-operations/graph.py`

- [ ] **Step 1: Implement graph.py**

Build the master orchestrator StateGraph following the spec's graph structure:

```python
def route_from_classifier(state):
    """Route after classification."""
    if state.get("current_task"):
        # Follow-up task — already classified, skip risk for low-risk
        # (risk_assessor handles it)
        return "risk_assessor"
    return "risk_assessor"

def route_from_risk(state):
    """Route based on risk level."""
    if state["risk_level"] == "high":
        return "approval_gate"
    return "dispatch_departments"

def dispatch_to_departments(state):
    """Fan out to department executors via Send."""
    departments = state["classification"].get("departments", [])
    return [Send("department_executor", {**state, "_target_dept": dept}) for dept in departments]

def route_from_task_queue(state):
    """check_task_queue uses Command internally, but declare possible destinations."""
    ...
```

Nodes registered:
- `request_classifier`, `risk_assessor`, `approval_gate` (with `ends=`), `department_executor`, `result_aggregator`, `check_task_queue` (with `ends=`), `compose_output`, `reporting_snapshot`

Edges:
- `START → request_classifier`
- `request_classifier → risk_assessor` (conditional, but always goes to risk_assessor)
- `risk_assessor → approval_gate | dispatch_departments` (conditional on risk_level)
- `approval_gate` uses Command internally (ends=["dispatch_departments", "compose_output"])
- `dispatch_departments` is a conditional edge returning Send objects
- `department_executor → result_aggregator`
- `result_aggregator → check_task_queue`
- `check_task_queue` uses Command internally (ends=["request_classifier", "compose_output"])
- `compose_output → reporting_snapshot`
- `reporting_snapshot → END`

`build_graph(checkpointer=None)` function returns compiled graph.

- [ ] **Step 2: Verify compilation**

```bash
cd projects/08-autonomous-operations && python -c "from graph import build_graph; g = build_graph(); print('OK:', type(g).__name__)"
```

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/graph.py
git commit -m "feat(p8): assemble master orchestrator StateGraph with task queue loop"
```

---

### Task 11: Integration tests

**Files:**
- Create: `projects/08-autonomous-operations/tests/test_graph.py`

- [ ] **Step 1: Write integration tests**

All with mocked LLMs, marked `@pytest.mark.integration`. Tests:

1. `test_single_department_low_risk` — Reporting request → auto-executes → returns metrics summary
2. `test_single_department_high_risk` — Tutor assignment → approval gate → approve → executes
3. `test_cascading_follow_ups` — Onboarding → follow-up to tutor management → both complete
4. `test_multi_department_parallel` — Request classified to 2 departments → Send fan-out → both results
5. `test_approval_rejection` — High-risk action → rejected → resolution_status = "rejected"
6. `test_task_queue_loop` — Chain of follow-up tasks processed in sequence
7. `test_reporting_metrics_update` — Verify metrics_store increments after request

Use the same mocking patterns as P6's test_graph.py. Mock `nodes._classifier_model` for the classifier, mock `ChatAnthropic` for department executors and compose_output, mock the DeepAgent invocations in `department_executor`.

- [ ] **Step 2: Run tests**

```bash
cd projects/08-autonomous-operations && python -m pytest tests/test_graph.py -v
```

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/tests/test_graph.py
git commit -m "test(p8): add integration tests for orchestrator graph flows"
```

---

## Phase 4: Evaluation + Streamlit + Docs

### Task 12: LangSmith evaluation

**Files:**
- Create: `projects/08-autonomous-operations/evaluation.py`

- [ ] **Step 1: Implement evaluation.py**

Three evaluators following P6's pattern:
1. `routing_accuracy_evaluator` — Deterministic: compare classified departments against expected
2. `response_quality_evaluator` — LLM-as-judge on final composed output
3. `task_chain_completeness_evaluator` — Deterministic: check all expected follow-ups were processed

Plus `create_dataset()` and `run_evaluation()` functions.

- [ ] **Step 2: Verify imports**

```bash
cd projects/08-autonomous-operations && python -c "from evaluation import routing_accuracy_evaluator, response_quality_evaluator, task_chain_completeness_evaluator; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/evaluation.py
git commit -m "feat(p8): add LangSmith evaluation with routing, quality, and task chain evaluators"
```

---

### Task 13: Streamlit adapter

**Files:**
- Create: `app/adapters/autonomous_ops.py`

- [ ] **Step 1: Implement adapter**

Follow the exact pattern from `app/adapters/content_moderation.py`:

```python
from adapters._importer import clear_project_modules
from adapters._env import ensure_repo_env

_PROJECT_DIR = _REPO_ROOT / "projects" / "08-autonomous-operations"
# ... path setup, clear_project_modules(), imports ...

_checkpointer = SqliteSaver(...)  # SqliteSaver for orchestrator persistence
_graph = build_graph(checkpointer=_checkpointer)
```

Functions:
- `start_request(thread_id, request_text, metadata)` → runs graph, returns result or interrupt
- `resume_approval(thread_id, decision)` → resumes after approval interrupt
- `get_state(thread_id)` → current state
- `get_metrics(thread_id)` → MetricsStore from state
- `get_task_queue(thread_id)` → task queue from state
- `get_sample_requests()` → SAMPLE_REQUESTS
- `create_thread_id()` → uuid

Check if `_importer.py` needs updates for new module names (risk, departments, evaluation). Add them to `_CONFLICTING` if needed.

- [ ] **Step 2: Verify imports**

```bash
cd app && python -c "from adapters import autonomous_ops; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/adapters/autonomous_ops.py
git commit -m "feat(app): add P8 autonomous operations Streamlit adapter"
```

---

### Task 14: Streamlit page

**Files:**
- Create: `app/pages/p8_operations.py`
- Modify: `app/app.py`

- [ ] **Step 1: Create p8_operations.py**

Three-panel layout:

```python
def render():
    st.header("Autonomous Operations")
    st.caption("Cross-department orchestration with autonomous task cascading")

    _init_state()

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_left:
        _render_console()      # Input + task queue + approval

    with col_center:
        _render_activity()     # Activity feed + results

    with col_right:
        _render_metrics()      # Dashboard + report button
```

Session state keys: `p8_stage`, `p8_thread_id`, `p8_activity_log`, `p8_result`, `p8_approval_pending`.

Include Reset button and doc viewer: `doc_viewer.render("docs/08-autonomous-operations.md", title="Documentation: Autonomous Operations")`.

- [ ] **Step 2: Update app.py**

Add import: `from pages import p8_operations`
Add tab: `"🚀 Autonomous Ops"` as tab 8
Update unpacking to include `tab8`

- [ ] **Step 3: Verify imports**

```bash
cd app && python -c "from pages import p8_operations; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app/pages/p8_operations.py app/app.py
git commit -m "feat(app): add P8 autonomous operations Streamlit page and tab"
```

---

### Task 15: Documentation

**Files:**
- Create: `projects/08-autonomous-operations/README.md`
- Create: `docs/08-autonomous-operations.md`

- [ ] **Step 1: Create README.md**

Quick project overview: what it demonstrates, LangGraph/DeepAgents concepts, how to run (pytest, evaluation, graph.py smoke test), file structure, the 6 departments, sample requests.

- [ ] **Step 2: Create docs/08-autonomous-operations.md**

Full educational document covering:
1. Introduction — from single agents to autonomous operations
2. Two-layer architecture — LangGraph outer + DeepAgents inner
3. Master orchestrator graph — nodes, edges, task queue loop
4. Parallel execution with Send — fan-out to departments
5. Autonomous task cascading — follow-up tasks, the task queue loop
6. Tiered approval — risk assessment, interrupt/Command pattern
7. Department agents — DeepAgent factories, SKILL.md, tools
8. State schema design — reducers, metrics, task queue
9. Reporting — structured metrics + narrative summary
10. Persistence — SqliteSaver vs InMemorySaver split
11. LangSmith observability — cross-agent tracing, evaluation
12. Testing strategy — unit, integration, evaluation
13. Code walkthrough — key snippets from actual files
14. Key takeaways — concept integration matrix

- [ ] **Step 3: Commit**

```bash
git add projects/08-autonomous-operations/README.md docs/08-autonomous-operations.md
git commit -m "docs(p8): add README and educational documentation for capstone project"
```

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| **1: Foundation** | 1-5 | Scaffold, models, 12 data modules, risk rules, 17 tools |
| **2: Departments** | 6-8 | Prompts, 6 SKILL.md files, 6 DeepAgent factories |
| **3: Orchestrator** | 9-11 | Node functions, graph assembly, integration tests |
| **4: Polish** | 12-15 | Evaluation, Streamlit adapter+page, documentation |

**Total: 15 tasks, ~45 files created/modified**
