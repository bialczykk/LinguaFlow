# LinguaFlow Autonomous Operations — Design Specification

**Project:** 08-autonomous-operations (Capstone)
**Date:** 2026-03-27
**Status:** Approved

## Overview

The capstone project integrates every concept from Projects 1–7 into a unified autonomous operations system for LinguaFlow. A master orchestrator manages 6 department agents that can plan their own work, delegate to sub-agents, generate follow-up tasks for other departments, and request human approval for high-risk actions. The system operates semi-autonomously: a single user request can cascade through multiple departments without further input, unless a high-risk action is encountered.

### Learning Goals

- Everything from P1–P7 integrated into a cohesive system
- Cross-department agent coordination with autonomous task cascading
- Autonomous task planning and execution at scale (DeepAgents TodoList)
- Advanced HITL patterns: tiered approval (auto-approve low-risk, human-approve high-risk)
- Full LangSmith observability: end-to-end tracing, automated evaluations, cost monitoring
- Persistent orchestrator state across sessions (SqliteSaver)

## Architecture

### Two-Layer Hybrid: LangGraph + DeepAgents

**Outer layer — LangGraph StateGraph (master orchestrator):**
- Request classification and routing
- Parallel department dispatch via `Send`
- Cross-department task queue management
- Tiered HITL approval gates
- Response aggregation and metrics tracking

**Inner layer — 6 DeepAgent department agents:**
- Each created via `create_deep_agent()` with department-specific tools
- SKILL.md files for domain knowledge
- TodoList for autonomous task planning
- SubAgentMiddleware for delegation within departments

This mirrors P7's hybrid pattern (LangGraph wrapping DeepAgents) but at a larger scale — the orchestrator coordinates 6 departments instead of 4 generation stages.

### Interaction Model: Hybrid (Request-Driven + Autonomous)

- User-initiated requests enter through the orchestrator
- Department agents can generate follow-up tasks that go back into the orchestrator's queue
- The orchestrator processes both paths identically — classify, assess risk, dispatch, aggregate

### Persistence Split

- **Master orchestrator:** `SqliteSaver` — survives restarts, manages task queue and metrics
- **Department agents:** `InMemorySaver` — ephemeral per-request

## Master Orchestrator Graph

```
START -> request_classifier
             |
             +-- (user request) -> risk_assessor
             |                        |
             |                        +-- (low risk) -> dispatch_departments [via Send]
             |                        |
             |                        +-- (high risk) -> approval_gate [interrupt] -> dispatch_departments
             |
             +-- (follow-up task from dept) -> dispatch_departments [direct, already classified]
             |
             v
         dispatch_departments -- [Send to 1-N dept agents in parallel]
             |
             v
         department_executor (runs DeepAgent, returns result)
             |
             v
         result_aggregator
             |
             +-- (has follow-up tasks) -> enqueue tasks -> check_task_queue
             |
             +-- (all done) -> compose_output
             |
             v
         check_task_queue
             |
             +-- (tasks pending) -> request_classifier (loop)
             |
             +-- (queue empty) -> compose_output
             |
             v
         compose_output -> reporting_snapshot -> END
```

### Nodes

| Node | Type | Purpose |
|------|------|---------|
| `request_classifier` | LLM | Classifies request into departments, detects complexity |
| `risk_assessor` | Logic | Checks action against risk rules |
| `approval_gate` | Interrupt | Pauses for human approval on high-risk actions |
| `dispatch_departments` | Send | Fans out to 1-N department executors in parallel |
| `department_executor` | DeepAgent | Runs the appropriate department's DeepAgent |
| `result_aggregator` | Logic | Collects results, extracts follow-up tasks |
| `check_task_queue` | Logic | Decides whether to loop back or finish |
| `compose_output` | LLM | Merges all results into unified response |
| `reporting_snapshot` | Logic | Updates metrics store with this request's data |

The task queue loop is what makes it "autonomous" — a single user request can cascade through multiple departments without further user input (unless a high-risk action is encountered).

## State Schema

```python
class DepartmentResult(TypedDict):
    department: str
    response: str
    resolved: bool
    follow_up_tasks: list[dict]  # [{"target_dept": "qa", "action": "review", "context": {...}}]
    metrics: dict                # {"actions_taken": 2, "tools_called": ["search_tutors", "assign_tutor"]}

class MetricsStore(TypedDict):
    students_onboarded: int
    tutors_assigned: int
    content_generated: int
    content_published: int
    qa_reviews: int
    qa_flags: int
    support_requests: int
    support_resolved: int
    total_requests: int
    department_invocations: dict[str, int]  # {"support": 5, "content_pipeline": 3, ...}

class OrchestratorState(TypedDict):
    # Input
    request: str
    request_metadata: dict                                  # user_id, priority, source

    # Classification
    classification: dict                                    # departments, action_type, complexity, summary
    risk_level: str                                         # "low" | "high"

    # Approval
    approval_status: str                                    # "approved" | "rejected" | "not_required"

    # Department results (reducer: operator.add for parallel Send)
    department_results: Annotated[list[DepartmentResult], operator.add]

    # Task queue (autonomous follow-ups)
    task_queue: list[dict]                                  # pending follow-up tasks
    current_task: dict | None                               # task being processed right now
    completed_tasks: list[dict]                             # finished tasks (for audit trail)

    # Metrics
    metrics_store: MetricsStore

    # Output
    final_response: str
    resolution_status: str                                  # "resolved" | "partial" | "pending_approval"
```

### Data Flow: Cascading Request Example

**Request:** "Onboard student Maria, B1 level, interested in business English"

1. `request_classifier` → `{"departments": ["student_onboarding"], "action_type": "onboard"}`
2. `risk_assessor` → `risk_level: "low"` (assessment is low risk)
3. `dispatch_departments` → `Send("department_executor", {dept: "student_onboarding", ...})`
4. `department_executor` runs Student Onboarding DeepAgent → returns result with `follow_up_tasks: [{"target_dept": "tutor_management", "action": "match_tutor", "context": {"student_id": "S010", "level": "B1"}}]`
5. `result_aggregator` → moves follow-up into `task_queue`
6. `check_task_queue` → task pending, pops it into `current_task`, loops back to `request_classifier`
7. `request_classifier` → detects it's a follow-up task, classifies to tutor_management
8. `risk_assessor` → `risk_level: "high"` (assign_tutor is high risk)
9. `approval_gate` → `interrupt("Assign tutor T003 to student S010?")`
10. User approves → `dispatch_departments` → tutor management runs
11. No more follow-ups → `compose_output` merges everything → `reporting_snapshot` updates counters → END

## Department Agents

Each is a DeepAgent created via `create_deep_agent()` with department-specific tools, skills, and system prompt. All return a structured `DepartmentResult` with optional follow-up tasks.

### Student Onboarding

**Tools:**
- `assess_student(profile)` — Evaluates student profile and determines CEFR level
- `create_study_plan(student_id, level, goals)` — Creates a personalized study plan

**SKILL.md:** CEFR assessment criteria, study plan templates

**Follow-ups generated:** → Tutor Management: "match tutor for student X"

### Tutor Management

**Tools:**
- `search_tutors(criteria)` — Search tutors by specialty, availability, rating
- `check_availability(tutor_id)` — Check a tutor's open slots
- `assign_tutor(student_id, tutor_id)` — Assign a tutor to a student

**SKILL.md:** Tutor matching heuristics, scheduling rules

**Follow-ups generated:** → Student Onboarding: "update study plan with tutor info"

### Content Pipeline

**Tools:**
- `generate_content(topic, type, level)` — Generate educational content
- `submit_for_review(content_id)` — Submit content to QA
- `publish_content(content_id)` — Publish approved content

**SKILL.md:** Content standards, CEFR-aligned writing guidelines

**Follow-ups generated:** → QA: "review content X" (auto-generated on submit)

### Quality Assurance

**Tools:**
- `review_content(content_id)` — Review content against quality standards
- `flag_issue(department, issue)` — Flag a quality issue to a department
- `check_satisfaction(student_id)` — Check student satisfaction metrics

**SKILL.md:** QA rubrics, quality thresholds

**Follow-ups generated:** → Content Pipeline: "revise content X" if fails QA

### Support

**Tools:**
- `lookup_invoice(student_id)` — Find invoices for a student
- `check_schedule(student_id)` — Check lesson schedule
- `check_system_status(service)` — Check platform service health
- `check_enrollment(student_id)` — Check course enrollments

**SKILL.md:** Escalation rules, response templates

**Follow-ups generated:** → Billing/Scheduling follow-ups as needed

**Reuse:** Tools adapted from P6's billing, tech support, scheduling, and content tools.

### Reporting

**Tools:**
- `aggregate_metrics(department, period)` — Pull metrics from the metrics store
- `get_department_state(department)` — Get recent activity for a department

**SKILL.md:** Report templates, KPI definitions

**Follow-ups generated:** None (terminal — produces output only)

**Output:** Structured metrics dict + LLM-generated narrative summary.

## Tiered Approval (HITL)

Two tiers, rule-based classification via `risk_assessor` node (pure logic, no LLM).

### Low Risk (auto-execute)

- Data lookups (invoices, schedules, enrollments, system status)
- Content generation (drafts, not publication)
- Student assessments
- Metrics aggregation
- Tutor searches and availability checks

### High Risk (human approval required)

- Refund processing
- Content publication
- New tutor assignment
- Flagging issues to departments
- Creating study plans (commits resources)

```python
HIGH_RISK_ACTIONS = {
    "content_pipeline": {"publish_content"},
    "support": {"process_refund"},
    "tutor_management": {"assign_tutor"},
    "quality_assurance": {"flag_issue"},
    "student_onboarding": {"create_study_plan"},
}
```

When high-risk is detected, `approval_gate` calls `interrupt()` with a payload showing what's about to happen, which department, and why it's flagged. The user approves or rejects via `Command(resume=...)`.

Follow-up tasks generated autonomously by departments also pass through `risk_assessor`. If onboarding creates a "match tutor" follow-up, that gets flagged as high-risk and pauses for approval before the tutor assignment happens.

## Reporting & Metrics

### Metrics Store

An in-memory dict persisted via SqliteSaver as part of orchestrator state. Accumulates structured metrics per department. Updated by `reporting_snapshot` node after every completed request.

### Reporting Agent

Invoked through the orchestrator like any other department. When a user requests an operations summary:

1. Calls `aggregate_metrics("all", "current")` to pull the metrics store
2. Calls `get_department_state(dept)` for each active department
3. Uses SKILL.md (KPI definitions, report templates) to generate a narrative summary
4. Returns structured metrics + narrative in its DepartmentResult

The Streamlit UI renders both: metric cards for structured data, narrative summary below.

## Mock Data

### Reused from P6 (adapted)

- `invoices.py` — Student invoices with varying statuses
- `accounts.py` — Student accounts
- `lessons.py` — Lesson schedules with dynamic dates
- `content_library.py` — Course catalog and enrollments
- `system_status.py` — Platform service health

### New Modules

| Module | Contents |
|--------|----------|
| `students.py` | 5-6 student profiles with name, email, CEFR level, goals, enrollment date |
| `tutors.py` | 6-8 tutors with specialties, availability slots, rating, max students |
| `study_plans.py` | Pre-built study plans linked to student IDs |
| `content_drafts.py` | Content items in various pipeline stages (draft, in_review, published, flagged) |
| `qa_records.py` | QA review history with pass/fail, flags, reviewer notes |
| `metrics_seed.py` | Seed data for the metrics store |
| `sample_requests.py` | Sample requests for demo and evaluation |

Student IDs are consistent across all data modules.

## Tools Summary

17 tools total, organized by department. Each uses `@tool` + `@traceable` with tag `["p8-autonomous-operations"]`.

| Department | Tools | Count |
|---|---|---|
| Student Onboarding | `assess_student`, `create_study_plan` | 2 |
| Tutor Management | `search_tutors`, `check_availability`, `assign_tutor` | 3 |
| Content Pipeline | `generate_content`, `submit_for_review`, `publish_content` | 3 |
| Quality Assurance | `review_content`, `flag_issue`, `check_satisfaction` | 3 |
| Support | `lookup_invoice`, `check_schedule`, `check_system_status`, `check_enrollment` | 4 |
| Reporting | `aggregate_metrics`, `get_department_state` | 2 |

## Sample Requests

| Request | Departments | Pattern |
|---------|-------------|---------|
| "Onboard new student Maria, B1 level, interested in business English" | student_onboarding → tutor_management | Cascade with approval |
| "Generate a B2 reading passage about climate change and publish it" | content_pipeline → quality_assurance | Cascade with QA + publish approval |
| "I was charged twice and can't access my recordings" | support (multi-dept parallel) | Parallel fan-out |
| "How is the platform performing this week?" | reporting | Single department |
| "Review all recently published content for quality" | quality_assurance | Single department |
| "Find me a tutor for advanced conversation practice" | tutor_management | Single with approval |
| "Cancel my Thursday lesson and refund me" | support (billing + scheduling) | Parallel + refund approval |
| "Check if student S001 is happy with their progress" | quality_assurance | Single, low risk |

## Streamlit UI

### Three-Panel Layout

**Left column (narrow) — Operations Console:**
- Request input: text area + sample request selector dropdown
- Task queue viewer: live list of pending/in-progress/completed tasks with department badges
- Approval panel: when high-risk action pending, shows context with Approve/Reject buttons

**Center column (wide) — Results & Activity:**
- Activity feed: chronological log of orchestrator actions
- Final response: composed output when request completes
- Behind the scenes: expandable showing department results, follow-up task chain, risk assessments

**Right column (narrow) — Metrics Dashboard:**
- Metric cards: students onboarded, content published, support resolved, etc.
- Department activity: table showing invocations per department
- Narrative summary: LLM-generated operations summary (on-demand via "Generate Report" button)

### Session State

All keys prefixed `p8_`. Reset button clears all.

### Adapter (`app/adapters/autonomous_ops.py`)

```python
def start_request(thread_id, request_text, metadata) -> dict | str
def resume_approval(thread_id, decision) -> dict | str
def get_state(thread_id) -> dict
def get_metrics(thread_id) -> MetricsStore
def get_task_queue(thread_id) -> list[dict]
def get_sample_requests() -> list[dict]
def create_thread_id() -> str
```

### Tab Registration

Add `"🚀 Autonomous Ops"` as tab 8 in `app/app.py`.

## Testing

### Unit Tests

| File | Coverage |
|------|----------|
| `test_models.py` | State schema, DepartmentResult with follow_up_tasks, MetricsStore |
| `test_tools.py` | All 17 tools in isolation against mock data |
| `test_nodes.py` | Each orchestrator node with mocked LLMs |
| `test_risk.py` | Risk assessor: correct classification for every action type |

### Integration Tests (`test_graph.py`)

All with mocked LLMs, marked `@pytest.mark.integration`:

| Test | Scenario |
|------|----------|
| `test_single_department_low_risk` | Simple lookup → auto-executes → returns result |
| `test_single_department_high_risk` | Publish request → approval gate → approve → executes |
| `test_cascading_follow_ups` | Onboarding → generates tutor match follow-up → processes both |
| `test_multi_department_parallel` | Support request hitting 2 departments via Send |
| `test_approval_rejection` | High-risk action → rejected → resolution_status = "rejected" |
| `test_task_queue_loop` | Chain of 3 follow-up tasks processed in sequence |
| `test_reporting_metrics_update` | Verify metrics_store increments after request completes |

### LangSmith Evaluation (`evaluation.py`)

- **Routing accuracy**: correct department classification (deterministic)
- **Response quality**: LLM-as-judge on final composed output
- **Task chain completeness**: did all follow-up tasks get processed?

## File Structure

```
projects/08-autonomous-operations/
├── models.py
├── graph.py
├── nodes.py
├── prompts.py
├── tools.py
├── risk.py
├── departments.py
├── evaluation.py
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
    ├── test_nodes.py
    ├── test_risk.py
    └── test_graph.py
```

Plus:
- `app/adapters/autonomous_ops.py`
- `app/pages/p8_operations.py`
- `app/app.py` (tab 8 registration)
- `docs/08-autonomous-operations.md`
- `projects/08-autonomous-operations/README.md`
