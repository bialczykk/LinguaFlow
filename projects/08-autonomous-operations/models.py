"""State schema and model definitions for the autonomous operations system.

OrchestratorState is the shared state for the master orchestrator graph.
DepartmentResult is the structured output from each department agent.
MetricsStore tracks cumulative platform metrics.

The department_results field uses an operator.add reducer so that parallel
sub-agents (dispatched via Send) can each append their result independently
without overwriting each other — LangGraph merges them by concatenation.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class DepartmentResult(TypedDict):
    """Structured result from a department agent.

    Each department returns one of these, appended to department_results.
    follow_up_tasks enables autonomous cascading — one department's output
    can trigger work in another department without human re-invocation.

    Example:
        Student onboarding completes and emits a follow-up task for
        tutor_management to assign a tutor for the newly onboarded student.
    """

    department: str              # One of DEPARTMENTS (e.g. "student_onboarding")
    response: str                # The agent's human-readable response text
    resolved: bool               # Whether the agent fully handled its part
    follow_up_tasks: list[dict]  # [{"target_dept": str, "action": str, "context": dict}]
    metrics: dict                # {"actions_taken": int, "tools_called": list[str]}


class MetricsStore(TypedDict):
    """Cumulative platform metrics, updated after each completed request.

    Persisted via SqliteSaver as part of orchestrator state so metrics
    survive across sessions and graph invocations. Each field corresponds
    to a measurable outcome from one of the six department agents.
    """

    students_onboarded: int          # How many students have been fully onboarded
    tutors_assigned: int             # How many tutor-student matches created
    content_generated: int           # Number of content pieces drafted
    content_published: int           # Number of content pieces published
    qa_reviews: int                  # Total QA review passes performed
    qa_flags: int                    # Number of QA issues flagged
    support_requests: int            # Total support tickets received
    support_resolved: int            # Support tickets successfully resolved
    total_requests: int              # Grand total of orchestrator invocations
    department_invocations: dict[str, int]  # Per-department call counts


class OrchestratorState(TypedDict):
    """Shared state for the master orchestrator graph.

    Flows through every node in the StateGraph lifecycle:
      1. Input fields are set at graph invocation.
      2. Classification fields are populated by the request_classifier node.
      3. Approval fields are managed by risk_assessor and approval_gate nodes.
      4. Department results are appended by parallel department nodes (via Send).
      5. Task queue fields drive autonomous follow-up cascading.
      6. Metrics are updated by the metrics_updater node.
      7. Output fields are set by the compose_output node.

    The Annotated[..., operator.add] on department_results is critical:
    when LangGraph dispatches multiple department agents in parallel using
    Send(), each agent returns a list with one result; operator.add concatenates
    all of those lists into a single unified list in state.
    """

    # --- Input (set at graph.invoke()) ---
    request: str                          # Natural language request from user/system
    request_metadata: dict                # {"user_id": str, "priority": str, "source": str}

    # --- Classification (set by request_classifier node) ---
    classification: dict                  # {"departments": list, "action_type": str, "complexity": str}
    risk_level: str                       # "low" | "high"

    # --- Approval (set by risk_assessor and approval_gate nodes) ---
    approval_status: str                  # "approved" | "rejected" | "not_required"

    # --- Department results (reducer: operator.add for parallel Send) ---
    # Each parallel department node returns {"department_results": [single_result]}.
    # operator.add concatenates all those single-element lists into one list.
    department_results: Annotated[list[DepartmentResult], operator.add]

    # --- Task queue (autonomous follow-up cascading) ---
    task_queue: list[dict]                # Pending follow-up tasks from department results
    current_task: dict | None             # Task currently being dispatched
    completed_tasks: list[dict]           # Finished tasks for audit trail

    # --- Metrics (updated by metrics_updater node) ---
    metrics_store: MetricsStore

    # --- Output (set by compose_output node) ---
    final_response: str                   # Consolidated human-readable response
    resolution_status: str                # "resolved" | "partial" | "pending_approval"


# The six department agents in the LinguaFlow platform.
# Each value maps to a DeepAgent with its own SKILL.md and toolset.
DEPARTMENTS: set[str] = {
    "student_onboarding",   # Intake, assessment, study plan creation
    "tutor_management",     # Tutor matching, scheduling, payroll
    "content_pipeline",     # Lesson content generation and publishing
    "quality_assurance",    # Review and flag content or tutor sessions
    "support",              # Handle student/tutor support requests
    "reporting",            # Metrics, dashboards, and platform analytics
}
