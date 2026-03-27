"""Tests for orchestrator node functions — Project 08 Autonomous Operations.

Tests use mocked LLM responses to verify node logic without API calls.
Follows TDD: tests are written first and define the expected interface.

Mocking strategy:
- LLM-based nodes: patch the module-level model or ChatAnthropic constructor
- Pure-logic nodes: call directly with crafted state dicts
- interrupt/Command nodes: test the inputs they produce and the routing they return
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from models import OrchestratorState, DepartmentResult, MetricsStore


# ---------------------------------------------------------------------------
# Helper: build a full OrchestratorState with sensible defaults
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> OrchestratorState:
    """Return an OrchestratorState populated with safe defaults.

    Any keyword argument overrides the corresponding default field, allowing
    tests to set only the fields they care about.
    """
    defaults: OrchestratorState = {
        "request": "Please onboard a new student",
        "request_metadata": {
            "user_id": "U001",
            "priority": "medium",
            "source": "api",
        },
        "classification": {},
        "risk_level": "low",
        "approval_status": "not_required",
        "department_results": [],
        "task_queue": [],
        "current_task": None,
        "completed_tasks": [],
        "metrics_store": {
            "students_onboarded": 0,
            "tutors_assigned": 0,
            "content_generated": 0,
            "content_published": 0,
            "qa_reviews": 0,
            "qa_flags": 0,
            "support_requests": 0,
            "support_resolved": 0,
            "total_requests": 5,
            "department_invocations": {},
        },
        "final_response": "",
        "resolution_status": "",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# TestRequestClassifier
# ---------------------------------------------------------------------------

class TestRequestClassifier:
    """Tests for the request_classifier node."""

    @patch("nodes._classifier_model")
    def test_classifies_user_request(self, mock_model):
        """Classifier should parse LLM JSON and return classification dict."""
        from nodes import request_classifier

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "departments": ["student_onboarding"],
            "action_type": "onboard",
            "complexity": "single",
            "summary": "New student onboarding request",
        })
        mock_model.invoke.return_value = mock_response

        state = _make_state(request="Please onboard student S001", current_task=None)
        result = request_classifier(state)

        assert "classification" in result
        assert result["classification"]["departments"] == ["student_onboarding"]
        assert result["classification"]["action_type"] == "onboard"
        assert result["classification"]["complexity"] == "single"

    @patch("nodes._classifier_model")
    def test_classifies_follow_up_task(self, mock_model):
        """When current_task is set, classifier should use follow-up context."""
        from nodes import request_classifier

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "departments": ["tutor_management"],
            "action_type": "assign_tutor",
            "complexity": "single",
            "summary": "Assign a tutor to newly onboarded student",
        })
        mock_model.invoke.return_value = mock_response

        state = _make_state(
            request="Please onboard student S001",
            current_task={
                "target_dept": "tutor_management",
                "action": "assign_tutor",
                "context": {"student_id": "S001", "level": "B1"},
            },
        )
        result = request_classifier(state)

        # The node must have invoked the model (follow-up context passed through)
        assert mock_model.invoke.called
        assert result["classification"]["departments"] == ["tutor_management"]

    @patch("nodes._classifier_model")
    def test_handles_markdown_fenced_json(self, mock_model):
        """Classifier should strip ```json fences before parsing."""
        from nodes import request_classifier

        mock_response = MagicMock()
        mock_response.content = (
            "```json\n"
            '{"departments": ["reporting"], "action_type": "aggregate_metrics",'
            ' "complexity": "single", "summary": "Metrics report"}\n'
            "```"
        )
        mock_model.invoke.return_value = mock_response

        state = _make_state()
        result = request_classifier(state)

        assert result["classification"]["departments"] == ["reporting"]

    @patch("nodes._classifier_model")
    def test_fallback_on_unparseable_response(self, mock_model):
        """Classifier should return a fallback classification for unparseable output."""
        from nodes import request_classifier

        mock_response = MagicMock()
        mock_response.content = "I cannot classify this."
        mock_model.invoke.return_value = mock_response

        state = _make_state()
        result = request_classifier(state)

        # Fallback: departments list must be present (may be empty)
        assert "classification" in result
        assert "departments" in result["classification"]


# ---------------------------------------------------------------------------
# TestRiskAssessor
# ---------------------------------------------------------------------------

class TestRiskAssessor:
    """Tests for the risk_assessor node — pure logic, no mocks needed."""

    def test_low_risk_returns_not_required(self):
        """Low-risk actions should set approval_status to not_required."""
        from nodes import risk_assessor

        state = _make_state(classification={
            "departments": ["reporting"],
            "action_type": "aggregate_metrics",
        })
        result = risk_assessor(state)

        assert result["risk_level"] == "low"
        assert result["approval_status"] == "not_required"

    def test_high_risk_action_is_flagged(self):
        """High-risk actions should return risk_level=high with empty approval_status."""
        from nodes import risk_assessor

        state = _make_state(classification={
            "departments": ["content_pipeline"],
            "action_type": "publish_content",
        })
        result = risk_assessor(state)

        assert result["risk_level"] == "high"
        # approval_status should be empty string (awaiting gate decision)
        assert result["approval_status"] == ""

    def test_high_risk_support_refund(self):
        """Support + process_refund must be flagged as high risk."""
        from nodes import risk_assessor

        state = _make_state(classification={
            "departments": ["support"],
            "action_type": "process_refund",
        })
        result = risk_assessor(state)

        assert result["risk_level"] == "high"

    def test_low_risk_unknown_action(self):
        """Unknown department/action combinations default to low risk."""
        from nodes import risk_assessor

        state = _make_state(classification={
            "departments": ["support"],
            "action_type": "lookup_invoice",
        })
        result = risk_assessor(state)

        assert result["risk_level"] == "low"


# ---------------------------------------------------------------------------
# TestApprovalGate
# ---------------------------------------------------------------------------

class TestApprovalGate:
    """Tests for the approval_gate node — uses interrupt/Command."""

    @patch("nodes.interrupt")
    def test_approved_routes_to_dispatch(self, mock_interrupt):
        """When the operator approves, gate should route to dispatch_departments."""
        from nodes import approval_gate
        from langgraph.types import Command

        # interrupt() returns the operator's resume value
        mock_interrupt.return_value = "approved"

        state = _make_state(
            classification={
                "departments": ["content_pipeline"],
                "action_type": "publish_content",
                "summary": "Publish B2 grammar lesson",
            },
            risk_level="high",
        )
        result = approval_gate(state)

        assert isinstance(result, Command)
        assert result.goto == "dispatch_departments"

    @patch("nodes.interrupt")
    def test_rejected_routes_to_compose_output(self, mock_interrupt):
        """When the operator rejects, gate should route to compose_output."""
        from nodes import approval_gate
        from langgraph.types import Command

        mock_interrupt.return_value = "rejected"

        state = _make_state(
            classification={
                "departments": ["support"],
                "action_type": "process_refund",
                "summary": "Refund invoice #1234",
            },
            risk_level="high",
        )
        result = approval_gate(state)

        assert isinstance(result, Command)
        assert result.goto == "compose_output"
        assert result.update["approval_status"] == "rejected"


# ---------------------------------------------------------------------------
# TestResultAggregator
# ---------------------------------------------------------------------------

class TestResultAggregator:
    """Tests for the result_aggregator node — pure logic."""

    def test_extracts_follow_up_tasks(self):
        """Follow-up tasks from department results should be moved to task_queue."""
        from nodes import result_aggregator

        follow_up = {
            "target_dept": "tutor_management",
            "action": "assign_tutor",
            "context": {"student_id": "S001", "level": "B1"},
        }
        dept_result: DepartmentResult = {
            "department": "student_onboarding",
            "response": "Student onboarded.",
            "resolved": True,
            "follow_up_tasks": [follow_up],
            "metrics": {"actions_taken": 2, "tools_called": ["assess_student"]},
        }
        state = _make_state(
            department_results=[dept_result],
            completed_tasks=[],
            task_queue=[],
        )
        result = result_aggregator(state)

        assert len(result["task_queue"]) == 1
        assert result["task_queue"][0]["target_dept"] == "tutor_management"
        # Original result should be in completed_tasks
        assert len(result["completed_tasks"]) == 1

    def test_no_follow_ups_leaves_queue_empty(self):
        """Department results with no follow_up_tasks should not populate task_queue."""
        from nodes import result_aggregator

        dept_result: DepartmentResult = {
            "department": "reporting",
            "response": "Metrics aggregated.",
            "resolved": True,
            "follow_up_tasks": [],
            "metrics": {"actions_taken": 1, "tools_called": ["aggregate_metrics"]},
        }
        state = _make_state(
            department_results=[dept_result],
            completed_tasks=[],
            task_queue=[],
        )
        result = result_aggregator(state)

        assert result["task_queue"] == []
        assert len(result["completed_tasks"]) == 1

    def test_multiple_results_aggregated(self):
        """Follow-ups from multiple department results should all be collected."""
        from nodes import result_aggregator

        results = [
            {
                "department": "student_onboarding",
                "response": "Student onboarded.",
                "resolved": True,
                "follow_up_tasks": [
                    {"target_dept": "tutor_management", "action": "assign_tutor", "context": {}}
                ],
                "metrics": {},
            },
            {
                "department": "content_pipeline",
                "response": "Content generated.",
                "resolved": True,
                "follow_up_tasks": [
                    {"target_dept": "quality_assurance", "action": "review_content", "context": {}}
                ],
                "metrics": {},
            },
        ]
        state = _make_state(department_results=results, completed_tasks=[], task_queue=[])
        result = result_aggregator(state)

        assert len(result["task_queue"]) == 2
        assert len(result["completed_tasks"]) == 2


# ---------------------------------------------------------------------------
# TestCheckTaskQueue
# ---------------------------------------------------------------------------

class TestCheckTaskQueue:
    """Tests for the check_task_queue node — returns Command objects."""

    def test_pops_next_task_when_queue_has_items(self):
        """Non-empty queue should pop the first task and route to request_classifier."""
        from nodes import check_task_queue
        from langgraph.types import Command

        task1 = {"target_dept": "tutor_management", "action": "assign_tutor", "context": {}}
        task2 = {"target_dept": "quality_assurance", "action": "review_content", "context": {}}
        state = _make_state(task_queue=[task1, task2], current_task=None)

        result = check_task_queue(state)

        assert isinstance(result, Command)
        assert result.goto == "request_classifier"
        assert result.update["current_task"] == task1
        assert result.update["task_queue"] == [task2]

    def test_empty_queue_routes_to_compose_output(self):
        """Empty queue should clear current_task and route to compose_output."""
        from nodes import check_task_queue
        from langgraph.types import Command

        state = _make_state(task_queue=[], current_task=None)
        result = check_task_queue(state)

        assert isinstance(result, Command)
        assert result.goto == "compose_output"
        assert result.update["current_task"] is None


# ---------------------------------------------------------------------------
# TestComposeOutput
# ---------------------------------------------------------------------------

class TestComposeOutput:
    """Tests for the compose_output node — LLM-based and rejection short-circuit."""

    @patch("nodes.ChatAnthropic")
    def test_compose_output_calls_llm(self, mock_chat_cls):
        """compose_output should invoke LLM and set final_response."""
        from nodes import compose_output

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The student has been successfully onboarded."
        mock_model_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_model_instance

        dept_result: DepartmentResult = {
            "department": "student_onboarding",
            "response": "Student onboarded with study plan.",
            "resolved": True,
            "follow_up_tasks": [],
            "metrics": {},
        }
        state = _make_state(
            department_results=[dept_result],
            approval_status="not_required",
            completed_tasks=[],
        )
        result = compose_output(state)

        assert result["final_response"] == "The student has been successfully onboarded."
        assert result["resolution_status"] == "resolved"

    def test_rejected_skips_llm(self):
        """When approval_status is rejected, compose_output should skip the LLM."""
        from nodes import compose_output

        state = _make_state(
            department_results=[],
            approval_status="rejected",
        )
        result = compose_output(state)

        # Should return a rejection message without calling the LLM
        assert result["resolution_status"] == "rejected"
        assert "rejected" in result["final_response"].lower()

    @patch("nodes.ChatAnthropic")
    def test_partial_resolution_status(self, mock_chat_cls):
        """If some department results are unresolved, status should be partial."""
        from nodes import compose_output

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Partial completion."
        mock_model_instance.invoke.return_value = mock_response
        mock_chat_cls.return_value = mock_model_instance

        dept_result: DepartmentResult = {
            "department": "support",
            "response": "Issue partially handled.",
            "resolved": False,
            "follow_up_tasks": [],
            "metrics": {},
        }
        state = _make_state(
            department_results=[dept_result],
            approval_status="not_required",
            completed_tasks=[],
        )
        result = compose_output(state)

        assert result["resolution_status"] == "partial"


# ---------------------------------------------------------------------------
# TestReportingSnapshot
# ---------------------------------------------------------------------------

class TestReportingSnapshot:
    """Tests for the reporting_snapshot node — pure metrics logic."""

    def test_increments_total_requests(self):
        """total_requests should be incremented by 1."""
        from nodes import reporting_snapshot

        state = _make_state(
            department_results=[],
            metrics_store={
                "students_onboarded": 0,
                "tutors_assigned": 0,
                "content_generated": 0,
                "content_published": 0,
                "qa_reviews": 0,
                "qa_flags": 0,
                "support_requests": 0,
                "support_resolved": 0,
                "total_requests": 5,
                "department_invocations": {},
            },
        )
        result = reporting_snapshot(state)

        assert result["metrics_store"]["total_requests"] == 6

    def test_increments_department_invocation_counts(self):
        """department_invocations counter should reflect departments that ran."""
        from nodes import reporting_snapshot

        dept_results = [
            {
                "department": "student_onboarding",
                "response": "Done.",
                "resolved": True,
                "follow_up_tasks": [],
                "metrics": {},
            },
            {
                "department": "tutor_management",
                "response": "Done.",
                "resolved": True,
                "follow_up_tasks": [],
                "metrics": {},
            },
        ]
        state = _make_state(
            department_results=dept_results,
            metrics_store={
                "students_onboarded": 2,
                "tutors_assigned": 1,
                "content_generated": 0,
                "content_published": 0,
                "qa_reviews": 0,
                "qa_flags": 0,
                "support_requests": 0,
                "support_resolved": 0,
                "total_requests": 10,
                "department_invocations": {"student_onboarding": 2},
            },
        )
        result = reporting_snapshot(state)

        invocations = result["metrics_store"]["department_invocations"]
        assert invocations["student_onboarding"] == 3
        assert invocations["tutor_management"] == 1

    def test_increments_students_onboarded(self):
        """student_onboarding department should increment students_onboarded."""
        from nodes import reporting_snapshot

        state = _make_state(
            department_results=[
                {
                    "department": "student_onboarding",
                    "response": "Student onboarded.",
                    "resolved": True,
                    "follow_up_tasks": [],
                    "metrics": {},
                }
            ],
            metrics_store={
                "students_onboarded": 3,
                "tutors_assigned": 0,
                "content_generated": 0,
                "content_published": 0,
                "qa_reviews": 0,
                "qa_flags": 0,
                "support_requests": 0,
                "support_resolved": 0,
                "total_requests": 3,
                "department_invocations": {},
            },
        )
        result = reporting_snapshot(state)

        assert result["metrics_store"]["students_onboarded"] == 4

    def test_increments_tutors_assigned(self):
        """tutor_management department should increment tutors_assigned."""
        from nodes import reporting_snapshot

        state = _make_state(
            department_results=[
                {
                    "department": "tutor_management",
                    "response": "Tutor assigned.",
                    "resolved": True,
                    "follow_up_tasks": [],
                    "metrics": {},
                }
            ],
            metrics_store={
                "students_onboarded": 0,
                "tutors_assigned": 1,
                "content_generated": 0,
                "content_published": 0,
                "qa_reviews": 0,
                "qa_flags": 0,
                "support_requests": 0,
                "support_resolved": 0,
                "total_requests": 1,
                "department_invocations": {},
            },
        )
        result = reporting_snapshot(state)

        assert result["metrics_store"]["tutors_assigned"] == 2
