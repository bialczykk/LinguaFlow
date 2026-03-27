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
