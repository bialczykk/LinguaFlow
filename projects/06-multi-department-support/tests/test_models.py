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
        from typing import get_type_hints, get_args
        full_hints = get_type_hints(SupportState, include_extras=True)
        dept_type = full_hints["department_results"]
        args = get_args(dept_type)
        assert args[1] is operator.add, "department_results must use operator.add reducer"
