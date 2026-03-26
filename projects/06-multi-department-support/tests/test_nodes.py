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
