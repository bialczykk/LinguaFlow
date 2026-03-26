"""Tests for graph node functions and routing logic.

Tests the node functions that wrap DeepAgents invocations and the
interrupt-based HITL review nodes. Uses mocking to avoid actual
LLM calls while verifying the control flow.
"""

import json
from unittest.mock import patch, MagicMock

import pytest


def _make_state(**overrides):
    """Create a minimal valid CurriculumEngineState dict."""
    base = {
        "curriculum_request": {
            "topic": "Test Topic",
            "level": "B1",
            "preferences": {},
        },
        "curriculum_plan": None,
        "plan_feedback": "",
        "lesson": None,
        "lesson_feedback": "",
        "exercises": None,
        "exercises_feedback": "",
        "assessment": None,
        "assessment_feedback": "",
        "assembled_module": None,
        "current_step": "idle",
    }
    base.update(overrides)
    return base


def test_review_plan_node_interrupts():
    """review_plan should call interrupt() with the plan data."""
    from nodes import review_plan_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test"},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "approve"}
        result = review_plan_node(state)

    mock_interrupt.assert_called_once()
    call_args = mock_interrupt.call_args[0][0]
    assert call_args["plan"]["title"] == "Test Module"


def test_review_plan_node_captures_feedback_on_revise():
    """When moderator requests revision, feedback should be captured."""
    from nodes import review_plan_node

    state = _make_state(
        curriculum_plan={"title": "Test"},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "revise", "feedback": "Add more exercises"}
        result = review_plan_node(state)

    assert result["plan_feedback"] == "Add more exercises"


def test_review_lesson_node_interrupts():
    """review_lesson should call interrupt() with the lesson artifact."""
    from nodes import review_lesson_node

    state = _make_state(
        lesson={"content": "# Lesson", "artifact_type": "lesson", "agent_todos": []},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "approve"}
        result = review_lesson_node(state)

    mock_interrupt.assert_called_once()


def test_route_after_review_approve():
    """Approve action should route to the next generation step."""
    from nodes import route_after_plan_review

    state = _make_state(
        curriculum_plan={"title": "Test"},
        plan_feedback="",
    )
    result = route_after_plan_review(state)
    assert result == "generate_lesson"


def test_route_after_review_revise():
    """Revise action should route back to the same generation step."""
    from nodes import route_after_plan_review

    state = _make_state(
        curriculum_plan={"title": "Test"},
        plan_feedback="Needs more detail",
    )
    result = route_after_plan_review(state)
    assert result == "plan_curriculum"


def test_assemble_module_combines_all_artifacts():
    """assemble_module should combine plan, lesson, exercises, and assessment."""
    from nodes import assemble_module_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test module"},
        lesson={"content": "# Lesson Content", "artifact_type": "lesson", "agent_todos": []},
        exercises={"content": "# Exercises", "artifact_type": "exercises", "agent_todos": []},
        assessment={"content": "# Assessment", "artifact_type": "assessment", "agent_todos": []},
    )

    result = assemble_module_node(state)
    assembled = result["assembled_module"]

    assert "Test Module" in assembled
    assert "# Lesson Content" in assembled
    assert "# Exercises" in assembled
    assert "# Assessment" in assembled


def test_assemble_module_handles_rejected_artifacts():
    """If an artifact was rejected (None), assembly should note it."""
    from nodes import assemble_module_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test"},
        lesson={"content": "# Lesson", "artifact_type": "lesson", "agent_todos": []},
        exercises=None,
        assessment={"content": "# Assessment", "artifact_type": "assessment", "agent_todos": []},
    )

    result = assemble_module_node(state)
    assembled = result["assembled_module"]

    assert "# Lesson" in assembled
    assert "# Assessment" in assembled
    assert "not included" in assembled.lower() or "skipped" in assembled.lower()
