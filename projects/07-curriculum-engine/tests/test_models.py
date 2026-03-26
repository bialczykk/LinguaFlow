"""Tests for Pydantic models and state schema."""

import pytest
from pydantic import ValidationError


def test_curriculum_request_valid():
    from models import CurriculumRequest

    req = CurriculumRequest(
        topic="Business English for meetings",
        level="B2",
        preferences={"teaching_style": "interactive", "focus_areas": ["vocabulary", "speaking"]},
    )
    assert req.topic == "Business English for meetings"
    assert req.level == "B2"
    assert req.preferences["teaching_style"] == "interactive"


def test_curriculum_request_defaults():
    from models import CurriculumRequest

    req = CurriculumRequest(topic="Grammar basics", level="A1")
    assert req.preferences == {}


def test_curriculum_request_invalid_level():
    from models import CurriculumRequest

    with pytest.raises(ValidationError):
        CurriculumRequest(topic="Test", level="X9")


def test_curriculum_plan_valid():
    from models import CurriculumPlan

    plan = CurriculumPlan(
        title="Business English Module",
        description="A module covering business meeting vocabulary and phrases",
        lesson_outline="Introduction to meeting vocabulary and common phrases",
        exercise_types=["fill-in-the-blank", "matching"],
        assessment_approach="Reading comprehension + writing prompt",
    )
    assert plan.title == "Business English Module"
    assert len(plan.exercise_types) == 2


def test_generated_artifact_valid():
    from models import GeneratedArtifact

    artifact = GeneratedArtifact(
        content="# Lesson: Business Meetings\n\n...",
        artifact_type="lesson",
        agent_todos=[
            {"content": "Draft lesson intro", "status": "completed"},
            {"content": "Add examples", "status": "completed"},
        ],
    )
    assert artifact.artifact_type == "lesson"
    assert len(artifact.agent_todos) == 2


def test_generated_artifact_invalid_type():
    from models import GeneratedArtifact

    with pytest.raises(ValidationError):
        GeneratedArtifact(content="test", artifact_type="invalid_type", agent_todos=[])


def test_curriculum_engine_state_has_required_keys():
    """CurriculumEngineState should be a TypedDict with all workflow fields."""
    from models import CurriculumEngineState

    annotations = CurriculumEngineState.__annotations__
    expected_keys = {
        "curriculum_request", "curriculum_plan", "plan_feedback",
        "lesson", "lesson_feedback",
        "exercises", "exercises_feedback",
        "assessment", "assessment_feedback",
        "assembled_module", "current_step",
    }
    assert expected_keys.issubset(set(annotations.keys()))


def test_cefr_levels_constant():
    from models import CEFR_LEVELS

    assert CEFR_LEVELS == ("A1", "A2", "B1", "B2", "C1", "C2")
