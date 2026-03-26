"""Unit tests for Pydantic models and LangGraph state schema.

Tests validate that all models enforce their constraints correctly:
- StudentProfile enforces CEFR levels and lesson types
- Activity validates duration and materials
- LessonPlan validates the complete lesson structure
"""

import pytest
from pydantic import ValidationError

from models import StudentProfile, Activity, LessonPlan


class TestStudentProfile:
    """Tests for StudentProfile Pydantic model."""

    def test_valid_profile(self):
        profile = StudentProfile(
            name="Maria",
            proficiency_level="B1",
            learning_goals=["improve fluency", "learn idioms"],
            preferred_topics=["travel", "food"],
            lesson_type="conversation",
        )
        assert profile.name == "Maria"
        assert profile.proficiency_level == "B1"
        assert profile.lesson_type == "conversation"

    def test_all_cefr_levels(self):
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            profile = StudentProfile(
                name="Test",
                proficiency_level=level,
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="grammar",
            )
            assert profile.proficiency_level == level

    def test_invalid_cefr_level(self):
        with pytest.raises(ValidationError):
            StudentProfile(
                name="Test",
                proficiency_level="D1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="grammar",
            )

    def test_all_lesson_types(self):
        for lesson_type in ["conversation", "grammar", "exam_prep"]:
            profile = StudentProfile(
                name="Test",
                proficiency_level="B1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type=lesson_type,
            )
            assert profile.lesson_type == lesson_type

    def test_invalid_lesson_type(self):
        with pytest.raises(ValidationError):
            StudentProfile(
                name="Test",
                proficiency_level="B1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="yoga",
            )


class TestActivity:
    """Tests for Activity Pydantic model."""

    def test_valid_activity(self):
        activity = Activity(
            name="Role-play: Ordering Food",
            description="Students practice ordering at a restaurant.",
            duration_minutes=15,
            materials=["menu handout", "vocabulary list"],
        )
        assert activity.name == "Role-play: Ordering Food"
        assert activity.duration_minutes == 15
        assert len(activity.materials) == 2

    def test_activity_empty_materials(self):
        activity = Activity(
            name="Free discussion",
            description="Open conversation on the topic.",
            duration_minutes=10,
            materials=[],
        )
        assert activity.materials == []


class TestLessonPlan:
    """Tests for LessonPlan Pydantic model."""

    def test_valid_lesson_plan(self):
        plan = LessonPlan(
            title="Travel English: At the Airport",
            level="B1",
            lesson_type="conversation",
            objectives=["Practice check-in dialogue", "Learn travel vocabulary"],
            warm_up="Discuss last travel experience",
            main_activities=[
                Activity(
                    name="Airport Role-play",
                    description="Simulate check-in counter interaction.",
                    duration_minutes=20,
                    materials=["dialogue script"],
                )
            ],
            wrap_up="Review new vocabulary learned today",
            homework="Write a short paragraph about your dream destination",
            estimated_duration_minutes=60,
        )
        assert plan.title == "Travel English: At the Airport"
        assert len(plan.main_activities) == 1
        assert plan.estimated_duration_minutes == 60

    def test_lesson_plan_multiple_activities(self):
        activities = [
            Activity(
                name=f"Activity {i}",
                description=f"Description {i}",
                duration_minutes=10,
                materials=[],
            )
            for i in range(3)
        ]
        plan = LessonPlan(
            title="Grammar Drills",
            level="A2",
            lesson_type="grammar",
            objectives=["Practice present tense"],
            warm_up="Warm up",
            main_activities=activities,
            wrap_up="Wrap up",
            homework="Homework",
            estimated_duration_minutes=45,
        )
        assert len(plan.main_activities) == 3
