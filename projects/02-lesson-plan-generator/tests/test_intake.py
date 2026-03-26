"""Integration tests for the intake conversation handler.

Tests simulate a multi-turn conversation where the student provides
their information, and verify that the handler produces a valid
StudentProfile when complete.

Running: pytest tests/test_intake.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from models import StudentProfile


class TestIntakeConversation:
    """Tests for IntakeConversation class."""

    def test_initial_state_not_complete(self):
        """A freshly created intake should not be complete."""
        from intake import IntakeConversation

        intake = IntakeConversation()
        assert intake.is_complete() is False

    def test_conversation_produces_profile(self):
        """A full conversation should produce a valid StudentProfile."""
        from intake import IntakeConversation

        intake = IntakeConversation()

        # First turn: introduce self and goals
        response1 = intake.ask(
            "Hi! I'm Maria, and I want to improve my English speaking."
        )
        assert len(response1) > 0

        # Second turn: provide level and topics
        response2 = intake.ask(
            "I think my level is around B1. I really like talking about "
            "travel and movies."
        )
        assert len(response2) > 0

        # Third turn: clarify goals
        response3 = intake.ask(
            "I mainly want to practice conversation and become more fluent."
        )
        assert len(response3) > 0

        # The LLM may need one more turn or may be done
        if not intake.is_complete():
            response4 = intake.ask("Yes, that sounds right! Let's go.")
            assert len(response4) > 0

        # If still not complete after 4 turns, force completion for test
        if not intake.is_complete():
            intake.ask("Yes, please create my lesson plan now.")

        assert intake.is_complete() is True

        profile = intake.get_profile()
        assert isinstance(profile, StudentProfile)
        assert profile.name == "Maria" or len(profile.name) > 0
        assert profile.proficiency_level in ["A1", "A2", "B1", "B2", "C1", "C2"]
        assert len(profile.learning_goals) > 0
        assert profile.lesson_type in ["conversation", "grammar", "exam_prep"]
