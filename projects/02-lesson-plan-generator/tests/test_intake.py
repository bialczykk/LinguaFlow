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
        """A full conversation should produce a valid StudentProfile.

        The LLM drives the conversation, so the number of turns can vary.
        We provide all required info upfront and then nudge completion
        with increasingly direct messages.
        """
        from intake import IntakeConversation

        intake = IntakeConversation()

        # Provide all info in one dense message to minimize turn count
        response1 = intake.ask(
            "Hi! I'm Maria. My English level is B1. I want to practice "
            "conversation and become more fluent. I'm interested in travel "
            "and movies. Please create my lesson plan!"
        )
        assert len(response1) > 0

        # If the LLM still wants to confirm or ask follow-ups, nudge it
        # toward completion with progressively more direct messages
        nudges = [
            "Yes, that's all correct! I'm ready for my lesson plan.",
            "I've given you everything you need. My name is Maria, level B1, "
            "I want conversation practice about travel and movies. Please proceed.",
            "Yes, confirmed. Please finalize my profile now.",
        ]

        for nudge in nudges:
            if intake.is_complete():
                break
            intake.ask(nudge)

        assert intake.is_complete() is True, (
            "Intake did not complete after providing all info and 3 nudges. "
            "The LLM may not be emitting [PROFILE_COMPLETE] reliably."
        )

        profile = intake.get_profile()
        assert isinstance(profile, StudentProfile)
        assert len(profile.name) > 0
        assert profile.proficiency_level in ["A1", "A2", "B1", "B2", "C1", "C2"]
        assert len(profile.learning_goals) > 0
        assert profile.lesson_type in ["conversation", "grammar", "exam_prep"]
