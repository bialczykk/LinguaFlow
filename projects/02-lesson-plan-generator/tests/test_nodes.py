"""Integration tests for individual node functions.

Each test calls a single node function with a pre-built state dict
and verifies it returns the expected state updates. These tests hit
the real Anthropic API.

Running: pytest tests/test_nodes.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from the repo root so ANTHROPIC_API_KEY is available.
_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from data.sample_profiles import INTERMEDIATE_CONVERSATION, INTERMEDIATE_GRAMMAR, EXAM_PREP_INTERMEDIATE
from models import LessonPlanState, LessonPlan


@pytest.fixture
def conversation_state() -> LessonPlanState:
    """A minimal state dict for testing with a conversation profile."""
    return {
        "student_profile": INTERMEDIATE_CONVERSATION,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


@pytest.fixture
def grammar_state() -> LessonPlanState:
    """A minimal state dict for testing with a grammar profile."""
    return {
        "student_profile": INTERMEDIATE_GRAMMAR,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


@pytest.fixture
def exam_state() -> LessonPlanState:
    """A minimal state dict for testing with an exam prep profile."""
    return {
        "student_profile": EXAM_PREP_INTERMEDIATE,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


class TestResearchNode:
    """Tests for the research node function."""

    def test_research_returns_notes(self, conversation_state):
        """Research node should populate research_notes with content."""
        from nodes import research_node

        result = research_node(conversation_state)
        assert "research_notes" in result
        assert len(result["research_notes"]) > 50

    def test_research_notes_mention_student_topics(self, conversation_state):
        """Research notes should be relevant to the student's topics."""
        from nodes import research_node

        result = research_node(conversation_state)
        notes_lower = result["research_notes"].lower()
        # Carlos's topics are travel and food — at least one should appear
        assert "travel" in notes_lower or "food" in notes_lower or "restaurant" in notes_lower


class TestDraftNodes:
    """Tests for the three drafting node functions."""

    def test_draft_conversation(self, conversation_state):
        """Conversation draft node should produce a draft plan."""
        from nodes import research_node, draft_conversation_node

        research_result = research_node(conversation_state)
        conversation_state.update(research_result)

        result = draft_conversation_node(conversation_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100

    def test_draft_grammar(self, grammar_state):
        """Grammar draft node should produce a draft plan."""
        from nodes import research_node, draft_grammar_node

        research_result = research_node(grammar_state)
        grammar_state.update(research_result)

        result = draft_grammar_node(grammar_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100

    def test_draft_exam_prep(self, exam_state):
        """Exam prep draft node should produce a draft plan."""
        from nodes import research_node, draft_exam_prep_node

        research_result = research_node(exam_state)
        exam_state.update(research_result)

        result = draft_exam_prep_node(exam_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100


class TestReviewNode:
    """Tests for the review node function."""

    def test_review_sets_approval_and_feedback(self, conversation_state):
        """Review node should set is_approved and review_feedback."""
        from nodes import research_node, draft_conversation_node, review_node

        conversation_state.update(research_node(conversation_state))
        conversation_state.update(draft_conversation_node(conversation_state))

        result = review_node(conversation_state)
        assert "is_approved" in result
        assert isinstance(result["is_approved"], bool)
        assert "review_feedback" in result
        assert len(result["review_feedback"]) > 0
        assert "revision_count" in result
        assert result["revision_count"] == 1


class TestFinalizeNode:
    """Tests for the finalize node function."""

    def test_finalize_produces_lesson_plan(self, conversation_state):
        """Finalize node should produce a valid LessonPlan."""
        from nodes import research_node, draft_conversation_node, finalize_node

        conversation_state.update(research_node(conversation_state))
        conversation_state.update(draft_conversation_node(conversation_state))
        conversation_state["is_approved"] = True

        result = finalize_node(conversation_state)
        assert "final_plan" in result
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0
        assert len(plan.main_activities) > 0
        assert plan.estimated_duration_minutes > 0
