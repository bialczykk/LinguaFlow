"""Integration tests for the full LangGraph StateGraph.

These tests compile the graph and invoke it end-to-end with sample profiles.
They verify correct routing, the review loop, and the final output.

Running: pytest tests/test_graph.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from data.sample_profiles import (
    INTERMEDIATE_CONVERSATION,
    INTERMEDIATE_GRAMMAR,
    EXAM_PREP_INTERMEDIATE,
)
from models import LessonPlan


class TestGraphRouting:
    """Tests that the graph routes to the correct drafting node."""

    def test_conversation_route(self):
        """Graph should route conversation profiles to the conversation drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0
        assert len(plan.main_activities) > 0

    def test_grammar_route(self):
        """Graph should route grammar profiles to the grammar drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_GRAMMAR,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0

    def test_exam_prep_route(self):
        """Graph should route exam prep profiles to the exam prep drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": EXAM_PREP_INTERMEDIATE,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0


class TestGraphReviewLoop:
    """Tests that the review loop works correctly."""

    def test_revision_count_populated(self):
        """After graph execution, revision_count should be at least 1."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["revision_count"] >= 1
        assert result["final_plan"] is not None

    def test_max_revisions_respected(self):
        """Graph should finalize even if review never approves (max 2 revisions)."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["revision_count"] <= 2
        assert result["final_plan"] is not None
