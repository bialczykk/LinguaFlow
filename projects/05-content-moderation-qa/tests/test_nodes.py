# tests/test_nodes.py
"""Tests for node functions and routing logic.

Unit tests for routing functions (no LLM).
Integration tests for generate node (hits LLM).
"""

import pytest
from langchain_core.messages import AIMessage

from models import ContentModerationState
from nodes import (
    route_after_draft_review,
    route_after_final_review,
    generate_node,
    publish_node,
)


class TestRouteAfterDraftReview:
    """Routing logic after draft_review interrupt."""

    def test_approve_routes_to_polish(self):
        state = {"draft_decision": {"action": "approve"}, "revision_count": 0}
        assert route_after_draft_review(state) == "polish"

    def test_edit_routes_to_polish(self):
        state = {
            "draft_decision": {"action": "edit", "edited_content": "Better version"},
            "revision_count": 0,
        }
        assert route_after_draft_review(state) == "polish"

    def test_reject_routes_to_revise(self):
        state = {
            "draft_decision": {"action": "reject", "feedback": "Too basic"},
            "revision_count": 0,
        }
        assert route_after_draft_review(state) == "revise"

    def test_reject_at_max_revisions_routes_to_end(self):
        state = {
            "draft_decision": {"action": "reject", "feedback": "Still bad"},
            "revision_count": 2,
        }
        assert route_after_draft_review(state) == "__end__"


class TestRouteAfterFinalReview:
    """Routing logic after final_review interrupt."""

    def test_approve_routes_to_publish(self):
        state = {"final_decision": {"action": "approve"}}
        assert route_after_final_review(state) == "publish"

    def test_reject_routes_to_end(self):
        state = {"final_decision": {"action": "reject", "feedback": "Not ready"}}
        assert route_after_final_review(state) == "__end__"


class TestPublishNode:
    """publish_node marks content as published."""

    def test_publish_sets_published_true(self):
        state = {
            "polished_content": "Some content",
            "revision_count": 1,
        }
        result = publish_node(state)
        assert result["published"] is True
        assert result["publish_metadata"] is not None
        assert result["publish_metadata"]["review_rounds"] == 1


@pytest.mark.integration
class TestGenerateNode:
    """generate_node produces content via LLM."""

    def test_generate_returns_content_and_confidence(self):
        state = {
            "content_request": {
                "topic": "Present Perfect Tense",
                "content_type": "grammar_explanation",
                "difficulty": "B1",
            },
            "revision_count": 0,
        }
        result = generate_node(state)
        assert "draft_content" in result
        assert len(result["draft_content"]) > 50
        assert "generation_confidence" in result
        assert 0.0 <= result["generation_confidence"] <= 1.0
