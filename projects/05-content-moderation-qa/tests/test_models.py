# tests/test_models.py
"""Tests for Pydantic models and state schema."""

import typing
import pytest
from models import ContentRequest, PublishMetadata, ContentModerationState


class TestContentRequest:
    def test_valid_request(self):
        req = ContentRequest(
            topic="Present Perfect Tense",
            content_type="grammar_explanation",
            difficulty="B1",
        )
        assert req.topic == "Present Perfect Tense"
        assert req.content_type == "grammar_explanation"
        assert req.difficulty == "B1"

    def test_difficulty_must_be_cefr(self):
        with pytest.raises(Exception):
            ContentRequest(
                topic="Test", content_type="grammar_explanation", difficulty="X9",
            )

    def test_content_type_constrained(self):
        with pytest.raises(Exception):
            ContentRequest(
                topic="Test", content_type="invalid_type", difficulty="A1",
            )


class TestPublishMetadata:
    def test_valid_metadata(self):
        meta = PublishMetadata(
            moderator_notes="Looks good",
            review_rounds=1,
        )
        assert meta.moderator_notes == "Looks good"
        assert meta.review_rounds == 1


class TestContentModerationState:
    def test_state_has_required_fields(self):
        hints = typing.get_type_hints(ContentModerationState)
        assert "content_request" in hints
        assert "draft_content" in hints
        assert "generation_confidence" in hints
        assert "draft_decision" in hints
        assert "revision_count" in hints
        assert "polished_content" in hints
        assert "final_decision" in hints
        assert "published" in hints
        assert "publish_metadata" in hints
