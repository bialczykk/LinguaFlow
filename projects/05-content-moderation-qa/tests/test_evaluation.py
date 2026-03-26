# tests/test_evaluation.py
"""Tests for custom evaluator functions.

These test the evaluator logic only — no LangSmith API calls.
Each evaluator takes a run-like dict and example-like dict and returns a score dict.
"""

import pytest


class TestTopicRelevanceEvaluator:
    def test_returns_score_dict(self):
        from evaluation import topic_relevance_evaluator

        # Simulate a run output and example
        run = type("Run", (), {"outputs": {"content": "The present perfect tense is used for..."}})()
        example = type("Example", (), {"inputs": {"topic": "Present Perfect Tense"}})()

        result = topic_relevance_evaluator(run, example)
        assert "key" in result
        assert result["key"] == "topic_relevance"
        assert "score" in result
        assert isinstance(result["score"], (int, float))


class TestDifficultyMatchEvaluator:
    def test_returns_score_dict(self):
        from evaluation import difficulty_match_evaluator

        run = type("Run", (), {"outputs": {"content": "Simple words. Easy grammar."}})()
        example = type("Example", (), {"inputs": {"difficulty": "A1"}})()

        result = difficulty_match_evaluator(run, example)
        assert result["key"] == "difficulty_match"
        assert isinstance(result["score"], (int, float))


class TestContentQualityEvaluator:
    def test_returns_score_dict(self):
        from evaluation import content_quality_evaluator

        run = type("Run", (), {"outputs": {"content": "A well-written grammar explanation."}})()
        example = type("Example", (), {"inputs": {}})()

        result = content_quality_evaluator(run, example)
        assert result["key"] == "content_quality"
        assert isinstance(result["score"], (int, float))
