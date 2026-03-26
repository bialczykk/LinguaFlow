"""Tests for Pydantic models and state schema.

Validates that all models accept correct data, enforce constraints,
and reject invalid inputs.
"""

import pytest
from models import (
    CriterionScore,
    CriteriaScores,
    SampleComparison,
    ComparativeAnalysis,
    Assessment,
)


# -- CriterionScore --

def test_criterion_score_valid():
    """CriterionScore accepts valid data with all required fields."""
    score = CriterionScore(
        dimension="Grammar & Accuracy",
        score=4,
        evidence=["Uses complex sentences correctly", "Minor article errors"],
        feedback="Strong grammar with occasional article misuse.",
    )
    assert score.dimension == "Grammar & Accuracy"
    assert score.score == 4
    assert len(score.evidence) == 2
    assert "Strong grammar" in score.feedback


def test_criterion_score_rejects_out_of_range():
    """CriterionScore rejects scores outside 1-5 range."""
    with pytest.raises(ValueError):
        CriterionScore(
            dimension="Grammar & Accuracy",
            score=0,
            evidence=["example"],
            feedback="feedback",
        )
    with pytest.raises(ValueError):
        CriterionScore(
            dimension="Grammar & Accuracy",
            score=6,
            evidence=["example"],
            feedback="feedback",
        )


# -- CriteriaScores --

def test_criteria_scores_valid():
    """CriteriaScores bundles multiple CriterionScore with a preliminary level."""
    scores = CriteriaScores(
        scores=[
            CriterionScore(
                dimension="Grammar & Accuracy",
                score=3,
                evidence=["example"],
                feedback="feedback",
            ),
            CriterionScore(
                dimension="Vocabulary Range & Precision",
                score=4,
                evidence=["example"],
                feedback="feedback",
            ),
        ],
        preliminary_level="B1",
        scoring_rationale="Scores average around B1 descriptors.",
    )
    assert scores.preliminary_level == "B1"
    assert len(scores.scores) == 2


def test_criteria_scores_rejects_invalid_level():
    """CriteriaScores rejects CEFR levels not in the valid set."""
    with pytest.raises(ValueError):
        CriteriaScores(
            scores=[],
            preliminary_level="D1",
            scoring_rationale="Invalid level.",
        )


# -- SampleComparison --

def test_sample_comparison_valid():
    """SampleComparison captures how a submission compares to one sample."""
    comp = SampleComparison(
        sample_level="B1",
        similarities=["Similar vocabulary range"],
        differences=["More complex sentence structure"],
        quality_position="above",
    )
    assert comp.quality_position == "above"


def test_sample_comparison_rejects_invalid_position():
    """SampleComparison rejects quality_position not in allowed set."""
    with pytest.raises(ValueError):
        SampleComparison(
            sample_level="B1",
            similarities=["x"],
            differences=["y"],
            quality_position="equal",
        )


# -- ComparativeAnalysis --

def test_comparative_analysis_valid():
    """ComparativeAnalysis bundles comparisons with a narrative."""
    analysis = ComparativeAnalysis(
        comparisons=[
            SampleComparison(
                sample_level="B1",
                similarities=["x"],
                differences=["y"],
                quality_position="comparable",
            ),
        ],
        narrative="The submission is comparable to B1 samples.",
    )
    assert len(analysis.comparisons) == 1
    assert "comparable" in analysis.narrative


# -- Assessment --

def test_assessment_valid():
    """Assessment is the complete final output with all fields."""
    assessment = Assessment(
        submission_text="The student wrote this essay.",
        overall_level="B1",
        criteria_scores=[
            CriterionScore(
                dimension="Grammar & Accuracy",
                score=3,
                evidence=["example"],
                feedback="feedback",
            ),
        ],
        comparative_summary="Comparable to B1 samples.",
        strengths=["Good vocabulary"],
        areas_to_improve=["Sentence variety"],
        recommendations=["Practice complex sentences"],
        confidence="high",
    )
    assert assessment.overall_level == "B1"
    assert assessment.confidence == "high"
    assert len(assessment.strengths) == 1


def test_assessment_rejects_invalid_confidence():
    """Assessment rejects confidence values not in allowed set."""
    with pytest.raises(ValueError):
        Assessment(
            submission_text="text",
            overall_level="B1",
            criteria_scores=[],
            comparative_summary="summary",
            strengths=[],
            areas_to_improve=[],
            recommendations=[],
            confidence="very_high",
        )


def test_assessment_rejects_invalid_level():
    """Assessment rejects CEFR levels not in the valid set."""
    with pytest.raises(ValueError):
        Assessment(
            submission_text="text",
            overall_level="X9",
            criteria_scores=[],
            comparative_summary="summary",
            strengths=[],
            areas_to_improve=[],
            recommendations=[],
            confidence="high",
        )
