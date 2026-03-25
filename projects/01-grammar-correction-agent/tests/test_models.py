"""Tests for grammar feedback Pydantic models."""

import pytest
from pydantic import ValidationError


def test_grammar_issue_valid():
    """GrammarIssue accepts valid data with all required fields."""
    from models import GrammarIssue

    issue = GrammarIssue(
        original_text="He go to school every day.",
        corrected_text="He goes to school every day.",
        error_category="subject-verb agreement",
        explanation="Third-person singular subjects require 's' on the verb in present simple.",
        severity="major",
    )
    assert issue.original_text == "He go to school every day."
    assert issue.severity == "major"


def test_grammar_issue_invalid_severity():
    """GrammarIssue rejects severity values outside the allowed literals."""
    from models import GrammarIssue

    with pytest.raises(ValidationError):
        GrammarIssue(
            original_text="test",
            corrected_text="test",
            error_category="test",
            explanation="test",
            severity="critical",
        )


def test_proficiency_assessment_valid():
    """ProficiencyAssessment accepts valid CEFR levels and lists."""
    from models import ProficiencyAssessment

    assessment = ProficiencyAssessment(
        cefr_level="B1",
        strengths=["Good vocabulary range", "Clear sentence structure"],
        areas_to_improve=["Article usage", "Verb tenses"],
        summary="Intermediate level with solid foundations.",
    )
    assert assessment.cefr_level == "B1"
    assert len(assessment.strengths) == 2


def test_proficiency_assessment_invalid_cefr():
    """ProficiencyAssessment rejects invalid CEFR levels."""
    from models import ProficiencyAssessment

    with pytest.raises(ValidationError):
        ProficiencyAssessment(
            cefr_level="D1",
            strengths=[],
            areas_to_improve=[],
            summary="test",
        )


def test_grammar_feedback_valid():
    """GrammarFeedback composes issues and proficiency into a full response."""
    from models import GrammarIssue, GrammarFeedback, ProficiencyAssessment

    feedback = GrammarFeedback(
        issues=[
            GrammarIssue(
                original_text="He go",
                corrected_text="He goes",
                error_category="subject-verb agreement",
                explanation="Third person singular needs 's'.",
                severity="major",
            )
        ],
        proficiency=ProficiencyAssessment(
            cefr_level="A2",
            strengths=["Simple vocabulary"],
            areas_to_improve=["Verb conjugation"],
            summary="Beginner with basic communication ability.",
        ),
        corrected_full_text="He goes to school every day.",
    )
    assert len(feedback.issues) == 1
    assert feedback.proficiency.cefr_level == "A2"


def test_grammar_feedback_empty_issues():
    """GrammarFeedback allows an empty issues list (perfect text)."""
    from models import GrammarFeedback, ProficiencyAssessment

    feedback = GrammarFeedback(
        issues=[],
        proficiency=ProficiencyAssessment(
            cefr_level="C2",
            strengths=["Flawless grammar"],
            areas_to_improve=[],
            summary="Native-like proficiency.",
        ),
        corrected_full_text="This text is perfect.",
    )
    assert len(feedback.issues) == 0
