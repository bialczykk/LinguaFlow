"""Pydantic models for structured grammar feedback.

These models define the schema for the grammar correction agent's output.
The LLM is constrained to return data matching these schemas via
ChatAnthropic's .with_structured_output() method.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GrammarIssue(BaseModel):
    """A single grammar issue found in the student's writing.

    Each issue captures the original error, the correction, and an
    educational explanation so the student understands the rule.
    """

    original_text: str = Field(
        description="The problematic fragment from the student's writing"
    )
    corrected_text: str = Field(
        description="The corrected version of the fragment"
    )
    error_category: str = Field(
        description=(
            "Grammar category, e.g. 'subject-verb agreement', "
            "'tense', 'article usage', 'punctuation', 'word order'"
        )
    )
    explanation: str = Field(
        description=(
            "Educational explanation of why this is wrong and how "
            "the grammar rule works — written for a language learner"
        )
    )
    severity: Literal["minor", "moderate", "major"] = Field(
        description="How impactful the error is on comprehension"
    )


class ProficiencyAssessment(BaseModel):
    """Overall CEFR proficiency assessment based on the writing sample.

    Uses the Common European Framework of Reference (CEFR) scale,
    the international standard for describing language ability.
    """

    cefr_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Assessed CEFR proficiency level"
    )
    strengths: list[str] = Field(
        description="What the student does well in their writing"
    )
    areas_to_improve: list[str] = Field(
        description="Key areas where the student should focus improvement"
    )
    summary: str = Field(
        description="Brief overall assessment of the student's writing level"
    )


class GrammarFeedback(BaseModel):
    """Complete grammar feedback for a student writing sample.

    This is the top-level model returned by the analysis chain.
    It contains individual grammar issues, an overall proficiency
    assessment, and the full corrected text.
    """

    issues: list[GrammarIssue] = Field(
        description="All grammar issues found in the writing sample"
    )
    proficiency: ProficiencyAssessment = Field(
        description="Overall CEFR proficiency assessment"
    )
    corrected_full_text: str = Field(
        description="The entire student submission with all corrections applied"
    )
