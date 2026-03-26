"""Pydantic models and LangGraph state schema for the Student Assessment Pipeline.

This module defines:
- CriterionScore: score for one assessment dimension (grammar, vocabulary, etc.)
- CriteriaScores: all dimension scores + preliminary CEFR level
- SampleComparison: how the submission compares to one sample essay
- ComparativeAnalysis: full comparative analysis across multiple samples
- Assessment: the complete final output
- AssessmentState: the TypedDict that flows through the LangGraph StateGraph

Key concepts demonstrated:
- Pydantic Field validators (ge/le for score range, Literal for constrained strings)
- TypedDict as LangGraph state schema with "last write wins" semantics
"""

from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Valid CEFR levels used across models
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


class CriterionScore(BaseModel):
    """Score for a single assessment dimension.

    Each dimension (grammar, vocabulary, coherence, task achievement) gets
    a 1-5 score, evidence quotes from the submission, and specific feedback.
    """

    dimension: str = Field(description="Assessment dimension name")
    score: int = Field(ge=1, le=5, description="Score from 1 (lowest) to 5 (highest)")
    evidence: list[str] = Field(description="Quotes/examples from the submission")
    feedback: str = Field(description="Specific feedback for this dimension")


class CriteriaScores(BaseModel):
    """Multi-criteria scoring results from the criteria_scoring node.

    Contains individual dimension scores and a preliminary CEFR level
    that drives the next retrieval phase (fetching level-appropriate samples).
    """

    scores: list[CriterionScore] = Field(description="Scores per dimension")
    preliminary_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Preliminary CEFR level based on aggregate scores"
    )
    scoring_rationale: str = Field(description="Why this level was assigned")


class SampleComparison(BaseModel):
    """Comparison of the submission against one retrieved sample essay.

    Captures what the submission shares with the sample, where it differs,
    and whether the submission is above, comparable to, or below the sample.
    """

    sample_level: str = Field(description="CEFR level of the sample essay")
    similarities: list[str] = Field(description="What the submission shares with this sample")
    differences: list[str] = Field(description="Where the submission diverges")
    quality_position: Literal["above", "comparable", "below"] = Field(
        description="Submission quality relative to this sample"
    )


class ComparativeAnalysis(BaseModel):
    """Full comparative analysis across all retrieved sample essays.

    Produced by the comparative_analysis node after comparing the submission
    against level-appropriate samples retrieved in the second retrieval phase.
    """

    comparisons: list[SampleComparison] = Field(description="Per-sample comparisons")
    narrative: str = Field(description="Overall narrative summary")


class Assessment(BaseModel):
    """Complete structured assessment — the final output of the pipeline.

    Merges criteria scores with comparative analysis into a single
    actionable assessment with recommendations for the student.
    """

    submission_text: str = Field(description="The original student submission")
    overall_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Final assessed CEFR level"
    )
    criteria_scores: list[CriterionScore] = Field(description="Per-dimension scores")
    comparative_summary: str = Field(description="How submission compares to samples")
    strengths: list[str] = Field(description="What the student does well")
    areas_to_improve: list[str] = Field(description="Key areas for improvement")
    recommendations: list[str] = Field(description="Actionable next steps")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Assessment confidence level"
    )


class AssessmentState(TypedDict):
    """State schema for the LangGraph StateGraph.

    This TypedDict defines the shared state flowing through every node.
    Each node reads what it needs and returns a partial dict updating only
    the fields it's responsible for.

    LangGraph concept: TypedDict state schemas
    - Every field uses "last write wins" (no reducers needed — each field
      is written by exactly one node)
    - Nodes return partial dicts like {"criteria_scores": ..., "preliminary_level": "B1"}
    """

    # -- Input (set at invocation) --
    submission_text: str
    submission_context: str
    student_level_hint: str

    # -- After retrieve_standards --
    retrieved_standards: list[Document]

    # -- After criteria_scoring --
    criteria_scores: CriteriaScores
    preliminary_level: str

    # -- After retrieve_samples --
    retrieved_samples: list[Document]

    # -- After comparative_analysis --
    comparative_analysis: ComparativeAnalysis

    # -- After synthesize --
    final_assessment: Assessment
