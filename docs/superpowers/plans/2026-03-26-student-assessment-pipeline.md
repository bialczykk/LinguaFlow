# Student Assessment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a RAG-powered LangGraph pipeline that assesses student writing by retrieving rubrics and sample essays from a Chroma vector store, scoring across criteria, and producing a comparative assessment.

**Architecture:** Two separate concerns — a standalone ingestion module that populates Chroma with rubrics, CEFR standards, and graded sample essays, and a 5-node LangGraph StateGraph (`retrieve_standards → criteria_scoring → retrieve_samples → comparative_analysis → synthesize`) where early scoring results drive later retrieval.

**Tech Stack:** LangGraph, LangChain (langchain-core, langchain-anthropic, langchain-chroma, langchain-huggingface), Chroma, sentence-transformers, LangSmith, pytest

---

## File Structure

```
projects/03-student-assessment-pipeline/
    requirements.txt            # Project dependencies
    models.py                   # Pydantic models + AssessmentState TypedDict
    ingestion.py                # Document ingestion into Chroma (build_vector_store, get_vector_store)
    prompts.py                  # All prompt templates for LLM nodes
    nodes.py                    # Node functions for the StateGraph
    graph.py                    # StateGraph assembly + compilation
    main.py                     # CLI entry point
    data/
        __init__.py
        rubrics.py              # CEFR rubric documents (12 docs: 3 level bands x 4 dimensions)
        standards.py            # CEFR level descriptors (6 docs: A1-C2)
        sample_essays.py        # Graded sample essays (~12-18 docs)
        sample_submissions.py   # Test submissions for running the pipeline
    tests/
        __init__.py
        conftest.py             # Shared fixtures (pre-populated Chroma store)
        test_models.py          # Model validation tests
        test_ingestion.py       # Ingestion pipeline tests
        test_nodes.py           # Individual node tests (integration, hits LLM)
        test_graph.py           # End-to-end graph tests (integration, hits LLM)
    chroma_db/                  # Persisted vector store (gitignored)
```

---

### Task 1: Scaffold project and install dependencies

**Files:**
- Create: `projects/03-student-assessment-pipeline/requirements.txt`
- Create: `projects/03-student-assessment-pipeline/data/__init__.py`
- Create: `projects/03-student-assessment-pipeline/tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create the project directory structure**

```bash
mkdir -p "projects/03-student-assessment-pipeline/data"
mkdir -p "projects/03-student-assessment-pipeline/tests"
```

- [ ] **Step 2: Create requirements.txt**

Create `projects/03-student-assessment-pipeline/requirements.txt`:
```
langchain-core>=0.3.0
langchain-anthropic>=0.3.0
langchain-chroma>=0.2.0
langchain-huggingface>=0.1.0
langgraph>=0.3.0
langsmith>=0.2.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 3: Create package init files**

Create `projects/03-student-assessment-pipeline/data/__init__.py`:
```python
"""Sample data for the Student Assessment Pipeline.

Contains CEFR rubrics, level descriptors, graded sample essays,
and test submissions used for both the vector store and testing.
"""
```

Create `projects/03-student-assessment-pipeline/tests/__init__.py`:
```python
```

- [ ] **Step 4: Add chroma_db to .gitignore**

Append to `.gitignore`:
```
# Chroma vector store (generated at runtime)
**/chroma_db/
```

- [ ] **Step 5: Install dependencies**

```bash
source .venv/bin/activate
pip install langchain-core langchain-anthropic langchain-chroma langchain-huggingface langgraph langsmith sentence-transformers chromadb python-dotenv pytest
```

- [ ] **Step 6: Commit**

```bash
git add projects/03-student-assessment-pipeline/requirements.txt \
       projects/03-student-assessment-pipeline/data/__init__.py \
       projects/03-student-assessment-pipeline/tests/__init__.py \
       .gitignore
git commit -m "feat(p3): scaffold project skeleton and dependencies"
```

---

### Task 2: Pydantic models and state schema (TDD)

**Files:**
- Create: `projects/03-student-assessment-pipeline/tests/test_models.py`
- Create: `projects/03-student-assessment-pipeline/models.py`

- [ ] **Step 1: Write failing tests for all Pydantic models**

Create `projects/03-student-assessment-pipeline/tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement all Pydantic models and state schema**

Create `projects/03-student-assessment-pipeline/models.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_models.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/03-student-assessment-pipeline/models.py \
       projects/03-student-assessment-pipeline/tests/test_models.py
git commit -m "feat(p3): add Pydantic models and state schema with tests"
```

---

### Task 3: Document data — rubrics, standards, and sample essays

**Files:**
- Create: `projects/03-student-assessment-pipeline/data/rubrics.py`
- Create: `projects/03-student-assessment-pipeline/data/standards.py`
- Create: `projects/03-student-assessment-pipeline/data/sample_essays.py`
- Create: `projects/03-student-assessment-pipeline/data/sample_submissions.py`

- [ ] **Step 1: Create CEFR rubrics**

Create `projects/03-student-assessment-pipeline/data/rubrics.py`:
```python
"""CEFR assessment rubrics for scoring student writing.

Each rubric describes what scores 1-5 look like for a specific assessment
dimension at a specific CEFR level band. These are stored in the Chroma
vector store and retrieved during the criteria_scoring phase.

4 dimensions x 3 level bands = 12 rubric documents.
"""

from langchain_core.documents import Document

# -- Grammar & Accuracy Rubrics --

GRAMMAR_A1_A2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Cannot form basic sentences. Frequent errors in word order, "
        "verb forms, and basic structures that prevent comprehension.\n"
        "Score 2: Attempts simple sentences but with frequent errors in subject-verb "
        "agreement, tense use, and articles. Meaning is often unclear.\n"
        "Score 3: Can produce simple sentences with some accuracy. Common errors in "
        "tenses and articles but meaning is generally clear. Uses present simple "
        "and past simple with moderate accuracy.\n"
        "Score 4: Good control of basic structures. Errors are infrequent and do not "
        "impede understanding. Beginning to use some complex structures.\n"
        "Score 5: Excellent control of A1-A2 grammar. Accurate use of present, past, "
        "and future tenses. Very few errors in basic structures."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "A1-A2"},
)

GRAMMAR_B1_B2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Struggles with intermediate structures. Frequent errors in "
        "conditionals, passive voice, and complex tenses impede meaning.\n"
        "Score 2: Attempts complex structures but with regular errors. Inconsistent "
        "use of reported speech, relative clauses, and modal verbs.\n"
        "Score 3: Reasonable control of intermediate grammar. Uses conditionals, "
        "passive voice, and relative clauses with some errors. Meaning is clear "
        "despite occasional mistakes.\n"
        "Score 4: Good range of structures used accurately. Occasional slips in "
        "complex areas but these do not impede communication.\n"
        "Score 5: Confident and accurate use of B1-B2 grammar. Complex sentences, "
        "conditionals, and passive constructions are well-formed."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "B1-B2"},
)

GRAMMAR_C1_C2 = Document(
    page_content=(
        "Grammar & Accuracy Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Errors in advanced structures undermine otherwise competent writing. "
        "Struggles with nuanced tense choices and complex subordination.\n"
        "Score 2: Some control of advanced grammar but noticeable errors in "
        "subjunctive mood, inversion, and cleft sentences.\n"
        "Score 3: Good control of advanced grammar. Uses a wide range of structures "
        "with occasional errors in the most complex constructions.\n"
        "Score 4: Consistently accurate use of advanced grammar. Minor slips "
        "are rare and stylistic rather than structural.\n"
        "Score 5: Near-native accuracy. Masterful use of nuanced grammar including "
        "inversions, cleft sentences, and sophisticated subordination."
    ),
    metadata={"type": "rubric", "dimension": "grammar", "level_band": "C1-C2"},
)

# -- Vocabulary Range & Precision Rubrics --

VOCABULARY_A1_A2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Extremely limited vocabulary. Relies on a few memorized words "
        "and phrases. Cannot express basic ideas without significant gaps.\n"
        "Score 2: Basic vocabulary for familiar topics but frequent gaps. Relies "
        "heavily on repetition and simple words.\n"
        "Score 3: Adequate vocabulary for everyday topics. Can describe basic "
        "situations using common words. Some word choice errors but meaning is clear.\n"
        "Score 4: Good range of basic vocabulary. Appropriate word choices for "
        "familiar contexts. Beginning to use some less common words.\n"
        "Score 5: Strong basic vocabulary. Precise word choices for everyday topics. "
        "Shows awareness of collocations at the basic level."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "A1-A2"},
)

VOCABULARY_B1_B2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Limited vocabulary for the level. Over-relies on basic words "
        "when intermediate vocabulary is expected. Frequent imprecision.\n"
        "Score 2: Some intermediate vocabulary but used inconsistently. "
        "Paraphrases when specific terms would be more appropriate.\n"
        "Score 3: Reasonable vocabulary range. Uses topic-specific terms and "
        "some idiomatic expressions. Occasional imprecision but generally effective.\n"
        "Score 4: Good intermediate vocabulary with appropriate use of less common "
        "words. Shows awareness of register and collocation.\n"
        "Score 5: Wide vocabulary range for the level. Precise, varied word choices "
        "including idiomatic expressions and topic-specific terminology."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "B1-B2"},
)

VOCABULARY_C1_C2 = Document(
    page_content=(
        "Vocabulary Range & Precision Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Vocabulary does not match the expected advanced level. "
        "Relies on intermediate-level words when sophisticated choices are needed.\n"
        "Score 2: Some advanced vocabulary but lacks precision. Occasional "
        "misuse of nuanced terms or inappropriate register.\n"
        "Score 3: Good advanced vocabulary. Uses sophisticated terms, academic "
        "language, and idiomatic expressions with reasonable precision.\n"
        "Score 4: Wide-ranging, precise vocabulary. Effective use of nuance, "
        "connotation, and register-appropriate language.\n"
        "Score 5: Near-native vocabulary mastery. Exceptional precision and range "
        "including rare words, technical terms, and subtle distinctions."
    ),
    metadata={"type": "rubric", "dimension": "vocabulary", "level_band": "C1-C2"},
)

# -- Coherence & Organization Rubrics --

COHERENCE_A1_A2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: No discernible organization. Isolated words or phrases "
        "without logical connection.\n"
        "Score 2: Minimal organization. Ideas are listed but not connected. "
        "No use of linking words.\n"
        "Score 3: Basic organization with simple linking words (and, but, because). "
        "Ideas follow a simple logical order. Paragraphing may be absent.\n"
        "Score 4: Clear basic organization. Uses simple connectors effectively. "
        "Ideas are logically sequenced with some paragraphing.\n"
        "Score 5: Well-organized for the level. Clear introduction and conclusion. "
        "Effective use of basic connectors and logical sequencing."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "A1-A2"},
)

COHERENCE_B1_B2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Poor organization for the level. Ideas are disjointed and "
        "transitions are missing or ineffective.\n"
        "Score 2: Some organization but inconsistent. Paragraphs exist but "
        "topic sentences and transitions are weak.\n"
        "Score 3: Reasonable organization. Clear paragraphs with topic sentences. "
        "Uses a range of linking devices (however, therefore, in addition) "
        "with some effectiveness.\n"
        "Score 4: Good organization with clear logical flow. Effective use of "
        "cohesive devices. Ideas are well-developed within paragraphs.\n"
        "Score 5: Excellent organization. Skillful use of transitions and "
        "cohesive devices. Strong paragraph structure with clear progression."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "B1-B2"},
)

COHERENCE_C1_C2 = Document(
    page_content=(
        "Coherence & Organization Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Organization does not match the expected advanced level. "
        "Lacks sophisticated structuring despite advanced language use.\n"
        "Score 2: Some advanced organizational features but inconsistent. "
        "Transitions between complex ideas are sometimes abrupt.\n"
        "Score 3: Good advanced organization. Effective use of discourse markers "
        "and referencing. Complex ideas are clearly structured.\n"
        "Score 4: Sophisticated organization. Skillful management of complex "
        "information with seamless transitions and cohesive referencing.\n"
        "Score 5: Masterful organization. Text flows naturally with implicit "
        "and explicit coherence devices. Complex arguments are elegantly structured."
    ),
    metadata={"type": "rubric", "dimension": "coherence", "level_band": "C1-C2"},
)

# -- Task Achievement Rubrics --

TASK_ACHIEVEMENT_A1_A2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band A1-A2\n\n"
        "Score 1: Does not address the task. Response is off-topic or "
        "incomprehensible.\n"
        "Score 2: Partially addresses the task. Some relevant content but "
        "significant parts of the prompt are ignored.\n"
        "Score 3: Addresses the main points of the task. Response is relevant "
        "though may lack detail or completeness.\n"
        "Score 4: Fully addresses the task with adequate detail. All parts "
        "of the prompt are covered.\n"
        "Score 5: Thoroughly addresses the task with good detail and development. "
        "Goes beyond the minimum requirements."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "A1-A2"},
)

TASK_ACHIEVEMENT_B1_B2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band B1-B2\n\n"
        "Score 1: Fails to address the task requirements. Response is superficial "
        "or largely irrelevant.\n"
        "Score 2: Partially addresses the task. Ideas are underdeveloped and "
        "some key points are missing.\n"
        "Score 3: Adequately addresses the task. Main ideas are developed with "
        "supporting details. Some aspects could be more thorough.\n"
        "Score 4: Fully addresses the task with well-developed ideas. Supporting "
        "evidence and examples are relevant and effective.\n"
        "Score 5: Comprehensively addresses the task. Ideas are fully developed "
        "with compelling evidence. Demonstrates critical thinking."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "B1-B2"},
)

TASK_ACHIEVEMENT_C1_C2 = Document(
    page_content=(
        "Task Achievement Rubric — CEFR Level Band C1-C2\n\n"
        "Score 1: Response does not meet advanced-level expectations for task "
        "completion. Analysis is superficial.\n"
        "Score 2: Some depth but insufficient for the level. Arguments lack "
        "nuance and critical analysis.\n"
        "Score 3: Good task achievement. Arguments are well-developed with "
        "supporting evidence. Shows analytical thinking.\n"
        "Score 4: Thorough task achievement. Sophisticated analysis with "
        "well-supported arguments and critical evaluation.\n"
        "Score 5: Exceptional task achievement. Insightful, nuanced response "
        "with compelling arguments and original thinking."
    ),
    metadata={"type": "rubric", "dimension": "task_achievement", "level_band": "C1-C2"},
)

# -- All rubrics collected for easy import --
ALL_RUBRICS = [
    GRAMMAR_A1_A2, GRAMMAR_B1_B2, GRAMMAR_C1_C2,
    VOCABULARY_A1_A2, VOCABULARY_B1_B2, VOCABULARY_C1_C2,
    COHERENCE_A1_A2, COHERENCE_B1_B2, COHERENCE_C1_C2,
    TASK_ACHIEVEMENT_A1_A2, TASK_ACHIEVEMENT_B1_B2, TASK_ACHIEVEMENT_C1_C2,
]
```

- [ ] **Step 2: Create CEFR level descriptors**

Create `projects/03-student-assessment-pipeline/data/standards.py`:
```python
"""CEFR level descriptors for writing proficiency.

Each descriptor explains what a learner at that CEFR level can do in writing.
Retrieved during the first retrieval phase to help the LLM ground its
scoring against official standards.

6 documents: one per CEFR level (A1-C2).
"""

from langchain_core.documents import Document

A1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level A1 — Writing Proficiency Descriptor\n\n"
        "Can write simple isolated phrases and sentences. Can fill in forms with "
        "personal details (name, nationality, address). Can write a short simple "
        "postcard (holiday greetings). Can write simple phrases and sentences about "
        "themselves and imaginary people, where they live and what they do. Writing "
        "is limited to isolated words and formulaic expressions with frequent "
        "spelling and grammar errors."
    ),
    metadata={"type": "standard", "cefr_level": "A1"},
)

A2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level A2 — Writing Proficiency Descriptor\n\n"
        "Can write short, simple notes and messages. Can write a very simple "
        "personal letter (thanking someone). Can write about everyday aspects of "
        "their environment (people, places, job, school) in linked sentences. "
        "Can describe plans and arrangements, habits and routines, past activities "
        "and personal experiences. Uses simple vocabulary and basic sentence patterns. "
        "Errors are common but meaning is usually clear."
    ),
    metadata={"type": "standard", "cefr_level": "A2"},
)

B1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level B1 — Writing Proficiency Descriptor\n\n"
        "Can write straightforward connected text on familiar topics. Can write "
        "personal letters describing experiences and impressions. Can write essays "
        "or reports presenting information and giving reasons for or against a point "
        "of view. Uses a reasonable range of vocabulary and grammar structures. "
        "Can link ideas using basic connectors (because, however, although). "
        "Errors occur but rarely impede communication."
    ),
    metadata={"type": "standard", "cefr_level": "B1"},
)

B2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level B2 — Writing Proficiency Descriptor\n\n"
        "Can write clear, detailed text on a wide range of subjects. Can write an "
        "essay or report passing on information or giving reasons for or against a "
        "particular point of view. Can write letters highlighting the personal "
        "significance of events and experiences. Uses a wide range of vocabulary "
        "and complex sentence structures. Good control of grammar with occasional "
        "errors in less common structures. Can develop arguments systematically."
    ),
    metadata={"type": "standard", "cefr_level": "B2"},
)

C1_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level C1 — Writing Proficiency Descriptor\n\n"
        "Can write well-structured, clear text about complex subjects. Can select "
        "appropriate style for the reader in mind. Can write detailed expositions, "
        "proposals, and reviews. Uses a broad range of vocabulary with precision and "
        "flexibility. Demonstrates consistent grammatical accuracy of complex "
        "language. Can use language effectively for social, academic, and professional "
        "purposes with only occasional minor errors."
    ),
    metadata={"type": "standard", "cefr_level": "C1"},
)

C2_DESCRIPTOR = Document(
    page_content=(
        "CEFR Level C2 — Writing Proficiency Descriptor\n\n"
        "Can write clear, smoothly flowing text in an appropriate style. Can write "
        "complex letters, reports, or articles presenting a case with effective "
        "logical structure. Can write summaries and reviews of professional or "
        "literary works. Demonstrates mastery of a very broad vocabulary including "
        "idiomatic expressions and colloquialisms. Virtually no errors. Writing "
        "is indistinguishable from that of an educated native speaker."
    ),
    metadata={"type": "standard", "cefr_level": "C2"},
)

ALL_STANDARDS = [
    A1_DESCRIPTOR, A2_DESCRIPTOR, B1_DESCRIPTOR,
    B2_DESCRIPTOR, C1_DESCRIPTOR, C2_DESCRIPTOR,
]
```

- [ ] **Step 3: Create sample graded essays**

Create `projects/03-student-assessment-pipeline/data/sample_essays.py`:
```python
"""Graded sample essays at various CEFR levels.

Each sample includes the essay, the writing prompt, the assigned CEFR level,
a quality score (1-5 within that level), and an assessor note explaining the grade.
Retrieved during the second retrieval phase for comparative analysis.

2 samples per level = 12 documents.
"""

from langchain_core.documents import Document

# -- A1 Samples --

SAMPLE_A1_LOW = Document(
    page_content=(
        "Writing Prompt: Write about your family.\n\n"
        "Student Essay:\n"
        "My family is big. I have mother and father. I have two brother. "
        "My mother name is Maria. She is nice. My father work in office. "
        "My brother play football. I love my family.\n\n"
        "Assessor Note: Very basic vocabulary and simple sentence patterns. "
        "Consistent errors in possessives and plurals but meaning is clear. "
        "Typical low A1 writing with memorized phrases."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A1", "score": 2},
)

SAMPLE_A1_HIGH = Document(
    page_content=(
        "Writing Prompt: Write about your daily routine.\n\n"
        "Student Essay:\n"
        "I wake up at 7 o'clock. I eat breakfast with my family. Then I go to "
        "school by bus. At school I study English and math. I have lunch at 12. "
        "After school I play with my friends. I go home and watch TV. I sleep "
        "at 10 o'clock.\n\n"
        "Assessor Note: Clear chronological structure using simple present tense "
        "accurately. Limited but appropriate vocabulary for daily routine topic. "
        "Strong A1 — approaching A2 in organization."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A1", "score": 4},
)

# -- A2 Samples --

SAMPLE_A2_LOW = Document(
    page_content=(
        "Writing Prompt: Describe your best holiday.\n\n"
        "Student Essay:\n"
        "Last summer I go to the beach with my family. The weather was very hot "
        "and sunny. We stayed in a small hotel near the sea. Every day we swimming "
        "and play in the sand. I eat many ice cream. The food in the restaurant "
        "was good. I liked this holiday very much because it was fun.\n\n"
        "Assessor Note: Attempts past tense but inconsistent (go vs was/stayed). "
        "Basic vocabulary appropriate for the topic. Simple linking with 'and' "
        "and 'because'. Typical A2 writing with expected tense errors."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A2", "score": 2},
)

SAMPLE_A2_HIGH = Document(
    page_content=(
        "Writing Prompt: Write about your favourite hobby.\n\n"
        "Student Essay:\n"
        "My favourite hobby is cooking. I started cooking when I was 12 years old "
        "because my grandmother taught me. I like to cook Italian food, especially "
        "pasta and pizza. Every weekend I try a new recipe from the internet. "
        "Last week I made chocolate cake for my friend's birthday and everyone "
        "liked it. Cooking makes me happy because I can share food with people "
        "I love.\n\n"
        "Assessor Note: Good range of past and present tenses with mostly accurate "
        "use. Nice personal detail and reasons given. Vocabulary is appropriate "
        "and varied for A2. Strong A2, near B1 threshold."
    ),
    metadata={"type": "sample_essay", "cefr_level": "A2", "score": 4},
)

# -- B1 Samples --

SAMPLE_B1_LOW = Document(
    page_content=(
        "Writing Prompt: Do you think social media is good or bad for young people?\n\n"
        "Student Essay:\n"
        "Social media is very popular with young people today. I think it has "
        "good things and bad things. The good thing is that you can talk with "
        "friends and learn new things. For example, I use Instagram to follow "
        "pages about science. But the bad thing is that some people spend too "
        "much time on their phone and they don't study. Also, sometimes there "
        "is cyberbullying which is very bad. I think social media is good if you "
        "use it carefully.\n\n"
        "Assessor Note: Presents both sides of the argument clearly. Uses basic "
        "connectors (but, also, for example). Vocabulary is adequate but repetitive "
        "(good/bad). Limited range of grammatical structures. Typical B1."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B1", "score": 3},
)

SAMPLE_B1_HIGH = Document(
    page_content=(
        "Writing Prompt: Should schools require students to wear uniforms?\n\n"
        "Student Essay:\n"
        "The question of school uniforms is a topic that many people disagree about. "
        "In my opinion, school uniforms are a good idea for several reasons.\n\n"
        "Firstly, uniforms help to reduce bullying because students cannot judge "
        "each other by their clothes. When everyone wears the same thing, there is "
        "less pressure to buy expensive brands. Secondly, uniforms save time in the "
        "morning because students don't need to choose what to wear.\n\n"
        "However, some people argue that uniforms limit self-expression. While I "
        "understand this point, I believe that students can express themselves "
        "through their hobbies and personality, not only through clothes.\n\n"
        "In conclusion, I think uniforms are beneficial because they create equality "
        "and save time, even though they may limit fashion choices.\n\n"
        "Assessor Note: Well-structured argumentative essay with clear introduction, "
        "body paragraphs, and conclusion. Good use of discourse markers (firstly, "
        "however, in conclusion). Acknowledges counterargument. Strong B1, approaching B2."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B1", "score": 5},
)

# -- B2 Samples --

SAMPLE_B2_LOW = Document(
    page_content=(
        "Writing Prompt: Should governments invest more in renewable energy?\n\n"
        "Student Essay:\n"
        "Renewable energy is becoming more important as climate change affects "
        "our planet. I strongly believe that governments should increase their "
        "investment in renewable energy sources such as solar, wind, and hydropower.\n\n"
        "One of the main reasons is that fossil fuels are running out and they "
        "cause pollution. If governments invest in renewable energy now, they can "
        "reduce carbon emissions and slow down global warming. Furthermore, the "
        "renewable energy sector creates many jobs, which is good for the economy.\n\n"
        "On the other hand, renewable energy can be expensive to develop and it "
        "depends on weather conditions. For example, solar panels don't work well "
        "in cloudy countries. Despite these challenges, technology is improving "
        "rapidly and costs are decreasing every year.\n\n"
        "To sum up, while there are some disadvantages, the long-term benefits of "
        "renewable energy make it a worthwhile investment for governments.\n\n"
        "Assessor Note: Clear argumentative structure with good use of evidence. "
        "Uses conditional (if...can) and passive (is becoming) accurately. "
        "Vocabulary is appropriate but could show more range. Solid B2."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B2", "score": 3},
)

SAMPLE_B2_HIGH = Document(
    page_content=(
        "Writing Prompt: How has technology changed the way we communicate?\n\n"
        "Student Essay:\n"
        "The way we communicate has undergone a dramatic transformation over the "
        "past two decades, primarily driven by advances in digital technology. "
        "While these changes have brought undeniable benefits, they have also "
        "introduced new challenges that deserve careful consideration.\n\n"
        "Perhaps the most significant change is the immediacy of communication. "
        "Through messaging apps and social media platforms, we can now connect with "
        "anyone, anywhere in the world, within seconds. This has been particularly "
        "valuable for maintaining relationships across distances — something that "
        "would have required expensive phone calls or slow postal services in the past.\n\n"
        "Nevertheless, there is a growing concern that digital communication has "
        "come at the expense of deeper, face-to-face interactions. Research suggests "
        "that despite being more connected than ever, many people report feeling "
        "increasingly isolated. The nuances of tone, body language, and emotional "
        "presence are inevitably lost in text-based exchanges.\n\n"
        "In my view, the key lies in balance. Technology should enhance our "
        "communication rather than replace traditional forms of human connection.\n\n"
        "Assessor Note: Sophisticated argument with excellent paragraph structure. "
        "Good range of complex structures (present perfect, passive, conditionals). "
        "Effective use of discourse markers and academic register. Top B2."
    ),
    metadata={"type": "sample_essay", "cefr_level": "B2", "score": 5},
)

# -- C1 Samples --

SAMPLE_C1_LOW = Document(
    page_content=(
        "Writing Prompt: Discuss the impact of artificial intelligence on employment.\n\n"
        "Student Essay:\n"
        "The rapid advancement of artificial intelligence has sparked intense debate "
        "about its potential impact on the global workforce. While some view AI as a "
        "harbinger of mass unemployment, others see it as a catalyst for economic "
        "transformation that will ultimately create more opportunities than it destroys.\n\n"
        "Proponents of AI argue that automation will eliminate repetitive, low-skill "
        "tasks, thereby freeing workers to engage in more creative and fulfilling roles. "
        "Historical precedent supports this view — the Industrial Revolution, despite "
        "initial disruption, eventually led to unprecedented job creation. However, the "
        "pace of AI development far exceeds that of previous technological shifts, which "
        "raises legitimate concerns about whether the workforce can adapt quickly enough.\n\n"
        "It is essential that governments and educational institutions take proactive "
        "measures to prepare workers for this transition. Investment in retraining "
        "programmes and lifelong learning initiatives will be crucial in ensuring that "
        "the benefits of AI are distributed equitably across society.\n\n"
        "Assessor Note: Well-structured academic essay with sophisticated vocabulary "
        "(harbinger, catalyst, equitably). Good use of historical analogy. Some areas "
        "could benefit from more nuanced analysis. Solid C1."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C1", "score": 3},
)

SAMPLE_C1_HIGH = Document(
    page_content=(
        "Writing Prompt: To what extent should freedom of speech be limited?\n\n"
        "Student Essay:\n"
        "The principle of freedom of speech, enshrined in democratic constitutions "
        "worldwide, has long been regarded as a cornerstone of open society. Yet the "
        "digital age has exposed fault lines in this ideal that demand careful "
        "re-examination rather than dogmatic adherence to absolutist positions.\n\n"
        "At its core, free expression serves two vital functions: it enables the "
        "marketplace of ideas, where truth emerges through open debate, and it acts "
        "as a check on governmental overreach. These principles remain as relevant "
        "today as when Mill articulated them in 'On Liberty.' However, the mechanisms "
        "through which speech is disseminated have changed fundamentally. Social media "
        "algorithms can amplify harmful content to millions within hours — a scenario "
        "that historical frameworks for free speech were never designed to address.\n\n"
        "The challenge, therefore, lies not in whether limits should exist — virtually "
        "all democracies already restrict incitement to violence and defamation — but "
        "in who should determine those limits and by what criteria. Delegating this "
        "responsibility to private technology companies, as is effectively the case "
        "today, raises profound questions about accountability and transparency.\n\n"
        "A nuanced approach would involve clear legislative frameworks that distinguish "
        "between the protection of individual expression and the regulation of systemic "
        "harms, while preserving robust judicial oversight.\n\n"
        "Assessor Note: Exceptional analytical depth with sophisticated argumentation. "
        "Masterful vocabulary (enshrined, dogmatic, disseminated). References to Mill "
        "demonstrate intellectual engagement. Near-C2 quality."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C1", "score": 5},
)

# -- C2 Samples --

SAMPLE_C2_LOW = Document(
    page_content=(
        "Writing Prompt: Evaluate the role of literature in shaping cultural identity.\n\n"
        "Student Essay:\n"
        "Literature has served as both mirror and architect of cultural identity "
        "throughout human history, its influence operating simultaneously on the "
        "individual psyche and the collective consciousness. To evaluate its role "
        "requires acknowledging this duality — literature reflects the values and "
        "anxieties of its era while actively shaping the narratives through which "
        "communities understand themselves.\n\n"
        "Consider the transformative impact of postcolonial literature. Writers such "
        "as Chinua Achebe and Ngũgĩ wa Thiong'o did not merely document the "
        "experience of colonialism; they fundamentally reconfigured the literary "
        "landscape by asserting the validity of African perspectives and narrative "
        "traditions. In doing so, they provided a framework for cultural reclamation "
        "that extended far beyond the literary sphere.\n\n"
        "Yet we must resist the temptation to romanticise literature's role uncritically. "
        "National literatures have frequently been co-opted to serve exclusionary "
        "agendas, reinforcing narrow definitions of cultural belonging that marginalise "
        "minority voices. The very notion of a literary canon is inherently political.\n\n"
        "Ultimately, literature's power lies in its capacity to hold multiple truths "
        "simultaneously — to celebrate cultural particularity while revealing universal "
        "human experiences that transcend borders.\n\n"
        "Assessor Note: Sophisticated academic prose with near-native fluency. "
        "Excellent use of literary references and critical vocabulary. Complex "
        "argumentation with appropriate hedging. C2 level."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C2", "score": 3},
)

SAMPLE_C2_HIGH = Document(
    page_content=(
        "Writing Prompt: Is economic growth compatible with environmental sustainability?\n\n"
        "Student Essay:\n"
        "The putative tension between economic growth and environmental sustainability "
        "has become one of the defining intellectual and policy challenges of our age. "
        "While the framing of this as a binary choice has a certain rhetorical "
        "convenience, it obscures the more nuanced reality that both the nature of "
        "growth and the definition of sustainability admit of considerable variation.\n\n"
        "The orthodox economic position — that technological innovation and market "
        "mechanisms can decouple growth from environmental degradation — finds partial "
        "support in the trajectory of certain developed economies, where GDP has "
        "continued to rise while absolute emissions have declined. Yet such examples "
        "remain the exception rather than the rule, and they invariably exclude the "
        "environmental costs embedded in global supply chains. A nation may appear to "
        "have decoupled when it has merely offshored its ecological footprint.\n\n"
        "What is required is not the abandonment of growth as a concept but its radical "
        "reconceptualisation. The emerging discourse around 'doughnut economics' — "
        "which posits a safe operating space between social foundations and ecological "
        "ceilings — offers a more sophisticated framework than the simplistic growth "
        "versus degrowth dichotomy. Within this model, certain forms of growth "
        "(in healthcare, education, renewable energy) are not merely compatible with "
        "sustainability but essential to it, while others (fossil fuel extraction, "
        "planned obsolescence) are inherently antithetical.\n\n"
        "The question, then, is not whether we can afford to grow but whether we can "
        "afford to continue growing in the same way.\n\n"
        "Assessor Note: Masterful academic writing indistinguishable from an educated "
        "native speaker. Exceptional vocabulary (putative, decouple, antithetical). "
        "Sophisticated argumentation with original synthesis. Top C2."
    ),
    metadata={"type": "sample_essay", "cefr_level": "C2", "score": 5},
)

ALL_SAMPLE_ESSAYS = [
    SAMPLE_A1_LOW, SAMPLE_A1_HIGH,
    SAMPLE_A2_LOW, SAMPLE_A2_HIGH,
    SAMPLE_B1_LOW, SAMPLE_B1_HIGH,
    SAMPLE_B2_LOW, SAMPLE_B2_HIGH,
    SAMPLE_C1_LOW, SAMPLE_C1_HIGH,
    SAMPLE_C2_LOW, SAMPLE_C2_HIGH,
]
```

- [ ] **Step 4: Create sample submissions for testing**

Create `projects/03-student-assessment-pipeline/data/sample_submissions.py`:
```python
"""Sample student submissions for testing the assessment pipeline.

These are NOT part of the vector store — they are inputs to the pipeline.
Each submission includes the student's writing, the prompt they were given,
and an optional self-reported level hint.
"""


SUBMISSION_B1_TRAVEL = {
    "submission_text": (
        "Traveling is one of the best things you can do in your life. Last year "
        "I traveled to Spain with my friends and it was an amazing experience. "
        "We visited Barcelona and Madrid. In Barcelona we saw the Sagrada Familia "
        "which is a very beautiful church designed by Gaudi. The weather was hot "
        "and sunny every day.\n\n"
        "I think traveling is important because you can learn about different "
        "cultures and meet new people. When you travel you also learn to be more "
        "independent because you need to solve problems by yourself. For example, "
        "in Spain I had to speak Spanish to order food in a restaurant even though "
        "my Spanish is not very good.\n\n"
        "However, traveling can be expensive. Not everyone can afford to go to "
        "other countries. I think governments should help young people to travel "
        "more by offering cheap flights or train tickets.\n\n"
        "In conclusion, traveling is a wonderful way to learn and grow as a person. "
        "I hope I can visit many more countries in the future."
    ),
    "submission_context": "Write an essay about the benefits of traveling.",
    "student_level_hint": "B1",
}

SUBMISSION_A2_HOBBY = {
    "submission_text": (
        "My hobby is play guitar. I started when I was 14 years old. My father "
        "give me a guitar for my birthday. At first it was very difficult because "
        "my fingers was hurting. But I practiced every day and now I can play many "
        "songs.\n\n"
        "I like to play rock music and pop music. My favorite band is Coldplay. "
        "I learn their songs from YouTube videos. Sometimes I play with my friends "
        "and we make a small concert in my house.\n\n"
        "Playing guitar make me happy and relaxed. When I am stressed from school "
        "I play guitar and I feel better. I want to learn more songs and maybe "
        "one day play in a real concert."
    ),
    "submission_context": "Write about your favourite hobby and why you enjoy it.",
    "student_level_hint": "",
}

SUBMISSION_C1_TECHNOLOGY = {
    "submission_text": (
        "The relationship between technological advancement and privacy has become "
        "one of the most pressing issues of the twenty-first century. As digital "
        "technologies permeate every aspect of our lives, the boundaries between "
        "public and private spheres have become increasingly blurred, raising "
        "fundamental questions about the nature of personal autonomy in a "
        "connected world.\n\n"
        "The collection of personal data by corporations and governments has "
        "reached an unprecedented scale. Every online interaction, purchase, and "
        "even physical movement can be tracked, aggregated, and analysed to create "
        "detailed profiles of individuals. Proponents of such surveillance argue "
        "that it is necessary for national security and enables personalised "
        "services that consumers value. However, this argument fails to account "
        "for the power asymmetry it creates — individuals rarely understand the "
        "extent to which their data is being harvested, let alone have meaningful "
        "control over how it is used.\n\n"
        "The implementation of regulations such as the GDPR represents an important "
        "step towards redressing this imbalance, yet enforcement remains inconsistent "
        "and the pace of technological change continually outstrips legislative "
        "responses. What is ultimately needed is a fundamental shift in how we "
        "conceptualise digital rights — treating privacy not as a commodity to be "
        "traded but as an inalienable right that underpins democratic participation."
    ),
    "submission_context": (
        "Discuss the challenges of maintaining personal privacy in the digital age."
    ),
    "student_level_hint": "C1",
}

ALL_SUBMISSIONS = [
    SUBMISSION_B1_TRAVEL,
    SUBMISSION_A2_HOBBY,
    SUBMISSION_C1_TECHNOLOGY,
]
```

- [ ] **Step 5: Commit**

```bash
git add projects/03-student-assessment-pipeline/data/
git commit -m "feat(p3): add document library — rubrics, standards, sample essays, and test submissions"
```

---

### Task 4: Ingestion module (TDD)

**Files:**
- Create: `projects/03-student-assessment-pipeline/tests/conftest.py`
- Create: `projects/03-student-assessment-pipeline/tests/test_ingestion.py`
- Create: `projects/03-student-assessment-pipeline/ingestion.py`

- [ ] **Step 1: Write shared test fixtures in conftest.py**

Create `projects/03-student-assessment-pipeline/tests/conftest.py`:
```python
"""Shared test fixtures for the Student Assessment Pipeline.

Provides a pre-populated Chroma vector store fixture that multiple
test modules can use without re-ingesting documents each time.
"""

import pytest
from ingestion import build_vector_store


@pytest.fixture(scope="session")
def vector_store(tmp_path_factory):
    """Build a Chroma vector store in a temp directory for the test session.

    Uses session scope so the (slow) embedding step only runs once
    across all tests.
    """
    persist_dir = str(tmp_path_factory.mktemp("chroma_test"))
    store = build_vector_store(persist_directory=persist_dir)
    return store
```

- [ ] **Step 2: Write failing tests for the ingestion module**

Create `projects/03-student-assessment-pipeline/tests/test_ingestion.py`:
```python
"""Tests for the document ingestion module.

Verifies that documents are correctly loaded, embedded, and stored
in Chroma with proper metadata for filtered retrieval.
"""

from langchain_core.documents import Document


def test_vector_store_has_documents(vector_store):
    """The vector store should contain all ingested documents."""
    # 12 rubrics + 6 standards + 12 sample essays = 30 base documents
    # After splitting, there may be more chunks, but at least 30
    collection = vector_store._collection
    count = collection.count()
    assert count >= 30, f"Expected at least 30 documents, got {count}"


def test_filter_by_rubric_type(vector_store):
    """Filtering by type='rubric' returns only rubric documents."""
    results = vector_store.similarity_search(
        "grammar accuracy scoring criteria",
        k=20,
        filter={"type": "rubric"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "rubric"


def test_filter_by_standard_type(vector_store):
    """Filtering by type='standard' returns only CEFR level descriptors."""
    results = vector_store.similarity_search(
        "what can a B1 learner do in writing",
        k=10,
        filter={"type": "standard"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "standard"


def test_filter_by_sample_essay_type(vector_store):
    """Filtering by type='sample_essay' returns only sample essays."""
    results = vector_store.similarity_search(
        "student essay about travel and culture",
        k=10,
        filter={"type": "sample_essay"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "sample_essay"


def test_filter_sample_essays_by_level(vector_store):
    """Can filter sample essays by CEFR level using metadata."""
    results = vector_store.similarity_search(
        "student writing sample",
        k=10,
        filter={"$and": [{"type": "sample_essay"}, {"cefr_level": "B1"}]},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "sample_essay"
        assert doc.metadata["cefr_level"] == "B1"


def test_similarity_search_returns_relevant_docs(vector_store):
    """Similarity search for grammar topics returns grammar-related documents."""
    results = vector_store.similarity_search(
        "grammar errors subject verb agreement tense usage",
        k=5,
    )
    assert len(results) > 0
    # At least one result should be grammar-related
    grammar_related = [
        doc for doc in results
        if "grammar" in doc.page_content.lower()
        or doc.metadata.get("dimension") == "grammar"
    ]
    assert len(grammar_related) > 0


def test_get_vector_store_loads_existing(vector_store, tmp_path):
    """get_vector_store loads an existing persisted Chroma store."""
    from ingestion import build_vector_store, get_vector_store

    # Build a store in a known location
    persist_dir = str(tmp_path / "reload_test")
    build_vector_store(persist_directory=persist_dir)

    # Load it back
    loaded = get_vector_store(persist_directory=persist_dir)
    count = loaded._collection.count()
    assert count >= 30
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_ingestion.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 4: Implement the ingestion module**

Create `projects/03-student-assessment-pipeline/ingestion.py`:
```python
"""Document ingestion module for the Student Assessment Pipeline.

Handles loading CEFR rubrics, level descriptors, and sample essays
into a Chroma vector store with metadata for filtered retrieval.

This module is separate from the assessment graph — ingestion is a
one-time setup concern, not part of the assessment workflow.

RAG concepts demonstrated:
- Document loading from Python data structures
- Text splitting with RecursiveCharacterTextSplitter
- Embedding with sentence-transformers (local, no API key)
- Storing in Chroma with metadata for filtered retrieval
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.rubrics import ALL_RUBRICS
from data.standards import ALL_STANDARDS
from data.sample_essays import ALL_SAMPLE_ESSAYS

# Embedding model — runs locally via sentence-transformers
# all-mpnet-base-v2 is a good general-purpose model (768 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Collection name used across ingestion and retrieval
COLLECTION_NAME = "linguaflow_assessment"

# Default persist directory (relative to project root)
DEFAULT_PERSIST_DIR = "./chroma_db"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model instance.

    Uses all-mpnet-base-v2 which runs locally — no API key needed.
    The same model must be used for both ingestion and retrieval.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Build and persist the Chroma vector store from all document sources.

    Loads rubrics, standards, and sample essays, splits longer documents,
    embeds them, and stores in Chroma with metadata.

    Args:
        persist_directory: Path to persist the Chroma database.

    Returns:
        The populated Chroma vector store instance.
    """
    # Collect all documents from the data modules
    all_documents = ALL_RUBRICS + ALL_STANDARDS + ALL_SAMPLE_ESSAYS

    # Split longer documents into chunks for better retrieval
    # Rubrics and standards are relatively short, but sample essays can be longer
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(all_documents)

    # Create the vector store with embeddings and metadata
    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Load an existing persisted Chroma vector store.

    Must use the same embedding model that was used during ingestion.

    Args:
        persist_directory: Path where the Chroma database is persisted.

    Returns:
        The loaded Chroma vector store instance.
    """
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_ingestion.py -v
```

Expected: All 7 tests PASS. (First run will be slow as it downloads the sentence-transformers model.)

- [ ] **Step 6: Commit**

```bash
git add projects/03-student-assessment-pipeline/ingestion.py \
       projects/03-student-assessment-pipeline/tests/conftest.py \
       projects/03-student-assessment-pipeline/tests/test_ingestion.py
git commit -m "feat(p3): add document ingestion module with Chroma and HuggingFace embeddings"
```

---

### Task 5: Prompt templates for all LLM nodes

**Files:**
- Create: `projects/03-student-assessment-pipeline/prompts.py`

- [ ] **Step 1: Create all prompt templates**

Create `projects/03-student-assessment-pipeline/prompts.py`:
```python
"""Prompt templates for every LLM node in the assessment graph.

All prompts are defined here in one place so they're easy to compare,
tweak, and review. Each prompt is a ChatPromptTemplate.

LangChain concept demonstrated:
- ChatPromptTemplate.from_messages() — reusable prompt templates
  with {variable} placeholders filled at invocation time.
"""

from langchain_core.prompts import ChatPromptTemplate

# -- Criteria Scoring Node --

CRITERIA_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English writing assessor for the LinguaFlow tutoring "
        "platform. You score student writing across four dimensions using CEFR "
        "standards as your reference.\n\n"
        "The four dimensions are:\n"
        "1. Grammar & Accuracy\n"
        "2. Vocabulary Range & Precision\n"
        "3. Coherence & Organization\n"
        "4. Task Achievement\n\n"
        "For each dimension:\n"
        "- Assign a score from 1 (lowest) to 5 (highest)\n"
        "- Cite specific evidence from the submission (direct quotes)\n"
        "- Provide specific, actionable feedback\n\n"
        "After scoring all dimensions, determine a preliminary CEFR level "
        "(A1, A2, B1, B2, C1, or C2) based on the aggregate scores and "
        "the standards provided.\n\n"
        "Use the following rubrics and CEFR standards as your reference:\n\n"
        "{retrieved_standards}",
    ),
    (
        "human",
        "Writing prompt given to the student:\n{submission_context}\n\n"
        "Student's submission:\n{submission_text}\n\n"
        "Please score this submission across all four dimensions and determine "
        "a preliminary CEFR level.",
    ),
])

# -- Comparative Analysis Node --

COMPARATIVE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English writing assessor for the LinguaFlow tutoring "
        "platform. You compare a student's submission against sample essays at "
        "similar CEFR levels.\n\n"
        "For each sample essay provided:\n"
        "- Note specific similarities between the submission and the sample\n"
        "- Note specific differences\n"
        "- Determine whether the submission is 'above', 'comparable' to, or "
        "'below' the sample in overall quality\n\n"
        "After comparing all samples, write a narrative summary that explains "
        "where the submission sits relative to the level. Use specific examples.\n\n"
        "The student was preliminarily scored at CEFR level {preliminary_level}.\n\n"
        "Sample essays for comparison:\n\n{retrieved_samples}",
    ),
    (
        "human",
        "Writing prompt given to the student:\n{submission_context}\n\n"
        "Student's submission:\n{submission_text}\n\n"
        "Please compare this submission against the sample essays and provide "
        "a detailed comparative analysis.",
    ),
])

# -- Synthesize Node --

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior assessor for the LinguaFlow tutoring platform. "
        "You produce final, comprehensive writing assessments by combining "
        "criteria-based scoring with comparative analysis.\n\n"
        "You have two sources of information:\n"
        "1. Criteria scores: per-dimension scores (1-5) with evidence and feedback\n"
        "2. Comparative analysis: how the submission compares to level-appropriate samples\n\n"
        "Produce a final assessment that:\n"
        "- Confirms or adjusts the preliminary CEFR level based on all evidence\n"
        "- Lists the student's key strengths\n"
        "- Lists specific areas for improvement\n"
        "- Provides actionable recommendations for what to study next\n"
        "- Rates your confidence in the assessment as 'high', 'medium', or 'low'\n\n"
        "Be encouraging but honest. The student should feel motivated to improve.",
    ),
    (
        "human",
        "Student's submission:\n{submission_text}\n\n"
        "Writing prompt:\n{submission_context}\n\n"
        "Criteria Scores:\n{criteria_scores}\n\n"
        "Comparative Analysis:\n{comparative_analysis}\n\n"
        "Please produce the final assessment.",
    ),
])
```

- [ ] **Step 2: Commit**

```bash
git add projects/03-student-assessment-pipeline/prompts.py
git commit -m "feat(p3): add prompt templates for all graph nodes"
```

---

### Task 6: Node functions (TDD)

**Files:**
- Create: `projects/03-student-assessment-pipeline/tests/test_nodes.py`
- Create: `projects/03-student-assessment-pipeline/nodes.py`

- [ ] **Step 1: Write failing integration tests for nodes**

Create `projects/03-student-assessment-pipeline/tests/test_nodes.py`:
```python
"""Integration tests for node functions.

These tests hit the real LLM (Anthropic) and the real vector store
to verify that each node produces correct state updates.

Mark with pytest markers so they can be skipped in fast test runs.
"""

import pytest
from langchain_core.documents import Document

from data.sample_submissions import SUBMISSION_B1_TRAVEL
from models import CriteriaScores, ComparativeAnalysis, Assessment


@pytest.fixture
def base_state(vector_store):
    """Base state with a B1 submission and pre-populated vector store."""
    return {
        "submission_text": SUBMISSION_B1_TRAVEL["submission_text"],
        "submission_context": SUBMISSION_B1_TRAVEL["submission_context"],
        "student_level_hint": SUBMISSION_B1_TRAVEL["student_level_hint"],
    }


# -- retrieve_standards --

@pytest.mark.integration
def test_retrieve_standards_returns_documents(vector_store, base_state):
    """retrieve_standards should return relevant rubrics and standards."""
    from nodes import retrieve_standards_node

    result = retrieve_standards_node(base_state, vector_store=vector_store)
    docs = result["retrieved_standards"]

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    # Should only contain rubrics and standards, not sample essays
    for doc in docs:
        assert doc.metadata["type"] in ("rubric", "standard")


# -- criteria_scoring --

@pytest.mark.integration
def test_criteria_scoring_returns_structured_scores(vector_store, base_state):
    """criteria_scoring should return CriteriaScores with a preliminary level."""
    from nodes import retrieve_standards_node, criteria_scoring_node

    # First retrieve standards to populate state
    retrieval_result = retrieve_standards_node(base_state, vector_store=vector_store)
    state = {**base_state, **retrieval_result}

    result = criteria_scoring_node(state)

    assert "criteria_scores" in result
    assert "preliminary_level" in result
    assert isinstance(result["criteria_scores"], CriteriaScores)
    assert result["preliminary_level"] in ("A1", "A2", "B1", "B2", "C1", "C2")
    assert len(result["criteria_scores"].scores) == 4  # 4 dimensions


# -- retrieve_samples --

@pytest.mark.integration
def test_retrieve_samples_returns_level_appropriate_essays(vector_store, base_state):
    """retrieve_samples should return sample essays filtered by level."""
    from nodes import retrieve_samples_node

    state = {**base_state, "preliminary_level": "B1"}

    result = retrieve_samples_node(state, vector_store=vector_store)
    docs = result["retrieved_samples"]

    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
    for doc in docs:
        assert doc.metadata["type"] == "sample_essay"


# -- comparative_analysis --

@pytest.mark.integration
def test_comparative_analysis_returns_structured_analysis(vector_store, base_state):
    """comparative_analysis should compare submission against samples."""
    from nodes import retrieve_samples_node, comparative_analysis_node

    state = {**base_state, "preliminary_level": "B1"}
    retrieval_result = retrieve_samples_node(state, vector_store=vector_store)
    state = {**state, **retrieval_result}

    result = comparative_analysis_node(state)

    assert "comparative_analysis" in result
    assert isinstance(result["comparative_analysis"], ComparativeAnalysis)
    assert len(result["comparative_analysis"].comparisons) > 0


# -- synthesize --

@pytest.mark.integration
def test_synthesize_returns_final_assessment(vector_store, base_state):
    """synthesize should produce a complete Assessment from scores + analysis."""
    from nodes import (
        retrieve_standards_node,
        criteria_scoring_node,
        retrieve_samples_node,
        comparative_analysis_node,
        synthesize_node,
    )

    # Run the full pipeline to build up state
    state = {**base_state}

    r1 = retrieve_standards_node(state, vector_store=vector_store)
    state.update(r1)

    r2 = criteria_scoring_node(state)
    state.update(r2)

    r3 = retrieve_samples_node(state, vector_store=vector_store)
    state.update(r3)

    r4 = comparative_analysis_node(state)
    state.update(r4)

    result = synthesize_node(state)

    assert "final_assessment" in result
    assessment = result["final_assessment"]
    assert isinstance(assessment, Assessment)
    assert assessment.overall_level in ("A1", "A2", "B1", "B2", "C1", "C2")
    assert len(assessment.criteria_scores) == 4
    assert len(assessment.strengths) > 0
    assert len(assessment.recommendations) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_nodes.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement node functions**

Create `projects/03-student-assessment-pipeline/nodes.py`:
```python
"""Node functions for the Student Assessment Pipeline StateGraph.

Each function represents a node in the graph. Retrieval nodes accept
the vector store as a parameter (injected at graph construction time).
LLM nodes use Anthropic Claude with structured output.

LangGraph concepts demonstrated:
- Node functions as building blocks of a StateGraph
- Each node performs one focused task and returns partial state updates
- Retrieval nodes query the vector store with metadata filters

RAG concepts demonstrated:
- Metadata-filtered similarity search
- Retrieved documents used as grounding context for LLM calls
- Phased retrieval: early results shape later queries

LangChain concepts demonstrated:
- Prompt | Model pipeline for LLM calls
- .with_structured_output() for structured generation
- @traceable for LangSmith observability
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langsmith import traceable

from models import (
    AssessmentState,
    CriteriaScores,
    ComparativeAnalysis,
    Assessment,
)
from prompts import (
    CRITERIA_SCORING_PROMPT,
    COMPARATIVE_ANALYSIS_PROMPT,
    SYNTHESIZE_PROMPT,
)

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# -- LangSmith Tags --
_TAGS = ["p3-student-assessment"]


def _format_documents(docs: list) -> str:
    """Format retrieved documents into a single string for prompt injection.

    Each document is separated by a divider and includes its metadata
    so the LLM can reference document types and levels.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
        parts.append(f"--- Document {i} [{meta}] ---\n{doc.page_content}")
    return "\n\n".join(parts)


@traceable(name="retrieve_standards", run_type="retriever", tags=_TAGS)
def retrieve_standards_node(
    state: AssessmentState, *, vector_store: Chroma
) -> dict:
    """Retrieve rubrics and CEFR level descriptors relevant to the submission.

    Queries the vector store with the submission text, filtering for
    rubric and standard document types. If a student_level_hint is provided,
    also performs a targeted retrieval for that level band.

    RAG concept: metadata-filtered similarity search.
    """
    query = state["submission_text"]

    # Retrieve rubrics and standards using Chroma's $or filter
    results = vector_store.similarity_search(
        query,
        k=10,
        filter={"$or": [{"type": "rubric"}, {"type": "standard"}]},
    )

    return {"retrieved_standards": results}


@traceable(name="criteria_scoring", run_type="chain", tags=_TAGS)
def criteria_scoring_node(state: AssessmentState) -> dict:
    """Score the submission across 4 dimensions using retrieved standards.

    Uses the LLM with structured output to produce CriteriaScores,
    which includes per-dimension scores and a preliminary CEFR level.
    The preliminary level drives the next retrieval phase.

    LangChain concept: .with_structured_output() for structured generation.
    """
    structured_model = _model.with_structured_output(
        CriteriaScores, method="json_schema"
    )
    chain = CRITERIA_SCORING_PROMPT | structured_model

    standards_text = _format_documents(state["retrieved_standards"])

    result = chain.invoke(
        {
            "retrieved_standards": standards_text,
            "submission_text": state["submission_text"],
            "submission_context": state["submission_context"],
        },
        config={"tags": _TAGS},
    )

    return {
        "criteria_scores": result,
        "preliminary_level": result.preliminary_level,
    }


@traceable(name="retrieve_samples", run_type="retriever", tags=_TAGS)
def retrieve_samples_node(
    state: AssessmentState, *, vector_store: Chroma
) -> dict:
    """Retrieve sample essays at the preliminary CEFR level for comparison.

    This is the second retrieval phase — it uses the preliminary_level
    from criteria_scoring to fetch level-appropriate sample essays.
    Also retrieves samples from adjacent levels for contrast.

    RAG concept: phased retrieval where early results inform later queries.
    """
    query = state["submission_text"]
    level = state["preliminary_level"]

    # Map levels to their neighbors for contrast retrieval
    level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    idx = level_order.index(level)
    target_levels = [level]
    if idx > 0:
        target_levels.append(level_order[idx - 1])
    if idx < len(level_order) - 1:
        target_levels.append(level_order[idx + 1])

    # Retrieve sample essays at the target levels
    results = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "$and": [
                {"type": "sample_essay"},
                {"cefr_level": {"$in": target_levels}},
            ]
        },
    )

    return {"retrieved_samples": results}


@traceable(name="comparative_analysis", run_type="chain", tags=_TAGS)
def comparative_analysis_node(state: AssessmentState) -> dict:
    """Compare the submission against retrieved sample essays.

    The LLM compares the submission to each sample, noting similarities,
    differences, and relative quality position. This grounds the assessment
    in concrete examples rather than abstract criteria alone.
    """
    structured_model = _model.with_structured_output(
        ComparativeAnalysis, method="json_schema"
    )
    chain = COMPARATIVE_ANALYSIS_PROMPT | structured_model

    samples_text = _format_documents(state["retrieved_samples"])

    result = chain.invoke(
        {
            "preliminary_level": state["preliminary_level"],
            "retrieved_samples": samples_text,
            "submission_text": state["submission_text"],
            "submission_context": state["submission_context"],
        },
        config={"tags": _TAGS},
    )

    return {"comparative_analysis": result}


@traceable(name="synthesize", run_type="chain", tags=_TAGS)
def synthesize_node(state: AssessmentState) -> dict:
    """Merge criteria scores and comparative analysis into a final Assessment.

    Combines all gathered evidence to produce the complete structured
    assessment with an overall CEFR level, strengths, areas to improve,
    and actionable recommendations.

    LangChain concept: .with_structured_output() for the final output model.
    """
    structured_model = _model.with_structured_output(
        Assessment, method="json_schema"
    )
    chain = SYNTHESIZE_PROMPT | structured_model

    # Format criteria scores as readable text for the prompt
    scores_text = state["criteria_scores"].model_dump_json(indent=2)
    analysis_text = state["comparative_analysis"].model_dump_json(indent=2)

    result = chain.invoke(
        {
            "submission_text": state["submission_text"],
            "submission_context": state["submission_context"],
            "criteria_scores": scores_text,
            "comparative_analysis": analysis_text,
        },
        config={"tags": _TAGS},
    )

    return {"final_assessment": result}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_nodes.py -v -m integration
```

Expected: All 5 integration tests PASS. (These hit the Anthropic API, so they'll take some time.)

- [ ] **Step 5: Commit**

```bash
git add projects/03-student-assessment-pipeline/nodes.py \
       projects/03-student-assessment-pipeline/tests/test_nodes.py
git commit -m "feat(p3): add node functions with retrieval and structured LLM output"
```

---

### Task 7: Graph assembly (TDD)

**Files:**
- Create: `projects/03-student-assessment-pipeline/tests/test_graph.py`
- Create: `projects/03-student-assessment-pipeline/graph.py`

- [ ] **Step 1: Write failing end-to-end graph test**

Create `projects/03-student-assessment-pipeline/tests/test_graph.py`:
```python
"""End-to-end integration tests for the assessment graph.

Runs the full pipeline: retrieve_standards → criteria_scoring →
retrieve_samples → comparative_analysis → synthesize.
"""

import pytest
from models import Assessment
from data.sample_submissions import SUBMISSION_B1_TRAVEL, SUBMISSION_A2_HOBBY


@pytest.mark.integration
def test_full_pipeline_produces_assessment(vector_store):
    """The full graph should produce a complete Assessment for a B1 submission."""
    from graph import build_graph

    graph = build_graph(vector_store)
    result = graph.invoke({
        "submission_text": SUBMISSION_B1_TRAVEL["submission_text"],
        "submission_context": SUBMISSION_B1_TRAVEL["submission_context"],
        "student_level_hint": SUBMISSION_B1_TRAVEL["student_level_hint"],
    })

    assessment = result["final_assessment"]
    assert isinstance(assessment, Assessment)
    assert assessment.overall_level in ("A1", "A2", "B1", "B2", "C1", "C2")
    assert len(assessment.criteria_scores) == 4
    assert len(assessment.strengths) > 0
    assert len(assessment.areas_to_improve) > 0
    assert len(assessment.recommendations) > 0
    assert assessment.confidence in ("high", "medium", "low")
    assert assessment.submission_text == SUBMISSION_B1_TRAVEL["submission_text"]


@pytest.mark.integration
def test_pipeline_with_no_level_hint(vector_store):
    """The graph should work when no student_level_hint is provided."""
    from graph import build_graph

    graph = build_graph(vector_store)
    result = graph.invoke({
        "submission_text": SUBMISSION_A2_HOBBY["submission_text"],
        "submission_context": SUBMISSION_A2_HOBBY["submission_context"],
        "student_level_hint": "",
    })

    assessment = result["final_assessment"]
    assert isinstance(assessment, Assessment)
    assert assessment.overall_level in ("A1", "A2", "B1", "B2", "C1", "C2")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_graph.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 3: Implement graph assembly**

Create `projects/03-student-assessment-pipeline/graph.py`:
```python
"""StateGraph assembly for the Student Assessment Pipeline.

Wires together the 5 node functions into a sequential graph:
retrieve_standards → criteria_scoring → retrieve_samples →
comparative_analysis → synthesize

LangGraph concepts demonstrated:
- StateGraph construction with TypedDict state schema
- Sequential edges (add_edge) for a linear pipeline
- functools.partial to inject the vector store into retrieval nodes
- Graph compilation and invocation

The key architectural insight: this is a linear graph, but the adaptive
behavior comes from criteria_scoring producing a preliminary_level that
retrieve_samples uses as a metadata filter. The graph structure is simple;
the intelligence is in how nodes use state from previous nodes.
"""

from functools import partial

from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from models import AssessmentState
from nodes import (
    retrieve_standards_node,
    criteria_scoring_node,
    retrieve_samples_node,
    comparative_analysis_node,
    synthesize_node,
)


def build_graph(vector_store: Chroma):
    """Build and compile the assessment StateGraph.

    The vector store is injected into retrieval nodes via functools.partial,
    so the graph nodes have a clean (state) -> dict signature while still
    accessing the store.

    Args:
        vector_store: Pre-populated Chroma instance with rubrics, standards,
                      and sample essays.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    # Bind the vector store to retrieval nodes using partial
    # This way, the graph nodes have the signature (state) -> dict
    # that LangGraph expects, while still accessing the vector store
    retrieve_standards = partial(retrieve_standards_node, vector_store=vector_store)
    retrieve_samples = partial(retrieve_samples_node, vector_store=vector_store)

    # Build the graph
    graph = (
        StateGraph(AssessmentState)
        .add_node("retrieve_standards", retrieve_standards)
        .add_node("criteria_scoring", criteria_scoring_node)
        .add_node("retrieve_samples", retrieve_samples)
        .add_node("comparative_analysis", comparative_analysis_node)
        .add_node("synthesize", synthesize_node)
        # Wire the sequential flow
        .add_edge(START, "retrieve_standards")
        .add_edge("retrieve_standards", "criteria_scoring")
        .add_edge("criteria_scoring", "retrieve_samples")
        .add_edge("retrieve_samples", "comparative_analysis")
        .add_edge("comparative_analysis", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )

    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_graph.py -v -m integration
```

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/03-student-assessment-pipeline/graph.py \
       projects/03-student-assessment-pipeline/tests/test_graph.py
git commit -m "feat(p3): assemble StateGraph with phased retrieval pipeline"
```

---

### Task 8: CLI entry point and README

**Files:**
- Create: `projects/03-student-assessment-pipeline/main.py`
- Create: `projects/03-student-assessment-pipeline/README.md`

- [ ] **Step 1: Create the CLI entry point**

Create `projects/03-student-assessment-pipeline/main.py`:
```python
"""CLI entry point for the Student Assessment Pipeline.

Ingests documents (if needed), takes a student submission, runs it
through the assessment graph, and prints the structured results.

Usage:
    python main.py                  # Run with default sample submission
    python main.py --sample 0      # Run with specific sample (0, 1, or 2)
    python main.py --rebuild        # Force rebuild of the vector store
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from repo root .env
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from ingestion import build_vector_store, get_vector_store, DEFAULT_PERSIST_DIR
from graph import build_graph
from data.sample_submissions import ALL_SUBMISSIONS


def _print_assessment(assessment):
    """Pretty-print the final assessment to the console."""
    print("\n" + "=" * 60)
    print("STUDENT WRITING ASSESSMENT")
    print("=" * 60)

    print(f"\nOverall CEFR Level: {assessment.overall_level}")
    print(f"Confidence: {assessment.confidence}")

    print("\n--- Criteria Scores ---")
    for score in assessment.criteria_scores:
        print(f"\n  {score.dimension}: {score.score}/5")
        print(f"  Feedback: {score.feedback}")
        if score.evidence:
            print(f"  Evidence: {score.evidence[0]}")

    print("\n--- Comparative Summary ---")
    print(f"  {assessment.comparative_summary}")

    print("\n--- Strengths ---")
    for s in assessment.strengths:
        print(f"  + {s}")

    print("\n--- Areas to Improve ---")
    for a in assessment.areas_to_improve:
        print(f"  - {a}")

    print("\n--- Recommendations ---")
    for r in assessment.recommendations:
        print(f"  > {r}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Student Assessment Pipeline")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Index of sample submission to use (0, 1, or 2)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of the vector store",
    )
    args = parser.parse_args()

    # Set up vector store
    persist_dir = os.path.join(os.path.dirname(__file__), DEFAULT_PERSIST_DIR)
    if args.rebuild or not os.path.exists(persist_dir):
        print("Building vector store (this may take a moment on first run)...")
        vector_store = build_vector_store(persist_directory=persist_dir)
        print("Vector store ready.")
    else:
        print("Loading existing vector store...")
        vector_store = get_vector_store(persist_directory=persist_dir)

    # Select submission
    submission = ALL_SUBMISSIONS[args.sample]
    print(f"\nAssessing submission {args.sample}...")
    print(f"Context: {submission['submission_context']}")
    if submission["student_level_hint"]:
        print(f"Student's self-reported level: {submission['student_level_hint']}")

    # Build and run the graph
    graph = build_graph(vector_store)
    result = graph.invoke(
        {
            "submission_text": submission["submission_text"],
            "submission_context": submission["submission_context"],
            "student_level_hint": submission["student_level_hint"],
        },
        config={"tags": ["p3-student-assessment"]},
    )

    # Display results
    _print_assessment(result["final_assessment"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the README**

Create `projects/03-student-assessment-pipeline/README.md`:
```markdown
# Project 3: Student Assessment Pipeline

A RAG-powered LangGraph pipeline that assesses student writing submissions by retrieving CEFR rubrics and sample essays from a Chroma vector store, scoring across multiple criteria, and producing a comparative assessment.

## Concepts Covered

- **RAG pipeline**: document ingestion, text splitting, embeddings, vector stores
- **Metadata-filtered retrieval**: querying Chroma with type and level filters
- **Phased retrieval**: early scoring results drive later sample retrieval
- **LangGraph StateGraph**: sequential node pipeline with shared state
- **Structured output**: Pydantic models for multi-criteria assessment

## Setup

```bash
# From repo root (shared venv)
source .venv/bin/activate
pip install -r projects/03-student-assessment-pipeline/requirements.txt
```

## Run

```bash
cd projects/03-student-assessment-pipeline

# Run with default sample submission
python main.py

# Choose a specific sample (0=B1 travel, 1=A2 hobby, 2=C1 technology)
python main.py --sample 1

# Force rebuild of the vector store
python main.py --rebuild
```

## Test

```bash
cd projects/03-student-assessment-pipeline

# Unit tests only (fast, no API calls)
python -m pytest tests/test_models.py tests/test_ingestion.py -v

# Integration tests (hits Anthropic API)
python -m pytest tests/ -v -m integration
```

## Architecture

```
retrieve_standards → criteria_scoring → retrieve_samples → comparative_analysis → synthesize
```

The key pattern: `criteria_scoring` determines a preliminary CEFR level, which `retrieve_samples` uses to fetch level-appropriate sample essays for comparison. Early results shape later retrieval.
```

- [ ] **Step 3: Commit**

```bash
git add projects/03-student-assessment-pipeline/main.py \
       projects/03-student-assessment-pipeline/README.md
git commit -m "feat(p3): add CLI entry point and project README"
```

---

### Task 9: Educational documentation

**Files:**
- Create: `docs/03-student-assessment-pipeline.md`

- [ ] **Step 1: Write the educational doc**

Create `docs/03-student-assessment-pipeline.md` — a comprehensive educational document explaining:

1. What RAG is and why it matters (the problem it solves)
2. The RAG pipeline: ingestion (load → split → embed → store) and retrieval (query → embed → search → return)
3. How Chroma works: collections, metadata, filtered search
4. How embeddings work: sentence-transformers, vector similarity
5. The phased retrieval pattern: early results informing later queries
6. How the StateGraph integrates retrieval with LLM generation
7. Key code walkthrough: ingestion.py, nodes.py, graph.py
8. LangSmith tracing for retrieval quality inspection

Follow the pattern from `docs/01-grammar-correction-agent.md` — this is a teaching document that explains every concept used in the project.

- [ ] **Step 2: Commit**

```bash
git add docs/03-student-assessment-pipeline.md
git commit -m "docs(p3): add comprehensive educational guide for RAG and the assessment pipeline"
```

---

### Task 10: Final validation

- [ ] **Step 1: Run all unit tests**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/test_models.py tests/test_ingestion.py -v
```

Expected: All unit tests PASS.

- [ ] **Step 2: Run all integration tests**

```bash
cd "projects/03-student-assessment-pipeline"
python -m pytest tests/ -v -m integration
```

Expected: All integration tests PASS.

- [ ] **Step 3: Run the CLI end-to-end**

```bash
cd "projects/03-student-assessment-pipeline"
python main.py --sample 0
```

Expected: Complete assessment output printed to console with CEFR level, criteria scores, comparative summary, strengths, areas to improve, and recommendations.

- [ ] **Step 4: Verify LangSmith traces**

Check LangSmith dashboard for traces tagged with `p3-student-assessment`. Verify:
- All 5 nodes appear in the trace
- Retrieval nodes show the documents retrieved
- Scoring nodes show structured output
