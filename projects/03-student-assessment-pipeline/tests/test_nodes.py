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
