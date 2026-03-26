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
