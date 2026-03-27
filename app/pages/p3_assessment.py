"""P3 — Student Assessment Pipeline tab.

Form-based interface for submitting writing and viewing CEFR assessment results.
"""

import streamlit as st
from adapters import assessment
from components import doc_viewer, overview, results

_DOC_PATH = "docs/03-student-assessment-pipeline.md"

# CEFR level colors for the badge
_LEVEL_COLORS = {
    "A1": "#e74c3c",
    "A2": "#e67e22",
    "B1": "#f1c40f",
    "B2": "#2ecc71",
    "C1": "#3498db",
    "C2": "#9b59b6",
}


def render() -> None:
    """Render the Student Assessment Pipeline interface."""
    st.header("Student Assessment Pipeline")
    st.caption("Submit student writing for comprehensive CEFR-level assessment")

    overview.render(
        business_scenario=(
            "Assessors submit student writing and receive a multi-criteria CEFR evaluation "
            "covering grammar, vocabulary, coherence, and task achievement. The system retrieves "
            "official CEFR rubrics from a vector store for grounding, then runs comparative "
            "analysis across proficiency levels. This standardizes assessment quality across "
            "different evaluators and provides evidence-backed scoring."
        ),
        tech_flowchart="""
            digraph {
                rankdir=LR
                node [shape=box style="rounded,filled" fillcolor="#f0f4ff" fontname="Helvetica" fontsize=11]
                edge [fontname="Helvetica" fontsize=10]

                input [label="Student\\nSubmission" shape=note fillcolor="#fff3cd"]
                retrieve [label="Retrieve\\nCEFR Rubrics"]
                vectordb [label="Chroma\\nVector Store" shape=cylinder fillcolor="#d6eaf8"]
                initial [label="Initial\\nAssessment"]
                comparative [label="Comparative\\nAnalysis"]
                final [label="Final\\nScoring"]
                output [label="CEFR Report\\n+ Scores" shape=note fillcolor="#fff3cd"]
                langsmith [label="LangSmith\\nTracing" shape=ellipse fillcolor="#e8daef"]

                input -> retrieve
                retrieve -> vectordb [dir=both label="similarity\\nsearch"]
                retrieve -> initial [label="rubrics +\\nsubmission"]
                initial -> comparative [label="adjacent\\nlevels"]
                comparative -> final
                final -> output
                initial -> langsmith [style=dashed]
                comparative -> langsmith [style=dashed]
            }
        """,
        key_prefix="p3",
    )

    # -- Sample submission selector --
    samples = assessment.get_sample_submissions()
    sample_labels = ["(enter your own text below)"] + [
        f"Sample {i + 1}: {s['submission_context'][:60]}..."
        for i, s in enumerate(samples)
    ]
    selected_idx = st.selectbox(
        "Quick-start with a sample submission:",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        key="p3_sample_select",
    )

    # -- Input form --
    if selected_idx > 0:
        sample = samples[selected_idx - 1]
        default_text = sample["submission_text"]
        default_context = sample["submission_context"]
        default_hint = sample["student_level_hint"]
    else:
        default_text = ""
        default_context = ""
        default_hint = ""

    submission_text = st.text_area(
        "Student writing:",
        value=default_text,
        height=200,
        key="p3_text_input",
        placeholder="Paste the student's writing here...",
    )

    col1, col2 = st.columns(2)
    with col1:
        submission_context = st.text_input(
            "Submission context (what they were asked to write):",
            value=default_context,
            key="p3_context_input",
        )
    with col2:
        level_options = ["", "A1", "A2", "B1", "B2", "C1", "C2"]
        hint_idx = level_options.index(default_hint) if default_hint in level_options else 0
        student_level_hint = st.selectbox(
            "Self-reported CEFR level (optional):",
            level_options,
            index=hint_idx,
            format_func=lambda x: "(none)" if x == "" else x,
            key="p3_hint_select",
        )

    # -- Action buttons --
    col_assess, col_reset = st.columns([1, 5])
    with col_assess:
        assess_clicked = st.button("📊 Assess", key="p3_assess", type="primary")
    with col_reset:
        if st.button("🔄 Reset", key="p3_reset"):
            st.session_state.pop("p3_result", None)
            st.rerun()

    # -- Run assessment --
    if assess_clicked:
        if not submission_text.strip():
            st.warning("Please enter some student writing to assess.")
        else:
            with st.spinner("Running assessment pipeline (this may take a moment)..."):
                try:
                    result = assessment.run_assessment(
                        submission_text=submission_text,
                        submission_context=submission_context,
                        student_level_hint=student_level_hint,
                    )
                    st.session_state["p3_result"] = result
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))

    # -- Display results --
    if "p3_result" in st.session_state:
        result = st.session_state["p3_result"]
        st.divider()

        # Overall level badge
        color = _LEVEL_COLORS.get(result.overall_level, "#95a5a6")
        results.badge("Overall CEFR Level", result.overall_level, color=color)

        if hasattr(result, "confidence"):
            st.caption(f"Confidence: {result.confidence}")

        st.markdown("")

        # Criteria scores
        st.subheader("📏 Criteria Scores")
        score_cols = st.columns(2)
        for i, criterion in enumerate(result.criteria_scores):
            with score_cols[i % 2]:
                results.score_card(
                    label=criterion.dimension,
                    score=criterion.score,
                    feedback=criterion.feedback,
                    evidence=criterion.evidence,
                )

        # Comparative summary
        if hasattr(result, "comparative_summary") and result.comparative_summary:
            st.subheader("📊 Comparative Analysis")
            st.markdown(result.comparative_summary)

        # Strengths, areas to improve, recommendations
        col_left, col_right = st.columns(2)
        with col_left:
            if result.strengths:
                results.bullet_list("💪 Strengths", result.strengths)
        with col_right:
            if result.areas_to_improve:
                results.bullet_list("🎯 Areas to Improve", result.areas_to_improve)

        if result.recommendations:
            st.markdown("")
            results.bullet_list("📋 Recommendations", result.recommendations)

    # -- Documentation --
    st.divider()
    doc_viewer.render(
        _DOC_PATH,
        title="Documentation: RAG, Vector Stores & Phased Retrieval",
    )
