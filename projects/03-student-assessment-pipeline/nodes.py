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
