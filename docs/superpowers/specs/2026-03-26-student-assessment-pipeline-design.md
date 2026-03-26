# Student Assessment Pipeline — Design Specification

**Project:** 03-student-assessment-pipeline
**Department:** Content
**Difficulty:** Intermediate
**Date:** 2026-03-26

## Overview

The Content department maintains a library of English learning materials — rubrics, CEFR standards, and graded sample essays. They need a system that assesses student writing submissions by retrieving relevant materials, scoring against criteria, and comparing against level-appropriate samples.

This project introduces **RAG** (Retrieval-Augmented Generation) integrated into a **LangGraph StateGraph**. The key architectural pattern: early assessment results shape later retrieval — the criteria scoring node determines a preliminary CEFR level, which drives which sample essays are retrieved for comparison.

## Concepts Introduced

- RAG pipeline: document loaders, text splitting, embeddings, vector stores
- Retrieval chains integrated into a LangGraph graph
- Combining retrieval with structured generation
- Metadata-filtered retrieval from Chroma
- Phased retrieval where early results inform later queries
- LangSmith: tracing retrieval quality, inspecting retrieved documents in traces

## Architecture

Two separate concerns:

1. **Document ingestion** (`ingestion.py`) — standalone module that loads rubrics, CEFR standards, and graded sample essays into a Chroma vector store with metadata tags. Run once as setup, separate from the assessment graph.

2. **Assessment graph** (`graph.py`) — a 5-node LangGraph StateGraph that takes a student submission and produces a multi-criteria, comparative assessment.

### Graph Flow

```
retrieve_standards → criteria_scoring → retrieve_samples → comparative_analysis → synthesize
```

All edges are sequential (no conditional routing in the main path). The adaptive behavior comes from `criteria_scoring` producing a `preliminary_level` that `retrieve_samples` uses as a metadata filter.

## Data Layer

### Document Library

Three categories of documents, defined as Python data structures in `data/` modules:

**1. CEFR Rubrics** (`type: "rubric"`)
Scoring criteria for each assessment dimension. One document per dimension per level band.

Dimensions:
- Grammar & Accuracy
- Vocabulary Range & Precision
- Coherence & Organization
- Task Achievement

Level bands: A1-A2, B1-B2, C1-C2 (3 bands x 4 dimensions = 12 documents)

Each rubric document describes what scores 1-5 look like for that dimension at that level band.

**2. CEFR Level Descriptors** (`type: "standard"`)
What a learner at each CEFR level can do in writing. One document per level (A1-C2). 6 documents.

**3. Sample Graded Essays** (`type: "sample_essay"`)
Example student writings with assigned grades. Each includes:
- The essay text
- The prompt/task that was given
- Assigned CEFR level
- Brief assessor note explaining the grade

Metadata: `cefr_level`, `score` (1-5 overall quality within that level)

~2-3 samples per level = 12-18 documents.

### Vector Store

- **Chroma** with a single collection
- Metadata filtering by `type` and `cefr_level`
- Persisted to `chroma_db/` directory within the project (gitignored)
- **Embeddings:** `sentence-transformers` via `langchain-huggingface` (local, no API key needed). LLM calls remain Anthropic-only.

### Ingestion Module (`ingestion.py`)

Standalone module responsible for:
1. Loading document data from `data/` modules
2. Splitting longer documents with `RecursiveCharacterTextSplitter`
3. Creating embeddings via `sentence-transformers`
4. Storing in Chroma with metadata

Exposes a `build_vector_store()` function that returns the populated Chroma instance, and a `get_vector_store()` that loads an existing persisted store.

## Graph State

```python
class AssessmentState(TypedDict):
    # -- Input --
    submission_text: str            # The student's writing to assess
    submission_context: str         # The prompt/task the student was responding to
    student_level_hint: str         # Optional self-reported CEFR level

    # -- After retrieve_standards --
    retrieved_standards: list[Document]  # Rubrics + level descriptors

    # -- After criteria_scoring --
    criteria_scores: CriteriaScores     # Multi-dimension scores
    preliminary_level: str              # CEFR level from scoring (drives next retrieval)

    # -- After retrieve_samples --
    retrieved_samples: list[Document]   # Sample essays at the preliminary level

    # -- After comparative_analysis --
    comparative_analysis: ComparativeAnalysis  # Comparison against samples

    # -- After synthesize --
    final_assessment: Assessment        # Complete structured output
```

## Graph Nodes

### 1. `retrieve_standards`

**Purpose:** Fetch rubrics and CEFR level descriptors relevant to the submission.

**Retrieval strategy:**
- Query Chroma with the submission text
- Filter: `type IN ["rubric", "standard"]`
- If `student_level_hint` is provided, also filter for rubrics at that level band to improve relevance
- Return top-k relevant documents (k=8-10)

**Returns:** `{"retrieved_standards": [Document, ...]}`

### 2. `criteria_scoring`

**Purpose:** Score the submission across 4 dimensions using retrieved standards as grounding.

**Process:**
- Receives the submission + retrieved rubrics/standards
- LLM evaluates each dimension (grammar, vocabulary, coherence, task achievement)
- For each dimension: assigns a score (1-5), cites evidence from the submission, provides specific feedback
- Determines a preliminary CEFR level based on the aggregate scores

**Returns:** `{"criteria_scores": CriteriaScores, "preliminary_level": "B1"}`

### 3. `retrieve_samples`

**Purpose:** Fetch sample essays at the preliminary CEFR level for comparison.

**Retrieval strategy:**
- Query Chroma with the submission text
- Filter: `type == "sample_essay"` AND `cefr_level == preliminary_level`
- Also retrieve 1-2 samples from adjacent levels (one above, one below) for contrast
- Return top-k results (k=3-5)

**Returns:** `{"retrieved_samples": [Document, ...]}`

### 4. `comparative_analysis`

**Purpose:** Compare the submission against retrieved sample essays.

**Process:**
- LLM compares the submission to each retrieved sample
- Notes similarities and differences in quality, style, and proficiency markers
- Produces a narrative summary of where the submission sits relative to the samples
- This grounds the assessment in concrete examples rather than abstract criteria alone

**Returns:** `{"comparative_analysis": ComparativeAnalysis}`

### 5. `synthesize`

**Purpose:** Merge criteria scores and comparative analysis into a final structured assessment.

**Process:**
- Combines the criteria scores (from node 2) with the comparative analysis (from node 4)
- Determines a final CEFR level (may adjust the preliminary level based on comparative evidence)
- Generates overall strengths, areas to improve, and actionable recommendations
- Outputs a fully structured `Assessment` object

**Returns:** `{"final_assessment": Assessment}`

## Pydantic Models

```python
class CriterionScore(BaseModel):
    """Score for a single assessment dimension."""
    dimension: str          # e.g., "Grammar & Accuracy"
    score: int              # 1-5
    evidence: list[str]     # Quotes/examples from the submission
    feedback: str           # Specific feedback for this dimension

class CriteriaScores(BaseModel):
    """Multi-criteria scoring results."""
    scores: list[CriterionScore]
    preliminary_level: str  # CEFR level based on aggregate scores
    scoring_rationale: str  # Why this level was assigned

class SampleComparison(BaseModel):
    """Comparison of the submission against one sample essay."""
    sample_level: str       # CEFR level of the sample
    similarities: list[str] # What the submission shares with this sample
    differences: list[str]  # Where the submission diverges
    quality_position: str   # "above", "comparable", or "below" this sample

class ComparativeAnalysis(BaseModel):
    """Full comparative analysis across all retrieved samples."""
    comparisons: list[SampleComparison]
    narrative: str          # Overall narrative summary

class Assessment(BaseModel):
    """Complete structured assessment — the final output."""
    submission_text: str
    overall_level: str                  # Final CEFR level
    criteria_scores: list[CriterionScore]
    comparative_summary: str            # How submission compares to samples
    strengths: list[str]
    areas_to_improve: list[str]
    recommendations: list[str]          # Actionable next steps
    confidence: str                     # "high", "medium", "low"
```

## Module Structure

```
projects/03-student-assessment-pipeline/
    requirements.txt
    models.py               # Pydantic models + AssessmentState TypedDict
    ingestion.py            # Document ingestion into Chroma
    prompts.py              # All prompt templates
    nodes.py                # Node functions for the graph
    graph.py                # StateGraph assembly + compilation
    main.py                 # CLI entry point
    data/
        __init__.py
        rubrics.py          # CEFR rubric documents
        standards.py        # CEFR level descriptors
        sample_essays.py    # Graded sample essays
        sample_submissions.py  # Test submissions for running the pipeline
    tests/
        __init__.py
        conftest.py         # Shared fixtures (pre-populated Chroma store)
        test_models.py      # Model validation tests
        test_ingestion.py   # Ingestion pipeline tests
        test_nodes.py       # Individual node tests
        test_graph.py       # End-to-end graph tests
    chroma_db/              # Persisted vector store (gitignored)
```

## Dependencies

```
langchain-core
langchain-anthropic
langchain-chroma
langchain-huggingface
langgraph
langsmith
sentence-transformers
chromadb
python-dotenv
pytest
```

## LangSmith Integration

- All nodes decorated with `@traceable`
- Retrieval nodes log: query text, filters used, number of results, document metadata
- Scoring/analysis nodes log: input documents, LLM reasoning
- End-to-end traces show the full pipeline including both retrieval phases

## Testing Strategy

1. **Model tests** — validate Pydantic models accept valid data and reject invalid
2. **Ingestion tests** — verify documents are loaded, split, embedded, and stored correctly; verify metadata filtering works
3. **Node tests** — each node tested individually with a pre-populated Chroma fixture (in `conftest.py`); integration tests that hit the real LLM
4. **Graph tests** — end-to-end: feed a sample submission, verify the full `Assessment` output has all required fields and sensible content
