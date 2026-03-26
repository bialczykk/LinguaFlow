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
