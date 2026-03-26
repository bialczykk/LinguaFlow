# Project 02: Lesson Plan Generator

A personalized lesson plan generator built with **LangGraph StateGraph** for the LinguaFlow English tutoring platform.

## What It Does

Takes a student through a short intake conversation to understand their needs, then generates a personalized lesson plan using a multi-node graph pipeline with:

- **Conditional routing** — routes to different drafting nodes based on lesson type (conversation, grammar, or exam prep)
- **Review loop** — an LLM reviewer critiques the draft and can send it back for revision (up to 2 cycles)
- **Structured output** — the final lesson plan is parsed into a validated Pydantic model

## Concepts Learned

- LangGraph StateGraph: defining graphs, state schemas, nodes, edges
- Conditional routing with `add_conditional_edges()`
- Graph cycles (review → draft loop)
- Graph compilation and invocation with streaming
- LangSmith: tracing graph execution, viewing node-level traces

## Project Structure

```
├── models.py          # Pydantic models + LangGraph state schema
├── intake.py          # Multi-turn intake conversation
├── prompts.py         # All prompt templates
├── nodes.py           # Node functions (research, draft, review, finalize)
├── graph.py           # StateGraph wiring and compilation
├── main.py            # Interactive CLI
├── data/
│   └── sample_profiles.py  # Test data
└── tests/
    ├── test_models.py      # Unit tests
    ├── test_nodes.py       # Node integration tests
    ├── test_graph.py       # Full graph integration tests
    └── test_intake.py      # Intake conversation tests
```

## Setup

```bash
# From the repo root (shared venv)
source .venv/bin/activate

# Ensure .env has ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

## Run

```bash
cd projects/02-lesson-plan-generator
python main.py
```

## Test

```bash
cd projects/02-lesson-plan-generator
python -m pytest tests/ -v
```

## Graph Topology

```
START → research → route → draft_conversation ─┐
                     │       draft_grammar ─────┤──→ review ──→ finalize → END
                     └──→    draft_exam_prep ───┘       ↑          │
                                                        └──────────┘
                                                     (revision loop,
                                                      max 2 cycles)
```
