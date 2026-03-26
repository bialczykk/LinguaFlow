# Project 5: Content Moderation & QA System

A content moderation pipeline that generates lesson content, pauses for human moderator review at two checkpoints, and supports revision loops. Includes a LangSmith evaluation pipeline with A/B prompt comparison.

## What This Teaches

- **Human-in-the-loop**: `interrupt()`, `Command(resume=...)`, approval workflows
- **Multiple interrupt points**: draft review + final review in one graph
- **Revision loops**: reject → revise → re-review with max revision guard
- **4-tier error handling**: RetryPolicy, interrupt for user-fixable, bubble up
- **LangSmith deep dive**: evaluation datasets, custom evaluators (LLM-as-judge), A/B prompt comparison

## How It Works

The graph flows through six nodes:
1. **Generate** — LLM creates a lesson snippet
2. **Draft Review** — `interrupt()` pauses for moderator (approve/edit/reject)
3. **Revise** — if rejected, LLM revises with feedback (max 2 rounds)
4. **Polish** — LLM does final cleanup on approved content
5. **Final Review** — `interrupt()` pauses for final approval (approve/reject)
6. **Publish** — marks content as approved

## Running Evaluations

```bash
# Create dataset and run evaluation
python evaluation.py

# A/B prompt comparison
python ab_comparison.py
```

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Just routing logic (no LLM)
python -m pytest tests/test_nodes.py::TestRouteAfterDraftReview -v

# Interrupt/resume workflows (requires API key)
python -m pytest tests/test_graph.py -v
```

## Project Structure

```
models.py        — State schema (TypedDict) and Pydantic models
prompts.py       — Prompt templates for generate, revise, polish + A/B variants
nodes.py         — 6 node functions + 2 routing functions
graph.py         — StateGraph with interrupt points and revision loop
evaluation.py    — LangSmith evaluation pipeline with custom evaluators
ab_comparison.py — A/B prompt comparison using LangSmith experiments
data/            — Sample content requests (scaffolding)
tests/           — Unit tests (routing) + integration tests (interrupt/resume)
```
