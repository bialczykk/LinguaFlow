# Project 5: Content Moderation & QA System — Design Specification

## Overview

**Department:** QA
**Difficulty:** Intermediate → Advanced

The QA department reviews all AI-generated lesson content before publication. The system generates short lesson snippets (vocabulary exercises, grammar explanations, reading passages), then pauses for human moderator review at two checkpoints. Moderators can approve, edit, or reject content. Rejected content loops back for regeneration with feedback. Nothing gets published without human sign-off.

This project introduces human-in-the-loop patterns (`interrupt()`, `Command(resume=...)`), multiple interrupt points in a single graph, revision loops, and the LangSmith deep dive (evaluation datasets, custom evaluators, A/B prompt comparison). It also integrates the 4-tier error handling strategy naturally into the workflow.

## Concepts Introduced

- **Human-in-the-loop:** `interrupt()`, `Command(resume=...)`, approval/edit/reject workflows
- **Multiple interrupt points** in a single graph — draft review and final review
- **Revision loops** — rejected content goes back to the generator with moderator feedback
- **4-tier error handling** — integrated naturally (RetryPolicy on LLM, interrupt for moderator decisions, errors bubble up)
- **LangSmith deep dive:** programmatic evaluation datasets, custom evaluators, A/B prompt comparison
- **`@traceable`** on all nodes, tagged `p5-content-moderation`

## Architecture: Linear Graph with Two Review Checkpoints

The graph is a linear pipeline with two interrupt points and a revision loop:

```
START → generate → draft_review (interrupt) → [decision?]
                                                 ├── approve → polish → final_review (interrupt) → [decision?]
                                                 │                                                    ├── approve → publish → END
                                                 │                                                    └── reject → END (killed)
                                                 ├── edit → polish → final_review → ...
                                                 └── reject → revise → draft_review (loop back)
```

### Six Nodes

1. **`generate`** — LLM produces a lesson snippet based on a content request (topic, type, difficulty level)
2. **`draft_review`** — calls `interrupt()` with the draft. Moderator sees the content and decides: approve, edit, or reject with feedback. Uses `Command(resume=...)` to return the decision. Approve/edit → forward to `polish`. Reject → loop to `revise`.
3. **`revise`** — LLM regenerates the content incorporating moderator feedback. Returns to `draft_review`. Max 2 revision rounds to prevent infinite loops.
4. **`polish`** — LLM does a final formatting/cleanup pass on the approved or edited content
5. **`final_review`** — second `interrupt()`. Moderator sees the polished content. Approve → `publish`. Reject → END (content is killed, not looped again — this is the final gate).
6. **`publish`** — marks content as approved, records metadata. Terminal node.

The two interrupts serve different purposes: `draft_review` is "is this content on the right track?" (with a revision loop), `final_review` is "is this ready to publish?" (binary yes/no gate).

The graph requires a checkpointer (mandatory for interrupts). Tests use `InMemorySaver`.

## State Schema

```python
class ContentModerationState(TypedDict):
    """State for the content moderation workflow."""

    # -- Input (set at invocation) --
    content_request: dict              # {"topic": ..., "type": ..., "difficulty": ...}

    # -- After generate/revise --
    draft_content: str                 # The generated lesson snippet
    generation_confidence: float       # LLM's self-assessed confidence (0-1)

    # -- After draft_review --
    draft_decision: dict               # {"action": "approve"|"edit"|"reject", "feedback": ..., "edited_content": ...}
    revision_count: int                # Tracks revision rounds (max 2)

    # -- After polish --
    polished_content: str              # Cleaned-up content ready for final review

    # -- After final_review --
    final_decision: dict               # {"action": "approve"|"reject", "feedback": ...}

    # -- After publish --
    published: bool                    # Whether content was published
    publish_metadata: dict | None      # Timestamp, moderator notes, etc.
```

Key design choices:
- **Plain `TypedDict`** — no `MessagesState` base, since this isn't a conversational agent. Content flows through a pipeline, not a chat.
- **`draft_decision` and `final_decision` as dicts** — these are the values returned from `Command(resume=...)`. Flexible and matches what `interrupt()` returns.
- **`revision_count`** — simple guard against infinite revision loops.
- **No reducers** — each field is written by exactly one node.

## Interrupt & Resume Mechanics

### `draft_review` node

Calls `interrupt()` with a payload containing the draft content, confidence score, and revision history. The moderator resumes with a decision dict:

```python
def draft_review(state):
    decision = interrupt({
        "content": state["draft_content"],
        "confidence": state["generation_confidence"],
        "revision_count": state["revision_count"],
        "prompt": "Review this draft. Approve, edit, or reject with feedback.",
    })
    # decision = {"action": "approve"|"edit"|"reject", "feedback": "...", "edited_content": "..."}
    return {"draft_decision": decision}
```

Routing after `draft_review` is a conditional edge:
- `action == "approve"` → `polish`
- `action == "edit"` → `polish` (the routing node copies `edited_content` from the decision into `draft_content` so `polish` always reads from `draft_content`)
- `action == "reject"` and `revision_count < 2` → `revise`
- `action == "reject"` and `revision_count >= 2` → `END` (too many revisions)

### `final_review` node

Same pattern, simpler payload. No revision loop — just approve or reject:

```python
def final_review(state):
    decision = interrupt({
        "content": state["polished_content"],
        "prompt": "Final review. Approve for publication or reject.",
    })
    # decision = {"action": "approve"|"reject", "feedback": "..."}
    return {"final_decision": decision}
```

Routing: approve → `publish`, reject → `END`.

### Resuming in code

```python
# Initial run — hits first interrupt
result = graph.invoke(initial_state, config)
# result contains __interrupt__ with the draft

# Moderator approves
result = graph.invoke(Command(resume={"action": "approve"}), config)
# Hits second interrupt with polished content

# Moderator approves final
result = graph.invoke(Command(resume={"action": "approve"}), config)
# Content published
```

## LangSmith Deep Dive

### 1. Evaluation Pipeline (`evaluation.py`)

A standalone script that:
- Creates a dataset of test cases programmatically via the LangSmith SDK — each case is a content request + expected quality criteria (topic relevance, difficulty match, appropriate length)
- Defines 3 custom evaluators:
  - **Topic relevance** — does the generated content match the requested topic? (LLM-as-judge)
  - **Difficulty match** — is the content appropriate for the requested CEFR level? (LLM-as-judge)
  - **Content quality** — grammar, clarity, completeness score (LLM-as-judge)
- Runs the `generate` node (not the full graph — HITL is skipped for automated evaluation) against the dataset with all evaluators
- Outputs a summary table of scores

### 2. A/B Prompt Comparison (`ab_comparison.py`)

A script that:
- Defines two prompt variants for the `generate` node (one more structured, one more creative)
- Runs both variants against the same evaluation dataset
- Compares scores side-by-side
- Demonstrates how to use LangSmith experiments to track which prompt performs better

Both scripts are standalone — they import from the project modules but run independently. Evaluation code is cleanly separated from graph code.

## Error Handling (4-Tier, Integrated Naturally)

- **Tier 1 (transient):** `RetryPolicy(max_attempts=3)` on LLM nodes (generate, revise, polish)
- **Tier 2 (LLM-recoverable):** Not applicable — no tool calls in this project
- **Tier 3 (user-fixable):** The two `interrupt()` checkpoints — moderators fix content issues
- **Tier 4 (unexpected):** Errors bubble up. Educational doc explains the full framework.

## Module Structure

```
projects/05-content-moderation-qa/
├── models.py              # ContentModerationState, ContentRequest, PublishMetadata
├── prompts.py             # Prompt templates for generate, revise, polish nodes
├── nodes.py               # generate, draft_review, revise, polish, final_review, publish + routing
├── graph.py               # build_graph() → compiled StateGraph with checkpointer
├── evaluation.py          # LangSmith evaluation pipeline: dataset, custom evaluators, run
├── ab_comparison.py       # A/B prompt comparison script
├── data/
│   ├── __init__.py
│   └── content_requests.py  # Sample content requests for testing and evaluation
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Shared fixtures (graph with checkpointer, sample requests)
│   ├── test_models.py     # State schema validation
│   ├── test_nodes.py      # Node unit tests (generate output, routing logic)
│   ├── test_graph.py      # Full interrupt/resume flows: approve, edit, reject, revision loop
│   └── test_evaluation.py # Evaluator function tests (no LangSmith API needed)
├── README.md
└── requirements.txt
```

## Dependencies

```
langchain-core
langchain-anthropic
langgraph
langsmith
python-dotenv
pytest
```

## LangSmith Integration

- `@traceable` decorator on all node functions
- All traces tagged with `["p5-content-moderation"]`
- Config passed through: `{"tags": ["p5-content-moderation"]}`
- Evaluation scripts use `langsmith.Client` for dataset and evaluator management

## Testing Strategy

1. **`test_models.py`** — State schema, Pydantic model validation
2. **`test_nodes.py`** — Generate node produces content, routing functions return correct targets, revision count guard works
3. **`test_graph.py`** — Full interrupt/resume workflows:
   - Happy path: generate → approve draft → approve final → published
   - Edit path: generate → edit draft → approve final → published
   - Reject path: generate → reject draft → revise → approve draft → approve final
   - Max revisions: reject twice → graph ends without publishing
   - Final rejection: approve draft → reject final → not published
4. **`test_evaluation.py`** — Custom evaluator functions return scores in expected range, work with sample data. No LangSmith API calls needed — just the evaluator logic.
