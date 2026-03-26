# Content Moderation & QA System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a content moderation graph with two human-in-the-loop review checkpoints, revision loops, and a LangSmith evaluation pipeline with A/B prompt comparison.

**Architecture:** Linear StateGraph with six nodes (generate → draft_review → revise/polish → final_review → publish). Two `interrupt()` points pause for moderator decisions. `Command(resume=...)` resumes with approve/edit/reject. Standalone evaluation scripts use the LangSmith SDK to create datasets, run custom evaluators, and compare prompt variants.

**Tech Stack:** LangGraph (StateGraph, interrupt, Command, RetryPolicy), LangChain (ChatAnthropic), LangSmith (Client, evaluate), Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-content-moderation-qa-design.md`

---

## File Structure

```
projects/05-content-moderation-qa/
├── models.py              # ContentModerationState TypedDict, ContentRequest, PublishMetadata
├── prompts.py             # Prompt templates for generate, revise, polish
├── nodes.py               # 6 node functions + 2 routing functions
├── graph.py               # build_graph() with checkpointer injection
├── evaluation.py          # LangSmith evaluation: dataset creation, custom evaluators, run
├── ab_comparison.py       # A/B prompt comparison using LangSmith experiments
├── data/
│   ├── __init__.py
│   └── content_requests.py  # Sample content requests
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Fixtures: graph_with_memory, sample_request
│   ├── test_models.py     # State schema validation
│   ├── test_nodes.py      # Node + routing function tests
│   ├── test_graph.py      # Full interrupt/resume workflow tests
│   └── test_evaluation.py # Evaluator function unit tests
├── README.md
└── requirements.txt
```

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `projects/05-content-moderation-qa/requirements.txt`
- Create: `projects/05-content-moderation-qa/data/__init__.py`
- Create: `projects/05-content-moderation-qa/tests/__init__.py`

- [ ] **Step 1: Create project directory and requirements.txt**

```
projects/05-content-moderation-qa/requirements.txt
```

```
langchain-core
langchain-anthropic
langgraph
langsmith
python-dotenv
pytest
```

- [ ] **Step 2: Create empty `__init__.py` files**

Create empty files:
- `projects/05-content-moderation-qa/data/__init__.py`
- `projects/05-content-moderation-qa/tests/__init__.py`

- [ ] **Step 3: Verify imports work**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -c "from langgraph.types import interrupt, Command; from langgraph.types import RetryPolicy; from langsmith import Client; print('All imports OK')"`

- [ ] **Step 4: Commit**

```bash
git add projects/05-content-moderation-qa/
git commit -m "feat(p5): scaffold project skeleton and dependencies"
```

---

### Task 2: Pydantic Models and State Schema (TDD)

**Files:**
- Create: `projects/05-content-moderation-qa/tests/test_models.py`
- Create: `projects/05-content-moderation-qa/models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# tests/test_models.py
"""Tests for Pydantic models and state schema."""

import typing
import pytest
from models import ContentRequest, PublishMetadata, ContentModerationState


class TestContentRequest:
    def test_valid_request(self):
        req = ContentRequest(
            topic="Present Perfect Tense",
            content_type="grammar_explanation",
            difficulty="B1",
        )
        assert req.topic == "Present Perfect Tense"
        assert req.content_type == "grammar_explanation"
        assert req.difficulty == "B1"

    def test_difficulty_must_be_cefr(self):
        with pytest.raises(Exception):
            ContentRequest(
                topic="Test", content_type="grammar_explanation", difficulty="X9",
            )

    def test_content_type_constrained(self):
        with pytest.raises(Exception):
            ContentRequest(
                topic="Test", content_type="invalid_type", difficulty="A1",
            )


class TestPublishMetadata:
    def test_valid_metadata(self):
        meta = PublishMetadata(
            moderator_notes="Looks good",
            review_rounds=1,
        )
        assert meta.moderator_notes == "Looks good"
        assert meta.review_rounds == 1


class TestContentModerationState:
    def test_state_has_required_fields(self):
        hints = typing.get_type_hints(ContentModerationState)
        assert "content_request" in hints
        assert "draft_content" in hints
        assert "generation_confidence" in hints
        assert "draft_decision" in hints
        assert "revision_count" in hints
        assert "polished_content" in hints
        assert "final_decision" in hints
        assert "published" in hints
        assert "publish_metadata" in hints
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

```python
# models.py
"""Pydantic models and LangGraph state schema for the Content Moderation system.

This module defines:
- ContentRequest: what kind of content to generate
- PublishMetadata: metadata attached to published content
- ContentModerationState: the TypedDict flowing through the graph

Key concepts demonstrated:
- Plain TypedDict state (not MessagesState) for non-conversational pipelines
- State fields map to specific nodes in the graph
- No reducers needed — each field is written by exactly one node
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Valid CEFR levels
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

# Valid content types the system can generate
CONTENT_TYPES = (
    "grammar_explanation",
    "vocabulary_exercise",
    "reading_passage",
)


class ContentRequest(BaseModel):
    """A request to generate lesson content."""

    topic: str = Field(description="The topic to cover (e.g., 'Present Perfect Tense')")
    content_type: Literal[
        "grammar_explanation", "vocabulary_exercise", "reading_passage"
    ] = Field(description="Type of content to generate")
    difficulty: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Target CEFR difficulty level"
    )


class PublishMetadata(BaseModel):
    """Metadata attached to published content."""

    moderator_notes: str = Field(default="", description="Notes from the moderator")
    review_rounds: int = Field(default=0, description="Number of review rounds")


class ContentModerationState(TypedDict):
    """State schema for the content moderation StateGraph.

    This is a pipeline state (not conversational), so we use a plain
    TypedDict instead of MessagesState. Each field is written by exactly
    one node — no reducers needed.

    LangGraph concept: TypedDict state for non-conversational workflows.
    Human-in-the-loop concept: draft_decision and final_decision hold
    the values returned from Command(resume=...) at each interrupt point.
    """

    # -- Input (set at invocation) --
    content_request: dict              # ContentRequest as dict

    # -- After generate/revise --
    draft_content: str                 # The generated lesson snippet
    generation_confidence: float       # LLM's self-assessed confidence (0-1)

    # -- After draft_review --
    draft_decision: dict               # {"action": "approve"|"edit"|"reject", ...}
    revision_count: int                # Tracks revision rounds (max 2)

    # -- After polish --
    polished_content: str              # Cleaned-up content ready for final review

    # -- After final_review --
    final_decision: dict               # {"action": "approve"|"reject", ...}

    # -- After publish --
    published: bool                    # Whether content was published
    publish_metadata: dict | None      # Timestamp, moderator notes, etc.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/05-content-moderation-qa/models.py projects/05-content-moderation-qa/tests/test_models.py
git commit -m "feat(p5): add Pydantic models and state schema with tests"
```

---

### Task 3: Sample Content Requests

**Files:**
- Create: `projects/05-content-moderation-qa/data/content_requests.py`

- [ ] **Step 1: Create sample content requests**

```python
# data/content_requests.py
"""Sample content requests for testing and evaluation.

Each request specifies a topic, content type, and CEFR difficulty level.
Used by tests and by the LangSmith evaluation pipeline.

This is scaffolding — it provides realistic inputs for the graph.
"""

SAMPLE_REQUESTS = [
    {
        "topic": "Present Perfect Tense",
        "content_type": "grammar_explanation",
        "difficulty": "B1",
    },
    {
        "topic": "Food and Cooking Vocabulary",
        "content_type": "vocabulary_exercise",
        "difficulty": "A2",
    },
    {
        "topic": "Climate Change",
        "content_type": "reading_passage",
        "difficulty": "B2",
    },
    {
        "topic": "Daily Routines",
        "content_type": "grammar_explanation",
        "difficulty": "A1",
    },
    {
        "topic": "Business Email Etiquette",
        "content_type": "reading_passage",
        "difficulty": "C1",
    },
    {
        "topic": "Travel and Transportation",
        "content_type": "vocabulary_exercise",
        "difficulty": "A2",
    },
    {
        "topic": "Conditional Sentences (Type 2)",
        "content_type": "grammar_explanation",
        "difficulty": "B2",
    },
    {
        "topic": "Technology and Innovation",
        "content_type": "reading_passage",
        "difficulty": "C1",
    },
]
```

- [ ] **Step 2: Verify data loads**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -c "from data.content_requests import SAMPLE_REQUESTS; print(f'{len(SAMPLE_REQUESTS)} requests'); assert len(SAMPLE_REQUESTS) == 8; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add projects/05-content-moderation-qa/data/
git commit -m "feat(p5): add sample content requests for testing and evaluation"
```

---

### Task 4: Prompt Templates

**Files:**
- Create: `projects/05-content-moderation-qa/prompts.py`

- [ ] **Step 1: Create prompt templates**

```python
# prompts.py
"""Prompt templates for the content generation and revision nodes.

Three prompts:
- GENERATE_PROMPT: creates a lesson snippet from a content request
- REVISE_PROMPT: regenerates content incorporating moderator feedback
- POLISH_PROMPT: final formatting/cleanup pass on approved content

LangChain concept demonstrated:
- ChatPromptTemplate with {variable} placeholders
- Different prompts for different stages of the same content
"""

from langchain_core.prompts import ChatPromptTemplate

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow, an English tutoring platform. "
        "You produce high-quality lesson content for students.\n\n"
        "Generate a short lesson snippet (150-300 words) based on the request below. "
        "The content should be appropriate for the specified CEFR difficulty level.\n\n"
        "After generating the content, assess your own confidence in the quality "
        "on a scale from 0.0 to 1.0. Be honest — flag uncertainty.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate the lesson content.",
    ),
])

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow. A moderator has reviewed your "
        "previous draft and rejected it with feedback. Please revise the content "
        "based on their feedback.\n\n"
        "The revised content should still be 150-300 words and appropriate for "
        "the specified CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the revised lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Original Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Previous Draft:\n{previous_draft}\n\n"
        "Moderator Feedback:\n{feedback}\n\n"
        "Please revise the content.",
    ),
])

POLISH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an editor for LinguaFlow. Do a final formatting and cleanup pass "
        "on this lesson content. Fix any remaining grammar issues, improve clarity, "
        "and ensure consistent formatting. Keep the content at the same difficulty "
        "level and approximately the same length.\n\n"
        "Return ONLY the polished content text (no JSON wrapping).",
    ),
    (
        "human",
        "Content Type: {content_type}\n"
        "Difficulty Level: {difficulty}\n\n"
        "Content to polish:\n{content}\n\n"
        "Please produce the final polished version.",
    ),
])

# -- A/B Prompt Variants for evaluation --
# These are alternative generate prompts used by ab_comparison.py

GENERATE_PROMPT_STRUCTURED = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow. Generate structured lesson content "
        "that follows a clear template:\n\n"
        "For grammar_explanation: Introduction → Rule → Examples → Common Mistakes\n"
        "For vocabulary_exercise: Word List → Definitions → Fill-in-the-Blank Exercises\n"
        "For reading_passage: Title → Passage → Comprehension Questions\n\n"
        "Keep content at 150-300 words, appropriate for the CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate the lesson content following the structured template.",
    ),
])

GENERATE_PROMPT_CREATIVE = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a creative content writer for LinguaFlow. Generate engaging lesson "
        "content that tells a story, uses humor, or creates a memorable scenario to "
        "teach the concept. Make the learning experience fun and memorable rather than "
        "formulaic.\n\n"
        "Keep content at 150-300 words, appropriate for the CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate creative, engaging lesson content.",
    ),
])
```

- [ ] **Step 2: Verify prompts load**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -c "from prompts import GENERATE_PROMPT, REVISE_PROMPT, POLISH_PROMPT, GENERATE_PROMPT_STRUCTURED, GENERATE_PROMPT_CREATIVE; print('5 prompts loaded OK')"`

- [ ] **Step 3: Commit**

```bash
git add projects/05-content-moderation-qa/prompts.py
git commit -m "feat(p5): add prompt templates for generate, revise, polish, and A/B variants"
```

---

### Task 5: Node Functions (TDD)

**Files:**
- Create: `projects/05-content-moderation-qa/tests/test_nodes.py`
- Create: `projects/05-content-moderation-qa/nodes.py`

- [ ] **Step 1: Write failing tests for nodes**

```python
# tests/test_nodes.py
"""Tests for node functions and routing logic.

Unit tests for routing functions (no LLM).
Integration tests for generate node (hits LLM).
"""

import pytest
from langchain_core.messages import AIMessage

from models import ContentModerationState
from nodes import (
    route_after_draft_review,
    route_after_final_review,
    generate_node,
    publish_node,
)


class TestRouteAfterDraftReview:
    """Routing logic after draft_review interrupt."""

    def test_approve_routes_to_polish(self):
        state = {"draft_decision": {"action": "approve"}, "revision_count": 0}
        assert route_after_draft_review(state) == "polish"

    def test_edit_routes_to_polish(self):
        state = {
            "draft_decision": {"action": "edit", "edited_content": "Better version"},
            "revision_count": 0,
        }
        assert route_after_draft_review(state) == "polish"

    def test_reject_routes_to_revise(self):
        state = {
            "draft_decision": {"action": "reject", "feedback": "Too basic"},
            "revision_count": 0,
        }
        assert route_after_draft_review(state) == "revise"

    def test_reject_at_max_revisions_routes_to_end(self):
        state = {
            "draft_decision": {"action": "reject", "feedback": "Still bad"},
            "revision_count": 2,
        }
        assert route_after_draft_review(state) == "__end__"


class TestRouteAfterFinalReview:
    """Routing logic after final_review interrupt."""

    def test_approve_routes_to_publish(self):
        state = {"final_decision": {"action": "approve"}}
        assert route_after_final_review(state) == "publish"

    def test_reject_routes_to_end(self):
        state = {"final_decision": {"action": "reject", "feedback": "Not ready"}}
        assert route_after_final_review(state) == "__end__"


class TestPublishNode:
    """publish_node marks content as published."""

    def test_publish_sets_published_true(self):
        state = {
            "polished_content": "Some content",
            "revision_count": 1,
        }
        result = publish_node(state)
        assert result["published"] is True
        assert result["publish_metadata"] is not None
        assert result["publish_metadata"]["review_rounds"] == 1


@pytest.mark.integration
class TestGenerateNode:
    """generate_node produces content via LLM."""

    def test_generate_returns_content_and_confidence(self):
        state = {
            "content_request": {
                "topic": "Present Perfect Tense",
                "content_type": "grammar_explanation",
                "difficulty": "B1",
            },
            "revision_count": 0,
        }
        result = generate_node(state)
        assert "draft_content" in result
        assert len(result["draft_content"]) > 50
        assert "generation_confidence" in result
        assert 0.0 <= result["generation_confidence"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement nodes.py**

```python
# nodes.py
"""Node functions for the Content Moderation & QA StateGraph.

Contains 6 node functions and 2 routing functions:
- generate_node: LLM generates lesson content
- draft_review_node: interrupt() for first moderator review
- revise_node: LLM revises content with moderator feedback
- polish_node: LLM does final cleanup
- final_review_node: interrupt() for final moderator review
- publish_node: marks content as published
- route_after_draft_review: conditional routing after draft review
- route_after_final_review: conditional routing after final review

LangGraph concepts demonstrated:
- interrupt() to pause execution for human input
- Command(resume=...) to provide the human's decision
- RetryPolicy for transient error handling on LLM nodes
- Revision loop with max revision guard

Human-in-the-loop concepts:
- Two interrupt points with different purposes
- draft_review: approve/edit/reject with revision loop
- final_review: approve/reject gate (no loop)
"""

import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.types import interrupt
from langsmith import traceable

from models import ContentModerationState
from prompts import GENERATE_PROMPT, REVISE_PROMPT, POLISH_PROMPT

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# -- LangSmith Tags --
_TAGS = ["p5-content-moderation"]

# -- Max revision rounds --
MAX_REVISIONS = 2


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response from the LLM, handling markdown code fences.

    The LLM sometimes wraps JSON in ```json ... ``` blocks.
    This helper strips those before parsing.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


@traceable(name="generate", run_type="chain", tags=_TAGS)
def generate_node(state: ContentModerationState) -> dict:
    """Generate a lesson snippet from a content request.

    Uses the LLM to produce content matching the requested topic, type,
    and CEFR difficulty level. The LLM also self-assesses its confidence.

    LangGraph concept: a standard LLM node that produces content
    for downstream human review.
    """
    request = state["content_request"]
    chain = GENERATE_PROMPT | _model

    response = chain.invoke(
        {
            "topic": request["topic"],
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
        },
        config={"tags": _TAGS},
    )

    parsed = _parse_json_response(response.content)

    return {
        "draft_content": parsed.get("content", response.content),
        "generation_confidence": float(parsed.get("confidence", 0.5)),
    }


@traceable(name="draft_review", run_type="chain", tags=_TAGS)
def draft_review_node(state: ContentModerationState) -> dict:
    """First human review checkpoint — pause for moderator decision.

    Calls interrupt() with the draft content and metadata. The graph
    pauses here until the moderator resumes with Command(resume=...).

    The moderator's decision dict must have:
    - action: "approve" | "edit" | "reject"
    - feedback: str (optional, used for reject)
    - edited_content: str (optional, used for edit)

    Human-in-the-loop concept: interrupt() pauses the graph. When resumed,
    the return value of interrupt() is the moderator's decision.
    """
    decision = interrupt({
        "content": state["draft_content"],
        "confidence": state["generation_confidence"],
        "revision_count": state["revision_count"],
        "prompt": "Review this draft. Approve, edit, or reject with feedback.",
    })

    # If moderator edited, update draft_content so polish reads from it
    updates = {"draft_decision": decision}
    if decision.get("action") == "edit" and decision.get("edited_content"):
        updates["draft_content"] = decision["edited_content"]

    return updates


def route_after_draft_review(
    state: ContentModerationState,
) -> Literal["polish", "revise", "__end__"]:
    """Route based on the moderator's draft review decision.

    - approve/edit → polish (content is ready for cleanup)
    - reject + revisions remaining → revise (loop back)
    - reject + max revisions reached → END (give up)
    """
    action = state["draft_decision"].get("action", "reject")

    if action in ("approve", "edit"):
        return "polish"

    # Reject — check revision budget
    if state["revision_count"] >= MAX_REVISIONS:
        return "__end__"

    return "revise"


@traceable(name="revise", run_type="chain", tags=_TAGS)
def revise_node(state: ContentModerationState) -> dict:
    """Revise content based on moderator feedback.

    Takes the previous draft and the moderator's feedback, generates
    a new version. Increments revision_count.

    LangGraph concept: revision loop — this node feeds back into
    draft_review, creating a cycle in the graph.
    """
    request = state["content_request"]
    feedback = state["draft_decision"].get("feedback", "Please improve the content.")

    chain = REVISE_PROMPT | _model

    response = chain.invoke(
        {
            "topic": request["topic"],
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
            "previous_draft": state["draft_content"],
            "feedback": feedback,
        },
        config={"tags": _TAGS},
    )

    parsed = _parse_json_response(response.content)

    return {
        "draft_content": parsed.get("content", response.content),
        "generation_confidence": float(parsed.get("confidence", 0.5)),
        "revision_count": state["revision_count"] + 1,
    }


@traceable(name="polish", run_type="chain", tags=_TAGS)
def polish_node(state: ContentModerationState) -> dict:
    """Final formatting and cleanup pass on approved/edited content.

    Produces the polished version that goes to final review.
    """
    request = state["content_request"]
    chain = POLISH_PROMPT | _model

    response = chain.invoke(
        {
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
            "content": state["draft_content"],
        },
        config={"tags": _TAGS},
    )

    return {"polished_content": response.content}


@traceable(name="final_review", run_type="chain", tags=_TAGS)
def final_review_node(state: ContentModerationState) -> dict:
    """Second human review checkpoint — final publication gate.

    Calls interrupt() with the polished content. Moderator sees the
    final version and decides: approve or reject. No revision loop
    at this stage — reject means the content is killed.

    Human-in-the-loop concept: a simpler interrupt point (binary decision)
    compared to draft_review's three-way decision.
    """
    decision = interrupt({
        "content": state["polished_content"],
        "prompt": "Final review. Approve for publication or reject.",
    })

    return {"final_decision": decision}


def route_after_final_review(
    state: ContentModerationState,
) -> Literal["publish", "__end__"]:
    """Route based on the moderator's final review decision.

    - approve → publish
    - reject → END (content killed)
    """
    action = state["final_decision"].get("action", "reject")

    if action == "approve":
        return "publish"

    return "__end__"


@traceable(name="publish", run_type="chain", tags=_TAGS)
def publish_node(state: ContentModerationState) -> dict:
    """Mark content as published and record metadata.

    This is the terminal node for successfully reviewed content.
    In a real system, this would write to a database or CMS.
    """
    return {
        "published": True,
        "publish_metadata": {
            "moderator_notes": state.get("final_decision", {}).get("feedback", ""),
            "review_rounds": state.get("revision_count", 0),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_nodes.py -v`
Expected: Routing tests (7) PASS. generate_node integration test (1) PASS if API key is set.

- [ ] **Step 5: Commit**

```bash
git add projects/05-content-moderation-qa/nodes.py projects/05-content-moderation-qa/tests/test_nodes.py
git commit -m "feat(p5): add node functions with interrupt, routing, and revision logic"
```

---

### Task 6: Graph Assembly and Interrupt/Resume Tests (TDD)

**Files:**
- Create: `projects/05-content-moderation-qa/tests/conftest.py`
- Create: `projects/05-content-moderation-qa/tests/test_graph.py`
- Create: `projects/05-content-moderation-qa/graph.py`

- [ ] **Step 1: Create shared fixtures**

```python
# tests/conftest.py
"""Shared pytest fixtures for the Content Moderation tests."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph


@pytest.fixture
def graph_with_memory():
    """Compiled graph with InMemorySaver for interrupt/resume tests."""
    checkpointer = InMemorySaver()
    return build_graph(checkpointer=checkpointer)


@pytest.fixture
def sample_initial_state():
    """Initial state for a grammar explanation request."""
    return {
        "content_request": {
            "topic": "Present Perfect Tense",
            "content_type": "grammar_explanation",
            "difficulty": "B1",
        },
        "draft_content": "",
        "generation_confidence": 0.0,
        "draft_decision": {},
        "revision_count": 0,
        "polished_content": "",
        "final_decision": {},
        "published": False,
        "publish_metadata": None,
    }
```

- [ ] **Step 2: Write failing tests for graph**

```python
# tests/test_graph.py
"""End-to-end interrupt/resume workflow tests.

These tests exercise the full graph with real interrupt/resume cycles.
Each test hits the LLM (Anthropic API).
"""

import pytest
from langgraph.types import Command

from graph import build_graph


@pytest.mark.integration
class TestHappyPath:
    """Generate → approve draft → approve final → published."""

    def test_full_approve_flow(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "happy-1"}, "tags": ["p5-content-moderation"]}

        # Step 1: invoke — should generate content and hit draft_review interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result
        interrupt_payload = result["__interrupt__"][0].value
        assert "content" in interrupt_payload
        assert len(interrupt_payload["content"]) > 50

        # Step 2: approve draft — should hit final_review interrupt
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert "__interrupt__" in result
        final_payload = result["__interrupt__"][0].value
        assert "content" in final_payload

        # Step 3: approve final — should publish
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True
        assert result.get("publish_metadata") is not None


@pytest.mark.integration
class TestEditPath:
    """Generate → edit draft → approve final → published."""

    def test_edit_replaces_content(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "edit-1"}, "tags": ["p5-content-moderation"]}

        # Generate and hit first interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Edit the draft
        edited_text = "This is the moderator's edited version of the content."
        result = graph_with_memory.invoke(
            Command(resume={
                "action": "edit",
                "edited_content": edited_text,
            }),
            config=config,
        )
        # Should hit final review with polished version of the edited content
        assert "__interrupt__" in result

        # Approve final
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True


@pytest.mark.integration
class TestRejectAndRevise:
    """Generate → reject → revise → approve → approve → published."""

    def test_reject_loops_back_to_revision(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "reject-1"}, "tags": ["p5-content-moderation"]}

        # Generate and hit first interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Reject with feedback — should revise and hit draft_review again
        result = graph_with_memory.invoke(
            Command(resume={
                "action": "reject",
                "feedback": "Too advanced for B1 level. Simplify the language.",
            }),
            config=config,
        )
        # Should hit draft_review interrupt again with revised content
        assert "__interrupt__" in result
        revised_payload = result["__interrupt__"][0].value
        assert revised_payload["revision_count"] == 1

        # Approve the revision
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        # Should hit final review
        assert "__interrupt__" in result

        # Approve final
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True


@pytest.mark.integration
class TestMaxRevisions:
    """Reject twice → graph ends without publishing."""

    def test_max_revisions_ends_graph(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "maxrev-1"}, "tags": ["p5-content-moderation"]}

        # Generate
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Reject #1
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Needs work"}),
            config=config,
        )
        assert "__interrupt__" in result  # revision 1, back to draft_review

        # Reject #2
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Still not good"}),
            config=config,
        )
        assert "__interrupt__" in result  # revision 2, back to draft_review

        # Reject #3 — should hit max revisions and end
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Giving up"}),
            config=config,
        )
        # Graph should have ended — no interrupt, not published
        assert result.get("published", False) is False


@pytest.mark.integration
class TestFinalRejection:
    """Approve draft → reject final → not published."""

    def test_final_rejection_kills_content(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "finalrej-1"}, "tags": ["p5-content-moderation"]}

        # Generate
        result = graph_with_memory.invoke(sample_initial_state, config=config)

        # Approve draft
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert "__interrupt__" in result  # final review

        # Reject final
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Not publication ready"}),
            config=config,
        )
        assert result.get("published", False) is False


class TestGraphStructure:
    """Verify graph wiring without LLM calls."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        for expected in ["generate", "draft_review", "revise", "polish", "final_review", "publish"]:
            assert expected in node_names, f"Missing node: {expected}"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 4: Implement graph.py**

```python
# graph.py
"""StateGraph assembly for the Content Moderation & QA System.

Wires together 6 nodes with conditional routing and a revision loop.
Two interrupt points pause for human moderator review.

LangGraph concepts demonstrated:
- StateGraph with interrupt() for human-in-the-loop
- Conditional edges for approve/edit/reject routing
- Revision loop (revise → draft_review cycle)
- RetryPolicy on LLM nodes for transient error handling
- Checkpointer injection (required for interrupts)

4-tier error handling:
- Tier 1: RetryPolicy on generate, revise, polish nodes
- Tier 3: interrupt() for user-fixable issues (moderator review)
- Tier 4: unexpected errors bubble up
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from models import ContentModerationState
from nodes import (
    generate_node,
    draft_review_node,
    revise_node,
    polish_node,
    final_review_node,
    publish_node,
    route_after_draft_review,
    route_after_final_review,
)

# RetryPolicy for LLM nodes — handles transient API errors
_llm_retry = RetryPolicy(max_attempts=3)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the content moderation StateGraph.

    Args:
        checkpointer: Required for interrupt/resume. Defaults to InMemorySaver
            if None is provided, since interrupts need a checkpointer.

    Returns:
        Compiled LangGraph graph ready for .invoke().
    """
    # Interrupts require a checkpointer — default to InMemorySaver
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = (
        StateGraph(ContentModerationState)
        # -- Nodes --
        # LLM nodes get RetryPolicy for transient error handling (Tier 1)
        .add_node("generate", generate_node, retry=_llm_retry)
        .add_node("draft_review", draft_review_node)
        .add_node("revise", revise_node, retry=_llm_retry)
        .add_node("polish", polish_node, retry=_llm_retry)
        .add_node("final_review", final_review_node)
        .add_node("publish", publish_node)
        # -- Edges --
        # Linear flow: generate → draft_review
        .add_edge(START, "generate")
        .add_edge("generate", "draft_review")
        # Conditional: draft_review → polish | revise | END
        .add_conditional_edges(
            "draft_review",
            route_after_draft_review,
            ["polish", "revise", "__end__"],
        )
        # Revision loop: revise → back to draft_review
        .add_edge("revise", "draft_review")
        # Linear: polish → final_review
        .add_edge("polish", "final_review")
        # Conditional: final_review → publish | END
        .add_conditional_edges(
            "final_review",
            route_after_final_review,
            ["publish", "__end__"],
        )
        # Terminal
        .add_edge("publish", END)
        # Compile with checkpointer (required for interrupts)
        .compile(checkpointer=checkpointer)
    )

    return graph
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_graph.py -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add projects/05-content-moderation-qa/graph.py projects/05-content-moderation-qa/tests/conftest.py projects/05-content-moderation-qa/tests/test_graph.py
git commit -m "feat(p5): add graph assembly with interrupt/resume and end-to-end tests"
```

---

### Task 7: LangSmith Evaluation Pipeline

**Files:**
- Create: `projects/05-content-moderation-qa/tests/test_evaluation.py`
- Create: `projects/05-content-moderation-qa/evaluation.py`

- [ ] **Step 1: Write tests for evaluator functions**

```python
# tests/test_evaluation.py
"""Tests for custom evaluator functions.

These test the evaluator logic only — no LangSmith API calls.
Each evaluator takes a run-like dict and example-like dict and returns a score dict.
"""

import pytest


class TestTopicRelevanceEvaluator:
    def test_returns_score_dict(self):
        from evaluation import topic_relevance_evaluator

        # Simulate a run output and example
        run = type("Run", (), {"outputs": {"content": "The present perfect tense is used for..."}})()
        example = type("Example", (), {"inputs": {"topic": "Present Perfect Tense"}})()

        result = topic_relevance_evaluator(run, example)
        assert "key" in result
        assert result["key"] == "topic_relevance"
        assert "score" in result
        assert isinstance(result["score"], (int, float))


class TestDifficultyMatchEvaluator:
    def test_returns_score_dict(self):
        from evaluation import difficulty_match_evaluator

        run = type("Run", (), {"outputs": {"content": "Simple words. Easy grammar."}})()
        example = type("Example", (), {"inputs": {"difficulty": "A1"}})()

        result = difficulty_match_evaluator(run, example)
        assert result["key"] == "difficulty_match"
        assert isinstance(result["score"], (int, float))


class TestContentQualityEvaluator:
    def test_returns_score_dict(self):
        from evaluation import content_quality_evaluator

        run = type("Run", (), {"outputs": {"content": "A well-written grammar explanation."}})()
        example = type("Example", (), {"inputs": {}})()

        result = content_quality_evaluator(run, example)
        assert result["key"] == "content_quality"
        assert isinstance(result["score"], (int, float))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement evaluation.py**

```python
# evaluation.py
"""LangSmith evaluation pipeline for the Content Moderation system.

This module provides:
1. Custom evaluator functions (LLM-as-judge) for content quality
2. A function to create an evaluation dataset in LangSmith
3. A function to run evaluations against the generate node

LangSmith concepts demonstrated:
- langsmith.Client for dataset/example management
- langsmith.evaluation.evaluate() for running evaluations
- Custom evaluator functions: (run, example) -> {"key": ..., "score": ...}
- LLM-as-judge pattern for subjective quality assessment

Usage:
    python evaluation.py              # Create dataset and run evaluation
    python evaluation.py --create     # Only create dataset
    python evaluation.py --evaluate   # Only run evaluation (dataset must exist)
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_anthropic import ChatAnthropic
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

from data.content_requests import SAMPLE_REQUESTS
from prompts import GENERATE_PROMPT

# -- LLM for evaluators --
_eval_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# -- Dataset name --
DATASET_NAME = "p5-content-generation-eval"


# -- Custom Evaluators --
# Each evaluator follows the LangSmith convention:
#   def evaluator(run, example) -> {"key": str, "score": float, "comment": str}

def topic_relevance_evaluator(run, example) -> dict:
    """Score whether generated content matches the requested topic.

    Uses LLM-as-judge to assess topic relevance on a 0-1 scale.
    """
    content = run.outputs.get("content", "")
    topic = example.inputs.get("topic", "")

    if not content or not topic:
        return {"key": "topic_relevance", "score": 0.0, "comment": "Missing content or topic"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, how relevant is this content to the topic '{topic}'?\n\n"
        f"Content:\n{content}\n\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = json.loads(response.content.strip())
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "topic_relevance", "score": score, "comment": f"Topic: {topic}"}


def difficulty_match_evaluator(run, example) -> dict:
    """Score whether content difficulty matches the requested CEFR level.

    Uses LLM-as-judge to assess difficulty appropriateness on a 0-1 scale.
    """
    content = run.outputs.get("content", "")
    difficulty = example.inputs.get("difficulty", "")

    if not content or not difficulty:
        return {"key": "difficulty_match", "score": 0.0, "comment": "Missing content or difficulty"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, how well does this content match CEFR level {difficulty}?\n\n"
        f"Content:\n{content}\n\n"
        f"Consider vocabulary complexity, grammar structures, and overall readability.\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = json.loads(response.content.strip())
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "difficulty_match", "score": score, "comment": f"Target: {difficulty}"}


def content_quality_evaluator(run, example) -> dict:
    """Score overall content quality (grammar, clarity, completeness).

    Uses LLM-as-judge to assess quality on a 0-1 scale.
    """
    content = run.outputs.get("content", "")

    if not content:
        return {"key": "content_quality", "score": 0.0, "comment": "No content"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, rate the overall quality of this lesson content.\n\n"
        f"Content:\n{content}\n\n"
        f"Consider: grammar correctness, clarity of explanation, completeness, "
        f"and usefulness as a learning resource.\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = json.loads(response.content.strip())
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "content_quality", "score": score}


# -- Target function for evaluation --
@traceable(name="generate_for_eval", tags=["p5-content-moderation"])
def generate_for_eval(inputs: dict) -> dict:
    """Wrapper around the generate prompt for LangSmith evaluation.

    Takes a content request dict and returns {"content": ...}.
    This bypasses the full graph (no HITL) for automated evaluation.
    """
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    chain = GENERATE_PROMPT | model

    response = chain.invoke(inputs)

    try:
        parsed = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        return {"content": parsed.get("content", response.content)}
    except json.JSONDecodeError:
        return {"content": response.content}


def create_dataset():
    """Create an evaluation dataset in LangSmith from sample requests."""
    client = Client()

    # Delete existing dataset if it exists (for idempotency)
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset: {DATASET_NAME}")
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Content generation evaluation dataset for P5",
    )

    for request in SAMPLE_REQUESTS:
        client.create_example(
            inputs=request,
            outputs={},  # No reference outputs — evaluators are LLM-as-judge
            dataset_id=dataset.id,
        )

    print(f"Created dataset '{DATASET_NAME}' with {len(SAMPLE_REQUESTS)} examples")
    return dataset


def run_evaluation():
    """Run the evaluation pipeline against the dataset."""
    results = evaluate(
        generate_for_eval,
        data=DATASET_NAME,
        evaluators=[
            topic_relevance_evaluator,
            difficulty_match_evaluator,
            content_quality_evaluator,
        ],
        experiment_prefix="p5-content-eval",
        metadata={"model": "claude-haiku-4-5-20251001", "version": "1.0"},
    )

    print(f"\nExperiment: {results.experiment_name}")
    print("-" * 60)
    for row in results:
        example_inputs = row["example"].inputs
        print(f"\nTopic: {example_inputs.get('topic', 'N/A')}")
        for eval_result in row["evaluation_results"]["results"]:
            print(f"  {eval_result.key}: {eval_result.score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangSmith evaluation for P5")
    parser.add_argument("--create", action="store_true", help="Only create dataset")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    if args.create:
        create_dataset()
    elif args.evaluate:
        run_evaluation()
    else:
        create_dataset()
        run_evaluation()
```

- [ ] **Step 4: Run evaluator tests**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/test_evaluation.py -v`
Expected: All 3 tests PASS (they hit the LLM for scoring but don't need LangSmith API).

- [ ] **Step 5: Commit**

```bash
git add projects/05-content-moderation-qa/evaluation.py projects/05-content-moderation-qa/tests/test_evaluation.py
git commit -m "feat(p5): add LangSmith evaluation pipeline with custom evaluators"
```

---

### Task 8: A/B Prompt Comparison Script

**Files:**
- Create: `projects/05-content-moderation-qa/ab_comparison.py`

- [ ] **Step 1: Implement A/B comparison script**

```python
# ab_comparison.py
"""A/B prompt comparison using LangSmith experiments.

Runs two prompt variants (structured vs creative) against the same
evaluation dataset and compares scores side-by-side.

LangSmith concepts demonstrated:
- Running multiple experiments against the same dataset
- Comparing experiment results programmatically
- Using experiment_prefix to organize runs

Usage:
    python ab_comparison.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_anthropic import ChatAnthropic
from langsmith import traceable
from langsmith.evaluation import evaluate

from evaluation import (
    DATASET_NAME,
    topic_relevance_evaluator,
    difficulty_match_evaluator,
    content_quality_evaluator,
    create_dataset,
)
from prompts import GENERATE_PROMPT_STRUCTURED, GENERATE_PROMPT_CREATIVE


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# -- Prompt Variant A: Structured --
@traceable(name="generate_structured", tags=["p5-content-moderation", "ab-test"])
def generate_structured(inputs: dict) -> dict:
    """Generate content using the structured prompt template."""
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    chain = GENERATE_PROMPT_STRUCTURED | model
    response = chain.invoke(inputs)
    try:
        parsed = _parse_json_response(response.content)
        return {"content": parsed.get("content", response.content)}
    except (json.JSONDecodeError, ValueError):
        return {"content": response.content}


# -- Prompt Variant B: Creative --
@traceable(name="generate_creative", tags=["p5-content-moderation", "ab-test"])
def generate_creative(inputs: dict) -> dict:
    """Generate content using the creative prompt template."""
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7)
    chain = GENERATE_PROMPT_CREATIVE | model
    response = chain.invoke(inputs)
    try:
        parsed = _parse_json_response(response.content)
        return {"content": parsed.get("content", response.content)}
    except (json.JSONDecodeError, ValueError):
        return {"content": response.content}


def run_ab_comparison():
    """Run both prompt variants and compare results."""
    evaluators = [
        topic_relevance_evaluator,
        difficulty_match_evaluator,
        content_quality_evaluator,
    ]

    # Ensure dataset exists
    try:
        from langsmith import Client
        Client().read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        print("Dataset not found. Creating it first...")
        create_dataset()

    print("=" * 60)
    print("A/B Prompt Comparison")
    print("=" * 60)

    # -- Run Variant A: Structured --
    print("\nRunning Variant A (Structured)...")
    results_a = evaluate(
        generate_structured,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="p5-ab-structured",
        metadata={"variant": "structured", "temperature": 0.3},
    )

    # -- Run Variant B: Creative --
    print("\nRunning Variant B (Creative)...")
    results_b = evaluate(
        generate_creative,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="p5-ab-creative",
        metadata={"variant": "creative", "temperature": 0.7},
    )

    # -- Compare Results --
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    def avg_scores(results):
        """Compute average score per evaluator across all examples."""
        totals = {}
        counts = {}
        for row in results:
            for eval_result in row["evaluation_results"]["results"]:
                key = eval_result.key
                totals[key] = totals.get(key, 0) + (eval_result.score or 0)
                counts[key] = counts.get(key, 0) + 1
        return {k: totals[k] / counts[k] for k in totals}

    scores_a = avg_scores(results_a)
    scores_b = avg_scores(results_b)

    print(f"\n{'Metric':<25} {'Structured':>12} {'Creative':>12} {'Winner':>12}")
    print("-" * 65)
    for metric in sorted(set(list(scores_a.keys()) + list(scores_b.keys()))):
        sa = scores_a.get(metric, 0)
        sb = scores_b.get(metric, 0)
        winner = "Structured" if sa > sb else "Creative" if sb > sa else "Tie"
        print(f"{metric:<25} {sa:>12.3f} {sb:>12.3f} {winner:>12}")

    print("\nView detailed results in LangSmith dashboard.")


if __name__ == "__main__":
    run_ab_comparison()
```

- [ ] **Step 2: Verify script parses**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -c "import ab_comparison; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add projects/05-content-moderation-qa/ab_comparison.py
git commit -m "feat(p5): add A/B prompt comparison script using LangSmith experiments"
```

---

### Task 9: Project README

**Files:**
- Create: `projects/05-content-moderation-qa/README.md`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add projects/05-content-moderation-qa/README.md
git commit -m "docs(p5): add project README"
```

---

### Task 10: Educational Documentation

**Files:**
- Create: `docs/05-content-moderation-qa.md`

- [ ] **Step 1: Write the educational doc**

Create `docs/05-content-moderation-qa.md` — a comprehensive educational document covering:

1. **Introduction** — Content moderation scenario, shift from P4's conversational agent to a pipeline with human review gates
2. **Human-in-the-Loop with interrupt()** — What `interrupt()` does (pauses graph, surfaces value), how `Command(resume=...)` resumes, the re-execution rule (code before interrupt re-runs on resume), idempotency requirements. Reference `draft_review_node` and `final_review_node` from `nodes.py`.
3. **Approval Workflows** — The three-way decision (approve/edit/reject) at draft review, binary decision at final review. How conditional edges route based on the moderator's response. The revision loop and max revision guard. Reference `route_after_draft_review` and `route_after_final_review`.
4. **4-Tier Error Handling** — Explain the full framework: RetryPolicy (transient), ToolNode errors (LLM-recoverable), interrupt (user-fixable), bubble up (unexpected). Show where each tier appears in the project. Reference `graph.py` for RetryPolicy usage.
5. **LangSmith Evaluation Deep Dive** — Creating datasets with `langsmith.Client`, custom evaluator functions (LLM-as-judge pattern), running evaluations with `evaluate()`, experiment tracking. Reference `evaluation.py`.
6. **A/B Prompt Comparison** — How to compare prompt variants using LangSmith experiments. Reference `ab_comparison.py`.

Reference specific code from the project. ~800-1200 words.

- [ ] **Step 2: Commit**

```bash
git add docs/05-content-moderation-qa.md
git commit -m "docs(p5): add educational guide for HITL, error handling, and LangSmith evaluation"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run the full test suite**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Verify the happy path workflow programmatically**

```python
cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -c "
from langgraph.types import Command
from graph import build_graph

graph = build_graph()
config = {'configurable': {'thread_id': 'verify-1'}, 'tags': ['p5-content-moderation']}

state = {
    'content_request': {'topic': 'Present Perfect', 'content_type': 'grammar_explanation', 'difficulty': 'B1'},
    'draft_content': '', 'generation_confidence': 0.0, 'draft_decision': {},
    'revision_count': 0, 'polished_content': '', 'final_decision': {},
    'published': False, 'publish_metadata': None,
}

r = graph.invoke(state, config=config)
print('Draft interrupt:', r['__interrupt__'][0].value['content'][:80])

r = graph.invoke(Command(resume={'action': 'approve'}), config=config)
print('Final interrupt:', r['__interrupt__'][0].value['content'][:80])

r = graph.invoke(Command(resume={'action': 'approve'}), config=config)
print('Published:', r.get('published'))
"
```

- [ ] **Step 3: Run all tests one final time**

Run: `cd projects/05-content-moderation-qa && source ../../.venv/bin/activate && python -m pytest tests/ -v`
Expected: All tests pass with no warnings.
