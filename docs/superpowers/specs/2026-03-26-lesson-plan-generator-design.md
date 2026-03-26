# Lesson Plan Generator — Design Spec

**Project:** 02-lesson-plan-generator
**Department:** Teaching (LinguaFlow)
**Difficulty:** Beginner → Intermediate
**Date:** 2026-03-26

---

## Overview

The Teaching department needs an agent that generates personalized lesson plans. Given a student's proficiency level, learning goals, and preferred topics, the agent researches appropriate material, drafts a lesson plan tailored to the lesson type, reviews it for quality (with a revision loop), and outputs a structured, finalized plan.

This is the first project to use **LangGraph StateGraph**, introducing stateful graphs, conditional routing, and graph cycles — building on Project 1's LangChain fundamentals.

## Concepts Introduced

- LangGraph StateGraph: defining graphs, state schemas, nodes, edges
- Conditional routing (branching logic based on state)
- Graph cycles (review loop with revision limit)
- Graph compilation and invocation
- LangSmith: tracing graph execution, viewing node-level traces

## Architecture: Two-Phase Design

### Phase 1: Intake Conversation

A conversational class (similar to Project 1's `ConversationHandler`) drives a short 3-4 turn dialogue with the student to gather:

- Name
- Proficiency level (A1-C2)
- Learning goals
- Preferred topics
- Lesson type (inferred from goals: conversation, grammar, or exam prep)

Once the LLM has enough information, it signals completion. The class exposes:
- `ask(user_message: str) -> str` — sends a message, returns the LLM's response
- `is_complete() -> bool` — checks if the LLM has gathered all required info
- `get_profile() -> StudentProfile` — extracts a structured profile from the conversation

### Phase 2: LangGraph StateGraph

The compiled graph receives a `StudentProfile` and produces a `LessonPlan`.

#### Graph Topology

```
START → research → route_by_type → draft_conversation ─┐
                        │              draft_grammar ───┤──→ review ──→ finalize → END
                        └──→          draft_exam_prep ──┘       ↑
                                                                │
                                                          (loop back if
                                                           not approved &
                                                           revision_count < 2)
```

#### Nodes

1. **`research`** — Takes the `StudentProfile`, asks the LLM to suggest relevant materials, activities, and themes for the student's level, goals, and topics. Writes `research_notes` to state.

2. **`draft_conversation`** — Drafts a conversation-focused lesson plan emphasizing dialogue scenarios, role-plays, and speaking exercises. Writes `draft_plan` to state.

3. **`draft_grammar`** — Drafts a grammar-focused lesson plan emphasizing exercises, rules, error correction, and drills. Writes `draft_plan` to state.

4. **`draft_exam_prep`** — Drafts an exam-prep lesson plan emphasizing practice questions, test strategies, time management, and mock exercises. Writes `draft_plan` to state.

5. **`review`** — A separate LLM call that critiques the draft against quality criteria: appropriate for proficiency level, covers stated objectives, realistic timing, well-structured. Sets `is_approved: bool` and `review_feedback: str`. Increments `revision_count`.

6. **`finalize`** — Parses the approved draft (or best-effort draft after 2 revision cycles) into a structured `LessonPlan` Pydantic model.

#### Edges

- `START → research` — unconditional
- `research → route_by_type` — conditional edge reading `state["student_profile"].lesson_type`:
  - `"conversation"` → `draft_conversation`
  - `"grammar"` → `draft_grammar`
  - `"exam_prep"` → `draft_exam_prep`
- `draft_* → review` — all three drafting nodes converge to `review`
- `review → ?` — conditional edge:
  - If `is_approved` is `True` → `finalize`
  - If `is_approved` is `False` and `revision_count < 2` → back to the same drafting node (with `review_feedback` included in the prompt)
  - If `is_approved` is `False` and `revision_count >= 2` → `finalize` (best effort)
- `finalize → END`

## Data Models

### `StudentProfile` (Pydantic)

```
name: str
proficiency_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
learning_goals: list[str]
preferred_topics: list[str]
lesson_type: Literal["conversation", "grammar", "exam_prep"]
```

### `Activity` (Pydantic)

```
name: str
description: str
duration_minutes: int
materials: list[str]
```

### `LessonPlan` (Pydantic, structured final output)

```
title: str
level: str
lesson_type: str
objectives: list[str]
warm_up: str
main_activities: list[Activity]
wrap_up: str
homework: str
estimated_duration_minutes: int
```

### `LessonPlanState` (TypedDict, graph state)

```
student_profile: StudentProfile
research_notes: str
draft_plan: str
review_feedback: str
revision_count: int
is_approved: bool
final_plan: LessonPlan | None
```

Note: `final_plan` is `None` until the `finalize` node parses the draft into a structured `LessonPlan`.

## Module Structure

```
projects/02-lesson-plan-generator/
├── models.py              # StudentProfile, Activity, LessonPlan, LessonPlanState
├── intake.py              # IntakeConversation class
├── graph.py               # StateGraph definition, edges, routing, compilation
├── nodes.py               # Node functions (research, draft_*, review, finalize)
├── prompts.py             # All prompt templates
├── main.py                # CLI entry point
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── sample_profiles.py # Pre-built StudentProfile instances for testing
└── tests/
    ├── __init__.py
    ├── test_models.py     # Unit tests for Pydantic models
    ├── test_nodes.py      # Integration tests for individual node functions
    ├── test_graph.py      # Integration tests for full graph execution
    └── test_intake.py     # Integration tests for intake conversation
```

### Module Responsibilities

- **`models.py`** — All Pydantic models and the TypedDict state schema. No logic, just data definitions.
- **`intake.py`** — `IntakeConversation` class that manages the multi-turn intake dialogue and produces a `StudentProfile`.
- **`prompts.py`** — All `ChatPromptTemplate` definitions: research prompt, three drafting prompts (one per lesson type), review prompt, finalize prompt, intake system prompt.
- **`nodes.py`** — Node functions that each take `LessonPlanState` and return a partial state update. Each function wires up the appropriate prompt + model call. Decorated with `@traceable`.
- **`graph.py`** — Builds the `StateGraph`, adds nodes and edges (including conditional routing and the review loop), compiles the graph. Exports a `build_graph()` function.
- **`main.py`** — CLI that runs the intake conversation interactively, then compiles and invokes the graph. Streams node execution status to terminal. Prints the final `LessonPlan`.
- **`data/sample_profiles.py`** — Pre-built `StudentProfile` instances at various levels and lesson types for testing.

## Testing Strategy

### Unit Tests (`test_models.py`)
- Valid and invalid `StudentProfile` construction (literal constraints on level and lesson_type)
- Valid and invalid `Activity` and `LessonPlan` construction
- No API calls required

### Integration Tests (`test_nodes.py`)
- Call each node function with a pre-built state dict
- Verify `research` populates `research_notes`
- Verify each `draft_*` populates `draft_plan`
- Verify `review` sets `is_approved` and `review_feedback`
- Verify `finalize` produces a valid `LessonPlan`

### Integration Tests (`test_graph.py`)
- Compile the graph and invoke with sample profiles
- One test per lesson type: conversation, grammar, exam_prep
- Verify correct routing (the right drafting node was reached)
- Verify the final output is a valid `LessonPlan`
- Test the review loop: verify `revision_count` can exceed 0

### Integration Tests (`test_intake.py`)
- Simulate a multi-turn conversation
- Verify `is_complete()` transitions from False to True
- Verify `get_profile()` returns a valid `StudentProfile`

## LangSmith Integration

- `@traceable` decorator on every node function and intake methods
- LangGraph's built-in LangSmith integration provides automatic graph-level traces with child spans per node
- The educational doc will explain how to read node-level traces: which node ran, input/output at each step, execution timing

## CLI Behavior (`main.py`)

1. Greets the user and starts the intake conversation
2. Loops: displays LLM response, takes user input, calls `ask()`
3. When `is_complete()` returns True, extracts the `StudentProfile`
4. Displays the profile for confirmation
5. Compiles and invokes the graph
6. Streams status updates: "Researching materials...", "Drafting grammar lesson...", "Reviewing draft...", etc.
7. Prints the final `LessonPlan` in a readable format

## Dependencies

- `langchain-core`
- `langchain-anthropic`
- `langgraph`
- `langsmith`
- `python-dotenv`
- `pydantic`
- `pytest` (dev)
