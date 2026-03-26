# Project 7: Intelligent Curriculum Engine

## What This Project Teaches

Project 7 introduces **DeepAgents** — Anthropic's framework for building autonomous,
file-system-aware agents that plan their own work, load domain skills on demand, and
manage hybrid memory. It pairs DeepAgents with a LangGraph outer workflow to show how
the two frameworks complement each other: LangGraph controls the pipeline and human
approval gates, while DeepAgents do the heavy autonomous content generation inside each step.

By the end of this project you will understand:

- How `create_deep_agent()` works and what the harness provides out of the box
- How SKILL.md files let agents load domain knowledge on demand
- How `CompositeBackend` routes files to ephemeral or persistent storage based on path
- How `TodoList` gives agents structured self-planning and visibility into their work
- How to nest DeepAgents inside LangGraph nodes for human-in-the-loop pipelines

---

## Key Concept: DeepAgents Architecture

DeepAgents wraps an LLM in a **harness** — a pre-built scaffold of built-in tools,
a virtual filesystem, and middleware — so the agent can act autonomously without
needing those capabilities wired manually.

### What the Harness Provides

When you call `create_deep_agent()`, the resulting agent automatically has access to:

| Built-in Tool | Purpose |
|---------------|---------|
| `read_file` / `write_file` | Read and write to a virtual filesystem |
| `list_files` | Discover what files exist in its workspace |
| `write_todos` | Create a structured todo list (self-planning) |
| `read_skill` | Load a SKILL.md file to gain domain knowledge |

These tools are injected by the harness, not by you. You just define what the agent
*does* via `system_prompt` and what domain knowledge it can access via `skills`.

### create_deep_agent()

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend

agent = create_deep_agent(
    name="curriculum-planner",
    model=ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3),
    system_prompt=PLANNER_PROMPT,
    skills=[str(SKILLS_DIR) + "/"],   # directory of SKILL.md files
    backend=create_composite_backend(),
    store=_store,                      # LangGraph Store for persistent memory
)
```

The returned `agent` is a compiled LangGraph graph — you invoke it exactly like any
other LangGraph graph with `.invoke({"messages": [...]})`. The harness sets up all the
internal nodes and edges.

### Invoking an Agent Inside a Node

In `nodes.py`, every generation node creates an agent, invokes it, and extracts the
file it wrote:

```python
def plan_curriculum_node(state: CurriculumEngineState) -> dict:
    agent = create_planner_agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": instruction}],
         "files": {}},          # pre-load any files the agent should see
        config={"tags": ["p7-curriculum-engine"]},
    )

    # The agent wrote /work/plan.json — extract it from the result
    plan_json = result.get("files", {}).get("/work/plan.json")
    plan = json.loads(plan_json)
    return {"curriculum_plan": plan}
```

The virtual filesystem is the contract between the outer graph and the agent: the graph
pre-loads context files, the agent writes output files, and the graph reads them back.

---

## SKILL.md — On-Demand Knowledge

**SKILL.md** files are plain markdown documents with a YAML front-matter header. They
give an agent domain expertise it can load on demand using `read_skill`.

```markdown
---
name: curriculum-design
description: Curriculum design principles — Bloom's taxonomy, CEFR levels, scaffolding
---

# Curriculum Design Principles

## CEFR Level Descriptors

- **A1 (Beginner):** Simple, everyday phrases. Present tense only ...
- **B2 (Upper Intermediate):** Complex texts, abstract topics ...

## Learning Objective Design (Bloom's Taxonomy)

1. **Remember:** Define, list, identify key vocabulary
2. **Understand:** Explain, summarize, paraphrase concepts
...
```

### Why Skills Matter

The `description` field is shown to the agent in a skills index. The agent reads this
index and decides which skill to load — only fetching the content it needs. This keeps
the context window small until the agent actually needs the knowledge.

This project has four skills:

| Skill | What it teaches the agent |
|-------|--------------------------|
| `curriculum-design` | CEFR levels, Bloom's taxonomy, scaffolding principles |
| `lesson-template` | Exact markdown structure for lessons |
| `exercise-template` | Exact format for fill-in-the-blank, MCQ, short answer, matching |
| `assessment-template` | Format for assessments with rubric and answer key |

Rather than stuffing all this into `system_prompt`, skills are separate files. This
means you can update a template without modifying agent code, and multiple agents can
share the same skill.

### Skill Directory Convention

Skills are grouped by subdirectory, each containing one `SKILL.md`:

```
skills/
  curriculum-design/SKILL.md
  lesson-template/SKILL.md
  exercise-template/SKILL.md
  assessment-template/SKILL.md
```

Pass the parent directory to `create_deep_agent(skills=[str(SKILLS_DIR) + "/"])` and
the harness registers all subdirectories automatically.

---

## CompositeBackend — Hybrid Memory

Agents need two kinds of storage: **ephemeral** working files that only matter for the
current task, and **persistent** files that should survive across sessions (a catalog,
preferences, history).

`CompositeBackend` handles both in a single backend by routing paths to different storage
strategies:

```python
def create_composite_backend():
    def factory(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),      # ephemeral: lives in LangGraph state
            routes={
                "/catalog/":     StoreBackend(runtime),   # persistent: LangGraph Store
                "/preferences/": StoreBackend(runtime),   # persistent: LangGraph Store
            },
        )
    return factory
```

### StateBackend vs StoreBackend

| Backend | Storage | Lifetime | Use for |
|---------|---------|----------|---------|
| `StateBackend` | LangGraph thread state | Current thread only | Drafts, intermediate work |
| `StoreBackend` | LangGraph Store (cross-thread) | Persistent | Catalogs, user preferences |

The agent doesn't need to know which backend it's using — it just reads and writes paths.
`CompositeBackend` inspects the path prefix and routes to the right backend automatically.

In this project:
- `/work/plan.json`, `/work/lesson.md`, `/work/exercises.md`, `/work/assessment.md` →
  `StateBackend` (ephemeral drafts for the current generation run)
- `/catalog/` → `StoreBackend` (could store previously approved modules across sessions)
- `/preferences/` → `StoreBackend` (could store learner or instructor preferences)

The factory pattern (`def factory(runtime): return CompositeBackend(...)`) is required
because the `runtime` object — which links the backend to the current thread's
checkpoint — is only available at agent invocation time, not at construction time.

---

## Architecture Pattern: LangGraph + DeepAgents

The central architectural decision in this project is to use **two frameworks together**
rather than one exclusively.

```
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph Outer Graph (workflow + HITL)                        │
│                                                                 │
│  START → plan_curriculum ─────────────────────────────────────┐ │
│              │                                                 │ │
│              ▼                                                 │ │
│          review_plan ──[revise]──────────────────────────────┘ │
│              │ [approve]                                        │
│              ▼                                                  │
│         generate_lesson ◄─────────────────────────────────────┐│
│              │                                                 ││
│              ▼                                                 ││
│          review_lesson ──[revise]──────────────────────────────┘│
│              │ [approve/reject]                                  │
│              ▼                                                   │
│       generate_exercises ◄────────────────────────────────────┐ │
│              │                                                 │ │
│              ▼                                                 │ │
│       review_exercises ──[revise]──────────────────────────────┘ │
│              │ [approve/reject]                                   │
│              ▼                                                    │
│      generate_assessment ◄────────────────────────────────────┐  │
│              │                                                 │  │
│              ▼                                                 │  │
│      review_assessment ──[revise]──────────────────────────────┘  │
│              │ [approve/reject]                                    │
│              ▼                                                     │
│       assemble_module → END                                        │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Each "generate_*" node creates and invokes a DeepAgent:      │ │
│  │   plan_curriculum_node → create_planner_agent().invoke(...)  │ │
│  │   generate_lesson_node → create_lesson_agent().invoke(...)   │ │
│  │   generate_exercises_node → create_exercise_agent().invoke() │ │
│  │   generate_assessment_node → create_assessment_agent().inv() │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### Why Hybrid?

| Concern | Handled by |
|---------|-----------|
| Workflow sequencing | LangGraph |
| Human approval gates | LangGraph (`interrupt()`) |
| Revision loops | LangGraph (conditional edges) |
| Content generation autonomy | DeepAgents |
| Domain knowledge loading | DeepAgents (SKILL.md) |
| Working file management | DeepAgents (virtual filesystem) |
| Agent self-planning | DeepAgents (TodoList) |

LangGraph is excellent at deterministic pipelines with human oversight. DeepAgents is
excellent at open-ended content generation where the agent needs to decide *how* to do
the work. Together they're more capable than either alone.

---

## HITL Pattern: Review at Each Stage

Each of the four review nodes uses `interrupt()` to pause the graph and surface the
generated artifact to a human moderator:

```python
@traceable(name="review_plan", run_type="chain", tags=_TAGS)
def review_plan_node(state: CurriculumEngineState) -> dict:
    decision = interrupt({
        "step": "review_plan",
        "plan": state["curriculum_plan"],
        "prompt": "Review the curriculum plan. Approve or request revisions with feedback.",
    })

    feedback = decision.get("feedback", "") if decision.get("action") == "revise" else ""
    return {"plan_feedback": feedback}
```

The dict passed to `interrupt()` is the **interrupt payload** — it contains everything
the moderator needs to make a decision. The graph suspends here and waits.

### Moderator Decisions

The moderator resumes the graph by calling `.invoke(Command(resume=decision))` where
`decision` is a dict:

| Action | What the moderator sends | What happens next |
|--------|------------------------|-------------------|
| `{"action": "approve"}` | No feedback | Moves to next stage |
| `{"action": "revise", "feedback": "..."}` | Feedback text | Re-runs the generator with feedback |
| `{"action": "reject"}` | No feedback | Moves forward, artifact set to `None` |

### Routing After Review

Conditional edges read the feedback fields in state to decide direction:

```python
def route_after_plan_review(
    state: CurriculumEngineState,
) -> Literal["generate_lesson", "plan_curriculum"]:
    if state.get("plan_feedback"):       # non-empty feedback = revise
        return "plan_curriculum"
    return "generate_lesson"             # empty feedback = approved
```

The pattern is consistent: non-empty `*_feedback` means revision requested,
empty (or `None`) means proceed. This is a clean encoding because it lets the
state field serve double duty as both the routing signal and the feedback text
passed to the next generation.

---

## TodoList — Agent Self-Planning

DeepAgents includes a `write_todos` built-in tool that lets agents create a structured
task list before they start working. This is not just for humans to observe — it
genuinely helps the agent stay on track for multi-step tasks.

Every system prompt in this project instructs the agent to use it:

```python
PLANNER_PROMPT = (
    "...Steps:\n"
    "1. Use write_todos to plan your work\n"
    "2. Read the curriculum-design skill for design principles\n"
    "3. Create a curriculum plan with: ...\n"
    "4. Write the plan as JSON to /work/plan.json\n"
    "..."
)
```

The agent calls `write_todos` with a list of tasks, then ticks them off as it proceeds.
The todos end up in the agent's final state under `result["todos"]`:

```python
def _extract_todos(result: dict) -> list[dict]:
    """Extract the agent's TodoList from its final state."""
    return result.get("todos", [])
```

In `nodes.py`, todos are stored in the `GeneratedArtifact` alongside the content:

```python
return {
    "lesson": {
        "content": lesson_content,
        "artifact_type": "lesson",
        "agent_todos": _extract_todos(result),   # planning trail preserved in state
    },
    ...
}
```

### Why This Matters

Persisting todos in the outer graph state gives you:
- **Observability:** You can see how the agent broke down its task
- **Debugging:** If the agent produced bad output, the todo trail shows where it went wrong
- **UI integration:** A Streamlit page can render the planning steps alongside the artifact

---

## LangSmith Integration

Every node in `nodes.py` is decorated with `@traceable` and tagged with the project name:

```python
_TAGS = ["p7-curriculum-engine"]

@traceable(name="plan_curriculum", run_type="chain", tags=_TAGS)
def plan_curriculum_node(state: CurriculumEngineState) -> dict:
    ...
```

### Nested Traces

When `plan_curriculum_node` calls `agent.invoke(config={"tags": _TAGS})`, the DeepAgent's
internal trace is nested under the outer node's trace in LangSmith. You see:

```
plan_curriculum (chain)
  └── curriculum-planner (deep_agent)
        ├── write_todos (tool)
        ├── read_skill: curriculum-design (tool)
        ├── write_file: /work/plan.json (tool)
        └── [final output]
```

This nested visibility is essential for autonomous agents. Without it, a DeepAgent is a
black box — you see the input and output but nothing in between. With nested traces, you
can see every tool call the agent made, the order it followed its todos, and exactly what
it wrote to disk.

### Filtering in LangSmith

All traces from this project are tagged `p7-curriculum-engine`. In the LangSmith UI,
filter by tag to see only this project's runs:

```
Tags: p7-curriculum-engine
```

This is especially useful when running multiple projects in the same LangSmith workspace —
each project's traces stay cleanly separated.

---

## State Schema Design

The `CurriculumEngineState` TypedDict reflects a **pipeline** rather than a conversation.
There are no reducers — each field is written by exactly one node:

```python
class CurriculumEngineState(TypedDict):
    curriculum_request: dict      # Input — set at invocation
    curriculum_plan: dict | None  # Written by plan_curriculum_node
    plan_feedback: str            # Written by review_plan_node
    lesson: dict | None           # Written by generate_lesson_node
    lesson_feedback: str          # Written by review_lesson_node
    exercises: dict | None        # Written by generate_exercises_node
    exercises_feedback: str       # Written by review_exercises_node
    assessment: dict | None       # Written by generate_assessment_node
    assessment_feedback: str      # Written by review_assessment_node
    assembled_module: str | None  # Written by assemble_module_node
    current_step: str             # Written by each node for UI progress
```

The `*_feedback` fields encode moderator decisions: empty string = approved, non-empty =
revise with this text. This double-duty encoding keeps the state schema lean while
preserving all the information needed for both routing and content regeneration.

---

## Key Takeaways

1. **DeepAgents = harness + virtual filesystem + skills + TodoList.** You define the
   agent's goal via `system_prompt` and domain knowledge via SKILL.md; the harness
   handles the rest.

2. **LangGraph and DeepAgents solve different problems.** LangGraph owns deterministic
   workflow, sequencing, and human approval. DeepAgents own autonomous, open-ended
   content generation within each step.

3. **SKILL.md decouples domain knowledge from code.** Update templates or add new
   curriculum guidelines without touching any Python.

4. **CompositeBackend makes file routing transparent.** The agent writes paths; the
   backend decides whether to store them ephemerally or persistently.

5. **interrupt() with conditional routing creates clean revision loops.** Non-empty
   feedback sends the graph backward; empty feedback moves it forward.

6. **Nested LangSmith traces are essential for autonomous agents.** They turn black
   boxes into auditable, debuggable pipelines.
