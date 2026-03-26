# Project 7: Intelligent Curriculum Engine — Design Specification

**Date:** 2026-03-26
**Department:** Content
**Difficulty:** Advanced

## Overview

The Intelligent Curriculum Engine autonomously creates entire curriculum modules for LinguaFlow. Given a topic and target level, it plans the curriculum structure, generates lessons, exercises, and assessments through specialized DeepAgents sub-agents, and pauses for human approval at each stage.

This project introduces the DeepAgents framework (`create_deep_agent()`, SKILL.md, persistent memory, sub-agent orchestration) while reusing LangGraph for workflow structure and HITL patterns from earlier projects.

## Architecture

**Hybrid approach: LangGraph outer graph + DeepAgents sub-agents.**

- **LangGraph** owns: workflow sequencing, state management, HITL interrupts (`interrupt()`), checkpointing
- **DeepAgents** own: autonomous content generation within each step, task planning (TodoList via `write_todos`), on-demand skill loading, persistent memory

### Workflow

```
[Input Form] → plan_curriculum → [HITL: approve plan]
  → generate_lesson → [HITL: approve lesson]
  → generate_exercises → [HITL: approve exercises]
  → generate_assessment → [HITL: approve assessment]
  → assemble_module → [Done]
```

Each content-generation node internally creates and invokes a `create_deep_agent()` with appropriate skills and tools. The outer graph handles state flow and HITL via `interrupt()` / `Command(resume=...)`.

### State Schema

The LangGraph state carries the full curriculum through the graph:

- `topic`, `level`, `preferences` — user input
- `curriculum_plan` — structured plan from the planning step
- `lesson` — generated lesson content
- `exercises` — generated exercise set
- `assessment` — generated assessment with rubric
- `assembled_module` — final assembled output
- `moderator_feedback` — feedback from HITL reviews at each checkpoint

## DeepAgents Configuration

### Specialized Agents

| Agent | Skills Loaded | Purpose |
|-------|--------------|---------|
| `curriculum-planner` | `curriculum-design` | Creates curriculum structure using `write_todos`, returns structured plan |
| `lesson-writer` | `curriculum-design` + `lesson-template` | Generates structured lesson with objectives, content, examples, takeaways |
| `exercise-creator` | `curriculum-design` + `exercise-template` | Creates fill-in-the-blank, multiple choice, short answer, matching exercises |
| `assessment-builder` | `curriculum-design` + `assessment-template` | Builds graded assessment with rubric, scoring criteria, answer key |

All agents use `claude-haiku-4-5-20251001`.

### Memory Layout (CompositeBackend)

| Path | Backend | Persistence | Purpose |
|------|---------|-------------|---------|
| `/work/` | `StateBackend` | Ephemeral (thread-scoped) | Drafts, scratch notes, working files |
| `/catalog/` | `StoreBackend` | Persistent (cross-session) | Catalog of created curriculum modules |
| `/preferences/` | `StoreBackend` | Persistent (cross-session) | User's teaching style, level focus, topic patterns |

On startup, the engine reads `/catalog/` to avoid duplicating existing content and `/preferences/` to tailor output. After successful assembly, it writes the new module to `/catalog/` and updates `/preferences/`.

## SKILL.md Files

### Domain Knowledge Skill

**`skills/curriculum-design/SKILL.md`** — Curriculum design principles:
- Learning objective taxonomy (Bloom's)
- Scaffolding techniques
- CEFR level descriptors (A1-C2)
- Content sequencing and progressive difficulty
- How exercises should reinforce lesson objectives

### Output Format Skills

**`skills/lesson-template/SKILL.md`** — Lesson markdown structure:
- Title, level, estimated duration
- Learning objectives (3-5 bullet points)
- Warm-up activity
- Core content sections (2-3, each with explanation + examples)
- Key takeaways
- Vocabulary list

**`skills/exercise-template/SKILL.md`** — Exercise format:
- Exercise set title, difficulty level
- 4 exercise types: fill-in-the-blank, multiple choice, short answer, matching
- Answer key section
- Difficulty progression within each type

**`skills/assessment-template/SKILL.md`** — Assessment format:
- Assessment title, level, time limit
- Sections: reading comprehension, grammar, writing prompt
- Scoring rubric with point values
- Grade boundaries
- Answer key with explanations

All SKILL.md files use standard frontmatter (`name`, `description`) for discovery via `SkillsMiddleware`.

## Streamlit UI

### Page: `app/pages/p7_curriculum.py`

**Left sidebar (form):**
- Topic input (text field, e.g. "Business English for meetings")
- Level selector (dropdown: A1-C2)
- Optional preferences: teaching style (conversational/formal/interactive), focus areas
- "Generate Module" button
- "Previous Modules" expander showing catalog entries from persistent memory

**Main area (live progress view):**
- **Step tracker** — horizontal progress indicator showing: Plan > Lesson > Exercises > Assessment > Done, with current step highlighted
- **Agent activity panel** — shows the current deep agent's TodoList items and their status (pending/in_progress/completed) as they update
- **Content preview** — renders generated artifact as markdown
- **HITL review panel** — appears at each checkpoint:
  - Generated content displayed for review
  - Three actions: Approve, Request Revision (with feedback text area), Reject
  - Revision: feedback passed back to agent for regeneration
  - Reject: step skipped, marked as rejected in final module

### Adapter: `app/adapters/curriculum_engine.py`

Thin wrapper following existing adapter patterns — imports the graph, manages thread IDs, translates between Streamlit session state and graph state.

### Session Persistence

Thread ID tied to Streamlit session. Refreshing resumes at the last checkpoint via LangGraph checkpointer.

## Testing

- `test_models.py` — Pydantic model and state schema validation
- `test_agents.py` — Each deep agent produces valid output given mock inputs; verify skill loading and TodoList usage
- `test_graph.py` — Outer graph routing: HITL interrupts fire at correct nodes, state flows correctly through pipeline
- `test_memory.py` — CompositeBackend routing: `/work/` writes are ephemeral, `/catalog/` and `/preferences/` writes persist across threads

## LangSmith Integration

- All traces tagged with `project:07-curriculum-engine`
- Nested traces: each deep agent invocation appears under the parent graph trace, showing TodoList planning and generation
- Monitoring focus: latency per step, cost per module generation, revision rates at HITL checkpoints
- No custom evaluators (evaluation was P5's deep-dive); this project focuses on observing autonomous agent behavior through tracing

## Project Structure

```
projects/07-curriculum-engine/
├── README.md
├── requirements.txt
├── models.py              # Pydantic models + LangGraph state schema
├── agents.py              # create_deep_agent() configs for all 4 agents
├── nodes.py               # Graph node functions (plan, generate_*, assemble)
├── graph.py               # StateGraph definition, edges, HITL interrupts
├── prompts.py             # System prompts for each agent
├── skills/
│   ├── curriculum-design/
│   │   └── SKILL.md       # Domain knowledge (Bloom's, CEFR, scaffolding)
│   ├── lesson-template/
│   │   └── SKILL.md       # Lesson markdown structure
│   ├── exercise-template/
│   │   └── SKILL.md       # Exercise format + types
│   └── assessment-template/
│       └── SKILL.md       # Assessment rubric + format
├── data/
│   └── sample_requests.py # Sample curriculum generation requests
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_memory.py
app/
├── adapters/
│   └── curriculum_engine.py
└── pages/
    └── p7_curriculum.py
```
