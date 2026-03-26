# Project 7: Intelligent Curriculum Engine

An autonomous curriculum module generator that uses DeepAgents sub-agents
orchestrated by a LangGraph workflow to create lessons, exercises, and
assessments — with human approval at each stage.

## Concepts Introduced

- **DeepAgents:** `create_deep_agent()`, harness architecture, built-in tools
- **SKILL.md:** On-demand skill loading for domain knowledge and output templates
- **CompositeBackend:** Hybrid storage — ephemeral working files + persistent catalog
- **TodoList:** Agent self-planning via `write_todos`
- **Sub-agent orchestration:** Specialized agents for distinct content types

## Quick Start

```bash
# From repo root
source .venv/bin/activate
cd projects/07-curriculum-engine

# Run tests
python -m pytest tests/ -v

# Or use via the Streamlit app
cd ../../app
streamlit run app.py
```

## Architecture

```
LangGraph Outer Graph (workflow + HITL)
├── plan_curriculum → DeepAgent (planner)
│   └── review_plan (interrupt)
├── generate_lesson → DeepAgent (lesson-writer)
│   └── review_lesson (interrupt)
├── generate_exercises → DeepAgent (exercise-creator)
│   └── review_exercises (interrupt)
├── generate_assessment → DeepAgent (assessment-builder)
│   └── review_assessment (interrupt)
└── assemble_module → Final markdown output
```

## Key Files

| File | Purpose |
|------|---------|
| `agents.py` | DeepAgent factory functions with CompositeBackend |
| `nodes.py` | Graph nodes wrapping agent invocations + HITL reviews |
| `graph.py` | StateGraph assembly with conditional routing |
| `skills/` | SKILL.md files for curriculum design + output templates |
| `models.py` | Pydantic models and LangGraph state schema |
