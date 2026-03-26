# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational workspace for learning the LangGraph ecosystem from basics to advanced. Covers LangChain, LangGraph, LangSmith, and DeepAgents through 8 progressive, hands-on subprojects set in the domain of **LinguaFlow** — a modern English tutoring platform.

The learning path is defined in `resources/LEARNING_PATH.md` — always consult it to understand which project comes next, what concepts it introduces, and how it connects to previous work.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain-core langchain-anthropic langgraph langsmith deepagents python-dotenv
```

## Commands

```bash
python <script_name>.py
pytest
pytest <test_file>::<test_name> -v
```

## Project Structure

```
projects/           # Each subproject gets its own folder with its own venv, dependencies, and code
  01-grammar-correction-agent/
  02-lesson-plan-generator/
  ...
resources/          # Reference materials used by Claude during development (LEARNING_PATH.md, etc.)
docs/               # Educational documentation — one doc per project explaining concepts in depth
.claude/skills/     # LangChain ecosystem skills (invoked automatically)
```

Each subproject in `projects/` is **self-contained** — its own virtual environment, `requirements.txt`, `.env`, and code. This keeps dependencies isolated and lets each project stand on its own. When starting a new project, always create it under `projects/<NN>-<name>/`.

Subprojects are developed sequentially following the learning path. Each builds on the previous:
- Projects 1-2: LangChain fundamentals, first LangGraph graphs
- Projects 3-4: RAG, tool use, persistence
- Projects 5-6: Human-in-the-loop, multi-agent orchestration
- Projects 7-8: DeepAgents, autonomous operations capstone

## Conventions

- Store API keys in `.env` (never commit this file)
- Each graph/agent should be in its own module
- **Well-commented code**: every significant block should have comments explaining what it does and why — this is a learning repo, clarity over brevity
- **Progressive complexity**: each new subproject introduces new concepts while reinforcing previous ones

## Documentation Rules

- **`docs/` is mandatory**: for every project in the learning path, create a corresponding document in `docs/` (e.g., `docs/01-grammar-correction-agent.md`) that explains what the project is about, highlights the most important parts of the code, and explains every concept used — written in an educational manner so that reading the doc alone teaches the concepts
- **Subproject READMEs**: each project directory must also have a README.md with a quick overview and how to run it

## Educational Focus Rules

- **This is NOT a production project.** Do not overcomplicate with infrastructure, deployment, CI/CD, or production-grade architecture. The sole purpose is to highlight the application of LangGraph ecosystem libraries.
- **Clean separation of concerns.** All mock/supporting code (mock APIs, sample data, simple data stores) must live in clearly separated modules (e.g., `mock_apis/`, `data/`) so that LangGraph ecosystem code reads cleanly on its own. The reader must always be able to distinguish what is LangGraph code vs. supporting scaffolding.
- **Business context serves learning, not the other way around.** The LinguaFlow domain provides realistic motivation for each project, but infrastructure and APIs should be low-to-moderate effort. Build them in a way that demonstrates how LangGraph libraries interact with external systems without the business logic polluting the LangGraph implementation.

## Model Selection

- **Always use the cheapest Anthropic model** (`claude-haiku-4-5-20251001`) for all LLM calls in project code. This is a learning repo — we don't need the most capable model, and costs add up across projects. Use `ChatAnthropic(model="claude-haiku-4-5-20251001")` everywhere.

## Critical Rules

- **Always use Context7 MCP** (`/find-docs` or the context7 tools directly) to retrieve the latest documentation for LangChain, LangGraph, LangSmith, and DeepAgents before writing any code that uses these libraries. Do not rely on training data — always fetch current reference docs.

- **Always use the LangChain skills** available in `.claude/skills/`. Any work involving LangChain, LangGraph, LangSmith, or DeepAgents **must** invoke the relevant skill before writing code. Skills to use:
  - `langchain-skills:framework-selection` — invoke **first** at the start of any new subproject to determine the right framework layer
  - `langchain-skills:langchain-fundamentals` — agents, tools, middleware basics
  - `langchain-skills:langchain-middleware` — human-in-the-loop, custom middleware, structured output
  - `langchain-skills:langchain-rag` — RAG pipelines, document loaders, vector stores, embeddings
  - `langchain-skills:langchain-dependencies` — package versions, installation, dependency management
  - `langchain-skills:langgraph-fundamentals` — StateGraph, nodes, edges, Command, streaming
  - `langchain-skills:langgraph-persistence` — checkpointers, state persistence, time travel, Store
  - `langchain-skills:langgraph-human-in-the-loop` — interrupt/resume, approval workflows, error handling
  - `langchain-skills:deep-agents-core` — create_deep_agent(), harness architecture, SKILL.md
  - `langchain-skills:deep-agents-memory` — StateBackend, StoreBackend, FilesystemMiddleware
  - `langchain-skills:deep-agents-orchestration` — subagents, task planning, HITL interrupts

- **Always consult `resources/LEARNING_PATH.md`** before starting any project to understand the current project's goals, concepts, and how it fits the progression.

- **Always use `resources/`** to store and retrieve reference materials that support development (learning path, domain context, shared data schemas, etc.).
