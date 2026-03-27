# Project 08: LinguaFlow Autonomous Operations (Capstone)

A fully autonomous, multi-department operations orchestrator for the LinguaFlow English tutoring platform. This capstone project integrates every concept from Projects 1–7 into a single cohesive system that classifies incoming requests, assesses risk, dispatches parallel department agents, cascades autonomous follow-up tasks, and reports cumulative metrics.

---

## What It Demonstrates

This project is the culmination of the eight-project learning path. It doesn't introduce entirely new primitives — it shows how all previous concepts compose into a production-like autonomous system:

- **P1–P2 foundations**: LLM calls, structured output, LangChain basics
- **P3 RAG**: domain knowledge loaded into department agents via SKILL.md files
- **P4 tool use**: six departments each with their own specialised tool sets
- **P5 persistence**: SqliteSaver for the orchestrator, InMemorySaver for department agents
- **P6 parallel fan-out**: `Send()` to dispatch multiple departments concurrently
- **P7 DeepAgents**: `create_deep_agent()`, `CompositeBackend`, SKILL.md, `TodoList`
- **P8 capstone**: tiered HITL approval, autonomous task cascading, cross-run metrics

---

## LangGraph / DeepAgents Concepts Covered

| Concept | Where Used |
|---------|-----------|
| `StateGraph` with typed state | `graph.py` — `OrchestratorState` |
| `operator.add` reducer | `models.py` — `department_results` parallel merge |
| `Send` for parallel fan-out | `graph.py` — `fan_out_to_departments` |
| Conditional edges + `path_map` | `graph.py` — `route_from_risk`, `fan_out_to_departments` |
| `interrupt` / `Command` HITL | `nodes.py` — `approval_gate` |
| `Command(goto=...)` dynamic routing | `nodes.py` — `approval_gate`, `check_task_queue` |
| `ends=` for Command targets | `graph.py` — `approval_gate`, `check_task_queue` node registration |
| SqliteSaver persistence | orchestrator checkpointer in `app.py` |
| InMemorySaver for sub-agents | department agents in `departments.py` |
| `create_deep_agent()` | `departments.py` — all six department agents |
| SKILL.md domain knowledge | `skills/*/SKILL.md` — one per department |
| `CompositeBackend` | `departments.py` — ephemeral + persistent routing |
| `StateBackend` / `StoreBackend` | `departments.py` — working files vs. records |
| Autonomous task cascading | `nodes.py` — `check_task_queue` loop |
| Tiered approval rules | `risk.py` — `HIGH_RISK_ACTIONS` deterministic lookup |
| LangSmith `@traceable` + tags | all nodes tagged `p8-autonomous-operations` |
| LangSmith `evaluate()` | `evaluation.py` — three evaluators |

---

## The Six Departments

| Department | Key Tools | High-Risk Actions |
|-----------|----------|------------------|
| `student_onboarding` | `check_eligibility`, `run_placement_test`, `provision_account` | `create_study_plan` |
| `tutor_management` | `query_tutor_availability`, `schedule_session`, `get_tutor_profile` | `assign_tutor` |
| `content_pipeline` | `draft_lesson`, `review_content`, `publish_content` | `publish_content` |
| `quality_assurance` | `run_qa_check`, `flag_issue`, `get_qa_report` | `flag_issue` |
| `support` | `create_ticket`, `resolve_ticket`, `process_refund` | `process_refund` |
| `reporting` | `get_metrics`, `generate_report`, `export_dashboard` | *(none)* |

---

## Sample Requests and Routing Patterns

| Request | Departments Routed | Risk | Cascades? |
|---------|--------------------|------|-----------|
| `"Onboard 3 new students and assign tutors"` | `student_onboarding` → `tutor_management` | High (study plan) | Yes — onboarding triggers tutor assignment |
| `"Generate a B2 grammar lesson on conditionals"` | `content_pipeline` | Low | No |
| `"Process refund for student S042"` | `support` | High (refund) | No |
| `"Run QA check and generate weekly report"` | `quality_assurance`, `reporting` | High (flag issue) | No |
| `"How many students enrolled this month?"` | `reporting` | Low | No |

---

## Graph Topology

```
START
  |
  v
request_classifier          <- LLM classifies request, selects departments + action_type
  |
  v
risk_assessor               <- Pure logic: lookup HIGH_RISK_ACTIONS table
  |
  +-- (low risk) ---------> dispatch_departments (pass-through)
  |                               |
  +-- (high risk) ------> approval_gate [interrupt]
                           |              |
                       approved        rejected
                           |              |
               dispatch_departments  compose_output
                       |
                  [Send x N]         <- fan out, one branch per department
                       |
               department_executor (parallel) <- DeepAgent per department
                       |
               result_aggregator     <- collect follow_up_tasks, update completed_tasks
                       |
               check_task_queue
               /              \
      (tasks pending)       (queue empty)
             |                    |
    request_classifier        compose_output
    (next cascade cycle)           |
                           reporting_snapshot
                                   |
                                  END
```

The task queue loop (`check_task_queue → request_classifier`) is what makes the system truly autonomous: a single user request can trigger a multi-department chain without any human re-invocation.

---

## How to Run

### Prerequisites

```bash
# From the repository root
source .venv/bin/activate
cp .env.example .env  # Add ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

### Run Tests

```bash
cd projects/08-autonomous-operations
pytest
pytest tests/test_nodes.py -v
pytest tests/test_risk.py -v
```

### Smoke Test the Graph

```bash
python graph.py
# Compiles both stateless and stateful graph variants, prints Mermaid diagram
```

### Run LangSmith Evaluation

```bash
python evaluation.py
# Pushes evaluation dataset, runs graph against examples, uploads scores
```

### Run the Streamlit App

```bash
# From the repository root
streamlit run app/main.py
# Navigate to page 8: Autonomous Operations
```

---

## File Structure

```
08-autonomous-operations/
├── graph.py            # Graph assembly: nodes, edges, routing functions, build_graph()
├── nodes.py            # All 8 node functions: classifier, risk, approval, executor, ...
├── models.py           # OrchestratorState, DepartmentResult, MetricsStore TypedDicts
├── departments.py      # DeepAgent factory functions (create_*_agent), DEPARTMENT_AGENTS dict
├── risk.py             # HIGH_RISK_ACTIONS table + assess_risk() deterministic function
├── prompts.py          # ChatPromptTemplate definitions for classifier and compose_output
├── tools.py            # Tool functions (mock APIs) for all 6 departments
├── evaluation.py       # LangSmith dataset, 3 evaluators, evaluate() call
├── requirements.txt    # Project dependencies
├── skills/             # SKILL.md domain knowledge files, one subdirectory per department
│   ├── student-onboarding/SKILL.md
│   ├── tutor-management/SKILL.md
│   ├── content-pipeline/SKILL.md
│   ├── quality-assurance/SKILL.md
│   ├── support/SKILL.md
│   └── reporting/SKILL.md
├── data/               # Mock data: students, tutors, content catalogue
└── tests/
    ├── conftest.py     # Shared fixtures (mock LLM, sample state)
    ├── test_models.py  # TypedDict structure validation
    ├── test_nodes.py   # Unit tests for each node function (mocked LLM)
    ├── test_risk.py    # Exhaustive risk assessment table tests
    └── test_tools.py   # Tool function tests with mock data
```

---

## Key Design Decisions

### Two-Layer Architecture

The outer layer is a LangGraph `StateGraph` — the master orchestrator. It handles routing, approval, and task cascading. The inner layer is six DeepAgents — each a self-contained domain specialist with its own SKILL.md, tools, and storage backend. This split keeps orchestration logic and domain logic fully separate.

### Task Queue Cascading

Department agents can emit `follow_up_tasks` in their response. The `result_aggregator` extracts these and places them in `task_queue`. The `check_task_queue` node pops them one at a time and loops back to `request_classifier`, enabling a single user request to chain across multiple departments without any re-invocation from outside.

### Tiered Approval

Risk is determined by a deterministic lookup (`risk.py`), not by the LLM. The `HIGH_RISK_ACTIONS` table maps department + action_type pairs to "high risk". Only those combinations pause the graph at `approval_gate` for human sign-off. Everything else executes autonomously. This makes the approval boundary auditable and consistent.

### Metrics Persistence

`metrics_store` is part of `OrchestratorState` and updated by `reporting_snapshot` after every request. With SqliteSaver as the checkpointer, this field accumulates across sessions — a running dashboard of platform activity without a separate database.
