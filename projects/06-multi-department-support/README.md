# Project 06: Multi-Department Support System

A LangGraph multi-agent orchestration system for LinguaFlow's student support desk. A single request may touch billing, technical issues, scheduling, and content — this project shows how to classify, fan-out, aggregate, and synthesise responses across all four departments in one coherent workflow.

---

## What This Project Demonstrates

**Core pattern: Supervisor Agent**

A `supervisor_router` node classifies each incoming request and decides which departments need to handle it. After all agents respond, a `supervisor_aggregator` checks for unresolved items and a `compose_response` node synthesises everything into a single user-facing reply. No department agent ever talks directly to another — all coordination flows through the supervisor.

**LangGraph concepts covered**

| Concept | Where It Appears |
|---------|-----------------|
| `StateGraph` with a typed state schema | `models.py`, `graph.py` |
| `Send` for parallel fan-out | `graph.py` — `route_from_supervisor`, `route_from_aggregator` |
| `Annotated[list, operator.add]` reducer | `models.py` — `SupportState.department_results` |
| Conditional edges with a routing function | `graph.py` — `add_conditional_edges` |
| `interrupt` / `Command` human-in-the-loop | `nodes.py` — `ask_clarification` |
| Supervisor-mediated escalation | `nodes.py` — `supervisor_aggregator` |
| Tool-calling agent loop (`bind_tools`) | `nodes.py` — `_run_agent_loop`, `_make_department_agent` |
| `@traceable` and LangSmith tagging | `nodes.py`, `tools.py` |
| LangSmith `evaluate()` | `evaluation.py` |

---

## The Four Departments

| Department | Domain | Tools |
|-----------|--------|-------|
| `billing` | Invoices, payments, refunds | `lookup_invoice`, `check_refund_status` |
| `tech_support` | Platform issues, login, connectivity | `check_system_status`, `lookup_user_account` |
| `scheduling` | Lessons, tutors, rescheduling | `check_lesson_schedule`, `reschedule_lesson` |
| `content` | Courses, enrolment, materials | `search_content_library`, `check_enrollment` |

---

## Sample Requests and Routing Patterns

```
"I can't log in"
  → single department: tech_support_agent

"My invoice is wrong and I need to reschedule my lesson"
  → parallel fan-out: [Send("billing_agent"), Send("scheduling_agent")]

"What are my upcoming lessons?"
  → single department: scheduling_agent

"..."  (empty or ambiguous)
  → needs clarification: ask_clarification [interrupt] → resume → supervisor_router
```

When a department agent returns `resolved=False` with an escalation dict, the supervisor re-dispatches that escalation to a second agent via another `Send` fan-out.

---

## Graph Structure

```
START
  │
  ▼
supervisor_router          ← LLM classifies request
  │
  ├─ needs clarification → ask_clarification ──[interrupt/resume]──► supervisor_router
  │
  ├─ single department   → <dept>_agent ──────────────────────────► supervisor_aggregator
  │
  └─ multi department    → [Send x N] → <dept>_agent (parallel) ──► supervisor_aggregator
                                              │
                              ┌── escalations pending ──┐
                              │                         ▼
                              │              [Send x M] → <dept>_agent ──► supervisor_aggregator
                              │
                              └── all resolved ──► compose_response ──► END
```

---

## File Structure

```
projects/06-multi-department-support/
├── graph.py              # StateGraph assembly: nodes, edges, routing functions
├── nodes.py              # All node functions: supervisor, department agents, HITL
├── models.py             # SupportState TypedDict + DepartmentResult TypedDict
├── tools.py              # LangChain @tool functions for each department
├── prompts.py            # ChatPromptTemplate definitions (classification, agents, compose)
├── evaluation.py         # LangSmith evaluation: dataset + routing_accuracy + quality evaluators
├── requirements.txt      # Python dependencies
├── pytest.ini            # pytest config with asyncio mode
├── data/
│   ├── invoices.py       # Mock invoice records
│   ├── accounts.py       # Mock student accounts
│   ├── lessons.py        # Mock lesson schedule
│   ├── system_status.py  # Mock platform service statuses
│   └── content_library.py # Mock courses and enrolments
└── tests/
    ├── conftest.py       # Shared fixtures and mock helpers
    ├── test_models.py    # Unit tests for state schema and TypedDicts
    ├── test_tools.py     # Unit tests for each department tool
    ├── test_nodes.py     # Unit tests for each node function (mocked LLMs)
    └── test_graph.py     # Integration tests for full graph flows
```

---

## How to Run

**Prerequisites:** Python 3.11+, a shared `.venv` at the repo root with all dependencies installed, `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` in `.env`.

**Run all tests (no API calls needed — LLMs are mocked):**

```bash
cd projects/06-multi-department-support
pytest
```

**Run a specific test:**

```bash
pytest tests/test_graph.py::test_single_department_routing -v
```

**Run the graph interactively (requires ANTHROPIC_API_KEY):**

```bash
python -c "
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

g = build_graph(checkpointer=MemorySaver())
result = g.invoke(
    {
        'request': 'I cannot login and my invoice is wrong',
        'request_metadata': {'sender_type': 'student', 'student_id': 'S001', 'priority': 'high'},
        'department_results': [],
        'escalation_queue': [],
        'classification': {},
        'clarification_needed': None,
        'user_clarification': None,
        'final_response': '',
        'resolution_status': '',
    },
    config={'configurable': {'thread_id': 'test-1'}},
)
print(result.get('final_response'))
"
```

**Run LangSmith evaluation (requires both API keys):**

```bash
python evaluation.py
```

**Visualise the graph topology:**

```bash
python graph.py
```

---

## Key Design Decisions

- **Supervisor-only coordination**: no direct agent-to-agent communication. All escalations are surfaced through `supervisor_aggregator` and re-dispatched centrally. This keeps the graph topology readable and auditable.
- **`operator.add` reducer**: allows multiple parallel `Send` branches to append their `DepartmentResult` to the same list without overwriting each other.
- **Factory pattern for department agents** (`_make_department_agent`): four agents share identical structure — only their prompt, tool list, and department name differ. Defined once, instantiated four times.
- **Stateless tools on mock data**: all `data/` modules are plain Python dicts/lists. This keeps the LangGraph code uncluttered by business logic.
