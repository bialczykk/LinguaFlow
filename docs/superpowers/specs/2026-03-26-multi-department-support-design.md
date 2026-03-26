# Multi-Department Support System — Design Specification

**Project:** 06-multi-department-support
**Date:** 2026-03-26
**Status:** Approved

## Overview

LinguaFlow needs a unified support system where a supervisor agent receives incoming requests, classifies them, routes to the right department agents, and orchestrates the response. The system supports parallel dispatch for multi-department requests, supervisor-mediated cross-agent escalation, and a hybrid interaction model where the system can ask the user for clarification before proceeding.

### Learning Goals

- Supervisor agent pattern
- Specialized sub-agents with distinct capabilities
- Agent handoff and escalation (supervisor-mediated)
- Shared state across agents
- Parallel execution with `Send`
- LangSmith: cross-agent tracing, latency monitoring, cost tracking

## Architecture

### Approach: Flat Graph with Supervisor Node

Single `StateGraph` where the supervisor is a node that routes to sub-agent nodes via conditional edges and `Send`. All nodes share one state schema. This keeps everything visible in one graph, makes `Send` straightforward, and produces clean LangSmith traces.

### Interaction Model: Hybrid

First turn is routed and processed automatically. If a sub-agent can't proceed or the request is ambiguous, the supervisor pauses via `interrupt()` to ask the user for clarification, then re-classifies and continues.

### Escalation Model: Supervisor-Mediated

Sub-agents never talk directly to each other. If one needs input from another department, it returns a structured escalation request. The supervisor processes the escalation queue and dispatches to the target department.

## State Schema

```python
class DepartmentResult(TypedDict):
    department: str           # "billing" | "tech_support" | "scheduling" | "content"
    response: str             # The sub-agent's response text
    resolved: bool            # Whether the sub-agent fully handled its part
    escalation: dict | None   # If not resolved: {"target": "scheduling", "context": "..."}

class SupportState(TypedDict):
    # Input
    request: str                              # User's support request text
    request_metadata: dict                    # sender type, priority, etc.

    # Supervisor analysis
    classification: dict                      # departments involved, complexity, summary

    # Sub-agent results (reducer: append)
    department_results: Annotated[list[DepartmentResult], operator.add]

    # Escalation tracking
    escalation_queue: list[dict]              # Pending escalations to process

    # Conversation (hybrid model)
    clarification_needed: str | None          # Question back to user if info is missing
    user_clarification: str | None            # User's response to clarification

    # Final output
    final_response: str                       # Aggregated response to the user
    resolution_status: str                    # "resolved" | "partial" | "escalated_to_human"
```

`department_results` uses `operator.add` reducer so parallel sub-agents can each append their result independently via `Send`.

## Graph Structure

```
START -> supervisor_router
             |
             +-- (needs clarification) -> ask_clarification --[interrupt]--> supervisor_router
             |
             +-- (single dept) -> [billing_agent | tech_support_agent | scheduling_agent | content_agent]
             |
             +-- (multi dept, Send) -> [agent_a + agent_b + ...] (parallel)
             |
             v
         supervisor_aggregator
             |
             +-- (has escalations) -> re-route via Send -> supervisor_aggregator
             |
             +-- (needs clarification) -> ask_clarification -> supervisor_router
             |
             +-- (all resolved) -> compose_response -> END
```

## Nodes

### supervisor_router

Classifies the user's request using an LLM with structured JSON output:

```json
{
  "departments": ["billing", "scheduling"],
  "needs_clarification": false,
  "clarification_question": null,
  "summary": "User wants to cancel a lesson and get a refund",
  "complexity": "multi"
}
```

On re-entry after clarification, merges the user's answer into context and re-classifies.

### Sub-Agent Nodes (billing, tech_support, scheduling, content)

Each sub-agent node:

1. Receives the full `SupportState`
2. Uses an LLM with department-specific tools bound
3. Runs an internal agent loop (invoke LLM -> call tools -> repeat until done, capped at 3 rounds)
4. Returns a `DepartmentResult` appended to `department_results`

If the sub-agent determines it needs another department's help, it sets `escalation` with the target department and context instead of resolving.

### supervisor_aggregator

Collects `department_results` and processes escalations:

- If unresolved escalations exist: populates `escalation_queue`, dispatches via `Send` to target departments with escalation context
- If a sub-agent flagged need for user clarification: routes to `ask_clarification`
- If all resolved: routes to `compose_response`

### compose_response

Takes all `department_results` and produces a single coherent `final_response` using an LLM that merges multi-department answers into a unified reply.

### ask_clarification

Calls `interrupt()` with the clarification question. Resumes when the user responds via `Command(resume=...)`. Routes back to `supervisor_router` for re-classification.

## Routing Logic

```python
def route_from_supervisor(state: SupportState) -> list[Send] | str:
    classification = state["classification"]

    if classification.get("needs_clarification"):
        return "ask_clarification"

    departments = classification["departments"]
    if len(departments) == 1:
        return departments[0] + "_agent"

    # Parallel dispatch via Send
    return [Send(dept + "_agent", state) for dept in departments]
```

## Tools Per Department

| Department | Tools | Description |
|------------|-------|-------------|
| Billing | `lookup_invoice(student_id)` | Find invoices by student |
| Billing | `check_refund_status(invoice_id)` | Check refund status for an invoice |
| Tech Support | `check_system_status(service)` | Check health of a platform service |
| Tech Support | `lookup_user_account(email)` | Look up student account details |
| Scheduling | `check_lesson_schedule(student_id)` | Get upcoming lessons for a student |
| Scheduling | `reschedule_lesson(lesson_id, new_date)` | Reschedule a lesson |
| Content | `search_content_library(query, level)` | Search course catalog |
| Content | `check_enrollment(student_id)` | Check student's enrolled courses |

All tools use `@tool` + `@traceable` decorators, following Project 4's pattern.

## Mock Data

All in `data/`:

| Module | Contents |
|--------|----------|
| `invoices.py` | 6-8 invoices with varying statuses (paid, pending, refunded, disputed). Links to student IDs and lesson IDs. |
| `system_status.py` | Service health map (video, chat, payments, content library). Most healthy, one degraded. |
| `accounts.py` | 4-5 student accounts with email, plan type, last login, known issues. |
| `lessons.py` | Upcoming lessons with dynamic dates (relative to today, like Project 4's calendar). Links to student and tutor IDs. |
| `content_library.py` | Course catalog (10-12 items) with topic, level, type. Plus enrollment records. |

Student IDs are consistent across all data modules to enable cross-department scenarios.

## Sample Support Requests

`data/support_requests.py`:

| Request | Departments | Pattern |
|---------|-------------|---------|
| "I can't log in to my account" | tech_support | Single dept |
| "Can I get a refund for last Tuesday's lesson?" | billing | Single dept |
| "I need to reschedule my Thursday lesson" | scheduling | Single dept |
| "What B2 materials do you have for business English?" | content | Single dept |
| "Cancel my Friday lesson and refund me" | billing + scheduling | Parallel (Send) |
| "I was charged twice and can't access my lesson recordings" | billing + tech_support | Parallel (Send) |
| "I want to change tutors" | (ambiguous) | Clarification flow |
| "My lesson was cancelled but I still got charged, and now I can't book a new one" | billing + scheduling + tech_support | Three-way parallel |

## Prompts

`prompts.py`:

| Prompt | Used By | Purpose |
|--------|---------|---------|
| `SUPERVISOR_CLASSIFICATION_PROMPT` | `supervisor_router` | Classify request into departments, detect ambiguity |
| `BILLING_PROMPT` | `billing_agent` | Department-specific system prompt with tool usage guidance and escalation instructions |
| `TECH_SUPPORT_PROMPT` | `tech_support_agent` | Same pattern, tech support focus |
| `SCHEDULING_PROMPT` | `scheduling_agent` | Same pattern, scheduling focus |
| `CONTENT_PROMPT` | `content_agent` | Same pattern, content focus |
| `COMPOSE_RESPONSE_PROMPT` | `compose_response` | Merge multi-department results into unified reply |

## Testing

### Unit Tests

| File | Coverage |
|------|----------|
| `test_models.py` | State schema validation, `DepartmentResult` construction, reducer behavior |
| `test_tools.py` | Each tool function in isolation against mock data |
| `test_nodes.py` | Each sub-agent node with mocked LLM responses — correct `DepartmentResult` structure, escalation detection, tool round capping |
| `test_supervisor.py` | Router classification, aggregator escalation logic, compose_response merging |

### Integration Tests (`test_graph.py`)

All marked `@pytest.mark.integration`:

| Test | Scenario |
|------|----------|
| `test_single_department_flow` | Single-dept request routes correctly and resolves |
| `test_parallel_dispatch` | Multi-dept request fans out via Send, both results collected |
| `test_escalation_flow` | Sub-agent escalates, supervisor re-routes to target dept |
| `test_clarification_interrupt` | Ambiguous request -> interrupt -> user clarifies -> re-routes and resolves |
| `test_three_way_parallel` | Request hitting three departments simultaneously |

### LangSmith Evaluation (`evaluation.py`)

Two LLM-as-judge evaluators following Project 5's pattern:

- **Routing accuracy**: Did the supervisor pick the correct departments for the request?
- **Response quality**: Is the final response coherent and does it address all parts of the request?

Uses sample requests as a dataset with expected department labels for routing evaluation.

## Streamlit Integration

### Adapter (`app/adapters/support_system.py`)

```python
def start_request(thread_id, request_text, metadata) -> dict | str:
    """Submit request. Returns final response or interrupt payload (clarification)."""

def resume_with_clarification(thread_id, user_response) -> dict | str:
    """Resume after user clarifies. Returns final response or another interrupt."""

def get_state(thread_id) -> dict:
    """Get current state for display (department results, status, etc.)."""
```

### Page (`app/pages/p6_support.py`)

Hybrid UI — chat-like for conversation, structured for results:

- **Input area**: Text input + sample request selector dropdown
- **Processing indicator**: Shows which departments are being consulted
- **Clarification flow**: System question displayed as chat bubble, user responds via text input
- **Results display**: Final unified response + expandable "Behind the scenes" showing departments consulted, individual responses, escalations, routing decision
- **Reset button**: Clears all `p6_` session state keys
- **Documentation viewer**: `doc_viewer.render("docs/06-multi-department-support.md")`

### Tab Registration

Add `"🎯 Support System"` as tab 6 in `app/app.py`.

## File Structure

```
projects/06-multi-department-support/
├── models.py
├── graph.py
├── nodes.py
├── prompts.py
├── tools.py
├── evaluation.py
├── data/
│   ├── __init__.py
│   ├── invoices.py
│   ├── system_status.py
│   ├── accounts.py
│   ├── lessons.py
│   ├── content_library.py
│   └── support_requests.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    ├── test_tools.py
    ├── test_nodes.py
    ├── test_supervisor.py
    └── test_graph.py
```
