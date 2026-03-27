# Project 06: Multi-Department Support System

## A Deep Dive into Multi-Agent Orchestration, Parallel Fan-Out with Send, and Supervisor-Mediated Coordination

---

## 1. Introduction

### What We Built

Project 6 builds a support desk system for LinguaFlow. A student submits a support request — "I can't log in and my invoice is wrong" — and the system automatically figures out which departments need to handle it, runs each department's specialist agent in parallel, merges the results, and returns a single coherent reply.

This sounds simple, but it forces us to answer several non-trivial questions:

- How do you classify an arbitrary text request into one or more department targets?
- How do you dispatch multiple agents concurrently when more than one department is involved?
- If agents run in parallel and all write to the same shared state, how do you prevent them from overwriting each other?
- What happens when a department agent can't fully resolve something and needs to hand off to another department?
- What if the request is so ambiguous that no routing decision can be made without asking the user first?

Each of these questions has a specific LangGraph answer, and this project demonstrates all of them in one connected system.

### The Shift from Project 5

Project 5 introduced human-in-the-loop pausing for a single automated pipeline. The human was an *approver* at fixed checkpoints. The graph was linear: generate → review → polish → publish.

Project 6 changes the shape of the graph entirely. Instead of a linear chain, we now have a *fan-out* topology: one supervisor dispatches to multiple agents in parallel, then aggregates their results. The critical new questions become "how do we run things in parallel" and "how do we safely merge parallel results back into one shared state." The HITL pattern from Project 5 is still present — but now it serves a different purpose: clarifying an ambiguous request before routing begins.

### The Graph Structure

```
START
  │
  ▼
supervisor_router          ← LLM classifies the request
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

The fan-out-then-aggregate shape is fundamentally different from everything in Projects 1-5. It is not a pipeline — it is a DAG (directed acyclic graph) with parallel branches that converge.

---

## 2. Supervisor Agent Pattern

### What a Supervisor Does

The supervisor agent pattern is the most common architecture for multi-agent systems. Instead of agents talking directly to each other, one central agent — the supervisor — is responsible for:

1. **Classification**: understanding the incoming request and deciding which specialists are needed.
2. **Routing**: dispatching to the right agents (and optionally in parallel).
3. **Aggregation**: collecting all agent responses, checking whether any problems were left unresolved.
4. **Synthesis**: merging individual specialist answers into one coherent reply.

This pattern keeps the system topology readable. You always know that information flows *through* the supervisor, never sideways between agents. It also makes the system easier to extend — adding a fifth department means registering one more node and adding it to the routing table, not re-wiring inter-agent communication.

### What the Supervisor Is Not Responsible For

The supervisor does not *know* how to answer billing questions or debug technical issues. It knows *which department* should handle them. This is a deliberate separation: the supervisor handles orchestration, the department agents handle domain knowledge.

### Our Supervisor's Two Roles

In this project the supervisor has two distinct nodes:

**`supervisor_router`** — The entry point. It calls an LLM to classify the request and populates `classification` in state. This is the only node that decides *which agents to invoke*.

**`supervisor_aggregator`** — A pure logic node (no LLM call). After all agents have run, it scans their results for unresolved items and builds the `escalation_queue`. It does not compose the final response — that is `compose_response`'s job.

Splitting routing and aggregation into two separate nodes makes each one easier to test, reason about, and replace independently.

---

## 3. Parallel Execution with Send

### The Problem: Fan-Out

When a request involves two departments — say, billing and scheduling — you have two options:

**Option A — Sequential**: run billing_agent, wait for it to finish, then run scheduling_agent. Simple, but slow. Billing and scheduling are independent; there is no reason to wait.

**Option B — Parallel**: dispatch both agents simultaneously and wait for both to finish before continuing. This is faster and is what real production systems do.

LangGraph's answer to option B is the `Send` API.

### How Send Works

`Send` is a special LangGraph type that tells the graph engine "invoke this node with this state snapshot right now, as a concurrent branch." When a routing function returns a list of `Send` objects, LangGraph launches all of them simultaneously, waits for all to complete, and then merges their state updates.

```python
# From graph.py — route_from_supervisor
from langgraph.types import Send

def route_from_supervisor(state: SupportState):
    departments = classification.get("departments", [])

    # Single department — normal routing
    if len(departments) == 1:
        dept = departments[0]
        return f"{dept}_agent"

    # Multiple departments — parallel fan-out via Send
    if departments:
        return [Send(f"{dept}_agent", state) for dept in departments]
```

`Send` takes two arguments: the node name to invoke, and the state to pass to it. Crucially, each `Send` receives its own *copy* of the current state — the agents do not share a mutable object. They run independently, each producing their own state update, which are then merged by LangGraph.

### Why Returns a List Matters

When a conditional edge's routing function returns a **string**, LangGraph routes to one node. When it returns a **list of `Send` objects**, LangGraph creates parallel branches. This is the entire API for fan-out — no special flags, no async/await, no thread management. You just return a list.

```python
# String → single branch
return "billing_agent"

# List of Send → parallel branches
return [Send("billing_agent", state), Send("scheduling_agent", state)]
```

The graph engine handles all concurrency internally.

### Send for Escalation Re-Dispatch

The same pattern appears in `route_from_aggregator`. After the aggregator builds the escalation queue, unresolved items are re-dispatched to their target departments via another round of `Send`:

```python
# From graph.py — route_from_aggregator
def route_from_aggregator(state: SupportState):
    escalation_queue = state.get("escalation_queue", [])

    if escalation_queue:
        # Re-dispatch each escalation to its target department
        return [Send(f"{e['target']}_agent", state) for e in escalation_queue]

    return "compose_response"
```

This means the graph can do multiple rounds of parallel dispatch — first pass for the original request, then a second pass for any escalations — all using the same mechanism.

---

## 4. State Schema Design

### The Core Problem: Parallel Writes

When multiple agents run in parallel and all return `{"department_results": [...]}`, what happens? If the last write wins, you lose all but one agent's result. You need a way to *merge* parallel writes, not overwrite them.

LangGraph solves this with **reducers** — Python callables attached to state fields via `Annotated` type hints. When multiple parallel branches all write to the same field, LangGraph applies the reducer to combine them.

### The operator.add Reducer

```python
# From models.py
import operator
from typing import Annotated
from typing_extensions import TypedDict

class SupportState(TypedDict):
    # ...
    department_results: Annotated[list[DepartmentResult], operator.add]
    # ...
```

`operator.add` on lists is list concatenation. When billing_agent returns `{"department_results": [billing_result]}` and scheduling_agent returns `{"department_results": [scheduling_result]}`, LangGraph concatenates them: `department_results = [billing_result, scheduling_result]`.

Without this reducer, the second write would overwrite the first. The `Annotated[..., operator.add]` annotation is the entire solution — one line of type annotation changes "last write wins" to "all writes are merged."

### Why Only department_results Gets a Reducer

Most fields in `SupportState` are written by a single node at a time, so they don't need a reducer. Default behaviour ("last write wins") is fine for them.

Only `department_results` is written to by multiple parallel branches simultaneously. It is the only field that needs the reducer.

This is a general principle: apply reducers surgically. Over-applying them can hide bugs where you accidentally write to a field from multiple places when you didn't intend to.

### The DepartmentResult TypedDict

```python
# From models.py
class DepartmentResult(TypedDict):
    department: str           # "billing" | "tech_support" | "scheduling" | "content"
    response: str             # The sub-agent's natural-language response
    resolved: bool            # Whether the sub-agent fully handled its part
    escalation: dict | None   # If not resolved: {"target": "<dept>", "context": "..."}
```

Each agent returns one of these. The `resolved` / `escalation` fields drive the escalation logic in `supervisor_aggregator`. A fully resolved result has `resolved=True` and `escalation=None`. An unresolved result has `resolved=False` and `escalation={"target": "billing", "context": "payment gateway is down"}`.

### Full State Schema Lifecycle

```python
class SupportState(TypedDict):
    # Set at invocation
    request: str
    request_metadata: dict

    # Set by supervisor_router
    classification: dict

    # Appended by each department agent (reducer: operator.add)
    department_results: Annotated[list[DepartmentResult], operator.add]

    # Managed by supervisor_aggregator
    escalation_queue: list[dict]

    # Used for hybrid clarification flow
    clarification_needed: str | None
    user_clarification: str | None

    # Set by compose_response
    final_response: str
    resolution_status: str
```

Reading this schema tells you the full lifecycle of a request: who sets what, and when. Each node only writes to the fields it owns.

---

## 5. Conditional Routing

### How Conditional Edges Work

In LangGraph, a conditional edge is a function that inspects state and returns the name of the next node to go to (or a list of `Send` objects for parallel dispatch). You register it with `add_conditional_edges`:

```python
# From graph.py
graph.add_conditional_edges(
    "supervisor_router",      # source node
    route_from_supervisor,    # routing function
    path_map={                # exhaustive map of all possible return values
        "ask_clarification": "ask_clarification",
        "billing_agent": "billing_agent",
        "tech_support_agent": "tech_support_agent",
        "scheduling_agent": "scheduling_agent",
        "content_agent": "content_agent",
    },
)
```

The `path_map` is used by LangGraph at compile time to validate the graph topology. It maps every string the routing function might return to a registered node. If your routing function could return a value not in `path_map`, the graph will fail to compile. This catches typos and missing nodes before you ever run the graph.

### route_from_supervisor: Three Cases

```python
# From graph.py
def route_from_supervisor(state: SupportState):
    classification = state.get("classification", {})

    # Case 1: clarification needed
    if classification.get("needs_clarification"):
        return "ask_clarification"

    departments = classification.get("departments", [])

    # Case 2: single department
    if len(departments) == 1:
        dept = departments[0]
        return f"{dept}_agent"

    # Case 3: multiple departments — parallel fan-out
    if departments:
        return [Send(f"{dept}_agent", state) for dept in departments]

    # Fallback: unexpected state → ask for clarification
    return "ask_clarification"
```

Notice the fallback at the end. If `classification` is somehow empty (e.g., the LLM returned malformed JSON), the routing function still produces a valid result rather than crashing.

### route_from_aggregator: Two Cases

```python
# From graph.py
def route_from_aggregator(state: SupportState):
    escalation_queue = state.get("escalation_queue", [])

    if escalation_queue:
        return [Send(f"{e['target']}_agent", state) for e in escalation_queue]

    return "compose_response"
```

This is simpler — either there are escalations to handle (another parallel fan-out), or there are none (proceed to final synthesis). The aggregator is the gate between "collecting results" and "composing the final reply."

---

## 6. Department Sub-Agents

### The Tool-Calling Agent Loop

Each department agent is a ReAct-style loop: present the request to the LLM, the LLM may request tool calls, execute those tools, feed the results back, repeat until the LLM produces a text response with no tool calls.

```python
# From nodes.py — _run_agent_loop
def _run_agent_loop(
    llm_with_tools: ChatAnthropic,
    system_prompt: str,
    request: str,
    tools: list[BaseTool],
    max_rounds: int = 3,
) -> str:
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request),
    ]

    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls → done
        if not getattr(response, "tool_calls", None):
            break

        # Execute each tool and append ToolMessages
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name in tool_map:
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"
            else:
                tool_result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
            )

    return response.content
```

The `max_rounds=3` guard prevents infinite loops — if the LLM keeps requesting tools without converging, the loop terminates and returns whatever the last response was.

### bind_tools

To enable tool calling, the LLM must be told which tools are available. In LangChain, this is done with `bind_tools`:

```python
# From nodes.py — inside _make_department_agent
model_with_tools = ChatAnthropic(
    model="claude-haiku-4-5-20251001"
).bind_tools(dept_tools)
```

`bind_tools` injects the tool schemas into the model's system context. The LLM can then respond with `tool_calls` in its message — structured requests to invoke specific tools with specific arguments. Without `bind_tools`, the LLM would not know the tools exist.

### The Factory Pattern

All four department agents share identical structure: classify request, format system prompt, bind tools, run agent loop, return `DepartmentResult`. Only the prompt, tool list, and department name differ.

Rather than writing four near-identical functions, a factory builds them:

```python
# From nodes.py
def _make_department_agent(
    department: str,
    prompt_template: str,
    dept_tools: list[BaseTool],
) -> Any:
    @traceable(name=f"{department}_agent", run_type="chain", tags=_TAGS)
    def agent_node(state: SupportState) -> dict:
        # ... format prompt, bind tools, run loop ...
        dept_result: DepartmentResult = {
            "department": department,
            "response": response_text,
            "resolved": True,
            "escalation": None,
        }
        return {"department_results": [dept_result]}

    return agent_node

billing_agent      = _make_department_agent("billing",      BILLING_PROMPT,      BILLING_TOOLS)
tech_support_agent = _make_department_agent("tech_support", TECH_SUPPORT_PROMPT, TECH_SUPPORT_TOOLS)
scheduling_agent   = _make_department_agent("scheduling",   SCHEDULING_PROMPT,   SCHEDULING_TOOLS)
content_agent      = _make_department_agent("content",      CONTENT_PROMPT,      CONTENT_TOOLS)
```

This is the DRY principle applied to node construction. The closure captures `department`, `prompt_template`, and `dept_tools` — each node function is uniquely configured at creation time while sharing the same implementation.

Notice that each node returns `{"department_results": [dept_result]}` — a list containing one item. This is intentional. Because `department_results` has an `operator.add` reducer, returning a list causes the reducer to *append* this item to the existing list. If you returned the item directly (not wrapped in a list), the reducer would try to concatenate a `DepartmentResult` with a list, which would fail.

---

## 7. Escalation Model

### Why Supervisor-Mediated Escalation?

In a naive multi-agent design, you might allow agents to call each other directly. Billing notices a payment gateway issue and invokes tech_support directly. This sounds efficient but creates problems:

- **Topology**: the graph becomes a mesh. Adding one agent potentially requires updating N other agents' routing tables.
- **Observability**: tracing which agent called which becomes complex. There is no single place to inspect the escalation decision.
- **State**: if billing invokes tech_support directly, when does tech_support's result get merged with billing's result? Who is responsible for that?

The supervisor-mediated model avoids all of this. Agents never call other agents. When an agent can't fully resolve something, it sets `resolved=False` and populates `escalation` in its `DepartmentResult`. The supervisor reads this and re-dispatches — centrally, auditably, with full visibility.

### How Escalation Flows

```
1. billing_agent runs → returns DepartmentResult(resolved=False,
     escalation={"target": "tech_support", "context": "payment gateway down"})

2. supervisor_aggregator scans results → builds escalation_queue

3. route_from_aggregator sees escalation_queue is non-empty →
     returns [Send("tech_support_agent", state)]

4. tech_support_agent runs with escalation_queue in state →
     sees escalation targeting it, includes context in its prompt

5. supervisor_aggregator scans results again → escalation_queue is empty

6. route_from_aggregator → "compose_response"
```

The `supervisor_aggregator` node is pure logic — no LLM, no API calls:

```python
# From nodes.py
def supervisor_aggregator(state: SupportState) -> dict:
    escalations = []
    for result in state.get("department_results", []):
        if not result.get("resolved") and result.get("escalation"):
            escalations.append(result["escalation"])
    return {"escalation_queue": escalations}
```

Pure logic nodes are fast, deterministic, and easy to unit test. This is deliberately kept as a simple scan — all the intelligence was already applied in the agent that decided to set `resolved=False`.

### Escalation Context

When an agent is re-invoked via an escalation, it receives the full state including the `escalation_queue`. The factory-built agent node checks for escalations targeting its department and injects the context into its system prompt:

```python
# From nodes.py — inside _make_department_agent
escalation_items = [
    e for e in state.get("escalation_queue", [])
    if e.get("target") == department
]
escalation_context = ""
if escalation_items:
    ctx_lines = [e.get("context", "") for e in escalation_items]
    escalation_context = "\nEscalation context:\n" + "\n".join(ctx_lines)
```

This means the second-pass agent has richer context than it would on a cold start. It knows *why* it was escalated to, not just what the user asked.

---

## 8. Human-in-the-Loop Clarification

### When Clarification Is Needed

The supervisor's LLM classification can fail to determine which department to route to when the request is too vague — "I need help", "something is wrong", or an empty message. In these cases, `classification.needs_clarification` is `True` and `route_from_supervisor` sends the graph to `ask_clarification`.

### The interrupt / Command Pattern

`ask_clarification` uses `interrupt()` to pause the graph and surface a question to the caller:

```python
# From nodes.py
def ask_clarification(state: SupportState) -> Command:
    question = state.get("clarification_needed", "Could you provide more details?")

    # Suspend graph execution and surface the question
    user_reply = interrupt(question)

    # On resume: store reply and re-route to supervisor for re-classification
    return Command(
        update={"user_clarification": user_reply, "clarification_needed": None},
        goto="supervisor_router",
    )
```

When `interrupt(question)` is called:
1. The graph pauses and returns to the caller with an interrupted status.
2. The `question` string is the payload — the caller sees it and can display it to the user.
3. Execution is frozen, all state is checkpointed.

When the caller resumes with `Command(resume=user_reply)`:
1. LangGraph restores state from the checkpoint.
2. The `ask_clarification` node re-runs from the top.
3. When execution reaches `interrupt()`, LangGraph detects pending resume data and returns it immediately.
4. `user_reply` is now the user's answer.
5. The node returns `Command(update=..., goto="supervisor_router")`.

### Command with goto

Notice that `ask_clarification` returns a `Command` object rather than a plain dict. This is a more powerful return type: it combines a state update (`update`) with an explicit routing decision (`goto`). The `goto="supervisor_router"` overrides normal edge traversal — regardless of what edges are registered from `ask_clarification`, the graph goes directly to `supervisor_router` after the Command.

This is why the node is registered with `ends=["supervisor_router"]` in `graph.py`:

```python
# From graph.py
graph.add_node("ask_clarification", ask_clarification, ends=["supervisor_router"])
```

The `ends` kwarg tells LangGraph at compile time that this node may dynamically route to `supervisor_router` via a Command. Without it, the graph compiler would see no edge from `ask_clarification` and raise a validation error.

### Re-classification After Clarification

When `supervisor_router` runs again with `user_clarification` populated in state, it appends that context to the classification prompt:

```python
# From nodes.py — supervisor_router
clarification_context = (
    f"\nUser clarification: {user_clarification}" if user_clarification else ""
)
messages = SUPERVISOR_CLASSIFICATION_PROMPT.format_messages(
    request=state["request"],
    # ...
    clarification_context=clarification_context,
)
```

The LLM now has both the original request and the user's clarifying answer, making it far more likely to produce a valid routing decision. If the second classification still returns `needs_clarification=True`, the graph will interrupt again — there is no explicit limit, which is intentional for a clarification flow (you keep asking until you can route).

---

## 9. Response Composition

### Merging Multi-Department Results

Once all agents have run and the escalation queue is empty, `compose_response` synthesises everything into a single user-facing reply:

```python
# From nodes.py
def compose_response(state: SupportState) -> dict:
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")

    dept_lines = []
    all_resolved = True

    for dr in state.get("department_results", []):
        dept_lines.append(f"[{dr['department'].upper()}] {dr['response']}")
        if not dr.get("resolved", True):
            all_resolved = False

    department_responses = "\n\n".join(dept_lines) if dept_lines else "No department responses available."

    messages = COMPOSE_RESPONSE_PROMPT.format_messages(
        request=state["request"],
        department_responses=department_responses,
    )
    response = model.invoke(messages)

    if all_resolved:
        resolution_status = "resolved"
    elif state.get("escalation_queue"):
        resolution_status = "escalated_to_human"
    else:
        resolution_status = "partial"

    return {
        "final_response": response.content,
        "resolution_status": resolution_status,
    }
```

The composition prompt receives the original request (for tone/context) and all department responses formatted with department labels. The LLM is asked to produce one coherent, empathetic message — not a concatenation of fragments.

### Why a Separate Composition Step?

You could ask the last agent to synthesise the final response. But if the last agent happens to be billing, it has no context about what scheduling answered. And if the agents ran in parallel, there is no "last agent" — they all finish at roughly the same time.

`compose_response` is the dedicated synthesis node. It runs *after* all agents have completed and all results are in state. It has full visibility into every department's answer and can produce a reply that addresses the user's complete request in a single voice.

---

## 10. LangSmith Observability

### Tagging All Traces

Every node and tool in this project shares the same LangSmith project tag:

```python
# From nodes.py and tools.py
_TAGS = ["p6-multi-department-support"]
```

This tag is applied via `@traceable`:

```python
@traceable(name="supervisor_router", run_type="chain", tags=_TAGS)
def supervisor_router(state: SupportState) -> dict:
    ...
```

In the LangSmith UI, filtering by `p6-multi-department-support` shows every trace from this project and nothing else. Because the graph can fan out to multiple agents in parallel, a single invocation produces multiple concurrent traces — the tag lets you see them all together and understand the full picture.

### run_type Values

`@traceable` accepts a `run_type` that categorises the trace in the LangSmith UI:

- `run_type="chain"` — used for nodes (orchestration steps)
- `run_type="tool"` — used for tool functions

This classification affects how LangSmith displays and groups traces. Tools appear nested under the chain that called them, making the tool-calling loop visually readable as a single coherent unit.

### LangSmith Evaluation

`evaluation.py` defines two evaluators and a dataset to run them against:

**`routing_accuracy_evaluator`** — Deterministic. Compares the supervisor's chosen departments against the expected departments stored in the dataset example. Returns 1.0 for exact match, 0.0 for no overlap, and a proportional score for partial overlap.

**`response_quality_evaluator`** — LLM-as-judge. Asks `claude-haiku` to rate the final unified response on coherence, completeness, and professionalism on a 0-1 scale.

```python
# From evaluation.py — routing_accuracy_evaluator (simplified)
def routing_accuracy_evaluator(run: Run, example: Example) -> dict:
    actual_depts   = set(run.outputs.get("classification", {}).get("departments", []))
    expected_depts = set(example.outputs.get("expected_departments", []))

    if not expected_depts:
        return {"key": "routing_accuracy", "score": 1.0}

    overlap = len(actual_depts & expected_depts)
    score   = overlap / len(expected_depts)
    return {"key": "routing_accuracy", "score": score}
```

The split between deterministic and LLM-as-judge evaluators is deliberate. Routing accuracy is a structural property — you can check it with set comparison. Response quality is subjective — only an LLM judge can assess "is this response coherent and complete?"

Running `python evaluation.py` pushes the dataset to LangSmith, runs the graph against every example, applies both evaluators, and uploads the results. You can view the experiment in the LangSmith dashboard and compare across runs as you change prompts or models.

---

## 11. Testing Strategy

### The Mocking Strategy

Real LLM calls in tests are slow, non-deterministic, and expensive. All tests in this project mock the LLM completely. There are two patching points:

1. `nodes._classification_model` — the module-level model used by `supervisor_router`.
2. `langchain_anthropic.ChatAnthropic` — the class constructor used by department agents and `compose_response`.

```python
# From tests/test_graph.py — typical test setup
def _classification_mock(departments, needs_clarification=False, ...) -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "departments": departments,
            "needs_clarification": needs_clarification,
            ...
        })
    )
    return mock
```

The mock's `.invoke()` returns a `MagicMock` whose `.content` is a pre-baked JSON string. The node parses this JSON exactly as it would parse a real LLM response — the mocking is transparent to the business logic.

### Unit vs Integration Tests

**Unit tests** (`test_nodes.py`, `test_tools.py`, `test_models.py`) test individual components in isolation. `test_nodes.py` calls each node function directly with a crafted state dict and asserts on the returned update dict. `test_tools.py` calls each tool function with mock data keys.

**Integration tests** (`test_graph.py`) compile and invoke the full graph. These tests verify end-to-end flows:

- Single department routing: one `Send` is dispatched, one result in `department_results`.
- Multi-department routing: two `Send`s are dispatched in parallel, two results in `department_results` (in any order).
- Clarification flow: graph interrupts, `Command(resume=...)` resumes it, routing re-runs.
- Escalation handling: first pass produces an unresolved result, aggregator fills escalation_queue, second pass re-dispatches.

The integration tests require a `MemorySaver` checkpointer for the HITL tests (interrupt/resume needs state persistence). Single-pass tests can use a stateless graph.

### What the Tests Do Not Cover

The tests mock away the LLM responses, which means they do not test the *quality* of the supervisor's classification or the department agents' answers. That is the job of `evaluation.py`. Tests verify structure and routing; evaluation verifies quality.

---

## 12. Code Walkthrough

### The Graph Assembly in graph.py

```python
# From graph.py — build_graph
def build_graph(checkpointer=None):
    graph = StateGraph(SupportState)

    # Supervisor layer
    graph.add_node("supervisor_router", supervisor_router)
    graph.add_node("supervisor_aggregator", supervisor_aggregator)

    # HITL node with dynamic routing declaration
    graph.add_node("ask_clarification", ask_clarification, ends=["supervisor_router"])

    # Department agents
    graph.add_node("billing_agent", billing_agent)
    graph.add_node("tech_support_agent", tech_support_agent)
    graph.add_node("scheduling_agent", scheduling_agent)
    graph.add_node("content_agent", content_agent)

    # Final synthesis
    graph.add_node("compose_response", compose_response)

    # Static edges
    graph.add_edge(START, "supervisor_router")
    graph.add_edge("billing_agent", "supervisor_aggregator")
    graph.add_edge("tech_support_agent", "supervisor_aggregator")
    graph.add_edge("scheduling_agent", "supervisor_aggregator")
    graph.add_edge("content_agent", "supervisor_aggregator")
    graph.add_edge("compose_response", END)

    # Dynamic routing
    graph.add_conditional_edges("supervisor_router",  route_from_supervisor,  path_map={...})
    graph.add_conditional_edges("supervisor_aggregator", route_from_aggregator, path_map={...})

    return graph.compile(checkpointer=checkpointer)
```

The structure is intentionally readable: first define all nodes, then all edges (static then dynamic), then compile. Reading `build_graph` from top to bottom gives you a complete picture of the graph topology.

### The operator.add Reducer in Action

```python
# Both of these run in parallel and both return:
{"department_results": [their_result]}

# LangGraph applies operator.add (list concatenation) to merge them:
# state["department_results"] = [billing_result] + [scheduling_result]
#                              = [billing_result, scheduling_result]
```

The order of results in `department_results` is not guaranteed when agents run in parallel. `compose_response` doesn't depend on order — it processes `department_results` as a set of responses, not an ordered sequence.

### Invoking the Graph

```python
from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

# With checkpointer (required for interrupt/resume)
g = build_graph(checkpointer=MemorySaver())

result = g.invoke(
    {
        "request": "I cannot login and my invoice is wrong",
        "request_metadata": {"sender_type": "student", "student_id": "S001", "priority": "high"},
        "department_results": [],
        "escalation_queue": [],
        "classification": {},
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    },
    config={"configurable": {"thread_id": "support-001"}},
)

print(result["final_response"])
# → A single unified response addressing both the login issue and the invoice issue
```

All `SupportState` fields must be initialised at invocation time — LangGraph does not set defaults for TypedDict fields. The `department_results: []` initialisation is especially important: without it, the `operator.add` reducer has nothing to append to.

---

## 13. Key Takeaways

| Concept | What It Does | Where to Look |
|---------|-------------|---------------|
| Supervisor agent pattern | Central coordinator for classification, routing, aggregation | `nodes.py` — `supervisor_router`, `supervisor_aggregator` |
| `Send` for fan-out | Parallel dispatch to multiple nodes simultaneously | `graph.py` — `route_from_supervisor`, `route_from_aggregator` |
| `Annotated[list, operator.add]` | Reducer that merges parallel writes by concatenation | `models.py` — `SupportState.department_results` |
| Conditional edges + `path_map` | Dynamic routing with compile-time topology validation | `graph.py` — `add_conditional_edges` |
| `bind_tools` | Injects tool schemas into LLM context to enable tool calling | `nodes.py` — inside `_make_department_agent` |
| ReAct agent loop | LLM → tool call → result → repeat until no more tool calls | `nodes.py` — `_run_agent_loop` |
| Factory pattern for nodes | Build N structurally identical nodes with one function | `nodes.py` — `_make_department_agent` |
| Supervisor-mediated escalation | Agents signal unresolved status, supervisor re-dispatches | `nodes.py` — `supervisor_aggregator` + `DepartmentResult.escalation` |
| `interrupt` / `Command(goto=...)` | HITL clarification with dynamic re-routing on resume | `nodes.py` — `ask_clarification` |
| `ends=["node"]` in `add_node` | Declare dynamic routing targets for Command-returning nodes | `graph.py` — `ask_clarification` registration |
| `@traceable` with tags | LangSmith observability with project-level filtering | `nodes.py`, `tools.py` — `_TAGS` |
| LangSmith `evaluate()` | Automated quality measurement against a fixed dataset | `evaluation.py` |

### The Patterns You Should Recognise

**Parallel fan-out:**
```
routing function returns [Send("node_a", state), Send("node_b", state)]
→ both nodes run concurrently
→ their state updates are merged (reducer applied to Annotated fields)
→ graph continues when all branches complete
```

**Supervisor-mediated coordination:**
```
agent sets resolved=False + escalation dict
→ aggregator collects → escalation_queue
→ routing function re-dispatches via Send to target agent
→ aggregator checks again → eventually escalation_queue is empty
→ compose_response
```

**HITL with re-routing:**
```
interrupt(question) → graph pauses
human answers
Command(resume=answer) → graph resumes at interrupt node
node returns Command(update={...}, goto="supervisor_router")
→ re-classification with added context
```

### What Comes Next: Project 7

Project 6 demonstrates how to orchestrate multiple specialised agents within a single LangGraph. **Project 7** introduces DeepAgents — a higher-level framework built on top of LangGraph. Rather than wiring nodes and edges manually, DeepAgents provides `create_deep_agent()` and a harness architecture with built-in memory, middleware, and skill composition. The multi-agent coordination patterns you learned here — fan-out, aggregation, escalation — are still present underneath, but expressed at a higher level of abstraction.

---

*This document covers Project 06 of the LinguaFlow learning path. Continue to `docs/07-...` when ready for DeepAgents.*
