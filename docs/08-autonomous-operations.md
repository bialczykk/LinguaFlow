# Project 08: LinguaFlow Autonomous Operations (Capstone)

## A Deep Dive into Autonomous Multi-Agent Orchestration — Integrating LangGraph, DeepAgents, HITL Approval, Persistent Metrics, and End-to-End LangSmith Observability

---

## 1. Introduction

### What We Built

Project 8 is the capstone of the LinguaFlow learning path. We built an **autonomous operations orchestrator** — a system that receives a high-level business request like "onboard 20 new students and assign tutors" and executes it across multiple specialized departments, without requiring the user to re-invoke anything.

The system:

1. **Classifies** the request — which departments need to act, what kind of action it is
2. **Assesses risk** — deterministically, using a rules table, not an LLM
3. **Gates high-risk actions** — pausing for human operator approval before executing
4. **Dispatches departments in parallel** — using `Send()` for concurrent fan-out
5. **Cascades autonomously** — when one department's output generates follow-up tasks for another
6. **Accumulates metrics** — persisting a running platform dashboard across sessions
7. **Reports** — synthesising a coherent final response with a task chain summary

### From Individual Agents to Autonomous Operations

The first seven projects each taught specific primitives in isolation:

- **P1–P2**: LLM chains, first graphs, structured output
- **P3**: RAG pipelines and document retrieval
- **P4**: Tool binding, the ReAct loop, agent planning
- **P5**: Human-in-the-loop with `interrupt` and `Command`
- **P6**: Parallel fan-out with `Send`, `operator.add` reducers, multi-agent routing
- **P7**: DeepAgents — `create_deep_agent()`, SKILL.md, `CompositeBackend`, `TodoList`

Project 8 does not introduce many new primitives. Instead, it shows how all of these compose into a system that feels qualitatively different: the machine takes over. A single human request triggers a chain of autonomous decisions — routing, risk assessment, parallel execution, follow-up dispatching — that can span six departments and multiple processing cycles, all without the human having to say "now do the next thing."

That is what "autonomous operations" means in this context.

### What Makes This a Capstone

A capstone is useful only if it forces you to integrate concepts, not just stack them. Several tensions arise that require careful design:

- The orchestrator needs persistent state across sessions. The department agents do not. These require *different checkpointers at different layers*.
- Risk assessment must be fast, auditable, and consistent. An LLM cannot be trusted for this. A deterministic rules lookup can.
- A department agent completing its task may discover that another department needs to act. Agents cannot call each other directly. The orchestrator must manage the cascade.
- Multiple department agents may run in parallel and all write to the same state field. Without reducers, only the last write survives.

Each of these tensions has a specific LangGraph or DeepAgents answer. Understanding *why* a particular mechanism was chosen — not just *what* it does — is the point of reading this document.

---

## 2. Two-Layer Architecture

### The Split

The system is built in two distinct layers:

**Layer 1 — The Master Orchestrator (LangGraph `StateGraph`)**

This is a hand-crafted `StateGraph` graph. It knows nothing about English tutoring. It manages orchestration: routing, risk assessment, approval gating, parallel dispatch, task queue management, and reporting. Every node in this graph is a Python function — either pure logic or an LLM call — wired together with conditional edges.

**Layer 2 — Department Agents (DeepAgents)**

Each of the six departments is a DeepAgent created with `create_deep_agent()`. The orchestrator does not care about their internals. It calls `agent.invoke({"message": request})` and gets back a text response. Each agent has its own SKILL.md for domain knowledge, its own tool set, and its own storage backend.

### Why This Split?

You could build the whole system as one giant StateGraph. But then all the domain knowledge for six departments would live in one set of nodes, one set of prompts, one set of tools. Adding a seventh department would require modifying the core graph. The graph would become unreadable.

The two-layer split enforces a clean interface: the orchestrator only knows department *names* (strings). It doesn't import department logic directly — it uses a `DEPARTMENT_AGENTS` dict that maps names to factory functions. Each factory returns a fully encapsulated DeepAgent. The orchestrator composes agents as black boxes.

```
                    ┌──────────────────────────────────────┐
                    │         Master Orchestrator           │
                    │    (LangGraph StateGraph, graph.py)   │
                    │                                       │
                    │  classify → risk → [approval] → Send  │
                    │     ↓                                 │
                    │  department_executor (per dept)        │
                    │     ↓                                 │
                    │  aggregate → task_queue → loop?       │
                    └───────────────┬──────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                         │
    ┌──────┴──────┐          ┌──────┴──────┐          ┌──────┴──────┐
    │  Onboarding │          │   Tutor Mgmt│          │    Content  │
    │  DeepAgent  │          │  DeepAgent  │          │  Pipeline   │
    │ (departments│          │             │          │  DeepAgent  │
    │    .py)     │          │             │          │             │
    └─────────────┘          └─────────────┘          └─────────────┘
```

### How P7 Pioneered This and P8 Scales It

Project 7 introduced DeepAgents as a higher-level abstraction for a single curriculum engine. It demonstrated `create_deep_agent()`, SKILL.md files, and `CompositeBackend` in the context of one specialised agent.

Project 8 takes that same pattern and scales it to six agents, each wrapping a different business domain, all orchestrated by a LangGraph graph that handles the cross-cutting concerns (routing, risk, approval, cascading) that are inappropriate to push inside any individual agent.

---

## 3. Master Orchestrator Graph

### Full Graph Walkthrough

The graph is assembled in `graph.py`'s `build_graph()` function. It is a `StateGraph[OrchestratorState]` with eight nodes and a combination of static edges, conditional edges, and `Command`-based dynamic routing.

**Nodes:**

| Node | Type | Purpose |
|------|------|---------|
| `request_classifier` | LLM | Parse request, determine departments and action_type |
| `risk_assessor` | Pure logic | Lookup risk level from `HIGH_RISK_ACTIONS` table |
| `approval_gate` | HITL interrupt | Pause for human operator, route on response |
| `dispatch_departments` | Pass-through | Stable `Command` target before fan-out |
| `department_executor` | DeepAgent invocation | Run the agent for a specific department |
| `result_aggregator` | Pure logic | Extract follow_up_tasks, update completed audit |
| `check_task_queue` | Pure logic | Pop next cascaded task or route to finish |
| `compose_output` | LLM | Merge all results into coherent final response |
| `reporting_snapshot` | Pure logic | Increment cumulative platform metrics |

**Static edges** describe the deterministic transitions that always happen:

```python
graph.add_edge(START, "request_classifier")
graph.add_edge("request_classifier", "risk_assessor")
graph.add_edge("department_executor", "result_aggregator")
graph.add_edge("result_aggregator", "check_task_queue")
graph.add_edge("compose_output", "reporting_snapshot")
graph.add_edge("reporting_snapshot", END)
```

**Conditional edges** describe branching points:

```python
# After risk assessment: low risk → dispatch, high risk → approval
graph.add_conditional_edges(
    "risk_assessor",
    route_from_risk,
    path_map={"approval_gate": "approval_gate", "dispatch_departments": "dispatch_departments"},
)

# After dispatch pass-through: fan out via Send to parallel department_executor nodes
graph.add_conditional_edges(
    "dispatch_departments",
    fan_out_to_departments,
    path_map={"department_executor": "department_executor", "compose_output": "compose_output"},
)
```

**Command-based routing** from `approval_gate` and `check_task_queue` is declared via `ends=`:

```python
graph.add_node("approval_gate", approval_gate, ends=["dispatch_departments", "compose_output"])
graph.add_node("check_task_queue", check_task_queue, ends=["request_classifier", "compose_output"])
```

### Why dispatch_departments Is a Pass-Through Node

`approval_gate` uses `Command(goto="dispatch_departments")` to route to the fan-out point after approval. But in LangGraph, `Command.goto` must target a registered node — you cannot route a `Command` directly to a conditional edge function.

The `dispatch_departments` node is a no-op (returns `{}`) that exists purely as a stable landing target. Immediately after it runs, the conditional edge `fan_out_to_departments` kicks in and produces the actual `Send` list for parallel dispatch. This is a clean way to combine `Command`-based routing with `Send`-based fan-out.

---

## 4. Parallel Execution with Send

### Revisiting the Fan-Out Pattern

Project 6 introduced `Send()` for dispatching multiple department agents concurrently. Project 8 uses the same mechanism, with one important difference: the departments are not hardcoded at the edge level. They are determined at runtime from the LLM's classification output.

```python
# From graph.py — fan_out_to_departments
def fan_out_to_departments(state: OrchestratorState):
    classification = state.get("classification", {})
    departments = classification.get("departments", [])

    if not departments:
        return "compose_output"  # graceful fallback

    return [
        Send("department_executor", {**state, "_target_dept": dept})
        for dept in departments
    ]
```

Two things to notice:

**Each `Send` injects `_target_dept`** into the state copy. This is how a single `department_executor` node can serve all six departments — it reads `state["_target_dept"]` to know which DeepAgent to instantiate. Without this injection, each branch would have no way to identify its department.

**The full state is copied per branch** (`{**state, ...}`). Parallel branches do not share mutable objects. Each branch gets its own state snapshot. The only synchronisation point is after all branches complete, when LangGraph merges their state updates using reducers.

### How operator.add Prevents Lost Results

When three departments run in parallel and each returns `{"department_results": [their_result]}`, LangGraph needs to merge three updates to the same field. Without a reducer, the last write wins — two results are lost.

The `operator.add` annotation on `department_results` in `models.py` changes the semantics from "last write wins" to "concatenate all writes":

```python
# From models.py
department_results: Annotated[list[DepartmentResult], operator.add]
```

When billing returns `[billing_result]` and tutor_management returns `[tutor_result]`, LangGraph applies `operator.add([billing_result], [tutor_result])` — which is list concatenation — yielding `[billing_result, tutor_result]`. Every parallel branch's result is preserved.

This one annotation is doing a lot of work. It is the difference between a broken system and a correct one.

### Why Each Node Returns a List with One Item

Because of how `operator.add` works, each department node must return its result wrapped in a list:

```python
# From nodes.py — department_executor
return {"department_results": [result]}   # correct: list of one
# NOT: return {"department_results": result}  # wrong: would concatenate chars of a string
```

If a node returned the `DepartmentResult` dict directly (not wrapped in a list), `operator.add` would try to concatenate a `dict` with a `list`, which would fail. The single-item list is the contract that makes the reducer work correctly across all branches.

---

## 5. Autonomous Task Cascading

### The Problem: Multi-Hop Operations Without Re-Invocation

Consider this request: "Onboard three new students and assign them tutors."

Onboarding a student and assigning a tutor are two different operations handled by two different departments. The user expressed them in one sentence. The straightforward approach — route to both departments simultaneously — doesn't work: the tutor assignment can only happen *after* the onboarding is complete, because the tutor needs to know which student was just enrolled.

This is a sequential dependency. The naive solution is to ask the user to submit two separate requests. But that's not autonomous — it's just a clever chatbot.

The task cascade mechanism solves this without user re-invocation.

### How It Works

**Step 1 — Department emits follow_up_tasks**

Department agents are prompted to include a `follow_up_tasks` JSON block in their response when they need another department to act. The `student_onboarding` agent, after completing enrollment, might include:

```json
{
  "follow_up_tasks": [
    {
      "target_dept": "tutor_management",
      "action": "assign_tutor",
      "context": {"student_id": "S123", "level": "B2", "schedule": "weekday evenings"}
    }
  ]
}
```

**Step 2 — result_aggregator extracts them**

```python
# From nodes.py — result_aggregator
def result_aggregator(state: OrchestratorState) -> dict:
    department_results = state.get("department_results", [])
    completed_tasks = list(state.get("completed_tasks", []))
    follow_ups: list[dict] = []

    for result in department_results:
        tasks = result.get("follow_up_tasks") or []
        follow_ups.extend(tasks)
        completed_tasks.append(result)   # move to audit trail

    return {"task_queue": follow_ups, "completed_tasks": completed_tasks}
```

`result_aggregator` is pure logic — no LLM. It scans every `DepartmentResult` for `follow_up_tasks` lists and gathers them all into `task_queue`. It also moves the current round's results into `completed_tasks` for the final audit trail.

**Step 3 — check_task_queue drives the loop**

```python
# From nodes.py — check_task_queue
def check_task_queue(state: OrchestratorState) -> Command[...]:
    task_queue = list(state.get("task_queue", []))

    if task_queue:
        next_task = task_queue[0]
        remaining = task_queue[1:]
        return Command(
            update={"current_task": next_task, "task_queue": remaining},
            goto="request_classifier",
        )
    else:
        return Command(update={"current_task": None}, goto="compose_output")
```

If the queue is non-empty, it pops the next task into `current_task` and loops back to `request_classifier`. This starts a new full cycle — classify the follow-up task, assess risk, dispatch to the target department, collect results, check queue again.

If the queue is empty, the cascade is complete. Route to `compose_output`.

**Step 4 — request_classifier handles follow-up context**

When `current_task` is set, `request_classifier` uses it as the classification input instead of the raw `request`:

```python
# From nodes.py — request_classifier (simplified)
if current_task:
    follow_up_context = (
        f"Action: {current_task.get('action', '')}\n"
        f"Target department: {current_task.get('target_dept', '')}\n"
        f"Context: {json.dumps(current_task.get('context', {}))}"
    )
```

This ensures the classifier routes the follow-up task to the correct department with all the context the emitting agent provided.

### The Loop Topology

The cascade loop is not a Python `for` loop. It is a graph cycle:

```
check_task_queue (tasks pending)
  → request_classifier
  → risk_assessor
  → dispatch_departments
  → department_executor
  → result_aggregator
  → check_task_queue  (pop next or end)
```

LangGraph supports graph cycles — the graph does not have to be a DAG. The cycle terminates when `check_task_queue` finds an empty queue and routes to `compose_output` instead of back to `request_classifier`.

---

## 6. Tiered Approval (HITL)

### Why Deterministic Rules, Not an LLM

You might expect a sophisticated system to use an LLM to judge whether an action is risky. There are good reasons not to:

1. **Inconsistency**: an LLM may approve the same action on one run and reject it on another. Approval decisions must be reproducible.
2. **Latency**: every request passes through risk assessment. An LLM call here adds 1–3 seconds unconditionally, even for trivially safe operations.
3. **Auditability**: when a compliance officer asks "why was this refund approved automatically?", you need a crisp answer. "The LLM said low risk" is not an answer.

The solution is a deterministic lookup table in `risk.py`:

```python
# From risk.py
HIGH_RISK_ACTIONS: dict[str, set[str]] = {
    "content_pipeline":   {"publish_content"},
    "support":            {"process_refund"},
    "tutor_management":   {"assign_tutor"},
    "quality_assurance":  {"flag_issue"},
    "student_onboarding": {"create_study_plan"},
}
```

The logic is equally simple:

```python
def assess_risk(classification: dict) -> str:
    action_type = classification.get("action_type", "")
    departments = classification.get("departments", [])

    for dept in departments:
        high_risk_set = HIGH_RISK_ACTIONS.get(dept, set())
        if action_type in high_risk_set:
            return "high"

    return "low"
```

If any department marks the action as high-risk, the whole task requires approval. Unknown departments and unknown actions default to low risk. This is a safe default — we only escalate what we *know* is dangerous.

### The Two Tiers

**Tier 1 — Low Risk (auto-execute)**

The `risk_assessor` node returns `risk_level="low"` and `approval_status="not_required"`. The routing function sends the graph directly to `dispatch_departments`. No human involvement. The departments execute immediately.

**Tier 2 — High Risk (human approval required)**

The `risk_assessor` node returns `risk_level="high"` and `approval_status=""` (empty, pending). The routing function sends the graph to `approval_gate`.

### The approval_gate Node

`approval_gate` uses `interrupt()` to pause the graph and surface a structured payload to the caller:

```python
# From nodes.py — approval_gate
def approval_gate(state: OrchestratorState) -> Command[...]:
    classification = state.get("classification", {})

    payload = {
        "department": classification.get("departments", []),
        "action_type": classification.get("action_type", ""),
        "summary": classification.get("summary", "No summary available"),
        "risk_reason": (
            f"Action '{classification.get('action_type', '')}' on department(s) "
            f"{classification.get('departments', [])} requires human approval."
        ),
    }

    operator_decision = interrupt(payload)

    if operator_decision == "approved":
        return Command(goto="dispatch_departments")
    else:
        return Command(update={"approval_status": "rejected"}, goto="compose_output")
```

When `interrupt(payload)` is called:

1. The graph pauses immediately. No further nodes execute.
2. The `payload` dict is returned to the caller in the graph's output.
3. All state is checkpointed (requires a checkpointer — use `InMemorySaver` or `SqliteSaver`).

The caller shows the payload to the human operator and collects a decision. To resume:

```python
# Resume with approval
graph.invoke(Command(resume="approved"), config=config)

# Resume with rejection
graph.invoke(Command(resume="rejected"), config=config)
```

LangGraph restores state from the checkpoint, re-runs `approval_gate` from its top, and when execution reaches `interrupt()`, sees that resume data is available and returns it immediately. The node then routes appropriately.

### Follow-Up Tasks Also Get Risk-Assessed

A critical property of the cascade architecture: because `check_task_queue` loops back to `request_classifier`, every follow-up task passes through `risk_assessor` too. If `student_onboarding` emits a follow-up task for `assign_tutor`, that follow-up will hit the approval gate before the tutor assignment executes.

This means the risk boundary is consistent across the entire cascade, not just the first request. The human approves (or rejects) each risky action individually, regardless of whether it was explicitly requested or autonomously cascaded.

---

## 7. Department Agents

### DeepAgent Factories

Each department agent is created by a factory function in `departments.py`. All six factories share the same structure — only the name, prompt, tools, and SKILL.md files differ:

```python
# From departments.py — create_onboarding_agent
def create_onboarding_agent():
    return create_deep_agent(
        name="student-onboarding",
        model=_MODEL,
        system_prompt=STUDENT_ONBOARDING_PROMPT,
        tools=STUDENT_ONBOARDING_TOOLS,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )
```

`create_deep_agent()` is the DeepAgents API introduced in Project 7. It constructs a fully wired agent with:
- An LLM with tools bound
- A `SkillsMiddleware` that loads SKILL.md files from the given directory
- A storage backend for reading/writing files
- A shared `InMemoryStore` for cross-thread persistent records

### SKILL.md for Domain Knowledge

Each department has a `SKILL.md` file in `skills/<dept-name>/SKILL.md`. This is the DeepAgents mechanism for injecting domain knowledge into an agent's context without hardcoding it in the system prompt. The `SkillsMiddleware` reads these files at agent creation time and appends them to the system context.

The `student-onboarding` SKILL.md, for example, contains LinguaFlow's eligibility rules, placement test scoring bands, and account provisioning checklist. The agent applies this knowledge when deciding which tools to call and in what order.

Using SKILL.md rather than a long system prompt has a practical advantage: you can update the domain knowledge by editing a markdown file without touching Python code.

### The DEPARTMENT_AGENTS Dispatcher

The orchestrator does not import each factory directly. It uses a dispatcher dict:

```python
# From departments.py
DEPARTMENT_AGENTS = {
    "student_onboarding": create_onboarding_agent,
    "tutor_management":   create_tutor_agent,
    "content_pipeline":   create_content_agent,
    "quality_assurance":  create_qa_agent,
    "support":            create_support_agent,
    "reporting":          create_reporting_agent,
}
```

In `department_executor`:

```python
# From nodes.py — department_executor
dept = state.get("_target_dept", "")
agent = DEPARTMENT_AGENTS[dept]()   # call the factory, get a fresh agent
agent_response = agent.invoke({"message": invoke_request})
```

This is the full interface between the orchestrator and the inner DeepAgent layer. The orchestrator only needs to know the department name string — the factory handles all the wiring.

### The DepartmentResult Interface

All six agents return results that are normalised into a `DepartmentResult` TypedDict:

```python
# From models.py
class DepartmentResult(TypedDict):
    department: str              # "student_onboarding" | "support" | ...
    response: str                # Human-readable response text from the agent
    resolved: bool               # Did the agent fully handle its task?
    follow_up_tasks: list[dict]  # Autonomous cascade instructions for other departments
    metrics: dict                # {"actions_taken": int, "tools_called": list[str]}
```

The `follow_up_tasks` field is the key that enables cascading. Without this standardised field in the result contract, the aggregator would have no consistent place to look for cascade instructions.

---

## 8. State Schema Design

### OrchestratorState: A Lifecycle View

The state schema in `models.py` is not just a bag of fields. Each field belongs to a phase of the processing lifecycle, written by a specific node and read by subsequent nodes.

```python
# From models.py
class OrchestratorState(TypedDict):
    # Phase 1 — Set at graph.invoke()
    request: str
    request_metadata: dict              # {"user_id", "priority", "source"}

    # Phase 2 — Set by request_classifier
    classification: dict                # {"departments": list, "action_type": str, ...}
    risk_level: str                     # "low" | "high"

    # Phase 3 — Set by risk_assessor and approval_gate
    approval_status: str                # "approved" | "rejected" | "not_required"

    # Phase 4 — Appended by parallel department_executor branches
    department_results: Annotated[list[DepartmentResult], operator.add]

    # Phase 5 — Managed by check_task_queue cascade loop
    task_queue: list[dict]              # Pending follow-up tasks
    current_task: dict | None           # Task being dispatched right now
    completed_tasks: list[dict]         # Audit trail of finished tasks

    # Phase 6 — Updated by reporting_snapshot (persists across invocations)
    metrics_store: MetricsStore

    # Phase 7 — Set by compose_output
    final_response: str
    resolution_status: str              # "resolved" | "partial" | "pending_approval"
```

Reading the schema tells you the full lifecycle. Each phase's fields are only written during that phase and read afterwards. This ownership discipline prevents accidental coupling between nodes.

### Why department_results Has a Reducer but Other Fields Don't

Only `department_results` needs `operator.add` because it is the only field that multiple parallel branches write to simultaneously. Every other field is written by exactly one node at a time — "last write wins" is fine for them.

Over-applying reducers is a mistake. If you annotated `task_queue` with `operator.add`, then any node that writes `{"task_queue": []}` to clear it would instead *extend* it with an empty list, which does nothing. Reducers change semantics — apply them only where parallel convergence is explicitly needed.

### MetricsStore as a Persistent Dashboard

```python
class MetricsStore(TypedDict):
    students_onboarded: int
    tutors_assigned: int
    content_generated: int
    content_published: int
    qa_reviews: int
    qa_flags: int
    support_requests: int
    support_resolved: int
    total_requests: int
    department_invocations: dict[str, int]
```

`MetricsStore` is part of `OrchestratorState`. When the orchestrator uses `SqliteSaver`, this field persists across graph invocations on the same `thread_id`. Every time `reporting_snapshot` runs, it increments the relevant counters. Over many requests, this builds a running tally of platform activity — a live dashboard without a separate database.

---

## 9. Reporting and Metrics

### Two Levels of Reporting

The system reports at two levels:

**Structured metrics accumulation** (`reporting_snapshot` node): pure logic, runs after every request, increments `metrics_store` counters based on which departments ran and whether they resolved successfully.

**Narrative summary** (`compose_output` node): LLM-generated, synthesises the human-readable response that includes what was done across all departments and any cascaded tasks.

### reporting_snapshot in Detail

```python
# From nodes.py — reporting_snapshot (simplified)
def reporting_snapshot(state: OrchestratorState) -> dict:
    metrics = dict(state.get("metrics_store", {}))
    metrics["total_requests"] = metrics.get("total_requests", 0) + 1

    for result in state.get("department_results", []):
        dept = result.get("department", "")
        resolved = result.get("resolved", False)

        dept_invocations[dept] = dept_invocations.get(dept, 0) + 1

        if dept == "student_onboarding":
            metrics["students_onboarded"] = metrics.get("students_onboarded", 0) + 1
        elif dept == "support":
            metrics["support_requests"] = metrics.get("support_requests", 0) + 1
            if resolved:
                metrics["support_resolved"] = metrics.get("support_resolved", 0) + 1
        # ... other departments ...

    return {"metrics_store": metrics}
```

This node runs unconditionally after every request. It is pure logic — fast, no LLM, no I/O. It uses `dict.get(key, 0)` rather than direct access because `metrics_store` starts as an empty dict and fields are created incrementally.

### compose_output: Task Chain Transparency

`compose_output` receives not just the current request's `department_results` but also the `completed_tasks` from the cascade — the audit trail of everything that was executed autonomously. It injects a task chain summary into the LLM's composition prompt:

```python
# From nodes.py — compose_output
completed = state.get("completed_tasks", [])
if completed:
    task_lines = [
        f"- {t.get('department', 'unknown')}: {t.get('response', '')[:80]}"
        for t in completed
    ]
    task_chain_summary = "\n".join(task_lines)
```

This transparency is important. The user asked for "onboard students and assign tutors." The response should explain that both the onboarding and the tutor assignment were handled — and specifically by which departments. The task chain summary gives the composition LLM the data to produce that explanation.

---

## 10. Persistence

### The Two-Checkpointer Design

LangGraph supports injecting a checkpointer at compile time. The choice of checkpointer determines what survives across invocations.

**Orchestrator — SqliteSaver (or InMemorySaver for tests)**

The master orchestrator uses `SqliteSaver` when run via the Streamlit app. This means:

- `OrchestratorState`, including `metrics_store`, persists across sessions.
- The `approval_gate` interrupt can survive process restarts — you can pause for human approval, shut down the server, restart it, and resume.
- Thread-level isolation via `thread_id` in config: each user session gets its own state.

**Department Agents — InMemorySaver**

Department agents created by `create_deep_agent()` use `InMemorySaver` internally (the default). They do not persist across sessions. This is intentional: department agents are stateless workers. They receive a task, execute it, and return a result. Their internal conversation history (the tool-calling ReAct loop) is not needed once the task is done.

### Why This Split Matters

Persisting both layers would introduce unintended coupling. If a department agent's internal state persisted, you'd need to manage thread IDs at both levels, and a stale checkpoint from a previous partial run might interfere with the current invocation.

The split is clean: the orchestrator remembers everything important (results, metrics, task queue state, approval decisions). The department agents are ephemeral executors. They start fresh each invocation and write their results back to the orchestrator's state.

### CompositeBackend for Agent File Storage

Department agents use a `CompositeBackend` to separate ephemeral working files from persistent records:

```python
# From departments.py
def create_composite_backend():
    def factory(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),          # ephemeral (scoped to this thread)
            routes={
                "/persistent/": StoreBackend(runtime),  # persistent (InMemoryStore)
            },
        )
    return factory
```

When a department agent writes to `/persistent/students/S123.json`, that file is stored in `_store` (an `InMemoryStore` shared across agents). When it writes to `/working/draft.txt`, that file disappears when the thread ends.

This is the `CompositeBackend` pattern from Project 7, now applied to six agents sharing the same `_store`. An onboarding agent can write a student record to `/persistent/students/`, and the tutor management agent — in a separate invocation — can read it, because they share the same store.

---

## 11. LangSmith Observability

### Cross-Agent Tracing

Every node in the orchestrator and every department agent execution is traced in LangSmith. The shared tag `p8-autonomous-operations` ties all traces from a single system invocation together:

```python
# From nodes.py — used on every node function
_TAGS = ["p8-autonomous-operations"]

@traceable(name="request_classifier", run_type="chain", tags=_TAGS)
def request_classifier(state: OrchestratorState) -> dict:
    ...
```

In the LangSmith UI, filtering by `p8-autonomous-operations` shows the full execution trace of any request — including parallel department branches and cascade cycles. Because a single user request may involve 3–4 orchestrator cycles and 6+ department agent invocations, this tag-based grouping is essential for understanding what happened.

### Three Evaluators

`evaluation.py` defines three evaluators and runs them against a fixed dataset.

**1. Routing Accuracy**

Deterministic. Compares the actual departments chosen against the expected departments in the dataset:

```python
def routing_accuracy_evaluator(run, example):
    actual   = set(run.outputs.get("classification", {}).get("departments", []))
    expected = set(example.outputs.get("expected_departments", []))

    if not expected:
        return {"key": "routing_accuracy", "score": 1.0}

    overlap = len(actual & expected)
    score   = overlap / len(expected)
    return {"key": "routing_accuracy", "score": score}
```

A score of 1.0 means all expected departments were selected. 0.5 means half were. The partial scoring is more useful than binary correct/incorrect for debugging the classifier prompt.

**2. Response Quality**

LLM-as-judge. Asks `claude-haiku-4-5-20251001` to rate the final response on coherence, completeness, and professionalism on a 0–1 scale:

```python
def response_quality_evaluator(run, example):
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    # Prompts the model to judge the response and return a numeric score
    ...
```

This evaluator captures what deterministic checks cannot: is the response *good*? Does it address the user's request in a professional tone?

**3. Task Chain Completeness**

Checks whether the completed_tasks audit trail covers all the departments that should have been involved — including cascaded ones:

```python
def task_chain_completeness_evaluator(run, example):
    completed = run.outputs.get("completed_tasks", [])
    completed_depts = {t.get("department") for t in completed}
    expected_chain = set(example.outputs.get("expected_task_chain", []))

    if not expected_chain:
        return {"key": "task_chain_completeness", "score": 1.0}

    overlap = len(completed_depts & expected_chain)
    return {"key": "task_chain_completeness", "score": overlap / len(expected_chain)}
```

This is the evaluator specific to Project 8. It verifies the cascade: a request that should trigger both onboarding and tutor assignment should show both departments in `completed_tasks`.

### Routing vs. Quality vs. Completeness

The three evaluators cover different failure modes:

- **Routing accuracy** catches classifier prompt regressions — if a refund request stops being routed to `support`.
- **Response quality** catches composition prompt regressions — if the final response becomes incoherent or incomplete.
- **Task chain completeness** catches cascade regressions — if follow_up_tasks stop being emitted or processed.

Running all three together gives confidence that the full pipeline is working, not just individual components.

---

## 12. Testing Strategy

### Unit Tests with Mocked LLMs

All tests mock the LLM to avoid real API calls. There are two patching points:

- `nodes._classifier_model` — the module-level model used by `request_classifier`
- `langchain_anthropic.ChatAnthropic` — the class constructor used by `compose_output`

```python
# Typical mock setup in tests/conftest.py
@pytest.fixture
def mock_classifier(monkeypatch):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=json.dumps({
        "departments": ["support"],
        "action_type": "lookup",
        "complexity": "single",
        "summary": "Student support lookup",
    }))
    monkeypatch.setattr("nodes._classifier_model", mock)
    return mock
```

The mock's `.invoke()` returns a `MagicMock` with a `.content` attribute containing pre-baked JSON — exactly as a real LLM would return. The node's parsing logic runs unchanged; it cannot tell the difference.

### Risk Tests Are Exhaustive

`tests/test_risk.py` tests every entry in `HIGH_RISK_ACTIONS` to confirm it returns `"high"`, and a selection of non-listed actions to confirm they return `"low"`:

```python
@pytest.mark.parametrize("dept, action, expected", [
    ("content_pipeline", "publish_content", "high"),
    ("support", "process_refund", "high"),
    ("tutor_management", "assign_tutor", "high"),
    ("quality_assurance", "flag_issue", "high"),
    ("student_onboarding", "create_study_plan", "high"),
    ("support", "lookup", "low"),
    ("reporting", "get_metrics", "low"),
    ("unknown_dept", "unknown_action", "low"),
])
def test_assess_risk(dept, action, expected):
    result = assess_risk({"departments": [dept], "action_type": action})
    assert result == expected
```

These tests are fast (no I/O), fully deterministic, and serve as executable documentation of the risk boundary. If you add a new high-risk action, you add a corresponding test case.

### Mocking DeepAgent Invocations

Tests for `department_executor` mock `DEPARTMENT_AGENTS` to return a fake agent that yields a predictable response:

```python
# From tests/test_nodes.py
def mock_agent_factory():
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "response": '{"follow_up_tasks": [{"target_dept": "tutor_management", "action": "assign_tutor", "context": {}}]}'
    }
    return mock_agent
```

This lets tests verify that `department_executor` correctly extracts `follow_up_tasks` from the agent's response without actually running a DeepAgent. The cascade logic is tested in isolation from the agents.

### Integration Tests for Graph Flows

`tests/test_nodes.py` also contains integration-style tests that compile and invoke the full graph with mocked LLMs. These verify end-to-end flows:

- Single department, low risk: graph runs without interrupting, produces `final_response`
- Single department, high risk: graph interrupts at `approval_gate`, resumes correctly
- Cascade: first cycle emits follow_up_task, second cycle processes it, `completed_tasks` contains both

Integration tests require a checkpointer for the HITL flow:

```python
from langgraph.checkpoint.memory import InMemorySaver
graph = build_graph(checkpointer=InMemorySaver())
```

---

## 13. Code Walkthrough

### The Graph Assembly in graph.py

The full `build_graph()` function reads as a complete, self-documenting topology specification:

```python
# From graph.py — build_graph (abridged)
def build_graph(checkpointer=None):
    graph = StateGraph(OrchestratorState)

    # --- Nodes ---
    graph.add_node("request_classifier", request_classifier)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("approval_gate", approval_gate,
                   ends=["dispatch_departments", "compose_output"])
    graph.add_node("dispatch_departments", _dispatch_pass_through)
    graph.add_node("department_executor", department_executor)
    graph.add_node("result_aggregator", result_aggregator)
    graph.add_node("check_task_queue", check_task_queue,
                   ends=["request_classifier", "compose_output"])
    graph.add_node("compose_output", compose_output)
    graph.add_node("reporting_snapshot", reporting_snapshot)

    # --- Static edges ---
    graph.add_edge(START, "request_classifier")
    graph.add_edge("request_classifier", "risk_assessor")
    graph.add_edge("department_executor", "result_aggregator")
    graph.add_edge("result_aggregator", "check_task_queue")
    graph.add_edge("compose_output", "reporting_snapshot")
    graph.add_edge("reporting_snapshot", END)

    # --- Conditional edges ---
    graph.add_conditional_edges("risk_assessor", route_from_risk,
        path_map={"approval_gate": "approval_gate",
                  "dispatch_departments": "dispatch_departments"})

    graph.add_conditional_edges("dispatch_departments", fan_out_to_departments,
        path_map={"department_executor": "department_executor",
                  "compose_output": "compose_output"})

    return graph.compile(checkpointer=checkpointer)
```

Notice the pattern: first all nodes, then all static edges, then all conditional edges. This structure makes the topology easy to read from top to bottom.

### The check_task_queue Loop

The cascade loop is worth seeing in full, because `Command` with `goto` is the key mechanism:

```python
# From nodes.py — check_task_queue
@traceable(name="check_task_queue", run_type="chain", tags=_TAGS)
def check_task_queue(
    state: OrchestratorState,
) -> Command[Literal["request_classifier", "compose_output"]]:
    task_queue = list(state.get("task_queue", []))

    if task_queue:
        next_task = task_queue[0]
        remaining = task_queue[1:]
        return Command(
            update={"current_task": next_task, "task_queue": remaining},
            goto="request_classifier",
        )
    else:
        return Command(
            update={"current_task": None},
            goto="compose_output",
        )
```

`Command` combines two things: a state update (`update=`) and an explicit routing decision (`goto=`). The `goto="request_classifier"` overrides any static edges from `check_task_queue` — the graph goes exactly where the node says.

This is why `check_task_queue` is registered with `ends=["request_classifier", "compose_output"]`. That `ends=` declaration is LangGraph's compile-time signal that this node may dynamically route to those targets. Without it, the graph compiler would raise a validation error because no `add_edge()` or `add_conditional_edges()` call mentions those targets.

### Invoking the Graph

```python
from graph import build_graph
from langgraph.checkpoint.memory import InMemorySaver

# Build with a checkpointer (required for interrupt/resume at approval_gate)
g = build_graph(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ops-session-001"}}

# First invocation — may interrupt if high-risk
result = g.invoke(
    {
        "request": "Onboard 10 new students and assign tutors",
        "request_metadata": {"user_id": "admin", "priority": "high", "source": "api"},
        "department_results": [],
        "task_queue": [],
        "completed_tasks": [],
        "metrics_store": {},
    },
    config=config,
)

# If approval_gate interrupted:
if result.get("__interrupt__"):
    print("Approval required:", result["__interrupt__"][0].value)
    # Human operator reviews and decides...
    final = g.invoke(Command(resume="approved"), config=config)
    print(final["final_response"])
```

All `OrchestratorState` fields must be initialised at invocation. `department_results: []` is especially important — if omitted, the `operator.add` reducer has nothing to concatenate to and the graph will raise a `TypeError`.

---

## 14. Key Takeaways

### Concept Integration Matrix

| Concept | Introduced | Applied in P8 |
|---------|-----------|--------------|
| LLM chains, ChatPromptTemplate | P1–P2 | `request_classifier`, `compose_output` |
| `StateGraph`, nodes, edges | P2 | The master orchestrator graph in `graph.py` |
| Structured output, JSON parsing | P2–P3 | Classification response parsing in `request_classifier` |
| Tool binding, ReAct loop | P4 | All six department DeepAgents via their tool sets |
| `interrupt` / `Command(resume=)` | P5 | `approval_gate` HITL node |
| `Send()` for parallel fan-out | P6 | `fan_out_to_departments` routing function |
| `Annotated[list, operator.add]` | P6 | `department_results` field in `OrchestratorState` |
| `create_deep_agent()` | P7 | All six department agents in `departments.py` |
| SKILL.md domain knowledge | P7 | `skills/*/SKILL.md` files for each department |
| `CompositeBackend` | P7 | `create_composite_backend()` in `departments.py` |
| SqliteSaver persistence | P5–P6 | Orchestrator checkpointer for metrics + HITL |
| `@traceable` + LangSmith tags | P5–P6 | Every node tagged `p8-autonomous-operations` |
| LangSmith `evaluate()` | P6 | Three evaluators in `evaluation.py` |
| Deterministic risk rules | New | `risk.py` `HIGH_RISK_ACTIONS` table |
| Task queue cascade loop | New | `check_task_queue` cycling back to `request_classifier` |

### The Three Architectural Insights

**Insight 1: Determinism where it matters**

LLMs are good at understanding intent and generating text. They are bad at being consistent gateways for consequential decisions. The risk assessment module (`risk.py`) deliberately avoids the LLM for the approval boundary — a lookup table returns the same answer every time. The LLM handles the ambiguous parts (classification, composition); the rules handle the auditable parts (risk, routing logic).

**Insight 2: Reducers are a contract, not a convenience**

The `operator.add` annotation on `department_results` is not a performance optimisation. It is the *only* reason parallel branches can write to the same field without losing each other's results. It is a contract between the graph designer and LangGraph's state merging mechanism. Apply it precisely — only where parallel convergence is needed — and your state will be predictable. Apply it everywhere and you may hide bugs.

**Insight 3: The cascade loop makes the system autonomous**

The difference between "a chatbot that can call tools" and "an autonomous operations system" is the task queue loop. When `check_task_queue` loops back to `request_classifier`, a single human request can trigger an unbounded chain of department actions, each risk-assessed, each potentially requiring approval, each adding to the completed audit trail. The human submitted one request. The machine executed many.

That loop is twelve lines of Python. Understanding why those twelve lines change the qualitative character of the system — from reactive to autonomous — is the most important lesson of this capstone.

### What Comes After

This is the end of the formal learning path. The eight projects have covered the full LangGraph ecosystem:

- Building graphs from scratch with fine-grained control over every node and edge
- Delegating to DeepAgents when domain complexity warrants a higher abstraction
- Persisting state across sessions at the orchestrator level
- Gating consequential actions behind human approval
- Parallelising independent work with `Send` and collecting results with reducers
- Cascading follow-up work autonomously through a task queue loop
- Measuring system quality continuously with LangSmith evaluation

The next step is to take these patterns to a domain you care about and combine them under the constraints of real requirements — performance budgets, error handling, partial failures, rate limits, and team code review. The fundamentals are here. The rest is engineering.

---

*This document covers Project 08 of the LinguaFlow learning path — the capstone integrating all concepts from P1–P7.*
