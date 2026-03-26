# Project 02: Lesson Plan Generator — LangGraph StateGraph Fundamentals

## 1. Introduction

Project 02 is the first project in this series to use **LangGraph** — the graph-based orchestration layer built on top of LangChain. The LinguaFlow teaching platform needs to generate personalized lesson plans for students. A lesson plan isn't a single LLM call — it's a multi-step pipeline that researches materials, drafts a plan tailored to the lesson type, reviews it for quality, and produces a validated structured output. These steps branch, loop, and converge. That's what LangGraph is built for.

Before running the graph, a short intake conversation (handled in `intake.py`) gathers the student's name, CEFR proficiency level, learning goals, preferred topics, and desired lesson type. Once the student profile is collected, it's handed off to a LangGraph `StateGraph` which runs the full generation pipeline.

This project builds directly on what Project 1 established. Project 1 showed how to construct **LangChain chains** — linear sequences of prompt templates and models connected with the `|` pipe operator. Those chains still appear inside individual node functions here (e.g., `RESEARCH_PROMPT | _model`). But chains alone can't branch: once you call `chain.invoke()`, you get one output and that's it. To branch based on a student's lesson type, loop back for revision, and then finalize a structured result, you need a graph.

---

## 2. From Chains to Graphs — Why LangGraph Exists

To understand why LangGraph exists, start with what a chain is good at:

```python
# From Project 1 — a simple linear chain
chain = CORRECTION_PROMPT | model | StrOutputParser()
result = chain.invoke({"text": user_input})
```

A chain is a **pipeline**: input flows in one end, output comes out the other. Every run follows exactly the same path. This works perfectly for single-turn tasks like grammar correction.

Lesson plan generation is fundamentally different. Consider the steps:

1. Research materials for the student's profile
2. Draft a lesson plan — but the *type* of draft depends on the lesson type (conversation? grammar? exam prep?)
3. Review the draft — if it's not good enough, send it back for revision
4. Once approved (or after 2 revision attempts), finalize into a structured output

Step 2 requires **branching**: three different drafting paths, selected at runtime. Step 3 requires **looping**: execution can flow backward from review to draft and run again. Neither of these is expressible in a simple LangChain chain.

LangGraph solves this by modeling your LLM application as a **directed graph**:
- **Nodes** are the processing steps (functions that read state and return updates)
- **Edges** are the connections between steps (static or conditional)
- **State** is a shared data structure that flows through the entire graph, updated by each node as it executes

The graph engine handles the orchestration: it calls nodes in the correct order, routes execution along the right edges, and manages the shared state. You define the topology; LangGraph runs it.

---

## 3. LangGraph StateGraph Fundamentals

### The State Schema

Every LangGraph graph has a **state schema** — a `TypedDict` that defines the shared data structure the graph operates on. From `models.py`:

```python
class LessonPlanState(TypedDict):
    student_profile: StudentProfile
    research_notes: str
    draft_plan: str
    review_feedback: str
    revision_count: int
    is_approved: bool
    final_plan: LessonPlan | None
```

This `TypedDict` is the contract between every node in the graph. Every node receives the **full** current state and can read any field. Every node returns a **partial dict** containing only the fields it updates. The graph engine merges those updates into the running state using "last write wins" semantics — the new value replaces the old one.

The fields tell the story of what the graph needs to track:
- `student_profile` — set before the graph starts, read by all nodes
- `research_notes` — written by the research node, read by draft nodes
- `draft_plan` — written by whichever draft node runs, read by review and finalize
- `review_feedback` — written by review, read by draft nodes on revision cycles
- `revision_count` — incremented by review, read by the routing logic
- `is_approved` — written by review, read by the post-review routing function
- `final_plan` — written by finalize, the end result of the entire graph

### Nodes — Functions That Return Partial State

A node is just a Python function. It receives the full `LessonPlanState` and returns a `dict` containing only the fields it wants to update. The function signature is always:

```python
def my_node(state: LessonPlanState) -> dict:
    # ... do work ...
    return {"field_name": new_value}
```

LangGraph calls this function when execution reaches that node, then merges the returned dict into the graph's state. The key insight is that nodes return **partial updates** — they don't need to return the entire state, only the fields they change. This keeps each node focused on one responsibility.

### Edges — Connecting Nodes

There are two kinds of edges in LangGraph:

**Static edges** (`add_edge`) always route to the same destination. They express "after this node, always go to that node":

```python
workflow.add_edge(START, "research")
workflow.add_edge("draft_conversation", "review")
workflow.add_edge("finalize", END)
```

**Conditional edges** (`add_conditional_edges`) call a routing function that inspects the current state and returns the name of the next node to execute. They express "after this node, go to one of these nodes, depending on the state":

```python
workflow.add_conditional_edges(
    "research",
    route_by_lesson_type,
    {
        "draft_conversation": "draft_conversation",
        "draft_grammar": "draft_grammar",
        "draft_exam_prep": "draft_exam_prep",
    },
)
```

The third argument is a **path map**: it translates return values from the routing function into actual node names. This is an optional but recommended practice — it makes the graph's routing options explicit and allows for better visualization.

`START` and `END` are special sentinel nodes imported from `langgraph.graph`. `START` is the entry point; `END` signals that the graph should terminate.

### Building and Compiling the Graph

You build a graph by creating a `StateGraph` instance, adding nodes and edges, then calling `compile()`:

```python
workflow = StateGraph(LessonPlanState)

workflow.add_node("research", research_node)
# ... add more nodes ...

workflow.add_edge(START, "research")
# ... add more edges ...

graph = workflow.compile()
```

`compile()` does two important things:
1. **Validates the topology** — it checks that every node is reachable, all edge destinations exist, and the graph has at least one path from `START` to `END`
2. **Returns an executable object** — the `CompiledGraph` supports `.invoke()`, `.stream()`, and `.ainvoke()` (async)

You cannot modify the graph after `compile()`. The compiled graph is immutable and thread-safe.

---

## 4. The Graph Walkthrough

Here is a node-by-node walkthrough of the full pipeline, following the execution path for a typical run.

### research_node

The graph always starts here. It reads the student profile from state, builds a LangChain chain from a prompt template and the Claude model, invokes it, and returns the research notes:

```python
@traceable(name="research", run_type="chain")
def research_node(state: LessonPlanState) -> dict:
    profile = state["student_profile"]
    chain = RESEARCH_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "lesson_type": profile.lesson_type,
    })
    return {"research_notes": result.content}
```

Notice that the `RESEARCH_PROMPT | _model` pattern is identical to Project 1's chain construction — nodes are where LangChain chains live inside a LangGraph graph. The node is a LangGraph concept (a callable that participates in graph routing); the chain inside it is a LangChain concept (a pipeline that makes an LLM call).

The `@traceable` decorator is from LangSmith. It wraps the function in a traced span so you can see its inputs and outputs in the LangSmith dashboard.

### route_by_lesson_type (conditional routing)

After research completes, the graph calls `route_by_lesson_type` to decide which drafting node to execute:

```python
def route_by_lesson_type(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep"
]:
    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"
```

This is a **pure routing function** — it reads state and returns a string. It does not call any LLM; it is simple Python logic. LangGraph calls it with the current state after the research node completes and uses the return value to select the next node.

The return type annotation (`Literal["draft_conversation", "draft_grammar", "draft_exam_prep"]`) documents the possible destinations and serves as a hint to LangGraph's type checking.

### draft_* nodes (three variants)

There are three specialized drafting nodes: `draft_conversation_node`, `draft_grammar_node`, and `draft_exam_prep_node`. They all follow the same structure, differing only in which prompt template they use. Here is the conversation variant:

```python
@traceable(name="draft_conversation", run_type="chain")
def draft_conversation_node(state: LessonPlanState) -> dict:
    profile = state["student_profile"]
    chain = DRAFT_CONVERSATION_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}
```

A few things to notice:

**`_creative_model` vs `_model`**: The draft nodes use a model configured with `temperature=0.3` rather than `temperature=0`. This is intentional — drafting creative lesson content benefits from some variation, while structured tasks like review and finalization are better served by deterministic output.

**`revision_context`**: The helper `_build_revision_context()` checks whether `revision_count > 0` and, if so, prepends the reviewer's feedback to the prompt. This is how the draft node "learns" from rejection — on a second pass, the LLM sees its previous output was rejected and what the specific criticism was.

**All three nodes write to `draft_plan`**: This is important. Only one draft node executes per run (based on routing), and all three update the same `draft_plan` field. The review and finalize nodes don't need to know which drafting node ran — they just read `draft_plan`.

### review_node

The review node is the quality gate. It calls an LLM to critique the draft plan and decides whether to approve it:

```python
@traceable(name="review", run_type="chain")
def review_node(state: LessonPlanState) -> dict:
    profile = state["student_profile"]
    chain = REVIEW_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })

    response_text = result.content
    is_approved = response_text.strip().startswith("APPROVED")

    return {
        "is_approved": is_approved,
        "review_feedback": response_text,
        "revision_count": state["revision_count"] + 1,
    }
```

The review node updates three fields:
- `is_approved` — a boolean derived from a simple text heuristic (does the response start with "APPROVED"?)
- `review_feedback` — the full reviewer response, stored for the draft node to use if revision is needed
- `revision_count` — incremented on every review pass (used by the routing logic to cap revisions at 2)

The approval detection (`response_text.strip().startswith("APPROVED")`) is deliberately simple. The REVIEW_PROMPT instructs the model to begin its response with either "APPROVED" or "REVISION REQUESTED". This structured response format makes parsing trivial without needing `.with_structured_output()`.

### route_after_review (the cycle)

After review, the graph calls `route_after_review` to decide what happens next:

```python
def route_after_review(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep", "finalize"
]:
    if state["is_approved"]:
        return "finalize"

    if state["revision_count"] >= 2:
        return "finalize"

    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"
```

This function creates the **cycle** in the graph. When it returns the name of a drafting node, execution flows backward — the draft node runs again with the feedback in state, then proceeds back to review. This is the same execution path as the first pass, but with `revision_context` now populated.

The `revision_count >= 2` guard is the safety valve. Without it, a draft node that consistently fails review would loop forever. With it, the graph gives up and finalizes the best-effort draft after 2 revision cycles.

### finalize_node

The finalize node converts the free-text `draft_plan` into a validated `LessonPlan` Pydantic model:

```python
@traceable(name="finalize", run_type="chain")
def finalize_node(state: LessonPlanState) -> dict:
    profile = state["student_profile"]
    structured_model = _model.with_structured_output(
        LessonPlan, method="json_schema"
    )
    chain = FINALIZE_PROMPT | structured_model
    plan = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })
    return {"final_plan": plan}
```

`_model.with_structured_output(LessonPlan, method="json_schema")` is a LangChain feature from Project 1. Here it converts the raw draft text into a fully validated `LessonPlan` instance with typed fields. After this node, `state["final_plan"]` holds the final output, and the graph terminates at `END`.

---

## 5. Conditional Routing — Deep Dive

Conditional routing is one of LangGraph's most powerful features. It lets the graph make runtime decisions based on the current state, turning a static pipeline into a dynamic one.

### How `add_conditional_edges()` Works

```python
workflow.add_conditional_edges(
    "research",           # The source node
    route_by_lesson_type, # The routing function
    {                     # The path map
        "draft_conversation": "draft_conversation",
        "draft_grammar": "draft_grammar",
        "draft_exam_prep": "draft_exam_prep",
    },
)
```

When the `research` node completes, LangGraph calls `route_by_lesson_type(current_state)`. The return value is looked up in the path map. The graph then routes execution to the matching destination node.

The path map serves two purposes:
1. **Explicit documentation** of the possible routing outcomes — at a glance you can see all the destinations
2. **Indirection** — the routing function returns a logical name (e.g., `"draft_conversation"`), and the path map translates it to the actual node name registered with `add_node()`. When node names change, you only update the path map, not the routing function.

### Routing Function Design

Routing functions should be **pure functions** — they read state and return a string, with no side effects. They should not make LLM calls, modify state, or raise exceptions. Keep them simple.

The return type annotation using `Literal` is a best practice. It documents what values the function can return and enables static analysis:

```python
def route_by_lesson_type(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep"
]:
```

### Three-Way Branch

The three-way split after research — `draft_conversation`, `draft_grammar`, `draft_exam_prep` — demonstrates a fan-out pattern. Three separate nodes exist because each lesson type needs a distinct prompt and different creative guidance. Rather than cramming all three into one node with an if-else, separate nodes keep each node's responsibility clean.

After the draft, all three paths converge back to review with a static edge from each:

```python
workflow.add_edge("draft_conversation", "review")
workflow.add_edge("draft_grammar", "review")
workflow.add_edge("draft_exam_prep", "review")
```

This fan-out-fan-in pattern (branch out to specialized nodes, converge back to a shared node) is a common and idiomatic LangGraph pattern.

---

## 6. The Review Loop — Graph Cycles

A **graph cycle** is any path through the graph that visits the same node more than once. In this project, the cycle is:

```
draft_* → review → draft_* → review → finalize
```

Cycles are what make LangGraph fundamentally different from LangChain chains and most other LLM orchestration frameworks. A chain is a DAG (directed acyclic graph) — no cycles allowed. LangGraph's graph engine explicitly supports cycles, which enables iterative refinement, retry logic, and human-in-the-loop approval workflows.

### How the Cycle Executes

On the first pass, the draft node runs, then review runs. If the reviewer approves, `is_approved = True` and routing sends execution to `finalize`. The graph terminates normally.

If the reviewer rejects, `is_approved = False` and `revision_count` is 1. The routing function returns the draft node name again. LangGraph routes execution back to that draft node, which now sees `revision_count = 1` and `review_feedback` populated in state. The `_build_revision_context()` helper builds a context string from this feedback and prepends it to the prompt, so the LLM knows what to fix:

```python
def _build_revision_context(state: LessonPlanState) -> str:
    if state["revision_count"] > 0 and state["review_feedback"]:
        return (
            f"REVISION REQUESTED (attempt {state['revision_count']}):\n"
            f"Reviewer feedback: {state['review_feedback']}\n\n"
            "Please address the feedback and improve the lesson plan.\n\n"
        )
    return ""
```

### The Infinite Loop Guard

Any cycle needs a termination condition. Without one, a review node that always rejects would produce an infinite loop. The guard in `route_after_review` prevents this:

```python
if state["revision_count"] >= 2:
    return "finalize"
```

`revision_count` is incremented by the review node on every pass. After 2 reviews, the routing function always returns `"finalize"` regardless of `is_approved`. The graph terminates with the best draft produced.

This pattern — a counter field in state, incremented by the looping node, checked by the routing function — is the standard way to cap cycles in LangGraph. It's simple, transparent, and easy to adjust (change `2` to a different limit in one place).

### Why Cycles Are Powerful

The review loop pattern is applicable to a wide range of real problems:

- **Quality control** — generate, evaluate, revise until a quality bar is met
- **Tool use retries** — attempt a tool call, check the result, retry with a different approach if it failed
- **Self-correction** — have an LLM check its own output and fix errors
- **Human-in-the-loop** (covered in Project 5) — pause the cycle at review to wait for a human decision

In all these cases, the pattern is the same: a generating node, a checking node, a routing function that decides whether to loop or continue. LangGraph handles the bookkeeping.

---

## 7. Streaming Graph Execution

LangGraph supports streaming, which allows you to observe the graph's progress as it runs rather than waiting for the final result. The CLI in `main.py` uses `stream_mode="updates"`:

```python
for chunk in graph.stream(initial_state, stream_mode="updates"):
    for node_name in chunk:
        label = node_labels.get(node_name, node_name)
        print(f"  [{node_name}] {label}")

        if node_name == "review":
            node_output = chunk[node_name]
            if not node_output.get("is_approved", False):
                count = node_output.get("revision_count", 0)
                if count < 2:
                    print(f"  [review] Requesting revision (attempt {count}/2)...")
```

### How `stream_mode="updates"` Works

`stream_mode="updates"` emits one chunk per node as each node completes. Each chunk is a dict with a single key — the node name — and the value is the partial state update that node returned.

For example, when the research node finishes, the stream emits:

```python
{"research": {"research_notes": "... suggested materials ..."}}
```

When the review node finishes:

```python
{"review": {"is_approved": False, "review_feedback": "REVISION REQUESTED...", "revision_count": 1}}
```

This makes it easy to show progress in the CLI: as each chunk arrives, you know which node just completed and what it produced.

### Other Streaming Modes

LangGraph supports other streaming modes too:

- `stream_mode="values"` — emits the full state snapshot after each node. Useful when you want to inspect the complete state at each step.
- `stream_mode="messages"` — streams individual LLM token chunks for real-time typing effects in chat interfaces.

For a background pipeline like this project's lesson plan generator, `"updates"` is the right choice — you want to know when each stage finishes, not every token.

---

## 8. LangSmith Graph Traces

LangSmith automatically traces LangGraph graph executions. When you run `main.py` with `LANGSMITH_API_KEY` set, every graph invocation creates a trace in LangSmith that shows the full execution path.

### What LangSmith Records for Graphs

For a LangGraph run, LangSmith creates a **hierarchical trace**:

- The top-level run represents the full graph invocation
- Each node that executed appears as a child span, in order
- Inside each node span, any LangChain chains (the `PROMPT | model` calls) appear as nested spans
- If a node is decorated with `@traceable`, it appears as an explicitly named span

The `nodes.py` file decorates every node with `@traceable`:

```python
@traceable(name="research", run_type="chain")
def research_node(state: LessonPlanState) -> dict:
    ...
```

This creates named spans with meaningful labels. Without `@traceable`, LangGraph still traces the nodes (it knows their names from `add_node()`), but the `@traceable` decorator gives you more control over the span name and run type.

### Reading a Graph Trace

When you open a trace in LangSmith for this project, you'll see:

1. **The graph root span** — shows total execution time, input state, output state
2. **Node spans in execution order** — for a typical run: `research → draft_grammar → review → finalize`
3. **LLM call spans inside each node** — the actual API calls to Claude, with prompt/response
4. **If a revision loop occurred** — the node spans repeat: `draft_grammar → review → draft_grammar → review → finalize`

The trace makes it easy to diagnose problems. If the review loop fires unexpectedly, you can click into the review node span and read the exact `review_feedback` value to understand what the LLM objected to. If the final lesson plan looks wrong, you can inspect the `draft_plan` field entering `finalize` to see whether the problem originated in drafting or in structured output parsing.

### What to Look For

When reviewing graph traces for this project:

- **Did the correct draft node run?** Check that the routing function selected the right branch for the lesson type.
- **How many review iterations occurred?** Count the review node spans. One means the first draft was approved. Two means one revision was needed.
- **What was the review feedback on rejection?** Click the review node span, look at the LLM response. This tells you whether the prompt needs tuning.
- **Did finalize produce a valid LessonPlan?** Look at the output of the finalize span. If structured output parsing failed, you'll see an error here.

LangSmith is especially valuable with graph-based applications because the execution path is non-deterministic (different students trigger different branches). The trace tells you exactly which path executed and why.

---

## 9. Key Takeaways

### Concepts Introduced in Project 2

**TypedDict state schemas** are the foundation of every LangGraph graph. The state schema is the contract between nodes — it defines what data exists, what each node can read, and what each node is responsible for updating. Designing a clear, well-named state schema is one of the most important steps in building a LangGraph application.

**Nodes as partial state updaters** is the core pattern. Nodes don't own the state — they receive it, do one focused job, and return only the fields they changed. This keeps nodes small, testable, and composable.

**Conditional routing** with `add_conditional_edges()` is how graphs make runtime decisions. Routing functions are simple Python: read state, return a string, no side effects. The path map makes routing options explicit. Together, they replace complex imperative branching logic with declarative graph topology.

**Graph cycles** are LangGraph's superpower over chains. Any iterative refinement pattern — generate, evaluate, revise — maps cleanly to a cycle. The key is always including a termination condition: a counter or flag in state, checked by the routing function.

**`compile()` and streaming** round out the execution model. Compile validates the graph and produces an executable object. Streaming with `stream_mode="updates"` lets you observe progress node by node — important for both user experience (showing progress in the CLI) and debugging (knowing which node produced unexpected output).

### LangChain Inside LangGraph

An important mental model: **LangChain and LangGraph are complementary layers**. Inside every node in this project, there is still a LangChain chain (`PROMPT | model`). LangChain handles the LLM call. LangGraph handles the orchestration — deciding which node runs, in what order, and what state it receives.

Project 1 built chains. Project 2 showed how to orchestrate chains into a graph. Going forward, the projects will keep building on both layers: more sophisticated chains (RAG, tool use) inside increasingly complex graph topologies.

### What Carries Forward to Project 3

Project 3 introduces **RAG (Retrieval-Augmented Generation)** and **tool use**. The graph topology from this project carries directly forward — nodes, edges, conditional routing, and cycles are all reused. What changes is what happens *inside* the nodes: instead of just calling a prompt + model chain, nodes will retrieve documents from a vector store and give the model access to external tools.

The state schema will also grow: it will need to track retrieved documents, tool call history, and intermediate reasoning steps. The `TypedDict` pattern from this project scales naturally to accommodate these additions.

---

## Summary

| Concept | Where It Appears |
|---|---|
| `TypedDict` state schema | `models.py` — `LessonPlanState` |
| `StateGraph` + `add_node()` | `graph.py` — `build_graph()` |
| Static edges `add_edge()` | `graph.py` — research→START, draft→review, finalize→END |
| Conditional routing `add_conditional_edges()` | `graph.py` — after research, after review |
| Graph cycles | `graph.py` — review → draft loop via `route_after_review` |
| Cycle termination guard | `graph.py` — `revision_count >= 2` check |
| Partial state updates | `nodes.py` — every node returns only its fields |
| LangChain chains inside nodes | `nodes.py` — `PROMPT | model` inside each node function |
| `@traceable` for LangSmith | `nodes.py` — decorator on every node |
| `stream_mode="updates"` | `main.py` — progress display in the CLI |
| Structured output at graph boundary | `nodes.py` — `finalize_node` using `.with_structured_output()` |
