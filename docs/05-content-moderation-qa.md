# Project 05: Content Moderation & QA

## A Deep Dive into Human-in-the-Loop, Error Handling, and LangSmith Evaluation

---

## 1. Introduction

### What We Built

This project builds a content moderation pipeline for LinguaFlow's editorial team. A content request arrives specifying a topic, content type, and CEFR difficulty level. An LLM generates a lesson snippet. Before that content ever reaches students, it passes through two mandatory human review checkpoints — a draft review and a final publication gate. If the draft is rejected, the LLM revises it based on feedback. If the polished version passes both reviews, it is published. If it fails, it is killed.

The central challenge this project solves is pausing an automated graph to wait for a human, then resuming from exactly where it left off. This is qualitatively different from everything in Projects 1-4. Previous projects were fully automated: you invoked the graph, it ran to completion, it returned a result. Here, the graph deliberately stops mid-execution, hands control back to a human, and only continues when the human provides a decision.

### The Shift from Project 4

Project 4's conversational tutor ran in a continuous message loop — every turn was a fresh invocation of the same chain with accumulated history. The human was always in control; the LLM was reactive.

Project 5 inverts this. The LLM drives the pipeline; the human is an *approver* at specific checkpoints, not a participant in every exchange. The graph runs autonomously until it hits an interrupt point, pauses, and waits. This is closer to how real production workflows operate — automated processing with human oversight at high-stakes decision points.

### The Graph Structure

```
START
  │
  ▼
generate          ← LLM creates lesson snippet
  │
  ▼
draft_review      ← INTERRUPT: moderator sees draft, decides approve/edit/reject
  │
  ├─ approve/edit → polish → final_review → INTERRUPT: moderator approves or rejects
  │                                │
  │                                ├─ approve → publish → END
  │                                └─ reject  → END
  │
  └─ reject (revisions remain) → revise → draft_review  (revision loop)
  └─ reject (max revisions)    → END
```

Two interrupt points, one revision loop, and a hard limit of two revision rounds create a workflow that is automated where possible and human-supervised where it matters.

---

## 2. Human-in-the-Loop with `interrupt()`

### What `interrupt()` Does

`interrupt()` is a LangGraph function that stops graph execution at the current node and serializes the graph's full state to the checkpointer. The graph process returns immediately — from the caller's perspective, the `graph.invoke()` call returns (with a special interrupted status) without completing. The state is frozen in the checkpoint store, waiting for a human decision.

```python
# From nodes.py — draft_review_node
def draft_review_node(state: ContentModerationState) -> dict:
    decision = interrupt({
        "content": state["draft_content"],
        "confidence": state["generation_confidence"],
        "revision_count": state["revision_count"],
        "prompt": "Review this draft. Approve, edit, or reject with feedback.",
    })

    updates = {"draft_decision": decision}
    if decision.get("action") == "edit" and decision.get("edited_content"):
        updates["draft_content"] = decision["edited_content"]

    return updates
```

The dict passed to `interrupt()` is the *payload* — the information shown to the human reviewer. It can contain anything: content, metadata, context. The call to `interrupt()` does not return immediately. It suspends execution. The function body after `interrupt()` only runs *after* the human resumes the graph.

When `interrupt()` eventually "returns", its return value is the `resume` data provided by `Command(resume=...)`. In `draft_review_node`, that return value is the moderator's decision dict — the `decision` variable.

### How `Command(resume=...)` Works

To resume a suspended graph, the caller invokes it again with a `Command` object instead of regular input:

```python
from langgraph.types import Command

# Resume with the moderator's decision
graph.invoke(
    Command(resume={"action": "approve"}),
    config={"configurable": {"thread_id": "some-thread-id"}},
)
```

The `thread_id` in the config is critical — it identifies which suspended execution to resume. The checkpointer looks up the saved state for that thread, restores the graph to exactly the node that called `interrupt()`, and re-runs that node from the top. This time, when execution reaches the `interrupt()` call, LangGraph detects that there is pending resume data and returns it immediately rather than suspending again. The node completes, writes `draft_decision` to state, and the graph continues to the routing function.

### The Re-execution Rule and Idempotency

This "re-run the node from the top" behavior is the most important thing to understand about `interrupt()`. When the graph resumes, the entire node function executes again — including any code before the `interrupt()` call. If you made an LLM call or an API call before `interrupt()`, it would run again on resume.

This is why `draft_review_node` and `final_review_node` contain *nothing* before the `interrupt()` call. They collect the relevant state fields, call `interrupt()`, and then process the result. There is no side-effectful code above the interrupt that could run twice.

If you ever need to do work before an interrupt (for example, sending a notification email), either:
1. Put that work in the previous node (so it runs once and is checkpointed), or
2. Make the operation idempotent so re-running it is harmless.

### The Two Interrupt Nodes Compared

`draft_review_node` uses a three-way interrupt with a revision loop. The moderator can approve, edit directly, or reject with feedback. `final_review_node` is simpler — binary decision only:

```python
# From nodes.py — final_review_node
def final_review_node(state: ContentModerationState) -> dict:
    decision = interrupt({
        "content": state["polished_content"],
        "prompt": "Final review. Approve for publication or reject.",
    })
    return {"final_decision": decision}
```

No edited content path, no revision loop — just approve or kill. This mirrors how real editorial pipelines work: early review is collaborative and iterative; the final gate is binary.

---

## 3. Approval Workflows

### Three-Way Decision at Draft Review

The routing function after `draft_review` handles three outcomes:

```python
# From nodes.py — route_after_draft_review
def route_after_draft_review(
    state: ContentModerationState,
) -> Literal["polish", "revise", "__end__"]:
    action = state["draft_decision"].get("action", "reject")

    if action in ("approve", "edit"):
        return "polish"

    # Reject — check revision budget
    if state["revision_count"] >= MAX_REVISIONS:
        return "__end__"

    return "revise"
```

Both `approve` and `edit` route to `polish` — the distinction is already resolved in `draft_review_node`, which overwrites `draft_content` with `edited_content` when the action is `edit`. By the time `route_after_draft_review` runs, the state already reflects the moderator's edits. The routing function only cares about the branch, not the editing detail.

### The Revision Loop and Max Revision Guard

`revise` → `draft_review` is a cycle in the graph. This is explicitly wired with `.add_edge("revise", "draft_review")`. The `revision_count` field in state tracks how many times the LLM has revised:

```python
# From nodes.py — revise_node
return {
    "draft_content": parsed.get("content", response.content),
    "generation_confidence": float(parsed.get("confidence", 0.5)),
    "revision_count": state["revision_count"] + 1,
}
```

The guard in `route_after_draft_review` checks `revision_count >= MAX_REVISIONS` (where `MAX_REVISIONS = 2`). If the moderator keeps rejecting and the budget is exhausted, the graph routes to `__end__` rather than looping again. Without this guard, a sufficiently demanding moderator could loop the graph indefinitely. The guard converts an infinite loop into a graceful terminal failure.

### Binary Final Decision

After `polish`, the `route_after_final_review` function has no loop and no revision path:

```python
# From nodes.py — route_after_final_review
def route_after_final_review(
    state: ContentModerationState,
) -> Literal["publish", "__end__"]:
    action = state["final_decision"].get("action", "reject")
    if action == "approve":
        return "publish"
    return "__end__"
```

`publish` marks `published: True` and records metadata. `__end__` quietly terminates. The final review is a gate, not a collaboration.

---

## 4. 4-Tier Error Handling

### The Framework

LangGraph supports four distinct strategies for handling errors in a graph. This project uses three of them, deliberately, at different layers:

| Tier | Mechanism | What It Handles |
|------|-----------|-----------------|
| 1 | `RetryPolicy` | Transient infrastructure errors (LLM API timeouts, rate limits) |
| 2 | `ToolNode` error handling | Tool call exceptions (not used in this project) |
| 3 | `interrupt()` | User-fixable issues that need human judgment |
| 4 | Bubble up | Unexpected errors that should crash loudly |

### Tier 1: RetryPolicy on LLM Nodes

LLM APIs fail transiently. A `503` response, a rate limit, a network dropout — these are temporary and should be retried automatically. LangGraph's `RetryPolicy` handles this at the node level:

```python
# From graph.py
_llm_retry = RetryPolicy(max_attempts=3)

graph = (
    StateGraph(ContentModerationState)
    .add_node("generate", generate_node, retry_policy=_llm_retry)
    .add_node("revise", revise_node, retry_policy=_llm_retry)
    .add_node("polish", polish_node, retry_policy=_llm_retry)
    ...
)
```

`RetryPolicy` is attached per-node, not globally. The interrupt nodes (`draft_review`, `final_review`) and `publish` do not get a retry policy — there is nothing to retry in them. Only the three LLM-calling nodes that can fail due to API issues receive it.

When a node covered by `RetryPolicy` raises an exception, LangGraph catches it and re-runs the node up to `max_attempts` times before allowing the exception to propagate. The caller sees a clean failure after three genuine attempts rather than a spurious failure on the first transient error.

### Tier 3: interrupt() as a Human Error Handler

`interrupt()` is listed as Tier 3 because it handles a class of issues that are neither transient infrastructure failures (Tier 1) nor code bugs (Tier 4). They are *quality* issues: content that is wrong, off-topic, or inappropriate. These cannot be handled by retrying the LLM and should not crash the system — they need human judgment.

By placing `interrupt()` at the draft review stage, the pipeline turns a potential quality failure into a structured decision point. The moderator is the error handler for content quality.

### Tier 4: Unexpected Errors Bubble Up

Anything not caught by Tiers 1-3 propagates as a normal Python exception. In this project, that includes `json.JSONDecodeError` from `_parse_json_response` (if the LLM returns non-JSON), `KeyError` from missing state fields, and any other unexpected runtime failure.

The deliberate choice not to catch everything is important. Swallowing unexpected errors hides bugs. A crash with a stack trace is more useful than a silent failure that produces a nonsensical state. Tier 4 isn't a tier you implement — it's the tier you leave in place by not over-engineering error handling.

---

## 5. LangSmith Evaluation Deep Dive

### Why Automated Evaluation?

The moderation pipeline has human review built in — but that review only happens after content is generated. The more interesting question is: is the *generator* producing good content in the first place? And is it doing so consistently across different topics and difficulty levels? Answering this requires running the generator against many inputs and measuring the results — which is exactly what LangSmith's evaluation framework provides.

### Creating a Dataset

LangSmith datasets are collections of input/output pairs stored in the LangSmith backend. You create them once, then run multiple experiments against the same fixed examples to get comparable results:

```python
# From evaluation.py — create_dataset()
client = Client()

dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="Content generation evaluation dataset for P5",
)

for request in SAMPLE_REQUESTS:
    client.create_example(
        inputs=request,
        outputs={},  # No reference outputs — evaluators are LLM-as-judge
        dataset_id=dataset.id,
    )
```

Notice that `outputs={}` — there are no reference answers. This is appropriate for a generative task where "correct" output is subjective and impossible to enumerate. Instead of comparing to a gold standard, we use *evaluators* that assess quality directly.

### Custom Evaluators: LLM-as-Judge

LangSmith evaluators are plain functions with the signature `(run, example) -> dict`. The `run` object contains the actual outputs; `example` contains the original inputs (and any reference outputs you provided). Your evaluator returns a dict with at minimum a `key` and a `score`:

```python
# From evaluation.py — topic_relevance_evaluator
def topic_relevance_evaluator(run, example) -> dict:
    content = run.outputs.get("content", "")
    topic = example.inputs.get("topic", "")

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, how relevant is this content to the topic '{topic}'?\n\n"
        f"Content:\n{content}\n\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    parsed = json.loads(response.content.strip())
    score = float(parsed.get("score", 0.5))
    return {"key": "topic_relevance", "score": score, "comment": f"Topic: {topic}"}
```

The LLM judge is asked to score on a 0-1 scale and respond with structured JSON. Using `temperature=0` for the evaluator model is important — you want deterministic, consistent judgments, not creative variation.

This project has three evaluators: `topic_relevance_evaluator`, `difficulty_match_evaluator`, and `content_quality_evaluator`. Each probes a different dimension of content quality. Splitting dimensions into separate evaluators (rather than asking for one combined score) makes results more actionable — you can see whether a low-scoring run failed on topic relevance, difficulty calibration, or general quality.

### Running Evaluations with `evaluate()`

LangSmith's `evaluate()` function runs your target function against every example in a dataset and applies your evaluators to each result:

```python
# From evaluation.py — run_evaluation()
results = evaluate(
    generate_for_eval,
    data=DATASET_NAME,
    evaluators=[
        topic_relevance_evaluator,
        difficulty_match_evaluator,
        content_quality_evaluator,
    ],
    experiment_prefix="p5-content-eval",
    metadata={"model": "claude-haiku-4-5-20251001", "version": "1.0"},
)
```

`generate_for_eval` is a thin wrapper around the generation chain that bypasses the full graph (no interrupts, no human review). This is intentional — you want to evaluate the generator in isolation, not the entire moderation workflow. For automated evaluation, human-in-the-loop steps need to be bypassed.

`experiment_prefix` names the experiment run in the LangSmith dashboard. Each call to `evaluate()` creates a new experiment — you can see how scores change across runs as you iterate on prompts or models. `metadata` attaches arbitrary key-value data to the experiment for filtering and comparison.

---

## 6. A/B Prompt Comparison

### The Pattern

One of the most common questions in LLM development is: "Which prompt is better?" A/B testing answers this by running two variants against the same dataset and comparing their evaluator scores.

`ab_comparison.py` defines two target functions — one using a structured prompt (`GENERATE_PROMPT_STRUCTURED`, `temperature=0.3`) and one using a creative prompt (`GENERATE_PROMPT_CREATIVE`, `temperature=0.7`) — and runs them both through `evaluate()`:

```python
# From ab_comparison.py — run_ab_comparison()
results_a = evaluate(
    generate_structured,
    data=DATASET_NAME,
    evaluators=evaluators,
    experiment_prefix="p5-ab-structured",
    metadata={"variant": "structured", "temperature": 0.3},
)

results_b = evaluate(
    generate_creative,
    data=DATASET_NAME,
    evaluators=evaluators,
    experiment_prefix="p5-ab-creative",
    metadata={"variant": "creative", "temperature": 0.7},
)
```

Both runs use the exact same `evaluators` list and the same `DATASET_NAME`. Because the examples are fixed, any difference in scores reflects a genuine difference in prompt performance — not a difference in the evaluation setup.

### Comparing Results

After both runs complete, `ab_comparison.py` averages evaluator scores across all examples for each variant and prints a comparison table:

```
Metric                    Structured      Creative        Winner
-----------------------------------------------------------------
content_quality                0.821         0.804     Structured
difficulty_match               0.873         0.841     Structured
topic_relevance                0.912         0.895     Structured
```

The comparison is programmatic — you don't need to read individual traces to know which variant won on each dimension. But the traces in LangSmith are still available if you want to understand *why* one variant outperformed the other on specific examples.

### Why the Same Dataset Matters

The strength of A/B comparison depends entirely on using the same dataset for both runs. If you generate different random examples for each variant, score differences could reflect lucky or unlucky inputs rather than prompt quality. LangSmith's dataset model — fixed examples stored server-side, referenced by name — makes this easy to enforce.

The `experiment_prefix` naming convention (`p5-ab-structured`, `p5-ab-creative`) keeps runs organized in the LangSmith dashboard and makes the A/B relationship explicit when you look at the experiment list.

---

## 7. Key Takeaways

### What You Learned

| Concept | What It Does | Where to Look |
|---------|-------------|---------------|
| `interrupt()` | Suspends graph execution for human input | `nodes.py` — `draft_review_node`, `final_review_node` |
| `Command(resume=...)` | Resumes a suspended graph with human's decision | Callers of the graph (tests, CLI) |
| Re-execution on resume | The interrupted node reruns from its start | `nodes.py` — no side effects before `interrupt()` |
| Three-way routing | Approve/edit/reject with different downstream paths | `nodes.py` — `route_after_draft_review` |
| Revision loop | `revise` → `draft_review` cycle with max revision guard | `graph.py`, `nodes.py` — `revision_count` |
| `RetryPolicy` | Automatic retry on transient LLM API errors | `graph.py` — LLM nodes only |
| 4-tier error handling | Layered strategy: retry, interrupt, bubble up | `graph.py` comments |
| LangSmith datasets | Fixed examples for reproducible evaluation | `evaluation.py` — `create_dataset()` |
| LLM-as-judge evaluators | Subjective quality assessment via another LLM | `evaluation.py` — evaluator functions |
| `evaluate()` | Runs target function against dataset with evaluators | `evaluation.py` — `run_evaluation()` |
| A/B prompt comparison | Comparing two prompt variants on the same dataset | `ab_comparison.py` — `run_ab_comparison()` |

### The Checkpointer Requirement

Interrupts require a checkpointer — without one, LangGraph has nowhere to save graph state when execution suspends. This project enforces this in `build_graph`:

```python
# From graph.py
def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    if checkpointer is None:
        checkpointer = InMemorySaver()
    ...
    .compile(checkpointer=checkpointer)
```

`InMemorySaver` is fine for development and testing. In production, you would use `PostgresSaver` or another persistent backend — `InMemorySaver` loses all state if the process restarts.

### The Pattern You Should Recognize

Human-in-the-loop in LangGraph follows a consistent three-step pattern:

```
1. Node calls interrupt(payload) — graph pauses, state is checkpointed
2. Human receives payload, makes a decision
3. Caller invokes graph with Command(resume=decision) and same thread_id
   — graph reruns the interrupted node, interrupt() returns the decision
   — execution continues normally
```

This pattern works at any scale — single-node interrupts, multi-interrupt workflows, nested interrupts in subgraphs. The mechanics are always the same: checkpoint, resume, re-execute.

### What Comes Next: Project 6

Project 5 adds human oversight to a single-agent pipeline. **Project 6** scales this to multiple agents — a coordinator that spawns specialized subagents for different tasks. You'll see how LangGraph's multi-agent primitives (subgraphs, message passing between agents) enable parallelism and specialization. The HITL patterns from this project carry forward; what's new is the orchestration layer above individual agents.

---

*This document covers Project 05 of the LinguaFlow learning path. Continue to `docs/06-...` when ready for multi-agent orchestration.*
