"""LangSmith evaluation module for the autonomous operations orchestrator — Project 08.

This module defines three evaluators and the scaffolding to run a LangSmith
evaluation suite against the compiled autonomous operations graph.

Evaluators
----------
1. routing_accuracy_evaluator  — Deterministic set-comparison check.
   Compares the departments chosen by the classifier against the expected
   departments recorded in the dataset. Returns a 0-1 Jaccard score
   (perfect match = 1.0, partial overlap = proportional, no overlap = 0.0).

2. response_quality_evaluator  — LLM-as-judge using claude-haiku.
   Asks the model to rate the final unified response on three dimensions:
   coherence, completeness (does it address every part of the request?), and
   professionalism. Returns a 0-1 normalised score plus qualitative feedback.

3. task_chain_completeness_evaluator  — Deterministic follow-up audit.
   Checks that every expected autonomous follow-up task was actually completed
   by comparing completed_tasks against the expected_follow_ups from the dataset.
   Returns 1.0 if all expected follow-ups appear in completed tasks, 0.0 otherwise.

Workflow
--------
1. `create_dataset()` — pushes SAMPLE_REQUESTS to a named LangSmith dataset so
   every example has well-defined inputs and reference outputs.
2. `run_evaluation()` — calls `langsmith.evaluate()`, which:
   a. iterates over every example in the dataset,
   b. invokes the compiled graph as the "target" function,
   c. runs all three evaluators against each (run, example) pair,
   d. uploads results to LangSmith for inspection.

LangSmith concepts demonstrated
---------------------------------
- `langsmith.Client` for programmatic dataset management
- `langsmith.evaluate()` for running an evaluation experiment
- Custom evaluator functions with the `(Run, Example) -> dict` signature
- LLM-as-judge pattern: structured prompts that ask the model to score output
- Deterministic evaluators for objective metrics (routing, task completion)
- Tagging traces with a project label for easy filtering in the LangSmith UI
"""

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

# ---------------------------------------------------------------------------
# Environment & shared resources
# ---------------------------------------------------------------------------

# Load .env so ANTHROPIC_API_KEY and LANGSMITH_API_KEY are available at runtime
load_dotenv()

# The LLM judge used by response_quality_evaluator.
# We use the cheapest Anthropic model — this is a learning repo, cost matters.
_llm_judge = ChatAnthropic(model="claude-haiku-4-5-20251001")

# LangSmith project tag — every trace will carry this label so you can filter
# all P8 runs in a single click inside the LangSmith UI.
LANGSMITH_TAGS = ["p8-autonomous-operations"]

# Name of the LangSmith dataset that stores our test cases
DATASET_NAME = "linguaflow-p8-autonomous-operations"


# ---------------------------------------------------------------------------
# 1. Dataset creation
# ---------------------------------------------------------------------------

def create_dataset(client: Client | None = None, dataset_name: str = DATASET_NAME) -> str:
    """Push SAMPLE_REQUESTS to LangSmith as a named evaluation dataset.

    Each SAMPLE_REQUEST becomes one Example in the dataset:
    - inputs  → the fields that will be fed to the graph (request + metadata)
    - outputs → the reference values an evaluator can compare against
                (expected_departments, expected_follow_ups, pattern)

    Args:
        client:       An optional pre-constructed LangSmith Client. If None, a
                      new Client() is created (reads LANGSMITH_API_KEY from env).
        dataset_name: Name of the dataset to create or reuse.

    Returns:
        The dataset ID string.

    Why create a dataset?
    ---------------------
    LangSmith `evaluate()` iterates over a dataset, invokes the target function
    for each example, and then calls each evaluator with the resulting Run and
    the original Example. Storing reference outputs in the dataset — rather than
    hard-coding them in evaluators — keeps the evaluation logic reusable and the
    ground truth easy to inspect or update in the LangSmith UI.
    """
    from data.sample_requests import SAMPLE_REQUESTS

    if client is None:
        client = Client()

    # Check whether the dataset already exists to avoid duplicates on re-runs
    existing = [ds for ds in client.list_datasets() if ds.name == dataset_name]
    if existing:
        print(f"Dataset '{dataset_name}' already exists (id={existing[0].id}). Skipping creation.")
        return str(existing[0].id)

    # Create a fresh dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "LinguaFlow P8 — autonomous operations routing and task chaining test cases. "
            "Each example has a request, metadata, expected routing, and expected follow-up cascades."
        ),
    )
    print(f"Created dataset '{dataset_name}' (id={dataset.id})")

    # Add one Example per sample request
    for req in SAMPLE_REQUESTS:
        client.create_example(
            dataset_id=dataset.id,
            # --- inputs: everything the graph needs to run ---
            inputs={
                "request": req["text"],
                "request_metadata": req["metadata"],
                "expected_departments": req["expected_departments"],
            },
            # --- outputs: reference values for evaluators ---
            outputs={
                "expected_departments": req["expected_departments"],
                "expected_follow_ups": req["expected_follow_ups"],
                "pattern": req["pattern"],
                "expected_risk": req["expected_risk"],
            },
        )

    print(f"Uploaded {len(SAMPLE_REQUESTS)} examples to dataset '{dataset_name}'.")
    return str(dataset.id)


# ---------------------------------------------------------------------------
# 2. Routing accuracy evaluator  (deterministic — no LLM needed)
# ---------------------------------------------------------------------------

def routing_accuracy_evaluator(run: Run, example: Example) -> dict:
    """Compare predicted departments against expected departments.

    This evaluator is purely deterministic — it performs a set comparison so
    there is no LLM call and no non-determinism. This is the preferred approach
    when the ground truth is objective (a list of strings to match).

    Scoring logic (Jaccard similarity)
    -----------------------------------
    - Perfect match  → 1.0   (predicted set == expected set)
    - Partial overlap → |predicted ∩ expected| / |predicted ∪ expected|
      Penalises both missing departments (under-routing) and extra departments
      (over-routing / hallucination).
    - No overlap     → 0.0

    Args:
        run:     The LangSmith Run object containing the graph's output.
        example: The LangSmith Example with reference inputs/outputs.

    Returns:
        dict with keys:
          - "key"     : metric name (used as column header in LangSmith UI)
          - "score"   : float in [0, 1]
          - "comment" : human-readable explanation of the score
    """
    # --- Extract predicted departments from the run output ---
    # The graph writes classification.departments to state; LangSmith captures
    # the full final state as run.outputs.
    run_outputs: dict[str, Any] = run.outputs or {}
    classification: dict = run_outputs.get("classification") or {}
    predicted: set[str] = set(classification.get("departments") or [])

    # --- Extract expected departments from the example reference outputs ---
    ref_outputs: dict[str, Any] = (example.outputs or {})
    expected: set[str] = set(ref_outputs.get("expected_departments") or [])

    # --- Edge case: both sets empty ---
    # If both predicted and expected are empty we consider routing correct.
    if not predicted and not expected:
        return {
            "key": "routing_accuracy",
            "score": 1.0,
            "comment": "Both predicted and expected departments are empty. Correct.",
        }

    # --- Jaccard similarity ---
    intersection = predicted & expected
    union = predicted | expected

    if not union:
        # Guard against ZeroDivision (shouldn't occur after the check above)
        score = 0.0
        comment = "Unable to compute score — both sets unexpectedly empty."
    else:
        score = len(intersection) / len(union)
        if score == 1.0:
            comment = f"Perfect match: {sorted(predicted)} == {sorted(expected)}."
        else:
            missing = sorted(expected - predicted)
            extra = sorted(predicted - expected)
            parts = []
            if missing:
                parts.append(f"missed: {missing}")
            if extra:
                parts.append(f"extra (hallucinated): {extra}")
            comment = f"Partial match (Jaccard={score:.2f}). " + "; ".join(parts)

    return {"key": "routing_accuracy", "score": score, "comment": comment}


# ---------------------------------------------------------------------------
# 3. Response quality evaluator  (LLM-as-judge)
# ---------------------------------------------------------------------------

# Prompt template for the LLM judge.
# We use a structured rubric so the model's score is consistent and auditable.
_QUALITY_JUDGE_PROMPT = """\
You are a quality assessor for an autonomous tutoring operations platform.

=== ORIGINAL REQUEST ===
{request}

=== SYSTEM RESPONSE ===
{response}

=== EVALUATION TASK ===
Rate the system response on the following three dimensions, each from 0 to 10:

1. COHERENCE (0-10)
   Does the response read naturally, follow a logical structure, and avoid
   internal contradictions?

2. COMPLETENESS (0-10)
   Does the response address every distinct issue or action requested?
   A partial response that ignores some issues should score below 7.

3. PROFESSIONALISM (0-10)
   Is the tone clear, helpful, and appropriate for an operations platform
   communicating outcomes to an administrator or student?

After scoring, compute the AVERAGE of the three scores.

Respond in the following exact format — nothing else:
COHERENCE: <score>
COMPLETENESS: <score>
PROFESSIONALISM: <score>
AVERAGE: <score>
FEEDBACK: <one or two sentences of qualitative feedback>
"""


def response_quality_evaluator(run: Run, example: Example) -> dict:
    """Use an LLM judge (claude-haiku) to rate the final unified response.

    Why LLM-as-judge?
    -----------------
    Response quality is subjective and multi-dimensional — hard to capture with
    a simple rule. We delegate the assessment to the same class of model that
    generated the response, but with a carefully structured rubric prompt that
    forces it to reason across three axes (coherence, completeness,
    professionalism) before producing a numeric score.

    Args:
        run:     LangSmith Run containing the graph's final state.
        example: LangSmith Example (used here only for the original request).

    Returns:
        dict with keys:
          - "key"     : "response_quality"
          - "score"   : float in [0, 1] (normalised from 0-10 average)
          - "comment" : verbatim FEEDBACK line from the judge + raw scores
    """
    # --- Pull request and response from run artefacts ---
    run_inputs: dict[str, Any] = run.inputs or {}
    run_outputs: dict[str, Any] = run.outputs or {}

    request_text: str = run_inputs.get("request", "")
    final_response: str = run_outputs.get("final_response", "")

    # Guard: if the graph produced no final_response (e.g. interrupted for
    # approval or failed mid-graph), we can't meaningfully rate quality.
    if not final_response:
        return {
            "key": "response_quality",
            "score": 0.0,
            "comment": "No final_response found in run output — graph may have been interrupted or failed.",
        }

    # --- Build the judge prompt ---
    prompt = _QUALITY_JUDGE_PROMPT.format(
        request=request_text,
        response=final_response,
    )

    # --- Invoke the LLM judge ---
    # We call the model directly (no chain) to keep the evaluation code simple
    # and readable — this is a learning repo, clarity matters.
    judge_message = _llm_judge.invoke(prompt)
    judge_text: str = judge_message.content

    # --- Parse the judge's structured output ---
    score_normalized, comment = _parse_judge_output(judge_text)

    return {
        "key": "response_quality",
        "score": score_normalized,
        "comment": comment,
    }


def _parse_judge_output(judge_text: str) -> tuple[float, str]:
    """Extract the AVERAGE score and FEEDBACK from the judge's structured reply.

    Parses lines of the form:
        AVERAGE: 8.5
        FEEDBACK: The response was clear but missed the billing issue.

    Returns (normalised_score, feedback_string) where normalised_score is
    in [0, 1] (divided by 10 from the 0-10 scale).

    Falls back gracefully if parsing fails — returns (0.0, raw_text).
    """
    lines = [line.strip() for line in judge_text.strip().splitlines()]
    average_score: float | None = None
    feedback: str = ""
    raw_scores: list[str] = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("AVERAGE:"):
            try:
                average_score = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif upper.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
        elif any(upper.startswith(k) for k in ("COHERENCE:", "COMPLETENESS:", "PROFESSIONALISM:")):
            raw_scores.append(line)

    if average_score is None:
        # Parsing failed — return 0 and the raw text for manual inspection
        return 0.0, f"(parse error) raw judge output: {judge_text[:300]}"

    # Normalise from 0-10 to 0-1
    normalised = max(0.0, min(1.0, average_score / 10.0))

    # Build a human-readable comment combining individual scores + feedback
    scores_summary = " | ".join(raw_scores) if raw_scores else "scores not parsed"
    comment = f"{scores_summary} → avg={average_score:.1f}/10. Feedback: {feedback}"

    return normalised, comment


# ---------------------------------------------------------------------------
# 4. Task chain completeness evaluator  (deterministic — no LLM needed)
# ---------------------------------------------------------------------------

def task_chain_completeness_evaluator(run: Run, example: Example) -> dict:
    """Check whether all expected autonomous follow-up tasks were completed.

    The autonomous operations graph supports cascading follow-up tasks:
    one department's output can trigger additional work in another department
    (e.g. student_onboarding triggers tutor_management). This evaluator checks
    that the expected cascade actually happened by inspecting completed_tasks.

    Scoring logic
    -------------
    - All expected follow-ups found in completed tasks → 1.0
    - None of the expected follow-ups found            → 0.0
    - Some but not all found                           → partial (fraction found)
    - No expected follow-ups (single/parallel patterns) → 1.0 (vacuously true)

    What we match on
    ----------------
    completed_tasks is a list of dicts with a "target_dept" key (set by the
    graph when it processes follow-up tasks). We check that each expected
    follow-up department appears in at least one completed task's target_dept.

    Args:
        run:     LangSmith Run containing the graph's final state.
        example: LangSmith Example with expected_follow_ups in reference outputs.

    Returns:
        dict with keys:
          - "key"     : "task_chain_completeness"
          - "score"   : float in [0, 1]
          - "comment" : human-readable explanation of the score
    """
    # --- Extract completed_tasks from the run output ---
    run_outputs: dict[str, Any] = run.outputs or {}
    completed_tasks: list[dict] = run_outputs.get("completed_tasks") or []

    # Build a set of all target departments that were actually processed
    completed_depts: set[str] = {
        task.get("target_dept", "")
        for task in completed_tasks
        if isinstance(task, dict)
    }

    # --- Extract expected follow-ups from the example reference outputs ---
    ref_outputs: dict[str, Any] = (example.outputs or {})
    expected_follow_ups: list[str] = ref_outputs.get("expected_follow_ups") or []

    # --- Vacuously true: no follow-ups expected ---
    if not expected_follow_ups:
        return {
            "key": "task_chain_completeness",
            "score": 1.0,
            "comment": "No follow-up tasks expected for this request pattern. Score is vacuously 1.0.",
        }

    expected_set: set[str] = set(expected_follow_ups)

    # --- Check which expected follow-ups were completed ---
    found = expected_set & completed_depts
    missing = expected_set - completed_depts

    if not found:
        score = 0.0
        comment = (
            f"No expected follow-ups were completed. "
            f"Expected: {sorted(expected_set)}. "
            f"Completed department tasks: {sorted(completed_depts) or 'none'}."
        )
    elif missing:
        # Partial: some follow-ups were completed, some were not
        score = len(found) / len(expected_set)
        comment = (
            f"Partial follow-up completion ({len(found)}/{len(expected_set)}). "
            f"Completed: {sorted(found)}. Missing: {sorted(missing)}."
        )
    else:
        score = 1.0
        comment = (
            f"All expected follow-up tasks completed: {sorted(expected_set)}."
        )

    return {"key": "task_chain_completeness", "score": score, "comment": comment}


# ---------------------------------------------------------------------------
# 5. Target function — wraps the graph for LangSmith evaluate()
# ---------------------------------------------------------------------------

def _build_target_fn():
    """Return a callable that LangSmith's `evaluate()` can call per example.

    `evaluate()` calls target(inputs) for each dataset example, where `inputs`
    is the dict stored in example.inputs. The return value becomes run.outputs.

    We build the graph once here and close over it to avoid re-compiling on
    every example invocation. No checkpointer is used for batch evaluation —
    each run is stateless and independent, which is what we want for a clean
    evaluation suite.
    """
    # Import here to avoid circular imports at module load time
    from graph import build_graph

    # Stateless graph — no checkpointer needed for batch evaluation.
    # Each evaluation run is independent; we don't need persistence.
    # Note: high-risk requests that would normally trigger the approval_gate
    # interrupt will be handled gracefully — the graph will pause and return
    # a partial state, which is expected behaviour for those test cases.
    graph = build_graph()

    def target(inputs: dict) -> dict:
        """Invoke the graph and return its final state as a plain dict."""
        result = graph.invoke(
            {
                "request": inputs.get("request", ""),
                "request_metadata": inputs.get("request_metadata", {}),
                # Initialise list/dict fields to avoid KeyError in nodes
                "department_results": [],
                "task_queue": [],
                "completed_tasks": [],
                "metrics_store": {},
            },
            # Tag every LangSmith trace with the project label so you can filter
            # evaluation runs alongside regular usage traces in one view.
            config={"tags": LANGSMITH_TAGS},
        )
        # result is an OrchestratorState TypedDict — return as plain dict for LangSmith
        return dict(result)

    return target


# ---------------------------------------------------------------------------
# 6. run_evaluation() — the main entry point
# ---------------------------------------------------------------------------

def run_evaluation(experiment_prefix: str = "p8-eval") -> None:
    """Run the full evaluation suite against the LangSmith dataset.

    Steps
    -----
    1. Ensure the dataset exists (create it if needed).
    2. Build the graph target function.
    3. Call `langsmith.evaluate()` with all three evaluators.
    4. Print a summary of results to stdout.

    Args:
        experiment_prefix: A string prefix for the experiment name shown in
                           the LangSmith UI. A UUID suffix is appended so
                           each run is uniquely identifiable.
    """
    # Step 1 — ensure dataset exists
    print(f"Ensuring dataset '{DATASET_NAME}' exists ...")
    create_dataset()

    # Step 2 — build target function
    target_fn = _build_target_fn()

    # Step 3 — run evaluation
    experiment_name = f"{experiment_prefix}-{uuid.uuid4().hex[:8]}"
    print(f"Starting LangSmith evaluation experiment: {experiment_name}")

    results = evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[
            routing_accuracy_evaluator,
            response_quality_evaluator,
            task_chain_completeness_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        # Metadata visible in the LangSmith UI experiment details panel
        metadata={
            "project": "08-autonomous-operations",
            "description": (
                "Routing accuracy, response quality, and task chain completeness "
                "evaluation for the LinguaFlow autonomous operations orchestrator."
            ),
        },
    )

    # Step 4 — print summary
    print("\n=== Evaluation complete ===")
    print(f"Experiment: {experiment_name}")
    print(f"Results object: {results}")
    print("\nView results at: https://smith.langchain.com  (filter by tag: p8-autonomous-operations)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run the evaluation pipeline end-to-end.

    Requires:
        ANTHROPIC_API_KEY  — for the LLM judge (response quality evaluator)
        LANGSMITH_API_KEY  — for dataset management and result upload

    Usage:
        python evaluation.py
    """
    run_evaluation()
