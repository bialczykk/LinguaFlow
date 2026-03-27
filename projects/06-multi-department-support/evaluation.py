"""LangSmith evaluation module for the multi-department support system.

This module defines two LLM-as-judge evaluators and the scaffolding to run a
LangSmith evaluation suite against the compiled support graph.

Evaluators
----------
1. routing_accuracy_evaluator  — Deterministic set-comparison check.
   Compares the departments chosen by the supervisor against the expected
   departments recorded in the dataset. Returns a 0-1 score (exact match = 1.0,
   partial overlap = proportional, no overlap = 0.0).

2. response_quality_evaluator  — LLM-as-judge using claude-haiku.
   Asks the model to rate the final unified response on three dimensions:
   coherence, completeness (does it address every part of the request?), and
   professionalism. Returns a 0-1 normalised score plus qualitative feedback.

Workflow
--------
1. `create_dataset()` — pushes SAMPLE_REQUESTS to a named LangSmith dataset so
   every example has well-defined inputs and reference outputs.
2. `run_evaluation()` — calls `langsmith.evaluate()`, which:
   a. iterates over every example in the dataset,
   b. invokes the compiled graph as the "target" function,
   c. runs both evaluators against each (run, example) pair,
   d. uploads results to LangSmith for inspection.

LangSmith concepts demonstrated
---------------------------------
- `langsmith.Client` for programmatic dataset management
- `langsmith.evaluate()` for running an evaluation experiment
- Custom evaluator functions with the `(Run, Example) -> dict` signature
- LLM-as-judge pattern: structured prompts that ask the model to score output
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
# all P6 runs in a single click inside the LangSmith UI.
LANGSMITH_TAGS = ["p6-multi-department-support"]

# Name of the LangSmith dataset that stores our test cases
DATASET_NAME = "linguaflow-p6-support-requests"


# ---------------------------------------------------------------------------
# 1. Dataset creation
# ---------------------------------------------------------------------------

def create_dataset() -> str:
    """Push SAMPLE_REQUESTS to LangSmith as a named evaluation dataset.

    Each SAMPLE_REQUEST becomes one Example in the dataset:
    - inputs  → the fields that will be fed to the graph (request + metadata)
    - outputs → the reference values an evaluator can compare against
                (expected_departments, pattern)

    Returns the dataset ID string.

    Why create a dataset?
    ---------------------
    LangSmith `evaluate()` iterates over a dataset, invokes the target function
    for each example, and then calls each evaluator with the resulting Run and
    the original Example. Storing reference outputs in the dataset — rather than
    hard-coding them in evaluators — keeps the evaluation logic reusable and the
    ground truth easy to inspect or update in the LangSmith UI.
    """
    from data.support_requests import SAMPLE_REQUESTS

    client = Client()

    # Check whether the dataset already exists to avoid duplicates on re-runs
    existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]
    if existing:
        print(f"Dataset '{DATASET_NAME}' already exists (id={existing[0].id}). Skipping creation.")
        return str(existing[0].id)

    # Create a fresh dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "LinguaFlow P6 — multi-department support routing test cases. "
            "Each example has a student request, metadata, and expected routing."
        ),
    )
    print(f"Created dataset '{DATASET_NAME}' (id={dataset.id})")

    # Add one Example per sample request
    for req in SAMPLE_REQUESTS:
        client.create_example(
            dataset_id=dataset.id,
            # --- inputs: everything the graph needs to run ---
            inputs={
                "request": req["text"],
                "request_metadata": req["metadata"],
            },
            # --- outputs: reference values for evaluators ---
            outputs={
                "expected_departments": req["expected_departments"],
                "pattern": req["pattern"],
            },
        )

    print(f"Uploaded {len(SAMPLE_REQUESTS)} examples to dataset '{DATASET_NAME}'.")
    return str(dataset.id)


# ---------------------------------------------------------------------------
# 2. Routing accuracy evaluator  (deterministic — no LLM needed)
# ---------------------------------------------------------------------------

def routing_accuracy_evaluator(run: Run, example: Example) -> dict:
    """Compare predicted departments against expected departments.

    This evaluator is purely deterministic — it performs a set comparison so
    there is no LLM call and no non-determinism. This is the preferred approach
    when the ground truth is objective (a list of strings to match).

    Scoring logic
    -------------
    - Perfect match  → 1.0   (sets are identical)
    - Partial overlap → |predicted ∩ expected| / |predicted ∪ expected|
      (Jaccard similarity — penalises both missing and extra departments)
    - Needs-clarification special case → if expected is [] (ambiguous request)
      and the graph also produced no departments, score = 1.0

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

    # --- Special case: ambiguous request (clarification pattern) ---
    # If both predicted and expected are empty we consider routing correct.
    if not predicted and not expected:
        return {
            "key": "routing_accuracy",
            "score": 1.0,
            "comment": "Both predicted and expected departments are empty (clarification pattern). Correct.",
        }

    # --- Jaccard similarity ---
    intersection = predicted & expected
    union = predicted | expected

    if not union:
        # Shouldn't happen after the check above, but guard against ZeroDivision
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
You are a quality assessor for a student support system at an English tutoring platform.

=== ORIGINAL STUDENT REQUEST ===
{request}

=== SYSTEM RESPONSE ===
{response}

=== EVALUATION TASK ===
Rate the system response on the following three dimensions, each from 0 to 10:

1. COHERENCE (0-10)
   Does the response read naturally and without internal contradictions?

2. COMPLETENESS (0-10)
   Does the response address every distinct issue or question raised in the request?
   A partial response (some issues ignored) should score below 7.

3. PROFESSIONALISM (0-10)
   Is the tone polite, empathetic, and appropriate for a customer-facing support reply?

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
    # clarification), we can't meaningfully rate quality.
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
# 4. Target function — wraps the graph for LangSmith evaluate()
# ---------------------------------------------------------------------------

def _build_target_fn():
    """Return a callable that LangSmith's `evaluate()` can call per example.

    `evaluate()` calls target(inputs) for each dataset example, where `inputs`
    is the dict stored in example.inputs. The return value becomes run.outputs.

    We build the graph once here and close over it to avoid re-compiling on
    every example invocation.
    """
    # Import here to avoid circular imports at module load time
    from graph import build_graph

    # Stateless graph — no checkpointer needed for batch evaluation.
    # Each evaluation run is independent; we don't need persistence.
    graph = build_graph()

    def target(inputs: dict) -> dict:
        """Invoke the graph and return its final state as a plain dict."""
        result = graph.invoke(
            {
                "request": inputs.get("request", ""),
                "request_metadata": inputs.get("request_metadata", {}),
            },
            # Tag every LangSmith trace with the project label so you can filter
            # evaluation runs alongside regular usage traces in one view.
            config={"tags": LANGSMITH_TAGS},
        )
        # result is a SupportState TypedDict — return as plain dict for LangSmith
        return dict(result)

    return target


# ---------------------------------------------------------------------------
# 5. run_evaluation() — the main entry point
# ---------------------------------------------------------------------------

def run_evaluation(experiment_prefix: str = "p6-eval") -> None:
    """Run the full evaluation suite against the LangSmith dataset.

    Steps
    -----
    1. Ensure the dataset exists (create it if needed).
    2. Build the graph target function.
    3. Call `langsmith.evaluate()` with both evaluators.
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
        evaluators=[routing_accuracy_evaluator, response_quality_evaluator],
        experiment_prefix=experiment_prefix,
        # Metadata visible in the LangSmith UI experiment details panel
        metadata={
            "project": "06-multi-department-support",
            "description": "Routing accuracy and response quality evaluation for the LinguaFlow support graph.",
        },
    )

    # Step 4 — print summary
    print("\n=== Evaluation complete ===")
    print(f"Experiment: {experiment_name}")
    print(f"Results object: {results}")
    print("\nView results at: https://smith.langchain.com  (filter by tag: p6-multi-department-support)")


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
