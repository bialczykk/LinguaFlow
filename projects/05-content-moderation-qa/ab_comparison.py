# ab_comparison.py
"""A/B prompt comparison using LangSmith experiments.

Runs two prompt variants (structured vs creative) against the same
evaluation dataset and compares scores side-by-side.

LangSmith concepts demonstrated:
- Running multiple experiments against the same dataset
- Comparing experiment results programmatically
- Using experiment_prefix to organize runs

Usage:
    python ab_comparison.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_anthropic import ChatAnthropic
from langsmith import traceable
from langsmith.evaluation import evaluate

from evaluation import (
    DATASET_NAME,
    topic_relevance_evaluator,
    difficulty_match_evaluator,
    content_quality_evaluator,
    create_dataset,
)
from prompts import GENERATE_PROMPT_STRUCTURED, GENERATE_PROMPT_CREATIVE


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# -- Prompt Variant A: Structured --
@traceable(name="generate_structured", tags=["p5-content-moderation", "ab-test"])
def generate_structured(inputs: dict) -> dict:
    """Generate content using the structured prompt template."""
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    chain = GENERATE_PROMPT_STRUCTURED | model
    response = chain.invoke(inputs)
    try:
        parsed = _parse_json_response(response.content)
        return {"content": parsed.get("content", response.content)}
    except (json.JSONDecodeError, ValueError):
        return {"content": response.content}


# -- Prompt Variant B: Creative --
@traceable(name="generate_creative", tags=["p5-content-moderation", "ab-test"])
def generate_creative(inputs: dict) -> dict:
    """Generate content using the creative prompt template."""
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7)
    chain = GENERATE_PROMPT_CREATIVE | model
    response = chain.invoke(inputs)
    try:
        parsed = _parse_json_response(response.content)
        return {"content": parsed.get("content", response.content)}
    except (json.JSONDecodeError, ValueError):
        return {"content": response.content}


def run_ab_comparison():
    """Run both prompt variants and compare results."""
    evaluators = [
        topic_relevance_evaluator,
        difficulty_match_evaluator,
        content_quality_evaluator,
    ]

    # Ensure dataset exists
    try:
        from langsmith import Client
        Client().read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        print("Dataset not found. Creating it first...")
        create_dataset()

    print("=" * 60)
    print("A/B Prompt Comparison")
    print("=" * 60)

    # -- Run Variant A: Structured --
    print("\nRunning Variant A (Structured)...")
    results_a = evaluate(
        generate_structured,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="p5-ab-structured",
        metadata={"variant": "structured", "temperature": 0.3},
    )

    # -- Run Variant B: Creative --
    print("\nRunning Variant B (Creative)...")
    results_b = evaluate(
        generate_creative,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="p5-ab-creative",
        metadata={"variant": "creative", "temperature": 0.7},
    )

    # -- Compare Results --
    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    def avg_scores(results):
        """Compute average score per evaluator across all examples."""
        totals = {}
        counts = {}
        for row in results:
            for eval_result in row["evaluation_results"]["results"]:
                key = eval_result.key
                totals[key] = totals.get(key, 0) + (eval_result.score or 0)
                counts[key] = counts.get(key, 0) + 1
        return {k: totals[k] / counts[k] for k in totals}

    scores_a = avg_scores(results_a)
    scores_b = avg_scores(results_b)

    print(f"\n{'Metric':<25} {'Structured':>12} {'Creative':>12} {'Winner':>12}")
    print("-" * 65)
    for metric in sorted(set(list(scores_a.keys()) + list(scores_b.keys()))):
        sa = scores_a.get(metric, 0)
        sb = scores_b.get(metric, 0)
        winner = "Structured" if sa > sb else "Creative" if sb > sa else "Tie"
        print(f"{metric:<25} {sa:>12.3f} {sb:>12.3f} {winner:>12}")

    print("\nView detailed results in LangSmith dashboard.")


if __name__ == "__main__":
    run_ab_comparison()
