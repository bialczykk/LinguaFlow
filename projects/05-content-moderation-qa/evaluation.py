# evaluation.py
"""LangSmith evaluation pipeline for the Content Moderation system.

This module provides:
1. Custom evaluator functions (LLM-as-judge) for content quality
2. A function to create an evaluation dataset in LangSmith
3. A function to run evaluations against the generate node

LangSmith concepts demonstrated:
- langsmith.Client for dataset/example management
- langsmith.evaluation.evaluate() for running evaluations
- Custom evaluator functions: (run, example) -> {"key": ..., "score": ...}
- LLM-as-judge pattern for subjective quality assessment

Usage:
    python evaluation.py              # Create dataset and run evaluation
    python evaluation.py --create     # Only create dataset
    python evaluation.py --evaluate   # Only run evaluation (dataset must exist)
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_anthropic import ChatAnthropic
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

from data.content_requests import SAMPLE_REQUESTS
from prompts import GENERATE_PROMPT

# -- LLM for evaluators --
_eval_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# -- Dataset name --
DATASET_NAME = "p5-content-generation-eval"


def _parse_json_from_llm(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences if present.

    Claude often wraps JSON in ```json ... ``` blocks. This helper
    strips those before parsing.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# -- Custom Evaluators --
# Each evaluator follows the LangSmith convention:
#   def evaluator(run, example) -> {"key": str, "score": float, "comment": str}

def topic_relevance_evaluator(run, example) -> dict:
    """Score whether generated content matches the requested topic.

    Uses LLM-as-judge to assess topic relevance on a 0-1 scale.
    """
    content = run.outputs.get("content", "")
    topic = example.inputs.get("topic", "")

    if not content or not topic:
        return {"key": "topic_relevance", "score": 0.0, "comment": "Missing content or topic"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, how relevant is this content to the topic '{topic}'?\n\n"
        f"Content:\n{content}\n\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = _parse_json_from_llm(response.content)
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "topic_relevance", "score": score, "comment": f"Topic: {topic}"}


def difficulty_match_evaluator(run, example) -> dict:
    """Score whether content difficulty matches the requested CEFR level.

    Uses LLM-as-judge to assess difficulty appropriateness on a 0-1 scale.
    """
    content = run.outputs.get("content", "")
    difficulty = example.inputs.get("difficulty", "")

    if not content or not difficulty:
        return {"key": "difficulty_match", "score": 0.0, "comment": "Missing content or difficulty"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, how well does this content match CEFR level {difficulty}?\n\n"
        f"Content:\n{content}\n\n"
        f"Consider vocabulary complexity, grammar structures, and overall readability.\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = _parse_json_from_llm(response.content)
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "difficulty_match", "score": score, "comment": f"Target: {difficulty}"}


def content_quality_evaluator(run, example) -> dict:
    """Score overall content quality (grammar, clarity, completeness).

    Uses LLM-as-judge to assess quality on a 0-1 scale.
    """
    content = run.outputs.get("content", "")

    if not content:
        return {"key": "content_quality", "score": 0.0, "comment": "No content"}

    response = _eval_model.invoke(
        f"On a scale from 0.0 to 1.0, rate the overall quality of this lesson content.\n\n"
        f"Content:\n{content}\n\n"
        f"Consider: grammar correctness, clarity of explanation, completeness, "
        f"and usefulness as a learning resource.\n"
        f"Respond with ONLY a JSON object: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    try:
        parsed = _parse_json_from_llm(response.content)
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {"key": "content_quality", "score": score}


# -- Target function for evaluation --
@traceable(name="generate_for_eval", tags=["p5-content-moderation"])
def generate_for_eval(inputs: dict) -> dict:
    """Wrapper around the generate prompt for LangSmith evaluation.

    Takes a content request dict and returns {"content": ...}.
    This bypasses the full graph (no HITL) for automated evaluation.
    """
    model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    chain = GENERATE_PROMPT | model

    response = chain.invoke(inputs)

    try:
        parsed = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        return {"content": parsed.get("content", response.content)}
    except json.JSONDecodeError:
        return {"content": response.content}


def create_dataset():
    """Create an evaluation dataset in LangSmith from sample requests."""
    client = Client()

    # Delete existing dataset if it exists (for idempotency)
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset: {DATASET_NAME}")
    except Exception:
        pass

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

    print(f"Created dataset '{DATASET_NAME}' with {len(SAMPLE_REQUESTS)} examples")
    return dataset


def run_evaluation():
    """Run the evaluation pipeline against the dataset."""
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

    print(f"\nExperiment: {results.experiment_name}")
    print("-" * 60)
    for row in results:
        example_inputs = row["example"].inputs
        print(f"\nTopic: {example_inputs.get('topic', 'N/A')}")
        for eval_result in row["evaluation_results"]["results"]:
            print(f"  {eval_result.key}: {eval_result.score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangSmith evaluation for P5")
    parser.add_argument("--create", action="store_true", help="Only create dataset")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    if args.create:
        create_dataset()
    elif args.evaluate:
        run_evaluation()
    else:
        create_dataset()
        run_evaluation()
