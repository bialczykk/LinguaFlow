"""CLI entry point for the Student Assessment Pipeline.

Ingests documents (if needed), takes a student submission, runs it
through the assessment graph, and prints the structured results.

Usage:
    python main.py                  # Run with default sample submission
    python main.py --sample 0      # Run with specific sample (0, 1, or 2)
    python main.py --rebuild        # Force rebuild of the vector store
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from repo root .env
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from ingestion import build_vector_store, get_vector_store, DEFAULT_PERSIST_DIR
from graph import build_graph
from data.sample_submissions import ALL_SUBMISSIONS


def _print_assessment(assessment):
    """Pretty-print the final assessment to the console."""
    print("\n" + "=" * 60)
    print("STUDENT WRITING ASSESSMENT")
    print("=" * 60)

    print(f"\nOverall CEFR Level: {assessment.overall_level}")
    print(f"Confidence: {assessment.confidence}")

    print("\n--- Criteria Scores ---")
    for score in assessment.criteria_scores:
        print(f"\n  {score.dimension}: {score.score}/5")
        print(f"  Feedback: {score.feedback}")
        if score.evidence:
            print(f"  Evidence: {score.evidence[0]}")

    print("\n--- Comparative Summary ---")
    print(f"  {assessment.comparative_summary}")

    print("\n--- Strengths ---")
    for s in assessment.strengths:
        print(f"  + {s}")

    print("\n--- Areas to Improve ---")
    for a in assessment.areas_to_improve:
        print(f"  - {a}")

    print("\n--- Recommendations ---")
    for r in assessment.recommendations:
        print(f"  > {r}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Student Assessment Pipeline")
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Index of sample submission to use (0, 1, or 2)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild of the vector store",
    )
    args = parser.parse_args()

    # Set up vector store
    persist_dir = os.path.join(os.path.dirname(__file__), DEFAULT_PERSIST_DIR)
    if args.rebuild or not os.path.exists(persist_dir):
        print("Building vector store (this may take a moment on first run)...")
        vector_store = build_vector_store(persist_directory=persist_dir)
        print("Vector store ready.")
    else:
        print("Loading existing vector store...")
        vector_store = get_vector_store(persist_directory=persist_dir)

    # Select submission
    submission = ALL_SUBMISSIONS[args.sample]
    print(f"\nAssessing submission {args.sample}...")
    print(f"Context: {submission['submission_context']}")
    if submission["student_level_hint"]:
        print(f"Student's self-reported level: {submission['student_level_hint']}")

    # Build and run the graph
    graph = build_graph(vector_store)
    result = graph.invoke(
        {
            "submission_text": submission["submission_text"],
            "submission_context": submission["submission_context"],
            "student_level_hint": submission["student_level_hint"],
        },
        config={"tags": ["p3-student-assessment"]},
    )

    # Display results
    _print_assessment(result["final_assessment"])


if __name__ == "__main__":
    main()
