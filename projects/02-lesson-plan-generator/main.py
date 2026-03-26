"""Interactive CLI for the Lesson Plan Generator.

This is the entry point that ties everything together:
1. Runs the intake conversation to gather student info
2. Compiles and invokes the LangGraph StateGraph
3. Streams status updates as each node executes
4. Prints the final structured LessonPlan

Run: python main.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from intake import IntakeConversation
from graph import build_graph
from models import LessonPlan


def print_header():
    """Print the welcome banner."""
    print("\n" + "=" * 60)
    print("  LinguaFlow Lesson Plan Generator")
    print("  Powered by LangGraph + Anthropic Claude")
    print("=" * 60)
    print()


def run_intake() -> dict:
    """Run the intake conversation and return the initial graph state."""
    print("Let's create a personalized lesson plan for you!")
    print("I'll ask a few questions to understand your needs.\n")

    intake = IntakeConversation()

    # Get the first greeting from the advisor
    first_response = intake.ask("Hello!")
    print(f"Advisor: {first_response}\n")

    while not intake.is_complete():
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            sys.exit(0)

        response = intake.ask(user_input)
        print(f"\nAdvisor: {response}\n")

    # Extract the student profile
    profile = intake.get_profile()
    print("\n" + "-" * 40)
    print("Profile gathered:")
    print(f"  Name: {profile.name}")
    print(f"  Level: {profile.proficiency_level}")
    print(f"  Goals: {', '.join(profile.learning_goals)}")
    print(f"  Topics: {', '.join(profile.preferred_topics)}")
    print(f"  Lesson Type: {profile.lesson_type}")
    print("-" * 40 + "\n")

    return {
        "student_profile": profile,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


def print_lesson_plan(plan: LessonPlan):
    """Print the final lesson plan in a readable format."""
    print("\n" + "=" * 60)
    print(f"  {plan.title}")
    print(f"  Level: {plan.level} | Type: {plan.lesson_type}")
    print(f"  Duration: {plan.estimated_duration_minutes} minutes")
    print("=" * 60)

    print("\nObjectives:")
    for obj in plan.objectives:
        print(f"  - {obj}")

    print(f"\nWarm-Up:\n  {plan.warm_up}")

    print("\nMain Activities:")
    for i, activity in enumerate(plan.main_activities, 1):
        print(f"\n  {i}. {activity.name} ({activity.duration_minutes} min)")
        print(f"     {activity.description}")
        if activity.materials:
            print(f"     Materials: {', '.join(activity.materials)}")

    print(f"\nWrap-Up:\n  {plan.wrap_up}")
    print(f"\nHomework:\n  {plan.homework}")
    print()


def main():
    """Run the full lesson plan generation pipeline."""
    print_header()

    # Phase 1: Intake conversation
    initial_state = run_intake()

    # Phase 2: Run the LangGraph pipeline
    print("Generating your lesson plan...\n")
    graph = build_graph()

    # Stream with "updates" mode to show progress as each node completes
    node_labels = {
        "research": "Researching materials...",
        "draft_conversation": "Drafting conversation lesson...",
        "draft_grammar": "Drafting grammar lesson...",
        "draft_exam_prep": "Drafting exam prep lesson...",
        "review": "Reviewing draft...",
        "finalize": "Finalizing lesson plan...",
    }

    final_state = None
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

        final_state = chunk

    # Get the complete final state by invoking (stream doesn't return full state easily)
    result = graph.invoke(initial_state)
    plan = result["final_plan"]

    if plan:
        print_lesson_plan(plan)
        print(f"Revisions needed: {result['revision_count']}")
    else:
        print("Something went wrong — no lesson plan was produced.")


if __name__ == "__main__":
    main()
