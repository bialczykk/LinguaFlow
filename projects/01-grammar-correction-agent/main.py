"""Interactive CLI for the Grammar Correction Agent.

This is the entry point for the application. It:
1. Loads environment variables (.env) for API keys and LangSmith config
2. Lets the student pick a sample text or enter their own
3. Runs the grammar analysis chain and displays structured feedback
4. Enters a conversation loop for follow-up questions
5. Supports submitting new text (prefix with 'new:') or quitting

Run: python main.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the root .env file
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from chains import analyze_grammar
from conversation import ConversationHandler
from data.sample_texts import SAMPLE_TEXTS
from models import GrammarFeedback


def display_feedback(feedback: GrammarFeedback) -> None:
    """Display grammar feedback in a clean, readable format."""
    print("\n" + "=" * 60)
    print("  GRAMMAR ANALYSIS RESULTS")
    print("=" * 60)

    # -- Proficiency Assessment --
    p = feedback.proficiency
    print(f"\n  CEFR Level: {p.cefr_level}")
    print(f"  {p.summary}")
    print(f"\n  Strengths: {', '.join(p.strengths)}")
    print(f"  Improve:   {', '.join(p.areas_to_improve)}")

    # -- Grammar Issues --
    if feedback.issues:
        print(f"\n  ISSUES FOUND: {len(feedback.issues)}")
        print("-" * 60)
        for i, issue in enumerate(feedback.issues, 1):
            severity_icon = {"minor": ".", "moderate": "!", "major": "!!"}[
                issue.severity
            ]
            print(f"\n  {i}. [{severity_icon} {issue.severity.upper()}] {issue.error_category}")
            print(f"     Original:  \"{issue.original_text}\"")
            print(f"     Corrected: \"{issue.corrected_text}\"")
            print(f"     Why: {issue.explanation}")
    else:
        print("\n  No grammar issues found — great writing!")

    # -- Corrected Full Text --
    print("\n" + "-" * 60)
    print("  CORRECTED TEXT:")
    print(f"  {feedback.corrected_full_text}")
    print("=" * 60)


def get_student_text() -> str:
    """Prompt the student to enter text or pick a sample."""
    print("\n  LINGUAFLOW GRAMMAR CORRECTION AGENT")
    print("=" * 60)
    print("\nChoose a sample text or enter your own:\n")

    for i, sample in enumerate(SAMPLE_TEXTS, 1):
        print(f"  {i}. {sample['label']}")

    print(f"  {len(SAMPLE_TEXTS) + 1}. Enter your own text")
    print()

    while True:
        choice = input("Your choice: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(SAMPLE_TEXTS):
                text = SAMPLE_TEXTS[idx - 1]["text"]
                print(f"\nSelected: {SAMPLE_TEXTS[idx - 1]['label']}")
                print(f"Text: {text[:80]}...")
                return text
            elif idx == len(SAMPLE_TEXTS) + 1:
                print("\nType or paste your text (press Enter twice to submit):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines:
                            break
                    else:
                        lines.append(line)
                return "\n".join(lines)
        print("Invalid choice, please try again.")


def conversation_loop(original_text: str, feedback: GrammarFeedback) -> str | None:
    """Run the follow-up conversation loop.

    Returns new text if student submits with 'new:', or None to quit.
    """
    handler = ConversationHandler(original_text=original_text, feedback=feedback)

    print("\n  Follow-up Chat")
    print("-" * 60)
    print("  Ask me about your feedback, or:")
    print("  - Type 'new: <your text>' to analyze new writing")
    print("  - Type 'quit' or 'exit' to end the session")
    print("-" * 60)

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep practicing your English!")
            return None

        if user_input.lower().startswith("new:"):
            new_text = user_input[4:].strip()
            if new_text:
                return new_text
            else:
                print("Tutor: Please include your text after 'new:'. For example: new: I goes to school.")
                continue

        print("\nTutor: ", end="")
        response = handler.ask(user_input)
        print(response)


def main() -> None:
    """Main application loop."""
    text = get_student_text()

    while True:
        print("\nAnalyzing your writing...")
        feedback = analyze_grammar(text)
        display_feedback(feedback)

        new_text = conversation_loop(text, feedback)

        if new_text is None:
            break
        else:
            text = new_text
            print(f"\nNew text received! Analyzing...")


if __name__ == "__main__":
    main()
