"""Adapter for Project 01 — Grammar Correction Agent.

Handles sys.path setup, environment loading, and wraps project functions
with error handling for use in the Streamlit app.
"""

import sys
from pathlib import Path

# -- Path setup: add P1 project directory to sys.path so its modules can be imported --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P1_DIR = _REPO_ROOT / "projects" / "01-grammar-correction-agent"
if str(_P1_DIR) not in sys.path:
    sys.path.insert(0, str(_P1_DIR))

# -- Load environment variables from repo root .env --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Import project modules (after path setup) --
from chains import analyze_grammar  # noqa: E402
from conversation import ConversationHandler  # noqa: E402
from data.sample_texts import SAMPLE_TEXTS  # noqa: E402
from models import GrammarFeedback  # noqa: E402


def get_sample_texts() -> list[dict[str, str]]:
    """Return the list of sample texts available for testing.

    Each dict has keys: "label" (display name) and "text" (student writing).
    """
    return SAMPLE_TEXTS


def run_analysis(student_text: str) -> GrammarFeedback:
    """Analyze student writing and return structured grammar feedback.

    Args:
        student_text: The student's writing to analyze.

    Returns:
        GrammarFeedback with issues, proficiency assessment, and corrected text.

    Raises:
        RuntimeError: If the analysis fails.
    """
    try:
        return analyze_grammar(student_text)
    except Exception as e:
        raise RuntimeError(f"Grammar analysis failed: {e}") from e


def create_conversation(
    original_text: str, feedback: GrammarFeedback
) -> ConversationHandler:
    """Create a new conversation handler for follow-up questions.

    Args:
        original_text: The student's original writing.
        feedback: The GrammarFeedback from run_analysis().

    Returns:
        A ConversationHandler ready to accept .ask() calls.
    """
    return ConversationHandler(original_text=original_text, feedback=feedback)


def ask_followup(handler: ConversationHandler, message: str) -> str:
    """Send a follow-up question and get the tutor's response.

    Args:
        handler: An active ConversationHandler.
        message: The user's follow-up question.

    Returns:
        The tutor's response string.

    Raises:
        RuntimeError: If the conversation call fails.
    """
    try:
        return handler.ask(message)
    except Exception as e:
        raise RuntimeError(f"Conversation failed: {e}") from e
