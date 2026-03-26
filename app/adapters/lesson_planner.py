"""Adapter for Project 02 — Lesson Plan Generator.

Handles sys.path setup, environment loading, and wraps project functions
with error handling for use in the Streamlit app.
"""

import sys
from pathlib import Path

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P2_DIR = _REPO_ROOT / "projects" / "02-lesson-plan-generator"
if str(_P2_DIR) not in sys.path:
    sys.path.insert(0, str(_P2_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Import project modules --
from intake import IntakeConversation  # noqa: E402
from graph import build_graph  # noqa: E402
from models import LessonPlan, StudentProfile  # noqa: E402
from data.sample_profiles import (  # noqa: E402
    BEGINNER_CONVERSATION,
    INTERMEDIATE_CONVERSATION,
    BEGINNER_GRAMMAR,
    INTERMEDIATE_GRAMMAR,
    EXAM_PREP_INTERMEDIATE,
    EXAM_PREP_ADVANCED,
)

# Pre-built list of sample profiles for the UI
SAMPLE_PROFILES: list[tuple[str, StudentProfile]] = [
    ("Yuki — A2 Conversation", BEGINNER_CONVERSATION),
    ("Carlos — B1 Conversation", INTERMEDIATE_CONVERSATION),
    ("Fatima — A1 Grammar", BEGINNER_GRAMMAR),
    ("Hans — B2 Grammar", INTERMEDIATE_GRAMMAR),
    ("Mei — B2 Exam Prep", EXAM_PREP_INTERMEDIATE),
    ("Olga — C1 Exam Prep", EXAM_PREP_ADVANCED),
]


def create_intake() -> IntakeConversation:
    """Create a new intake conversation instance."""
    return IntakeConversation()


def ask_intake(intake: IntakeConversation, message: str) -> str:
    """Send a message in the intake conversation.

    Returns:
        The intake assistant's response.

    Raises:
        RuntimeError: If the intake call fails.
    """
    try:
        return intake.ask(message)
    except Exception as e:
        raise RuntimeError(f"Intake conversation failed: {e}") from e


def is_intake_complete(intake: IntakeConversation) -> bool:
    """Check whether the intake has gathered enough information."""
    return intake.is_complete()


def extract_profile(intake: IntakeConversation) -> StudentProfile:
    """Extract the student profile from a completed intake.

    Raises:
        RuntimeError: If extraction fails.
    """
    try:
        return intake.get_profile()
    except Exception as e:
        raise RuntimeError(f"Profile extraction failed: {e}") from e


def generate_plan(profile: StudentProfile) -> LessonPlan:
    """Run the lesson plan generation graph for a student profile.

    Args:
        profile: A complete StudentProfile.

    Returns:
        The generated LessonPlan.

    Raises:
        RuntimeError: If graph execution fails.
    """
    try:
        graph = build_graph()
        initial_state = {
            "student_profile": profile,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        }
        result = graph.invoke(initial_state, config={"tags": ["p2-lesson-plan-generator"]})
        return result["final_plan"]
    except Exception as e:
        raise RuntimeError(f"Lesson plan generation failed: {e}") from e
