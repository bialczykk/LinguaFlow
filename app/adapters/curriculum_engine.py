"""Adapter for Project 07 — Intelligent Curriculum Engine.

Handles sys.path setup, environment loading, and wraps the LangGraph
pipeline with interrupt/resume support for use in the Streamlit app.

The graph has four interrupt points (review_plan, review_lesson,
review_exercises, review_assessment) where the moderator must provide
a decision via Command(resume=...).
"""

import sys
import traceback
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "07-curriculum-engine"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment (root .env + Streamlit Cloud secrets) --
from adapters._env import ensure_repo_env  # noqa: E402
ensure_repo_env()

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.sample_requests import SAMPLE_REQUESTS  # noqa: E402
from models import CEFR_LEVELS  # noqa: E402

# -- Build a shared graph with in-memory checkpointer --
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_checkpointer = InMemorySaver()
# Project graph only wires checkpointer; DeepAgents use the shared store inside agents.py.
_graph = build_graph(checkpointer=_checkpointer)

# -- Step names for progress tracking --
STEPS = [
    ("plan_curriculum", "Planning"),
    ("review_plan", "Review Plan"),
    ("generate_lesson", "Lesson"),
    ("review_lesson", "Review Lesson"),
    ("generate_exercises", "Exercises"),
    ("review_exercises", "Review Exercises"),
    ("generate_assessment", "Assessment"),
    ("review_assessment", "Review Assessment"),
    ("assemble_module", "Assembling"),
]

# Review steps where HITL interrupt happens
REVIEW_STEPS = {"review_plan", "review_lesson", "review_exercises", "review_assessment"}


def _normalize_resume_decision(decision: object) -> dict:
    """Normalize moderator decision to the dict format expected by review nodes."""
    if isinstance(decision, dict):
        return decision
    if isinstance(decision, str):
        return {"action": decision}
    raise TypeError(f"Invalid decision type: {type(decision).__name__}")


def get_sample_requests() -> list[dict]:
    """Return sample curriculum requests for quick testing."""
    return list(SAMPLE_REQUESTS)


def get_cefr_levels() -> tuple:
    """Return valid CEFR levels."""
    return CEFR_LEVELS


def create_thread_id() -> str:
    """Generate a new unique thread ID for a pipeline run."""
    return str(uuid.uuid4())


def start_pipeline(thread_id: str, curriculum_request: dict) -> dict | None:
    """Start the curriculum engine pipeline.

    Runs the graph until it hits the first interrupt (review_plan).

    Returns:
        Dict with interrupt info, or None if pipeline finished.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p7-curriculum-engine"],
    }

    initial_state = {
        "curriculum_request": curriculum_request,
        "curriculum_plan": None,
        "plan_feedback": "",
        "lesson": None,
        "lesson_feedback": "",
        "exercises": None,
        "exercises_feedback": "",
        "assessment": None,
        "assessment_feedback": "",
        "assembled_module": None,
        "current_step": "plan_curriculum",
    }

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Curriculum engine pipeline failed: {e}") from e

    return _get_interrupt_value(thread_id)


def resume_pipeline(thread_id: str, decision: dict) -> dict | None:
    """Resume the pipeline after a moderator decision.

    Returns:
        Dict with next interrupt info, or None if pipeline finished.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p7-curriculum-engine"],
    }

    try:
        normalized_decision = _normalize_resume_decision(decision)
        _graph.invoke(Command(resume=normalized_decision), config=config)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Pipeline resume failed: {e}") from e

    return _get_interrupt_value(thread_id)


def get_state(thread_id: str) -> dict:
    """Get the current state values for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return dict(snapshot.values) if snapshot.values else {}
    except Exception:
        return {}


def get_current_step(thread_id: str) -> str | None:
    """Get the name of the next pending node."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return list(snapshot.next)[0] if snapshot.next else None
    except Exception:
        return None


def _get_interrupt_value(thread_id: str) -> dict | None:
    """Extract the interrupt payload from the current graph state."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        # Newer LangGraph snapshots may expose interrupts directly.
        snapshot_interrupts = getattr(snapshot, "interrupts", None)
        if snapshot_interrupts:
            first = snapshot_interrupts[0]
            return first.value if hasattr(first, "value") else first

        if snapshot.next:
            for task in snapshot.tasks:
                task_interrupts = getattr(task, "interrupts", None)
                if task_interrupts:
                    first = task_interrupts[0]
                    return first.value if hasattr(first, "value") else first
        return None
    except Exception:
        return None
