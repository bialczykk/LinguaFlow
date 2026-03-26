"""Adapter for Project 05 — Content Moderation & QA System.

Handles sys.path setup, environment loading, and wraps the LangGraph
pipeline with interrupt/resume support for use in the Streamlit app.

The graph has two interrupt points (draft_review, final_review) where
the moderator must provide a decision via Command(resume=...).
"""

import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "05-content-moderation-qa"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from adapters._env import ensure_repo_env  # noqa: E402
ensure_repo_env()

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.content_requests import SAMPLE_REQUESTS  # noqa: E402
from models import CONTENT_TYPES, CEFR_LEVELS  # noqa: E402

# -- Build a shared graph with in-memory checkpointer --
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_checkpointer = InMemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


def get_sample_requests() -> list[dict]:
    """Return the sample content requests for quick testing."""
    return list(SAMPLE_REQUESTS)


def get_content_types() -> tuple:
    """Return valid content types."""
    return CONTENT_TYPES


def get_cefr_levels() -> tuple:
    """Return valid CEFR levels."""
    return CEFR_LEVELS


def create_thread_id() -> str:
    """Generate a new unique thread ID for a pipeline run."""
    return str(uuid.uuid4())


def start_pipeline(thread_id: str, content_request: dict) -> dict:
    """Start the content moderation pipeline with a content request.

    Runs the graph until it hits the first interrupt (draft_review).

    Args:
        thread_id: Unique thread ID for this pipeline run.
        content_request: Dict with keys: topic, content_type, difficulty.

    Returns:
        Dict with interrupt info: content, confidence, revision_count, prompt.

    Raises:
        RuntimeError: If the graph invocation fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p5-content-moderation"],
    }

    initial_state = {
        "content_request": content_request,
        "draft_content": "",
        "generation_confidence": 0.0,
        "draft_decision": {},
        "revision_count": 0,
        "polished_content": "",
        "final_decision": {},
        "published": False,
        "publish_metadata": None,
    }

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Content moderation pipeline failed: {e}") from e

    return _get_interrupt_value(thread_id)


def resume_pipeline(thread_id: str, decision: dict) -> dict | None:
    """Resume the pipeline after a moderator decision.

    Args:
        thread_id: The thread ID of the paused pipeline.
        decision: The moderator's decision dict (action, feedback, etc.).

    Returns:
        Dict with next interrupt info, or None if the pipeline finished.

    Raises:
        RuntimeError: If resume fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p5-content-moderation"],
    }

    try:
        _graph.invoke(Command(resume=decision), config=config)
    except Exception as e:
        raise RuntimeError(f"Pipeline resume failed: {e}") from e

    return _get_interrupt_value(thread_id)


def get_state(thread_id: str) -> dict:
    """Get the current state values for a thread.

    Returns the full state dict, or empty dict if thread doesn't exist.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return dict(snapshot.values) if snapshot.values else {}
    except Exception:
        return {}


def get_next_task(thread_id: str) -> list[str]:
    """Get the list of next pending tasks (node names) for a thread.

    Returns an empty list if the graph has finished.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return list(snapshot.next) if snapshot.next else []
    except Exception:
        return []


def _get_interrupt_value(thread_id: str) -> dict | None:
    """Extract the interrupt payload from the current graph state.

    Returns the interrupt value dict, or None if the graph finished
    without interrupting.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        # Check if graph is at an interrupt point
        if snapshot.next:
            # The interrupt value is in the state's tasks
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return task.interrupts[0].value
        return None
    except Exception:
        return None
