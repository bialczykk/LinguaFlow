"""Adapter for Project 06 — Multi-Department Support System.

Handles sys.path setup, environment loading, and wraps the LangGraph
pipeline with interrupt/resume support for use in the Streamlit app.

The graph can interrupt at the ask_clarification node when the supervisor
cannot classify the request without more information from the user.
After the user responds, execution resumes via Command(resume=...) and
the classifier re-routes to the appropriate department(s).
"""

import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "06-multi-department-support"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from adapters._env import ensure_repo_env  # noqa: E402
ensure_repo_env()

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.support_requests import SAMPLE_REQUESTS  # noqa: E402

# -- Build a shared graph with in-memory checkpointer --
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_checkpointer = InMemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


def get_sample_requests() -> list[dict]:
    """Return the sample support requests for quick testing."""
    return list(SAMPLE_REQUESTS)


def create_thread_id() -> str:
    """Generate a new unique thread ID for a support request run."""
    return str(uuid.uuid4())


def start_request(thread_id: str, request_text: str, metadata: dict) -> dict | str:
    """Submit a support request to the graph.

    Runs the graph until it either completes or hits a clarification interrupt.

    Args:
        thread_id: Unique thread ID for this run.
        request_text: The user's support request text.
        metadata: Dict with sender context (sender_type, student_id, priority).

    Returns:
        - A string clarification question if the graph interrupted.
        - A dict with final state values if the graph completed.

    Raises:
        RuntimeError: If the graph invocation fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p6-multi-department-support"],
    }

    initial_state = {
        "request": request_text,
        "request_metadata": metadata,
        "classification": {},
        "department_results": [],
        "escalation_queue": [],
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Support request failed: {e}") from e

    # Check if the graph interrupted for clarification
    question = _get_interrupt_value(thread_id)
    if question is not None:
        return question  # string — the clarification question

    # Graph completed — return the final state
    return get_state(thread_id)


def resume_with_clarification(thread_id: str, user_response: str) -> dict | str:
    """Resume the graph after the user provides clarification.

    Args:
        thread_id: The thread ID of the paused graph.
        user_response: The user's answer to the clarification question.

    Returns:
        - A string clarification question if another interrupt occurs.
        - A dict with final state values if the graph completed.

    Raises:
        RuntimeError: If resume fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p6-multi-department-support"],
    }

    try:
        _graph.invoke(Command(resume=user_response), config=config)
    except Exception as e:
        raise RuntimeError(f"Resume after clarification failed: {e}") from e

    # Check if the graph interrupted again
    question = _get_interrupt_value(thread_id)
    if question is not None:
        return question  # string — another clarification question

    # Graph completed — return the final state
    return get_state(thread_id)


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


def _get_interrupt_value(thread_id: str) -> str | None:
    """Extract the clarification question from an interrupted graph state.

    Returns the interrupt value string (the question), or None if the graph
    finished without interrupting.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        # Graph is paused at an interrupt point when snapshot.next is non-empty
        if snapshot.next:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return task.interrupts[0].value
        return None
    except Exception:
        return None
