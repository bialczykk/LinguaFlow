"""Adapter for Project 08 — Autonomous Operations Orchestrator.

Handles sys.path setup, environment loading, and wraps the LangGraph
orchestrator with interrupt/resume support for use in the Streamlit app.

The graph may interrupt at the approval_gate node (for high-risk actions)
where an operator must provide an approve/reject decision via Command(resume=...).
State is persisted via SqliteSaver so metrics and thread state survive restarts.
"""

import sqlite3
import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "08-autonomous-operations"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from adapters._env import ensure_repo_env  # noqa: E402
ensure_repo_env()

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.sample_requests import SAMPLE_REQUESTS  # noqa: E402

# -- Build a shared graph with SQLite checkpointer for persistence --
# SqliteSaver persists state (including cumulative metrics_store) across
# sessions and app restarts — critical for the metrics dashboard.
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_db_path = _PROJECT_DIR / "orchestrator.db"
_conn = sqlite3.connect(str(_db_path), check_same_thread=False)
_checkpointer = SqliteSaver(conn=_conn)
_graph = build_graph(checkpointer=_checkpointer)


def get_sample_requests() -> list[dict]:
    """Return the sample operation requests for quick testing."""
    return list(SAMPLE_REQUESTS)


def create_thread_id() -> str:
    """Generate a new unique thread ID for a graph run."""
    return str(uuid.uuid4())


def start_request(thread_id: str, request_text: str, metadata: dict) -> dict | None:
    """Start an autonomous operations request.

    Invokes the graph with the request text and metadata. Runs until it
    either completes or hits the approval_gate interrupt (for high-risk actions).

    Args:
        thread_id: Unique thread ID for this run (enables HITL + persistence).
        request_text: The natural language operations request.
        metadata: Dict with user_id, priority, source fields.

    Returns:
        Interrupt payload dict if approval is required, None if completed.

    Raises:
        RuntimeError: If the graph invocation fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p8-autonomous-operations"],
    }

    initial_state = {
        "request": request_text,
        "request_metadata": metadata,
        "department_results": [],
        "task_queue": [],
        "current_task": None,
        "completed_tasks": [],
        "metrics_store": {},
        "classification": {},
        "risk_level": "low",
        "approval_status": "not_required",
        "final_response": "",
        "resolution_status": "",
    }

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Autonomous operations request failed: {e}") from e

    return _get_interrupt_value(thread_id)


def resume_approval(thread_id: str, decision: dict) -> dict | None:
    """Resume the graph after an operator approval/rejection decision.

    Args:
        thread_id: The thread ID of the paused graph.
        decision: Dict with at minimum {"action": "approve" | "reject"}.
                  May include "reason" for rejections.

    Returns:
        Next interrupt payload if another approval is needed,
        None if the graph completed.

    Raises:
        RuntimeError: If resume fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p8-autonomous-operations"],
    }

    try:
        _graph.invoke(Command(resume=decision), config=config)
    except Exception as e:
        raise RuntimeError(f"Approval resume failed: {e}") from e

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


def get_metrics(thread_id: str) -> dict:
    """Get the metrics_store from the current thread state.

    Returns the cumulative metrics dict, or empty dict if unavailable.
    """
    state = get_state(thread_id)
    return state.get("metrics_store", {})


def get_task_queue(thread_id: str) -> list:
    """Get the current task_queue from thread state.

    Returns the list of pending autonomous follow-up tasks,
    or an empty list if the thread has no pending tasks.
    """
    state = get_state(thread_id)
    return state.get("task_queue", [])


def _get_interrupt_value(thread_id: str) -> dict | None:
    """Extract the interrupt payload from the current graph state.

    Returns the interrupt value dict if the graph is paused at an interrupt
    (i.e., waiting for human approval), or None if the graph has finished.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        # Graph is at an interrupt point if snapshot.next is non-empty
        if snapshot.next:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return task.interrupts[0].value
        return None
    except Exception:
        return None
