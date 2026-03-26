"""Adapter for Project 04 — Tutor Matching & Scheduling Agent.

Handles sys.path setup, environment loading, and wraps the LangGraph
agent with error handling for use in the Streamlit app.
"""

import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "04-tutor-matching-agent"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from adapters._env import ensure_repo_env  # noqa: E402
ensure_repo_env()

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.tutors import TUTORS  # noqa: E402

# -- Build a shared graph with in-memory checkpointer --
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

_checkpointer = InMemorySaver()
_graph = build_graph(checkpointer=_checkpointer)


def get_sample_scenarios() -> list[dict[str, str]]:
    """Return sample opening messages for quick testing.

    Each dict has keys: "label" (display name) and "message" (opening text).
    """
    return [
        {
            "label": "Grammar tutor in Europe",
            "message": "Hi! I'm looking for a grammar tutor in a European timezone. I'm preparing for my FCE exam.",
        },
        {
            "label": "Business English tutor",
            "message": "I need a business English tutor who can help me with professional presentations. My name is Alex.",
        },
        {
            "label": "Conversation practice, any timezone",
            "message": "I want to practice conversation skills. I'm flexible on timezone and my budget is around $35/hour. My name is Sam.",
        },
    ]


def create_thread_id() -> str:
    """Generate a new unique thread ID for a conversation session."""
    return str(uuid.uuid4())


def send_message(thread_id: str, user_input: str, is_first_turn: bool = False) -> str:
    """Send a user message to the agent and return the assistant's response.

    Args:
        thread_id: Unique thread ID for conversation isolation.
        user_input: The user's message text.
        is_first_turn: If True, includes initial state fields in the input.

    Returns:
        The agent's text response (may include tool call indicators).

    Raises:
        RuntimeError: If the graph invocation fails.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p4-tutor-matching"],
    }

    # Build input for this turn
    turn_input = {"messages": [HumanMessage(content=user_input)]}

    # On the first turn, include initial state
    if is_first_turn:
        turn_input.update({
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        })

    try:
        # Collect the agent's response from streaming
        response_parts = []
        tool_calls_made = []

        for chunk in _graph.stream(turn_input, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                if node_name == "agent_node" and "messages" in updates:
                    for msg in updates["messages"]:
                        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                            response_parts.append(msg.content)
                        elif hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls_made.append(tc["name"])

        # Build the response text
        lines = []
        if tool_calls_made:
            tool_labels = {
                "search_tutors": "Searching tutors",
                "check_availability": "Checking availability",
                "book_session": "Booking session",
            }
            for tc in tool_calls_made:
                label = tool_labels.get(tc, tc)
                lines.append(f"*{label}...*")
            lines.append("")

        if response_parts:
            lines.append(response_parts[-1])  # Use the final response
        elif not tool_calls_made:
            lines.append("I'm processing your request...")

        return "\n".join(lines)

    except Exception as e:
        raise RuntimeError(f"Tutor matching agent failed: {e}") from e


def get_phase(thread_id: str) -> str:
    """Get the current conversation phase for a thread.

    Returns one of: gather, search, present, book, done.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = _graph.get_state(config)
        return state.values.get("phase", "gather")
    except Exception:
        return "gather"
