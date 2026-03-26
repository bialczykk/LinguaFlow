"""CLI entry point for the Tutor Matching & Scheduling Agent.

Provides an interactive conversation loop where students can find and
book tutors. Supports two persistence modes:
- Default (InMemorySaver): state persists within the session only
- --persist (SqliteSaver): state survives process restarts

Usage:
    python main.py                          # New conversation, in-memory
    python main.py --persist                # New conversation, durable
    python main.py --persist --thread ID    # Resume a durable conversation

LangGraph concepts demonstrated:
- Checkpointer swapping: one-line change between InMemorySaver and SqliteSaver
- Thread management: thread_id in config for conversation isolation
- graph.invoke() with message appending for multi-turn conversation
"""

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from repo root .env
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph


def main():
    parser = argparse.ArgumentParser(
        description="LinguaFlow Tutor Matching Agent — find and book the right tutor"
    )
    parser.add_argument(
        "--persist", action="store_true",
        help="Use SQLite for durable persistence (state survives restarts)",
    )
    parser.add_argument(
        "--thread", type=str, default=None,
        help="Resume an existing conversation by thread ID (requires --persist)",
    )
    args = parser.parse_args()

    # -- Select checkpointer --
    if args.persist:
        db_path = Path(__file__).parent / "tutor_matching.db"
        checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
        persistence_label = f"SQLite ({db_path.name})"
    else:
        checkpointer = InMemorySaver()
        persistence_label = "in-memory (lost on exit)"

    # -- Set up thread --
    thread_id = args.thread or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p4-tutor-matching"],
    }

    # -- Build graph --
    graph = build_graph(checkpointer=checkpointer)

    # -- Print session info --
    print("=" * 60)
    print("LinguaFlow Tutor Matching Agent")
    print("=" * 60)
    print(f"Thread ID:   {thread_id}")
    print(f"Persistence: {persistence_label}")
    if args.thread:
        print("(Resuming existing conversation)")
    print("-" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    # -- Check if resuming an existing conversation --
    is_first_turn = args.thread is None

    # -- Conversation loop --
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Your thread ID for resuming: " + thread_id)
            break

        # Build the input for this turn
        turn_input = {"messages": [HumanMessage(content=user_input)]}

        # On the very first turn of a new conversation, include initial state
        if is_first_turn:
            turn_input.update({
                "phase": "gather",
                "preferences": {},
                "search_results": [],
                "selected_tutor": None,
                "booking_confirmation": None,
            })
            is_first_turn = False

        # Stream the graph execution to show progress
        for chunk in graph.stream(turn_input, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                if node_name == "agent_node" and "messages" in updates:
                    for msg in updates["messages"]:
                        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                            print(f"\nAgent: {msg.content}\n")
                        elif hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"  [Calling tool: {tc['name']}...]")

        # Check if conversation is done
        state = graph.get_state(config)
        if state.values.get("phase") == "done":
            print("\n" + "=" * 60)
            print("Session complete! Thank you for using LinguaFlow.")
            print(f"Thread ID for reference: {thread_id}")
            print("=" * 60)
            break


if __name__ == "__main__":
    main()
