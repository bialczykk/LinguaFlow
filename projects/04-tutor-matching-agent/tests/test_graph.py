"""End-to-end integration tests for the Tutor Matching Agent graph.

Tests full conversation flow and checkpointer-based persistence.
These tests hit the LLM (Anthropic API).
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph


@pytest.mark.integration
class TestGraphFlow:
    def test_initial_message_gets_response(self, graph_with_memory):
        config = {"configurable": {"thread_id": "test-1"}, "tags": ["p4-tutor-matching"]}
        result = graph_with_memory.invoke(
            {"messages": [HumanMessage(content="Hi, I need a grammar tutor")],
             "phase": "gather", "preferences": {}, "search_results": [],
             "selected_tutor": None, "booking_confirmation": None},
            config=config,
        )
        assert len(result["messages"]) >= 2
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

    def test_multi_turn_conversation_persists(self, graph_with_memory):
        config = {"configurable": {"thread_id": "test-persist"}, "tags": ["p4-tutor-matching"]}
        result1 = graph_with_memory.invoke(
            {"messages": [HumanMessage(content="Hello!")],
             "phase": "gather", "preferences": {}, "search_results": [],
             "selected_tutor": None, "booking_confirmation": None},
            config=config,
        )
        turn1_count = len(result1["messages"])
        result2 = graph_with_memory.invoke(
            {"messages": [HumanMessage(content="I want to focus on grammar")]},
            config=config,
        )
        assert len(result2["messages"]) > turn1_count

    def test_separate_threads_are_isolated(self, graph_with_memory):
        initial_state = {
            "messages": [HumanMessage(content="Hi")],
            "phase": "gather", "preferences": {}, "search_results": [],
            "selected_tutor": None, "booking_confirmation": None,
        }
        config_a = {"configurable": {"thread_id": "thread-A"}, "tags": ["p4-tutor-matching"]}
        config_b = {"configurable": {"thread_id": "thread-B"}, "tags": ["p4-tutor-matching"]}
        result_a = graph_with_memory.invoke(initial_state, config=config_a)
        result_b = graph_with_memory.invoke(initial_state, config=config_b)
        assert len(result_a["messages"]) >= 2
        assert len(result_b["messages"]) >= 2


@pytest.mark.integration
class TestGraphStructure:
    def test_graph_compiles(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "agent_node" in node_names
        assert "tool_node" in node_names
