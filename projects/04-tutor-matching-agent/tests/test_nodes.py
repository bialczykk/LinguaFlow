"""Integration tests for node functions.

These tests hit the LLM (Anthropic API) to verify agent_node behavior,
and unit-test should_continue routing logic.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from models import TutorMatchingState
from nodes import agent_node, should_continue


@pytest.mark.integration
class TestAgentNode:
    def test_gather_phase_responds_conversationally(self):
        state: TutorMatchingState = {
            "messages": [HumanMessage(content="Hi, I need help finding a tutor")],
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        }
        result = agent_node(state)
        assert "messages" in result
        ai_msg = result["messages"][-1]
        assert isinstance(ai_msg, AIMessage)
        assert not ai_msg.tool_calls

    def test_gather_phase_calls_search_when_ready(self):
        state: TutorMatchingState = {
            "messages": [
                HumanMessage(content="I want to improve my grammar skills"),
                AIMessage(content="I'd love to help you find a grammar tutor! What timezone are you in?"),
                HumanMessage(content="I'm in London timezone. Can you find me someone?"),
            ],
            "phase": "gather",
            "preferences": {},
            "search_results": [],
            "selected_tutor": None,
            "booking_confirmation": None,
        }
        result = agent_node(state)
        ai_msg = result["messages"][-1]
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) > 0
        assert ai_msg.tool_calls[0]["name"] == "search_tutors"


class TestShouldContinue:
    def test_routes_to_tools_when_tool_calls_present(self):
        ai_msg = AIMessage(content="", tool_calls=[{
            "id": "call_1", "name": "search_tutors",
            "args": {"specialization": "grammar"},
        }])
        state = {"messages": [ai_msg], "phase": "gather"}
        assert should_continue(state) == "tool_node"

    def test_routes_to_end_when_done(self):
        state = {"messages": [AIMessage(content="Booking confirmed!")], "phase": "done"}
        assert should_continue(state) == "__end__"

    def test_routes_to_end_when_waiting_for_user(self):
        state = {"messages": [AIMessage(content="What specialization are you looking for?")], "phase": "gather"}
        assert should_continue(state) == "__end__"
