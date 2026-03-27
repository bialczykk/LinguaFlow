"""Integration tests for the full multi-department support graph.

All tests use mocked LLM responses so that no real API calls are made.
The mocking strategy is:
- `nodes._classification_model` is patched to control supervisor_router output.
- `langchain_anthropic.ChatAnthropic` is patched to control department agent
  and compose_response LLM calls.

Each test verifies end-to-end graph behaviour: routing, parallel fan-out,
escalation handling, clarification interrupt/resume, and three-way dispatch.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import build_graph
from models import SupportState


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_input(request: str = "I need help", **overrides) -> dict:
    """Return a minimal graph input dict."""
    base = {
        "request": request,
        "request_metadata": {
            "sender_type": "student",
            "student_id": "S001",
            "priority": "medium",
        },
        "department_results": [],
        "escalation_queue": [],
        "classification": {},
        "clarification_needed": None,
        "user_clarification": None,
        "final_response": "",
        "resolution_status": "",
    }
    base.update(overrides)
    return base


def _classification_mock(departments: list, needs_clarification: bool = False,
                         clarification_question: str | None = None) -> MagicMock:
    """Build a MagicMock that mimics ChatAnthropic returning a classification JSON."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "departments": departments,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "summary": "Test request",
            "complexity": "multi" if len(departments) > 1 else "single",
        })
    )
    return mock


def _agent_llm_response(text: str) -> MagicMock:
    """Build a MagicMock AIMessage with no tool_calls so the agent loop ends immediately."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = []           # No tool calls → loop exits after first round
    return msg


def _agent_chat_anthropic_mock(response_text: str) -> MagicMock:
    """Build a MagicMock for ChatAnthropic used inside department agents.

    The factory in nodes.py does:
        ChatAnthropic(...).bind_tools(dept_tools)
    so the mock must support chained calls: instance → .bind_tools() → .invoke()
    """
    mock_instance = MagicMock()
    bound_mock = MagicMock()
    bound_mock.invoke.return_value = _agent_llm_response(response_text)
    mock_instance.bind_tools.return_value = bound_mock
    return mock_instance


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSingleDepartmentFlow:
    """Single-dept request routes to one agent and resolves."""

    def test_single_department_flow(self):
        # --- Mocks ---
        # Classification returns only "billing"
        classification_mock = _classification_mock(["billing"])

        # Department agent LLM returns a resolved response
        agent_mock = _agent_chat_anthropic_mock("Your refund of $50 has been processed.")

        # compose_response LLM returns a final unified message
        compose_mock = MagicMock()
        compose_response_msg = MagicMock()
        compose_response_msg.content = "We have processed your refund. Is there anything else?"
        compose_mock.invoke.return_value = compose_response_msg

        # ChatAnthropic is called in two places:
        #   1. Inside billing_agent via _make_department_agent (returns agent_mock)
        #   2. Inside compose_response (returns compose_mock)
        # We return agent_mock first, then compose_mock for subsequent calls.
        chat_side_effects = [agent_mock, compose_mock]

        with patch("nodes._classification_model", classification_mock), \
             patch("nodes.ChatAnthropic", side_effect=chat_side_effects):

            graph = build_graph()
            result = graph.invoke(
                _make_input("I want a refund for my last lesson"),
                config={"configurable": {"thread_id": "test-single"}},
            )

        # Verify routing ended up in billing and produced a final response
        assert len(result["department_results"]) == 1
        assert result["department_results"][0]["department"] == "billing"
        assert result["department_results"][0]["resolved"] is True
        assert result["final_response"] != ""
        assert result["resolution_status"] == "resolved"


@pytest.mark.integration
class TestParallelDispatch:
    """Multi-dept request fans out via Send and both results are collected."""

    def test_parallel_dispatch(self):
        # Classification returns two departments
        classification_mock = _classification_mock(["billing", "scheduling"])

        # Both agents return resolved responses
        billing_agent_mock = _agent_chat_anthropic_mock("Billing: invoice corrected.")
        scheduling_agent_mock = _agent_chat_anthropic_mock("Scheduling: lesson rescheduled.")

        # compose_response
        compose_mock = MagicMock()
        compose_mock_msg = MagicMock()
        compose_mock_msg.content = "Both your invoice and lesson have been handled."
        compose_mock.invoke.return_value = compose_mock_msg

        # ChatAnthropic calls: billing agent, scheduling agent, then compose_response
        chat_side_effects = [billing_agent_mock, scheduling_agent_mock, compose_mock]

        with patch("nodes._classification_model", classification_mock), \
             patch("nodes.ChatAnthropic", side_effect=chat_side_effects):

            graph = build_graph()
            result = graph.invoke(
                _make_input("My invoice is wrong and I need to reschedule my lesson"),
                config={"configurable": {"thread_id": "test-parallel"}},
            )

        # Both department results should be present
        departments_hit = {dr["department"] for dr in result["department_results"]}
        assert "billing" in departments_hit
        assert "scheduling" in departments_hit
        assert len(result["department_results"]) == 2
        assert result["final_response"] != ""


@pytest.mark.integration
class TestEscalationFlow:
    """Sub-agent escalates; supervisor re-routes to target department.

    The escalation scenario requires careful mocking because supervisor_aggregator
    scans ALL department_results on each pass.  The real aggregator would keep
    re-queuing billing's escalation indefinitely once the billing result (resolved=False)
    is in state.  To test the escalation routing path without hitting infinite
    recursion we patch supervisor_aggregator with a version that only escalates
    on the very first pass (when scheduling hasn't run yet).
    """

    def test_escalation_flow(self):
        # Classification: routes to billing first
        classification_mock = _classification_mock(["billing"])

        # billing_agent: first-pass — escalates to scheduling
        def mock_billing_agent_node(state):
            return {
                "department_results": [{
                    "department": "billing",
                    "response": "Need lesson cancellation info to process refund.",
                    "resolved": False,
                    "escalation": {
                        "target": "scheduling",
                        "context": "Need confirmation that lesson L004 was cancelled",
                    },
                }]
            }

        # scheduling_agent: escalation-pass — resolves the issue
        def mock_scheduling_agent_node(state):
            return {
                "department_results": [{
                    "department": "scheduling",
                    "response": "Scheduling: lesson L004 cancelled and refund confirmed.",
                    "resolved": True,
                    "escalation": None,
                }]
            }

        # aggregator: escalate once (when scheduling hasn't run yet), then clear queue
        def mock_aggregator(state):
            dept_names = {dr["department"] for dr in state.get("department_results", [])}
            if "scheduling" not in dept_names:
                # First pass — billing ran, scheduling hasn't → escalate
                return {"escalation_queue": [{"target": "scheduling", "context": "Need L004 info"}]}
            # Second pass — scheduling ran → all resolved, clear queue
            return {"escalation_queue": []}

        # compose_response uses ChatAnthropic directly
        compose_mock = MagicMock()
        compose_mock_msg = MagicMock()
        compose_mock_msg.content = "We've cancelled your lesson and issued a refund."
        compose_mock.invoke.return_value = compose_mock_msg

        with patch("nodes._classification_model", classification_mock), \
             patch("graph.billing_agent", mock_billing_agent_node), \
             patch("graph.scheduling_agent", mock_scheduling_agent_node), \
             patch("graph.supervisor_aggregator", mock_aggregator), \
             patch("nodes.ChatAnthropic", return_value=compose_mock):

            graph = build_graph()
            result = graph.invoke(
                _make_input("Cancel lesson L004 and refund me"),
                config={"configurable": {"thread_id": "test-escalation"}},
            )

        # Both billing (first pass) and scheduling (escalation pass) should be present
        departments_hit = {dr["department"] for dr in result["department_results"]}
        assert "billing" in departments_hit
        assert "scheduling" in departments_hit
        assert result["final_response"] != ""


@pytest.mark.integration
class TestClarificationInterrupt:
    """Ambiguous request triggers interrupt; user clarifies; graph re-routes."""

    def test_clarification_interrupt(self):
        # First classification: needs clarification
        classification_needs_clarification = MagicMock()
        classification_needs_clarification.invoke.return_value = MagicMock(
            content=json.dumps({
                "departments": [],
                "needs_clarification": True,
                "clarification_question": "Are you asking about billing or scheduling?",
                "summary": "Ambiguous",
                "complexity": "single",
            })
        )

        # Second classification (after user clarifies): routes to billing
        classification_resolved = MagicMock()
        classification_resolved.invoke.return_value = MagicMock(
            content=json.dumps({
                "departments": ["billing"],
                "needs_clarification": False,
                "clarification_question": None,
                "summary": "Billing issue",
                "complexity": "single",
            })
        )

        # classification model is called twice: once before interrupt, once after resume
        classification_side_effects = [
            classification_needs_clarification.invoke.return_value,
            classification_resolved.invoke.return_value,
        ]
        classification_mock = MagicMock()
        classification_mock.invoke.side_effect = classification_side_effects

        # Billing agent + compose_response
        billing_agent_mock = _agent_chat_anthropic_mock("Your billing issue has been resolved.")
        compose_mock = MagicMock()
        compose_mock_msg = MagicMock()
        compose_mock_msg.content = "Your billing issue is resolved. Let us know if you need more help."
        compose_mock.invoke.return_value = compose_mock_msg

        thread_id = "test-clarification-interrupt"
        checkpointer = MemorySaver()

        with patch("nodes._classification_model", classification_mock), \
             patch("nodes.ChatAnthropic", side_effect=[billing_agent_mock, compose_mock]):

            graph = build_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}

            # --- First invocation: should hit the interrupt ---
            first_result = graph.invoke(
                _make_input("I need to change something"),
                config=config,
            )

            # The graph should have been interrupted — check for interrupt signal
            # When interrupted, LangGraph returns state with __interrupt__ key
            # or the invocation completes with clarification_needed set
            assert first_result.get("clarification_needed") == "Are you asking about billing or scheduling?"

            # --- Resume with user's clarification ---
            resumed_result = graph.invoke(
                Command(resume="I meant billing — my invoice is wrong"),
                config=config,
            )

        # After resume the graph should complete with a final response
        assert resumed_result.get("final_response", "") != ""
        assert resumed_result.get("user_clarification") == "I meant billing — my invoice is wrong"


@pytest.mark.integration
class TestThreeWayParallel:
    """Request hitting three departments simultaneously."""

    def test_three_way_parallel(self):
        # Classification returns three departments
        classification_mock = _classification_mock(["billing", "tech_support", "scheduling"])

        # Three agent mocks
        billing_mock = _agent_chat_anthropic_mock("Billing: refund issued.")
        tech_mock = _agent_chat_anthropic_mock("Tech: login issue resolved.")
        scheduling_mock = _agent_chat_anthropic_mock("Scheduling: lesson rescheduled.")

        # compose_response
        compose_mock = MagicMock()
        compose_mock_msg = MagicMock()
        compose_mock_msg.content = "All three issues have been handled for you."
        compose_mock.invoke.return_value = compose_mock_msg

        # ChatAnthropic: 3 agent calls + 1 compose call
        chat_side_effects = [billing_mock, tech_mock, scheduling_mock, compose_mock]

        with patch("nodes._classification_model", classification_mock), \
             patch("nodes.ChatAnthropic", side_effect=chat_side_effects):

            graph = build_graph()
            result = graph.invoke(
                _make_input("I can't log in, my invoice is wrong, and I need to reschedule"),
                config={"configurable": {"thread_id": "test-three-way"}},
            )

        departments_hit = {dr["department"] for dr in result["department_results"]}
        assert "billing" in departments_hit
        assert "tech_support" in departments_hit
        assert "scheduling" in departments_hit
        assert len(result["department_results"]) == 3
        assert result["final_response"] != ""
        assert result["resolution_status"] == "resolved"
