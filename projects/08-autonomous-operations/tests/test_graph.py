"""Integration tests for the autonomous operations orchestrator graph — Project 08.

All tests use mocked LLMs and mocked DeepAgents so that no real API calls are made.

Mocking strategy:
- `nodes._classifier_model`  : patch to control request_classifier's LLM output.
- `nodes.ChatAnthropic`      : patch to control compose_output's LLM (it instantiates
                               a fresh ChatAnthropic() on each call, so we patch the
                               class-level reference imported in nodes.py).
- `nodes.DEPARTMENT_AGENTS`  : patch the dispatcher dict imported by nodes.py so that
                               department_executor receives a mock agent factory, keeping
                               tests fully isolated from DeepAgent internals.

Each test exercises end-to-end graph behaviour: low-risk auto-execution, high-risk
interrupt/resume, cascading follow-ups, parallel fan-out, rejection, chained tasks,
and metrics accumulation.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import build_graph


# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> dict:
    """Return a minimal valid OrchestratorState dict for graph invocation.

    All fields required by the state schema are present. Override any of them
    for test-specific setup via keyword arguments.
    """
    defaults = {
        "request": "Test request",
        "request_metadata": {"user_id": "admin", "priority": "medium", "source": "test"},
        "classification": {},
        "risk_level": "",
        "approval_status": "",
        "department_results": [],
        "task_queue": [],
        "current_task": None,
        "completed_tasks": [],
        "metrics_store": {
            "students_onboarded": 0,
            "tutors_assigned": 0,
            "content_generated": 0,
            "content_published": 0,
            "qa_reviews": 0,
            "qa_flags": 0,
            "support_requests": 0,
            "support_resolved": 0,
            "total_requests": 0,
            "department_invocations": {},
        },
        "final_response": "",
        "resolution_status": "",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _classifier_mock(classification_dict: dict) -> MagicMock:
    """Build a mock for nodes._classifier_model that returns a JSON classification."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=json.dumps(classification_dict))
    return mock


def _compose_mock(response_text: str = "All tasks completed successfully.") -> MagicMock:
    """Build a mock for the ChatAnthropic instance created inside compose_output."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(content=response_text)
    return mock_instance


def _dept_agent_mock(
    response_text: str = "Department task completed.",
) -> MagicMock:
    """Build a mock department agent whose .invoke() returns a canned result dict.

    The agent response contains only plain text. For tests that need follow-up
    tasks, use _make_dept_executor_node() to mock the full department_executor
    node directly, bypassing the regex parsing in nodes.py.

    Args:
        response_text: Human-readable response from the agent.

    Returns:
        A MagicMock configured to behave like a DeepAgent.
    """
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"response": response_text}
    return mock_agent


def _make_dept_executor_node(dept_result_map: dict):
    """Build a replacement for the department_executor node function.

    department_executor is called with a state that has `_target_dept` injected
    by Send(). This factory builds a node function that returns pre-canned
    DepartmentResult dicts, bypassing DeepAgent and regex parsing entirely.

    This is the cleanest approach for integration tests involving follow-up tasks,
    since the real regex in nodes.py only matches flat JSON (no nested objects).

    IMPORTANT — operator.add accumulation and result_aggregator behaviour:
    OrchestratorState.department_results uses operator.add, so results accumulate
    across task-queue cycles. result_aggregator scans ALL department_results on
    every pass. To prevent infinite re-queuing, each department entry in
    dept_result_map should only emit follow_up_tasks on its FIRST invocation.
    This factory tracks call counts per department and strips follow_up_tasks on
    subsequent invocations.

    Args:
        dept_result_map: Maps department name → DepartmentResult dict to return.
                         follow_up_tasks are only returned on the first call for
                         each department; subsequent calls get empty follow_up_tasks.

    Returns:
        A node function (state → dict) compatible with the graph's node API.
    """
    call_counts: dict[str, int] = {}

    def _executor(state):
        dept = state.get("_target_dept", "")
        call_counts[dept] = call_counts.get(dept, 0) + 1
        base = dept_result_map.get(dept, {
            "department": dept,
            "response": "Mocked response",
            "resolved": True,
            "follow_up_tasks": [],
            "metrics": {"actions_taken": 1, "tools_called": []},
        })
        # Only emit follow_up_tasks on the FIRST call for this department.
        # On subsequent calls (re-encountered via accumulated department_results
        # in result_aggregator), return an empty follow_up_tasks list to prevent
        # infinite re-queuing.
        result = dict(base)
        if call_counts[dept] > 1:
            result = {**result, "follow_up_tasks": []}
        return {"department_results": [result]}

    return _executor


def _dept_agents_dict(dept_name_to_agent: dict) -> dict:
    """Build a mock DEPARTMENT_AGENTS dict where each value is a factory lambda.

    nodes.py calls `DEPARTMENT_AGENTS[dept]()` (note the call: it invokes the
    factory to get the agent instance). So we wrap each mock agent in a
    zero-argument callable.

    Args:
        dept_name_to_agent: Mapping from department name to a pre-built mock agent.

    Returns:
        Dict suitable for patching `nodes.DEPARTMENT_AGENTS`.
    """
    return {dept: (lambda agent=agent: agent) for dept, agent in dept_name_to_agent.items()}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSingleDepartmentLowRisk:
    """Low-risk single-department request auto-executes without approval gate."""

    def test_single_department_low_risk(self):
        # --- Setup mocks ---
        # Classifier returns "reporting" + "aggregate_metrics" → low risk (not in HIGH_RISK_ACTIONS)
        classifier_mock = _classifier_mock({
            "departments": ["reporting"],
            "action_type": "aggregate_metrics",
            "complexity": "single",
            "summary": "Metrics request",
        })

        # Department agent returns a resolved result with no follow-ups
        reporting_agent = _dept_agent_mock("Metrics dashboard generated successfully.")
        dept_agents = _dept_agents_dict({"reporting": reporting_agent})

        # compose_output LLM response
        compose_chat = _compose_mock("Here are your platform metrics for this week.")

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph()
            result = graph.invoke(
                _make_state(request="Show me this week's platform metrics"),
                config={"configurable": {"thread_id": "test-low-risk-single"}},
            )

        # Should have executed without going through approval_gate
        assert len(result["department_results"]) == 1
        assert result["department_results"][0]["department"] == "reporting"
        assert result["department_results"][0]["resolved"] is True
        assert result["final_response"] != ""
        assert result["resolution_status"] == "resolved"
        # Risk should be low and approval should be auto-set to not_required
        assert result["risk_level"] == "low"
        assert result["approval_status"] == "not_required"


@pytest.mark.integration
class TestSingleDepartmentHighRisk:
    """High-risk request triggers approval_gate interrupt; resume with approval."""

    def test_single_department_high_risk_approved(self):
        # content_pipeline + publish_content is HIGH RISK (see risk.py)
        classifier_mock = _classifier_mock({
            "departments": ["content_pipeline"],
            "action_type": "publish_content",
            "complexity": "single",
            "summary": "Publish lesson content",
        })

        content_agent = _dept_agent_mock("Content published to the platform.")
        dept_agents = _dept_agents_dict({"content_pipeline": content_agent})
        compose_chat = _compose_mock("Content has been published successfully.")

        checkpointer = MemorySaver()
        thread_id = "test-high-risk-approval"
        config = {"configurable": {"thread_id": thread_id}}

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph(checkpointer=checkpointer)

            # --- First invoke: graph should interrupt at approval_gate ---
            first_result = graph.invoke(
                _make_state(request="Publish the new grammar lesson to all students"),
                config=config,
            )

            # After interrupt the graph returns the current state snapshot.
            # The graph is paused — risk_level should be "high".
            assert first_result.get("risk_level") == "high"

            # --- Resume with operator approval ---
            resumed_result = graph.invoke(
                Command(resume="approved"),
                config=config,
            )

        # After approval the department should have run and completed
        assert len(resumed_result["department_results"]) == 1
        assert resumed_result["department_results"][0]["department"] == "content_pipeline"
        assert resumed_result["final_response"] != ""
        assert resumed_result["resolution_status"] == "resolved"


@pytest.mark.integration
class TestCascadingFollowUps:
    """A follow-up task drives a second department cycle via the autonomous cascade.

    The autonomous cascade works by:
    1. department_executor returns a DepartmentResult with follow_up_tasks
    2. result_aggregator extracts them into task_queue
    3. check_task_queue pops a task, sets current_task, loops to request_classifier
    4. request_classifier classifies the follow-up → department B runs

    To inject follow_up_tasks without relying on the regex parser inside
    department_executor, we mock result_aggregator to emit the desired task_queue
    on its first call, then clear it on the second call (so the loop terminates).
    """

    def test_cascading_follow_ups(self):
        # Classifier is called twice:
        # - First call: initial request → student_onboarding
        # - Second call: follow-up task → tutor_management
        classification_responses = [
            MagicMock(content=json.dumps({
                "departments": ["student_onboarding"],
                "action_type": "enroll_student",
                "complexity": "single",
                "summary": "Onboard new student",
            })),
            MagicMock(content=json.dumps({
                "departments": ["tutor_management"],
                "action_type": "get_tutor_info",
                "complexity": "single",
                "summary": "Assign tutor follow-up",
            })),
        ]
        classifier_mock = MagicMock()
        classifier_mock.invoke.side_effect = classification_responses

        dept_agents = _dept_agents_dict({
            "student_onboarding": _dept_agent_mock("Student S999 enrolled successfully."),
            "tutor_management": _dept_agent_mock("Tutor assigned to student S999."),
        })
        compose_chat = _compose_mock("Student onboarded and tutor assigned.")

        # Mock result_aggregator to inject a follow-up task on the first pass,
        # then clear task_queue on subsequent passes. This simulates a department
        # agent having produced a follow_up_tasks entry without relying on the
        # regex text parser in the real department_executor.
        aggregator_call_count = [0]

        def _mock_result_aggregator(state):
            aggregator_call_count[0] += 1
            dept_results = state.get("department_results", [])
            completed = list(state.get("completed_tasks", []))
            completed.extend(dept_results)
            if aggregator_call_count[0] == 1:
                # First pass: emit one follow-up task for tutor_management
                follow_ups = [{"target_dept": "tutor_management",
                               "action": "assign_tutor", "context": {}}]
            else:
                # Subsequent passes: no more follow-ups
                follow_ups = []
            return {"task_queue": follow_ups, "completed_tasks": completed}

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("graph.result_aggregator", _mock_result_aggregator), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph()
            result = graph.invoke(
                _make_state(request="Onboard new student Sarah"),
                config={"configurable": {"thread_id": "test-cascade"}},
            )

        # Both departments should have run — student_onboarding first (initial request),
        # then tutor_management (driven by the injected follow-up task).
        dept_names = {dr["department"] for dr in result["department_results"]}
        assert "student_onboarding" in dept_names
        assert "tutor_management" in dept_names
        assert len(result["department_results"]) == 2
        assert result["final_response"] != ""
        assert result["resolution_status"] == "resolved"


@pytest.mark.integration
class TestMultiDepartmentParallel:
    """Request dispatched to two departments simultaneously via Send."""

    def test_multi_department_parallel(self):
        # Two departments in the classification → two parallel Send() branches
        classifier_mock = _classifier_mock({
            "departments": ["reporting", "quality_assurance"],
            "action_type": "aggregate_metrics",
            "complexity": "multi",
            "summary": "Metrics and QA review",
        })

        reporting_agent = _dept_agent_mock("Metrics report generated.")
        qa_agent = _dept_agent_mock("QA review passed — no issues found.")
        dept_agents = _dept_agents_dict({
            "reporting": reporting_agent,
            "quality_assurance": qa_agent,
        })
        compose_chat = _compose_mock("Metrics and QA completed.")

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph()
            result = graph.invoke(
                _make_state(request="Run a weekly QA check and generate metrics report"),
                config={"configurable": {"thread_id": "test-parallel"}},
            )

        # Both parallel branches should have contributed results
        dept_names = {dr["department"] for dr in result["department_results"]}
        assert "reporting" in dept_names
        assert "quality_assurance" in dept_names
        assert len(result["department_results"]) == 2
        assert result["final_response"] != ""


@pytest.mark.integration
class TestApprovalRejection:
    """High-risk request is rejected by operator — no department executes."""

    def test_approval_rejection(self):
        # support + process_refund is HIGH RISK
        classifier_mock = _classifier_mock({
            "departments": ["support"],
            "action_type": "process_refund",
            "complexity": "single",
            "summary": "Process student refund",
        })

        support_agent = _dept_agent_mock("Refund processed.")
        dept_agents = _dept_agents_dict({"support": support_agent})
        # compose_output short-circuits on rejection — no LLM call needed.
        # We still provide a mock in case the code path changes.
        compose_chat = _compose_mock()

        checkpointer = MemorySaver()
        thread_id = "test-rejection"
        config = {"configurable": {"thread_id": thread_id}}

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph(checkpointer=checkpointer)

            # First invoke — should interrupt at approval_gate
            first_result = graph.invoke(
                _make_state(request="Issue a refund of $100 to student S001"),
                config=config,
            )
            assert first_result.get("risk_level") == "high"

            # Resume with a non-approval answer (anything other than "approved")
            rejected_result = graph.invoke(
                Command(resume={"approved": False}),
                config=config,
            )

        # Graph should have routed to compose_output and short-circuited
        assert rejected_result["resolution_status"] == "rejected"
        assert "rejected" in rejected_result["final_response"].lower()
        # Department should NOT have been called — no department_results
        assert len(rejected_result["department_results"]) == 0


@pytest.mark.integration
class TestTaskQueueLoop:
    """Chain of 3 departments — initial request + 2 follow-ups process sequentially.

    Tests that check_task_queue correctly loops back to request_classifier until the
    queue is drained, then routes to compose_output. Three classifier invocations
    produce three different departments; all three execute and all results appear.

    We mock result_aggregator to inject follow-up tasks without relying on the regex
    text parser inside department_executor. The aggregator emits one task on the first
    call, another on the second, and nothing on the third — so the queue drains.
    """

    def test_task_queue_loop(self):
        # Three classifier invocations:
        # 1. Initial request → student_onboarding
        # 2. Follow-up emitted by first aggregator pass → tutor_management
        # 3. Follow-up emitted by second aggregator pass → reporting
        classification_responses = [
            MagicMock(content=json.dumps({
                "departments": ["student_onboarding"],
                "action_type": "enroll_student",
                "complexity": "single",
                "summary": "Onboard student",
            })),
            MagicMock(content=json.dumps({
                "departments": ["tutor_management"],
                "action_type": "get_tutor_info",
                "complexity": "single",
                "summary": "Assign tutor",
            })),
            MagicMock(content=json.dumps({
                "departments": ["reporting"],
                "action_type": "aggregate_metrics",
                "complexity": "single",
                "summary": "Log new student",
            })),
        ]
        classifier_mock = MagicMock()
        classifier_mock.invoke.side_effect = classification_responses

        dept_agents = _dept_agents_dict({
            "student_onboarding": _dept_agent_mock("Student enrolled."),
            "tutor_management": _dept_agent_mock("Tutor assigned."),
            "reporting": _dept_agent_mock("Reported to dashboard."),
        })
        compose_chat = _compose_mock("All three tasks completed in sequence.")

        # Mock result_aggregator to emit follow-up tasks on the first 2 passes,
        # then nothing on the third — driving a 3-department sequential chain.
        aggregator_call_count = [0]

        def _mock_result_aggregator(state):
            aggregator_call_count[0] += 1
            dept_results = state.get("department_results", [])
            completed = list(state.get("completed_tasks", []))
            completed.extend(dept_results)
            if aggregator_call_count[0] == 1:
                follow_ups = [{"target_dept": "tutor_management",
                               "action": "assign", "context": {}}]
            elif aggregator_call_count[0] == 2:
                follow_ups = [{"target_dept": "reporting",
                               "action": "log_new_student", "context": {}}]
            else:
                follow_ups = []
            return {"task_queue": follow_ups, "completed_tasks": completed}

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("graph.result_aggregator", _mock_result_aggregator), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph()
            result = graph.invoke(
                _make_state(request="Onboard student and run full pipeline"),
                config={"configurable": {"thread_id": "test-task-chain"}},
            )

        # All three departments should appear in results
        dept_names = {dr["department"] for dr in result["department_results"]}
        assert "student_onboarding" in dept_names
        assert "tutor_management" in dept_names
        assert "reporting" in dept_names
        assert len(result["department_results"]) == 3
        assert result["final_response"] != ""
        assert result["resolution_status"] == "resolved"


@pytest.mark.integration
class TestReportingMetricsUpdate:
    """After a request completes, metrics_store.total_requests increments
    and department_invocations is updated for the department that ran."""

    def test_reporting_metrics_update(self):
        # Low-risk single department request
        classifier_mock = _classifier_mock({
            "departments": ["reporting"],
            "action_type": "aggregate_metrics",
            "complexity": "single",
            "summary": "Metrics snapshot",
        })

        reporting_agent = _dept_agent_mock("Metrics captured.")
        dept_agents = _dept_agents_dict({"reporting": reporting_agent})
        compose_chat = _compose_mock("Metrics updated.")

        # Start with non-zero counters so we can confirm increment
        initial_metrics = {
            "students_onboarded": 5,
            "tutors_assigned": 3,
            "content_generated": 2,
            "content_published": 1,
            "qa_reviews": 7,
            "qa_flags": 1,
            "support_requests": 10,
            "support_resolved": 8,
            "total_requests": 4,           # Should become 5 after this run
            "department_invocations": {"reporting": 2},  # Should become 3
        }

        with patch("nodes._classifier_model", classifier_mock), \
             patch("nodes.DEPARTMENT_AGENTS", dept_agents), \
             patch("nodes.ChatAnthropic", return_value=compose_chat):

            graph = build_graph()
            result = graph.invoke(
                _make_state(
                    request="Show platform metrics",
                    metrics_store=initial_metrics,
                ),
                config={"configurable": {"thread_id": "test-metrics-update"}},
            )

        updated_metrics = result["metrics_store"]

        # total_requests must have been incremented by exactly 1
        assert updated_metrics["total_requests"] == 5

        # reporting invocation counter must have been incremented by 1
        assert updated_metrics["department_invocations"]["reporting"] == 3

        # Other counters should be unchanged (reporting dept doesn't update student/tutor/etc. metrics)
        assert updated_metrics["students_onboarded"] == 5
        assert updated_metrics["support_requests"] == 10
