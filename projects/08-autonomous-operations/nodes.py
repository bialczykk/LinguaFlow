"""Node functions for the autonomous operations orchestrator graph — Project 08.

Each function is a LangGraph node: it receives the full OrchestratorState and
returns a partial state dict (or a Command) with only the fields it updates.

Node overview:
- request_classifier   : LLM-based routing → sets classification
- risk_assessor        : Pure logic → sets risk_level and approval_status
- approval_gate        : HITL interrupt → waits for human approval, routes accordingly
- department_executor  : Runs a DeepAgent for the target department (via Send)
- result_aggregator    : Pure logic → extracts follow-up tasks, updates completed list
- check_task_queue     : Pure logic → pops the next queued task or ends the chain
- compose_output       : LLM-based → merges department results into final_response
- reporting_snapshot   : Pure logic → updates cumulative platform metrics

LangSmith tag: p8-autonomous-operations
"""

import json
import re
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langsmith import traceable
from langgraph.types import Command, interrupt

from departments import DEPARTMENT_AGENTS
from models import DepartmentResult, OrchestratorState
from prompts import CLASSIFIER_PROMPT, COMPOSE_OUTPUT_PROMPT
from risk import assess_risk

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# LangSmith trace tag shared across all nodes in this project
_TAGS = ["p8-autonomous-operations"]

# ---------------------------------------------------------------------------
# Module-level classification model — no tools needed, only JSON generation.
# Tests patch `nodes._classifier_model` to avoid real API calls.
# ---------------------------------------------------------------------------
_classifier_model = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# 1. request_classifier
# ---------------------------------------------------------------------------

@traceable(name="request_classifier", run_type="chain", tags=_TAGS)
def request_classifier(state: OrchestratorState) -> dict:
    """Classify the incoming request and decide which department(s) to dispatch to.

    If state['current_task'] is set, we're processing a follow-up task generated
    by a department agent — in that case we classify from the follow-up instruction
    instead of the original request, passing it as follow_up_context.

    Uses CLASSIFIER_PROMPT and the module-level _classifier_model.
    Strips markdown code fences (```json ... ```) before JSON parsing.

    Returns:
        {"classification": dict} — the parsed classification from the LLM.
    """
    metadata = state.get("request_metadata", {})
    current_task = state.get("current_task")

    # Build the follow-up context string from current_task (if any)
    if current_task:
        follow_up_context = (
            f"Action: {current_task.get('action', '')}\n"
            f"Target department: {current_task.get('target_dept', '')}\n"
            f"Context: {json.dumps(current_task.get('context', {}))}"
        )
    else:
        follow_up_context = ""

    # Format and invoke the classification prompt
    messages = CLASSIFIER_PROMPT.format_messages(
        request=state["request"],
        user_id=metadata.get("user_id", "unknown"),
        priority=metadata.get("priority", "medium"),
        source=metadata.get("source", "api"),
        follow_up_context=follow_up_context,
    )
    response = _classifier_model.invoke(messages)

    # Parse the JSON response; the LLM sometimes wraps it in markdown fences
    raw = response.content
    try:
        classification = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Strip markdown code fences (```json ... ```) and retry
        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            classification = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Fallback: safe default so the graph can continue
            classification = {
                "departments": [],
                "action_type": "unknown",
                "complexity": "single",
                "summary": "Unable to classify request",
            }

    return {"classification": classification}


# ---------------------------------------------------------------------------
# 2. risk_assessor
# ---------------------------------------------------------------------------

@traceable(name="risk_assessor", run_type="chain", tags=_TAGS)
def risk_assessor(state: OrchestratorState) -> dict:
    """Determine whether the classified action requires human approval.

    Pure logic node — calls assess_risk() from risk.py (deterministic lookup).
    No LLM is involved; this keeps the approval decision fast and auditable.

    Returns:
        {"risk_level": "low", "approval_status": "not_required"}  — if low risk
        {"risk_level": "high", "approval_status": ""}             — if high risk
              (approval_status is empty pending the approval_gate decision)
    """
    classification = state.get("classification", {})
    risk_level = assess_risk(classification)

    if risk_level == "low":
        return {"risk_level": "low", "approval_status": "not_required"}
    else:
        # Empty string signals approval_gate that it must fill this in
        return {"risk_level": "high", "approval_status": ""}


# ---------------------------------------------------------------------------
# 3. approval_gate
# ---------------------------------------------------------------------------

@traceable(name="approval_gate", run_type="chain", tags=_TAGS)
def approval_gate(
    state: OrchestratorState,
) -> Command[Literal["dispatch_departments", "compose_output"]]:
    """Pause execution and ask a human operator to approve or reject a high-risk action.

    Uses LangGraph's interrupt() to suspend the graph and surface a payload
    to the caller. The payload provides enough context for the operator to make
    an informed decision without needing to read raw state.

    Resume values:
    - "approved" → routes to dispatch_departments so the task can proceed
    - anything else → routes to compose_output with approval_status="rejected"
    """
    classification = state.get("classification", {})

    # Build a human-readable payload for the approval request
    payload = {
        "department": classification.get("departments", []),
        "action_type": classification.get("action_type", ""),
        "summary": classification.get("summary", "No summary available"),
        "risk_reason": (
            f"Action '{classification.get('action_type', '')}' on department(s) "
            f"{classification.get('departments', [])} requires human approval."
        ),
    }

    # Suspend the graph — the caller receives `payload` and must resume with a decision
    operator_decision = interrupt(payload)

    # Route based on the operator's response
    if operator_decision == "approved":
        return Command(goto="dispatch_departments")
    else:
        return Command(
            update={"approval_status": "rejected"},
            goto="compose_output",
        )


# ---------------------------------------------------------------------------
# 4. department_executor
# ---------------------------------------------------------------------------

@traceable(name="department_executor", run_type="chain", tags=_TAGS)
def department_executor(state: OrchestratorState) -> dict:
    """Execute the department DeepAgent for the target department.

    Reads `_target_dept` from state — this key is injected by the Send()
    dispatch in graph.py so each parallel branch knows which agent to run.

    Creates the agent using the DEPARTMENT_AGENTS factory dict, invokes it
    with the request, and parses the response to extract any follow_up_tasks
    embedded in the agent's output (JSON block within the response text).

    Returns:
        {"department_results": [DepartmentResult]} — single-item list so the
        operator.add reducer can safely concatenate results from parallel branches.
    """
    dept = state.get("_target_dept", "")
    request = state.get("request", "")
    current_task = state.get("current_task")

    # Build a rich user message with all context the agent needs.
    # Context-specific details go in the user message (not the system prompt)
    # because agents are created once with a static system prompt.
    metadata = state.get("request_metadata", {})

    parts = [f"Request: {request}"]
    # Only include student_id if explicitly present in metadata (not user_id —
    # admin-initiated requests have user_id="admin" which isn't a student).
    if "student_id" in metadata:
        parts.append(f"Student ID: {metadata['student_id']}")
    if current_task:
        parts.append(f"Follow-up context: {json.dumps(current_task)}")

    invoke_request = "\n".join(parts)

    # Instantiate the department agent using its factory function
    agent = DEPARTMENT_AGENTS[dept]()

    # Invoke the DeepAgent with the request — DeepAgents expect messages input
    # (same format as LangGraph graphs), following the P7 invocation pattern.
    invoke_input = {
        "messages": [{"role": "user", "content": invoke_request}],
    }
    config = {"tags": _TAGS}
    agent_response = agent.invoke(invoke_input, config=config)

    # Extract the text response from the agent's output.
    # DeepAgents return state dicts with "messages" containing the conversation.
    if isinstance(agent_response, dict):
        messages = agent_response.get("messages", [])
        if messages:
            # Last message is the agent's final response
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                response_text = last_msg.content
            elif isinstance(last_msg, dict):
                response_text = last_msg.get("content", str(last_msg))
            else:
                response_text = str(last_msg)
        else:
            response_text = agent_response.get("response", str(agent_response))
    else:
        response_text = str(agent_response)

    # Attempt to extract follow_up_tasks from the agent's response.
    # Department agents are prompted to emit follow_up_tasks as a JSON block.
    follow_up_tasks: list[dict] = []
    try:
        # Look for a JSON object anywhere in the response text
        json_match = re.search(r'\{[^{}]*"follow_up_tasks"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            follow_up_tasks = parsed.get("follow_up_tasks", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass  # No follow-up tasks found — that's fine

    result: DepartmentResult = {
        "department": dept,
        "response": response_text,
        "resolved": True,
        "follow_up_tasks": follow_up_tasks,
        "metrics": {"actions_taken": 1, "tools_called": []},
    }

    # Return as a list so operator.add concatenates parallel results
    return {"department_results": [result]}


# ---------------------------------------------------------------------------
# 5. result_aggregator
# ---------------------------------------------------------------------------

@traceable(name="result_aggregator", run_type="chain", tags=_TAGS)
def result_aggregator(state: OrchestratorState) -> dict:
    """Collect follow-up tasks from department results and update completed tasks.

    Pure logic node — no LLM call. Scans department_results for any follow_up_tasks
    lists and gathers them all into task_queue for the next processing cycle.
    Moves the current batch of department_results into completed_tasks for audit.

    Returns:
        {
          "task_queue": list of follow-up tasks extracted from all department results,
          "completed_tasks": existing completed_tasks + this round's department_results,
        }
    """
    department_results = state.get("department_results", [])
    completed_tasks = list(state.get("completed_tasks", []))

    # Collect all follow-up tasks from every department result
    follow_ups: list[dict] = []
    for result in department_results:
        tasks = result.get("follow_up_tasks") or []
        follow_ups.extend(tasks)
        # Move each result into the completed audit trail
        completed_tasks.append(result)

    return {
        "task_queue": follow_ups,
        "completed_tasks": completed_tasks,
    }


# ---------------------------------------------------------------------------
# 6. check_task_queue
# ---------------------------------------------------------------------------

@traceable(name="check_task_queue", run_type="chain", tags=_TAGS)
def check_task_queue(
    state: OrchestratorState,
) -> Command[Literal["request_classifier", "compose_output"]]:
    """Check whether autonomous follow-up tasks remain in the queue.

    Pure logic node. Two branches:
    - Queue non-empty → pop the first task into current_task, route back to
      request_classifier so the task is classified and dispatched as a new cycle.
    - Queue empty     → clear current_task and route to compose_output to finalise
      the response for the user.

    This node enables fully autonomous multi-hop processing: a single user request
    can cascade through multiple departments without any human re-invocation.
    """
    task_queue = list(state.get("task_queue", []))

    if task_queue:
        # Pop the first task (FIFO)
        next_task = task_queue[0]
        remaining = task_queue[1:]
        return Command(
            update={"current_task": next_task, "task_queue": remaining},
            goto="request_classifier",
        )
    else:
        # No more tasks — proceed to final output composition
        return Command(
            update={"current_task": None},
            goto="compose_output",
        )


# ---------------------------------------------------------------------------
# 7. compose_output
# ---------------------------------------------------------------------------

@traceable(name="compose_output", run_type="chain", tags=_TAGS)
def compose_output(state: OrchestratorState) -> dict:
    """Merge all department results into a single unified final response.

    Short-circuits immediately if approval_status is "rejected" — in that case
    we return a rejection message without calling the LLM (no point composing
    results that were never executed).

    Otherwise, uses COMPOSE_OUTPUT_PROMPT and a fresh ChatAnthropic instance
    to synthesise all department responses and completed task summaries into
    one coherent, professional reply.

    Returns:
        {"final_response": str, "resolution_status": str}
    """
    approval_status = state.get("approval_status", "")

    # Short-circuit: rejected before any department ran
    if approval_status == "rejected":
        return {
            "final_response": (
                "This request was reviewed and rejected by an operator. "
                "No actions were taken. Please contact support if you believe this was an error."
            ),
            "resolution_status": "rejected",
        }

    # Build department response summary for the LLM
    dept_lines = []
    all_resolved = True

    for dr in state.get("department_results", []):
        dept_lines.append(f"[{dr['department'].upper()}] {dr['response']}")
        if not dr.get("resolved", True):
            all_resolved = False

    department_responses = (
        "\n\n".join(dept_lines) if dept_lines else "No department responses available."
    )

    # Build a task chain summary from completed_tasks for transparency
    completed = state.get("completed_tasks", [])
    if completed:
        task_lines = [
            f"- {t.get('department', 'unknown')}: {t.get('response', '')[:80]}"
            for t in completed
        ]
        task_chain_summary = "\n".join(task_lines)
    else:
        task_chain_summary = "No cascading tasks were executed."

    # Invoke the composition model
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    messages = COMPOSE_OUTPUT_PROMPT.format_messages(
        request=state["request"],
        department_responses=department_responses,
        task_chain_summary=task_chain_summary,
    )
    response = model.invoke(messages)

    # Determine resolution status based on results
    if all_resolved and dept_lines:
        resolution_status = "resolved"
    elif not dept_lines:
        resolution_status = "pending_approval"
    else:
        resolution_status = "partial"

    return {
        "final_response": response.content,
        "resolution_status": resolution_status,
    }


# ---------------------------------------------------------------------------
# 8. reporting_snapshot
# ---------------------------------------------------------------------------

@traceable(name="reporting_snapshot", run_type="chain", tags=_TAGS)
def reporting_snapshot(state: OrchestratorState) -> dict:
    """Update cumulative platform metrics based on this request's department results.

    Pure logic node — no LLM call. Reads department_results and increments the
    relevant counters in metrics_store. This gives a running tally of platform
    activity that persists across graph invocations via the SqliteSaver checkpointer.

    Metric update rules:
    - total_requests            : always +1 per orchestrator invocation
    - department_invocations    : +1 for each department that produced a result
    - students_onboarded        : +1 per student_onboarding result
    - tutors_assigned           : +1 per tutor_management result
    - content_generated         : +1 per content_pipeline result
    - qa_reviews                : +1 per quality_assurance result
    - support_requests          : +1 per support result (regardless of resolution)
    - support_resolved          : +1 per resolved support result

    Returns:
        {"metrics_store": updated MetricsStore}
    """
    # Deep-copy the metrics store so we don't mutate the original state dict
    metrics = dict(state.get("metrics_store", {}))
    dept_invocations = dict(metrics.get("department_invocations", {}))

    # Always increment the grand total for this orchestrator invocation
    metrics["total_requests"] = metrics.get("total_requests", 0) + 1

    # Scan each department result and increment the relevant specific counters
    for result in state.get("department_results", []):
        dept = result.get("department", "")
        resolved = result.get("resolved", False)

        # Per-department invocation counter
        dept_invocations[dept] = dept_invocations.get(dept, 0) + 1

        # Department-specific metric increments
        if dept == "student_onboarding":
            metrics["students_onboarded"] = metrics.get("students_onboarded", 0) + 1

        elif dept == "tutor_management":
            metrics["tutors_assigned"] = metrics.get("tutors_assigned", 0) + 1

        elif dept == "content_pipeline":
            metrics["content_generated"] = metrics.get("content_generated", 0) + 1

        elif dept == "quality_assurance":
            metrics["qa_reviews"] = metrics.get("qa_reviews", 0) + 1

        elif dept == "support":
            metrics["support_requests"] = metrics.get("support_requests", 0) + 1
            if resolved:
                metrics["support_resolved"] = metrics.get("support_resolved", 0) + 1

    metrics["department_invocations"] = dept_invocations

    return {"metrics_store": metrics}
