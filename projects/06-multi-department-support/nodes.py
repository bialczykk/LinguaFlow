"""Node functions for the multi-department support graph.

Each function is a LangGraph node — it receives the full SupportState and
returns a partial state dict with only the fields it updates.

Node overview:
- supervisor_router:     LLM-based request classifier → sets classification
- supervisor_aggregator: Pure logic → scans results for escalations
- compose_response:      LLM-based response merger → sets final_response
- ask_clarification:     interrupt/resume node for hybrid HITL clarification
- billing_agent etc.:    Department sub-agents with tool-calling loops
"""

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langsmith import traceable
from langgraph.types import Command, interrupt

from models import DepartmentResult, SupportState
from prompts import (
    BILLING_PROMPT,
    COMPOSE_RESPONSE_PROMPT,
    CONTENT_PROMPT,
    SCHEDULING_PROMPT,
    SUPERVISOR_CLASSIFICATION_PROMPT,
    TECH_SUPPORT_PROMPT,
)
from tools import (
    BILLING_TOOLS,
    CONTENT_TOOLS,
    SCHEDULING_TOOLS,
    TECH_SUPPORT_TOOLS,
)

# LangSmith trace tag shared by all nodes in this project
_TAGS = ["p6-multi-department-support"]

# ---------------------------------------------------------------------------
# Module-level classification model — no tools needed, just JSON generation.
# Tests patch `nodes._classification_model` to mock LLM responses.
# ---------------------------------------------------------------------------
_classification_model = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# Helper: run a tool-calling agent loop
# ---------------------------------------------------------------------------

def _run_agent_loop(
    llm_with_tools: ChatAnthropic,
    system_prompt: str,
    request: str,
    tools: list[BaseTool],
    max_rounds: int = 3,
) -> str:
    """Execute a simple ReAct-style tool-calling loop.

    Each round:
    1. Invoke the LLM with the current message history.
    2. If the LLM requests tool calls, execute them and append results.
    3. Repeat until no more tool calls or max_rounds is reached.

    Returns the final text response from the LLM.
    """
    # Build a tool lookup map for fast dispatch
    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    # Start with system + human messages
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request),
    ]

    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)  # append AI message to history

        # If no tool calls are requested, we're done
        if not getattr(response, "tool_calls", None):
            break

        # Execute each requested tool and append ToolMessages
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name in tool_map:
                # Invoke the LangChain tool — it expects keyword args
                try:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"
            else:
                tool_result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                )
            )

    # Return the final AI text response
    return response.content if isinstance(response.content, str) else str(response.content)


# ---------------------------------------------------------------------------
# Supervisor nodes
# ---------------------------------------------------------------------------

@traceable(name="supervisor_router", run_type="chain", tags=_TAGS)
def supervisor_router(state: SupportState) -> dict:
    """Classify the incoming support request and decide which departments to route to.

    Uses an LLM (module-level _classification_model) to parse the request and
    return a JSON classification with:
    - departments: list of department names to dispatch to
    - needs_clarification: bool
    - clarification_question: string or null
    - summary: brief description of the request
    - complexity: "single" | "multi"

    If clarification is needed, also sets clarification_needed in state so
    the graph can route to ask_clarification instead of department nodes.
    """
    metadata = state.get("request_metadata", {})
    user_clarification = state.get("user_clarification")

    # Include any user clarification as extra context in the prompt
    clarification_context = (
        f"\nUser clarification: {user_clarification}" if user_clarification else ""
    )

    # Format the prompt and invoke the classification model
    messages = SUPERVISOR_CLASSIFICATION_PROMPT.format_messages(
        request=state["request"],
        sender_type=metadata.get("sender_type", "unknown"),
        student_id=metadata.get("student_id", "unknown"),
        priority=metadata.get("priority", "medium"),
        clarification_context=clarification_context,
    )

    response = _classification_model.invoke(messages)

    # Parse the JSON response from the LLM.
    # Haiku sometimes wraps JSON in markdown code fences (```json ... ```),
    # so we strip those before parsing.
    raw = response.content
    try:
        # Try direct parse first
        classification = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Strip markdown code fences and retry
        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        try:
            classification = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # Fallback: treat as a general request needing clarification
            classification = {
                "departments": [],
                "needs_clarification": True,
                "clarification_question": "Could you please describe your issue in more detail?",
                "summary": "Unparseable request",
                "complexity": "single",
            }

    result: dict = {"classification": classification}

    # If clarification is needed, surface the question so the graph can route
    # to ask_clarification instead of dispatching to department nodes.
    if classification.get("needs_clarification"):
        result["clarification_needed"] = classification.get("clarification_question")

    return result


@traceable(name="supervisor_aggregator", run_type="chain", tags=_TAGS)
def supervisor_aggregator(state: SupportState) -> dict:
    """Collect department results and build the escalation queue.

    Pure logic node — no LLM call. Scans department_results for any that
    were not fully resolved and have an escalation dict. Those escalations
    are gathered into escalation_queue so the graph can handle them.
    """
    escalations = []

    for result in state.get("department_results", []):
        # Only unresolved results with explicit escalation info go into the queue
        if not result.get("resolved") and result.get("escalation"):
            escalations.append(result["escalation"])

    return {"escalation_queue": escalations}


# ---------------------------------------------------------------------------
# Response composition
# ---------------------------------------------------------------------------

@traceable(name="compose_response", run_type="chain", tags=_TAGS)
def compose_response(state: SupportState) -> dict:
    """Merge all department responses into a single unified reply.

    Uses an LLM to synthesise the individual department outputs into one
    coherent, empathetic message for the user. Also sets resolution_status
    based on whether every department fully resolved their part.
    """
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")

    # Build a human-readable summary of all department responses
    dept_lines = []
    all_resolved = True

    for dr in state.get("department_results", []):
        dept_lines.append(f"[{dr['department'].upper()}] {dr['response']}")
        if not dr.get("resolved", True):
            all_resolved = False

    department_responses = "\n\n".join(dept_lines) if dept_lines else "No department responses available."

    # Invoke the composition model
    messages = COMPOSE_RESPONSE_PROMPT.format_messages(
        request=state["request"],
        department_responses=department_responses,
    )
    response = model.invoke(messages)

    # Determine resolution status
    if all_resolved:
        resolution_status = "resolved"
    elif state.get("escalation_queue"):
        resolution_status = "escalated_to_human"
    else:
        resolution_status = "partial"

    return {
        "final_response": response.content,
        "resolution_status": resolution_status,
    }


# ---------------------------------------------------------------------------
# Clarification node (Human-in-the-Loop via interrupt)
# ---------------------------------------------------------------------------

@traceable(name="ask_clarification", run_type="chain", tags=_TAGS)
def ask_clarification(state: SupportState) -> Command:
    """Pause execution and ask the user for clarification.

    Uses LangGraph's interrupt() to suspend the graph. When the graph is
    resumed (via Command(resume=...)), the user's reply is stored in
    user_clarification and control goes back to supervisor_router so the
    request can be re-classified with the extra context.
    """
    question = state.get("clarification_needed", "Could you provide more details?")

    # interrupt() suspends the graph and surfaces the question to the caller.
    # The return value is whatever the caller passes in Command(resume=<value>).
    user_reply = interrupt(question)

    # On resume: store the reply and route back for re-classification
    return Command(
        update={"user_clarification": user_reply, "clarification_needed": None},
        goto="supervisor_router",
    )


# ---------------------------------------------------------------------------
# Department sub-agent nodes
# ---------------------------------------------------------------------------

def _make_department_agent(
    department: str,
    prompt_template: str,
    dept_tools: list[BaseTool],
) -> Any:
    """Factory that builds a department sub-agent node function.

    Each department agent:
    1. Formats its system prompt with the student_id, request, and any
       escalation context from the escalation_queue.
    2. Binds the department-specific tools to a fresh ChatAnthropic instance.
    3. Runs a tool-calling loop (up to 3 rounds).
    4. Returns a DepartmentResult appended to department_results via the
       operator.add reducer (parallel-safe).
    """
    @traceable(name=f"{department}_agent", run_type="chain", tags=_TAGS)
    def agent_node(state: SupportState) -> dict:
        metadata = state.get("request_metadata", {})
        student_id = metadata.get("student_id", "unknown")

        # Build escalation context string from any pending escalations that
        # target this department (from a previous aggregation pass)
        escalation_items = [
            e for e in state.get("escalation_queue", [])
            if e.get("target") == department
        ]
        escalation_context = ""
        if escalation_items:
            ctx_lines = [e.get("context", "") for e in escalation_items]
            escalation_context = "\nEscalation context:\n" + "\n".join(ctx_lines)

        # Format the department's system prompt
        system_prompt = prompt_template.format(
            student_id=student_id,
            request=state["request"],
            escalation_context=escalation_context,
        )

        # Bind tools to a fresh model instance so tool schemas are included
        model_with_tools = ChatAnthropic(
            model="claude-haiku-4-5-20251001"
        ).bind_tools(dept_tools)

        # Run the agent loop
        response_text = _run_agent_loop(
            llm_with_tools=model_with_tools,
            system_prompt=system_prompt,
            request=state["request"],
            tools=dept_tools,
        )

        # Wrap in a DepartmentResult (always resolved=True here; the LLM may
        # include escalation JSON in its text, but we keep it simple for now)
        dept_result: DepartmentResult = {
            "department": department,
            "response": response_text,
            "resolved": True,
            "escalation": None,
        }

        # Use operator.add reducer — return a list so it gets appended
        return {"department_results": [dept_result]}

    return agent_node


# Create the four department agent nodes using the factory
billing_agent = _make_department_agent("billing", BILLING_PROMPT, BILLING_TOOLS)
tech_support_agent = _make_department_agent("tech_support", TECH_SUPPORT_PROMPT, TECH_SUPPORT_TOOLS)
scheduling_agent = _make_department_agent("scheduling", SCHEDULING_PROMPT, SCHEDULING_TOOLS)
content_agent = _make_department_agent("content", CONTENT_PROMPT, CONTENT_TOOLS)
