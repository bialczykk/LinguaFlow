"""Node functions for the Tutor Matching Agent StateGraph.

Contains:
- agent_node: the conversational LLM node that drives the agent loop
- should_continue: routing function for conditional edges
- get_tool_node: factory for the prebuilt ToolNode

LangGraph concepts demonstrated:
- Agent loop pattern: LLM node → conditional edge → tool node → back to LLM
- bind_tools(): attaching tool definitions to the model so it can request calls
- ToolNode: prebuilt node that executes tool calls from the LLM response
- Phase-based state updates within a single node function

LangChain concepts demonstrated:
- ChatAnthropic with bound tools
- SystemMessage for phase-aware behavior
- @traceable for LangSmith observability
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from models import TutorMatchingState
from prompts import get_system_prompt
from tools import search_tutors, check_availability, book_session

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# Bind all three tools to the model so it can request tool calls
_tools = [search_tutors, check_availability, book_session]
_model_with_tools = _model.bind_tools(_tools)

# -- LangSmith Tags --
_TAGS = ["p4-tutor-matching"]


@traceable(name="agent_node", run_type="chain", tags=_TAGS)
def agent_node(state: TutorMatchingState) -> dict:
    """Conversational agent node — the brain of the matching workflow.

    Reads the current phase from state, constructs a phase-aware system
    prompt, and invokes the model with bound tools. After the LLM responds,
    infers the next phase from deterministic heuristics.

    LangGraph concept: a single agentic node that drives multi-turn
    conversation through tool calling, contrasting with P2/P3's
    deterministic one-node-per-step approach.
    """
    phase = state.get("phase", "gather")

    # Build message list with phase-aware system prompt
    system_msg = SystemMessage(content=get_system_prompt(phase))
    messages = [system_msg] + state["messages"]

    # Invoke the LLM with bound tools
    response = _model_with_tools.invoke(messages, config={"tags": _TAGS})

    # -- Deterministic phase inference --
    new_phase = phase
    updates: dict = {"messages": [response]}

    if phase == "gather" and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "search_tutors" in tool_names:
            new_phase = "present"

    elif phase == "present" and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "check_availability" in tool_names:
            new_phase = "book"
        elif "search_tutors" in tool_names:
            new_phase = "present"

    elif phase == "book" and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        if "book_session" in tool_names:
            new_phase = "done"

    updates["phase"] = new_phase
    return updates


def should_continue(state: TutorMatchingState) -> Literal["tool_node", "__end__"]:
    """Routing function for the conditional edge after agent_node.

    - tool_calls present → route to tool_node
    - Otherwise → END (done or waiting for user)
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "__end__"


def get_tool_node() -> ToolNode:
    """Create the prebuilt ToolNode for executing tool calls.

    LangGraph concept: ToolNode from langgraph.prebuilt automatically
    dispatches tool calls to matching tool functions. handle_tool_errors=True
    means errors are returned as ToolMessages so the LLM can recover.
    """
    return ToolNode(_tools, handle_tool_errors=True)
