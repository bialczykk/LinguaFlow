"""State schema and model definitions for the multi-department support system.

SupportState is the shared state for the entire graph. All nodes read from
and write to this state. The department_results field uses an operator.add
reducer so that parallel sub-agents (dispatched via Send) can each append
their result independently.

DepartmentResult is the structured output from each sub-agent node.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class DepartmentResult(TypedDict):
    """Structured result from a department sub-agent.

    Each sub-agent returns one of these, appended to department_results.
    If the sub-agent can't fully resolve the request, it sets resolved=False
    and populates escalation with the target department and context.
    """

    department: str           # "billing" | "tech_support" | "scheduling" | "content"
    response: str             # The sub-agent's response text
    resolved: bool            # Whether the sub-agent fully handled its part
    escalation: dict | None   # If not resolved: {"target": "<dept>", "context": "..."}


class SupportState(TypedDict):
    """Shared state for the multi-department support graph.

    Fields are grouped by lifecycle stage:
    - Input: set at invocation
    - Supervisor analysis: set by supervisor_router
    - Sub-agent results: appended by each sub-agent (reducer: operator.add)
    - Escalation: managed by supervisor_aggregator
    - Conversation: used for hybrid clarification flow
    - Output: set by compose_response
    """

    # --- Input ---
    request: str                                            # User's support request text
    request_metadata: dict                                  # sender_type, student_id, priority

    # --- Supervisor analysis ---
    classification: dict                                    # departments, complexity, summary

    # --- Sub-agent results (reducer: append for parallel Send) ---
    department_results: Annotated[list[DepartmentResult], operator.add]

    # --- Escalation tracking ---
    escalation_queue: list[dict]                            # Pending cross-dept escalations

    # --- Conversation (hybrid clarification) ---
    clarification_needed: str | None                        # Question to ask the user
    user_clarification: str | None                          # User's response

    # --- Final output ---
    final_response: str                                     # Unified response to the user
    resolution_status: str                                  # "resolved" | "partial" | "escalated_to_human"


# Valid department names — used for validation in routing
DEPARTMENTS = {"billing", "tech_support", "scheduling", "content"}
