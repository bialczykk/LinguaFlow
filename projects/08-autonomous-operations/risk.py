"""Risk assessment rules for the tiered approval system.

Two tiers define how the orchestrator routes each task:
- Low risk:  auto-execute — data lookups, draft generation, assessments, metrics
- High risk: requires human approval — refunds, publication, tutor assignment, flagging

The assess_risk() function is called by the risk_assessor node in the
orchestrator graph immediately after classification. It is pure logic —
no LLM is involved — which keeps latency near zero for this decision.

Design rationale
----------------
Rather than embedding risk rules inside the LLM classification prompt
(which could be inconsistent across runs), we use a deterministic lookup.
This makes the approval boundary auditable and easy to extend: adding a
new high-risk action is a one-line change to HIGH_RISK_ACTIONS.
"""

# ---------------------------------------------------------------------------
# Risk map
# ---------------------------------------------------------------------------

# Maps each department to the set of action_types that require human approval.
# Only state-changing, hard-to-reverse, or user-impactful actions are listed.
# Read/lookup/aggregate actions are implicitly low-risk (not listed here).
HIGH_RISK_ACTIONS: dict[str, set[str]] = {
    # Publishing content is irreversible once live on the platform.
    "content_pipeline": {"publish_content"},
    # Issuing a refund moves money — must always have a human sign-off.
    "support": {"process_refund"},
    # Assigning a tutor directly affects a student's learning relationship.
    "tutor_management": {"assign_tutor"},
    # Flagging a QA issue can escalate to account suspension or content removal.
    "quality_assurance": {"flag_issue"},
    # Creating a study plan locks in a personalised curriculum — high stakes.
    "student_onboarding": {"create_study_plan"},
}


# ---------------------------------------------------------------------------
# Risk assessment function
# ---------------------------------------------------------------------------

def assess_risk(classification: dict) -> str:
    """Determine the risk level from a classification dict.

    The function checks whether the resolved action_type is listed as high-risk
    for **any** of the departments in the classification. If even one department
    marks the action as high-risk, the whole task requires approval. This is a
    conservative (safe-default) approach: when in doubt, ask a human.

    Args:
        classification: A dict with at minimum:
            - "departments" (list[str]): departments involved in handling the task
            - "action_type"  (str):       the specific action to be performed
            May also contain "is_follow_up" (bool) or other keys — these are
            ignored; risk is determined solely by department + action_type.

    Returns:
        "high" if the action matches any HIGH_RISK_ACTIONS entry for any
               of the listed departments.
        "low"  otherwise (unknown departments and unknown actions are safe defaults).

    Examples:
        >>> assess_risk({"departments": ["support"], "action_type": "lookup"})
        'low'
        >>> assess_risk({"departments": ["content_pipeline"], "action_type": "publish_content"})
        'high'
    """
    action_type = classification.get("action_type", "")
    departments = classification.get("departments", [])

    # Iterate over each department and check if the action is in its high-risk set.
    # We use dict.get(dept, set()) so that unknown departments return an empty set
    # rather than raising a KeyError — unknown = low risk by default.
    for dept in departments:
        high_risk_set = HIGH_RISK_ACTIONS.get(dept, set())
        if action_type in high_risk_set:
            return "high"

    # No department flagged this action as high risk → auto-execute tier.
    return "low"
