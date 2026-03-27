"""P6 — Multi-Department Support System tab.

Hybrid UI combining chat-like clarification flow with structured results.
Users submit a support request, and the system routes it to one or more
department agents. If the request is ambiguous, the system asks for
clarification before processing.
"""

import streamlit as st
from adapters import support_system
from components import doc_viewer

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/06-multi-department-support.md"

# Department display labels
_DEPT_LABELS = {
    "billing": "Billing",
    "tech_support": "Tech Support",
    "scheduling": "Scheduling",
    "content": "Content Library",
}

# Pattern display labels for sample requests
_PATTERN_LABELS = {
    "single": "Single Dept",
    "parallel": "Multi-Dept (Parallel)",
    "clarification": "Needs Clarification",
}


def _reset_state() -> None:
    """Clear all p6_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p6_"):
            del st.session_state[key]


def _init_state() -> None:
    """Initialize session state defaults."""
    if "p6_stage" not in st.session_state:
        st.session_state["p6_stage"] = "idle"
        st.session_state["p6_thread_id"] = None
        st.session_state["p6_result"] = None
        st.session_state["p6_clarification_question"] = None
        st.session_state["p6_chat_history"] = []  # list of {"role": str, "content": str}


def _add_message(role: str, content: str) -> None:
    """Append a message to the chat history."""
    st.session_state["p6_chat_history"].append({"role": role, "content": content})


def _render_chat_history() -> None:
    """Render the chat history using st.chat_message."""
    for msg in st.session_state.get("p6_chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_request_form() -> None:
    """Render the initial support request input form (idle stage)."""
    st.subheader("Submit a Support Request")

    # Sample request selector for quick testing
    samples = support_system.get_sample_requests()
    sample_labels = ["(custom — type your own)"] + [
        f"{s['text'][:60]}... [{_PATTERN_LABELS.get(s['pattern'], s['pattern'])}]"
        for s in samples
    ]

    selected = st.selectbox(
        "Quick-start with a sample request:",
        sample_labels,
        key="p6_sample",
    )

    if selected != "(custom — type your own)":
        idx = sample_labels.index(selected) - 1
        sample = samples[idx]
        request_text = sample["text"]
        metadata = sample["metadata"]
        st.info(f"**Request:** {request_text}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Sender type: **{metadata['sender_type']}**")
        with col2:
            st.caption(f"Student ID: **{metadata['student_id']}**")
        with col3:
            st.caption(f"Priority: **{metadata['priority']}**")
        can_submit = True
    else:
        # Custom request entry
        request_text = st.text_area(
            "Describe your support issue:",
            placeholder="e.g. I was charged twice for my last lesson...",
            height=100,
            key="p6_custom_request",
        )
        metadata = {"sender_type": "student", "student_id": "S999", "priority": "medium"}
        can_submit = bool(request_text and request_text.strip())

    if st.button("Submit Request", key="p6_submit", type="primary", disabled=not can_submit):
        thread_id = support_system.create_thread_id()
        st.session_state["p6_thread_id"] = thread_id
        _add_message("user", request_text)

        with st.spinner("Routing your request to the right department(s)..."):
            try:
                result = support_system.start_request(thread_id, request_text, metadata)

                if isinstance(result, str):
                    # Graph interrupted — result is the clarification question
                    st.session_state["p6_stage"] = "clarification"
                    st.session_state["p6_clarification_question"] = result
                    _add_message("assistant", f"Before I route your request, I need a bit more information:\n\n{result}")
                else:
                    # Graph completed — result is the final state dict
                    st.session_state["p6_stage"] = "done"
                    st.session_state["p6_result"] = result
                    final_response = result.get("final_response", "Your request has been processed.")
                    _add_message("assistant", final_response)

                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_clarification() -> None:
    """Render the clarification flow — system asked a question, user responds."""
    _render_chat_history()

    st.divider()

    # User response input
    user_response = st.text_area(
        "Your response:",
        placeholder="Please provide the additional information requested above...",
        height=80,
        key="p6_clarification_input",
    )

    if st.button("Send", key="p6_clarification_submit", type="primary", disabled=not bool(user_response and user_response.strip())):
        _add_message("user", user_response)

        with st.spinner("Processing your response and routing to the right department(s)..."):
            try:
                result = support_system.resume_with_clarification(
                    st.session_state["p6_thread_id"], user_response
                )

                if isinstance(result, str):
                    # Another clarification round
                    st.session_state["p6_clarification_question"] = result
                    _add_message("assistant", f"Thanks! One more question:\n\n{result}")
                else:
                    # Completed
                    st.session_state["p6_stage"] = "done"
                    st.session_state["p6_result"] = result
                    final_response = result.get("final_response", "Your request has been processed.")
                    _add_message("assistant", final_response)

                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_done() -> None:
    """Render the completed request with final response and behind-the-scenes details."""
    _render_chat_history()

    st.divider()
    state = st.session_state.get("p6_result") or {}

    # Resolution status badge
    status = state.get("resolution_status", "")
    if status == "resolved":
        st.success("Request resolved successfully")
    elif status == "partial":
        st.warning("Request partially resolved — some items may need follow-up")
    elif status == "escalated_to_human":
        st.error("Request escalated to a human agent for further assistance")

    # Behind the scenes expander — shows the routing/agent details
    classification = state.get("classification", {})
    department_results = state.get("department_results", [])
    escalation_queue = state.get("escalation_queue", [])

    with st.expander("Behind the scenes", expanded=False):
        # Routing decision
        st.markdown("**Routing Decision (Classification)**")
        departments = classification.get("departments", [])
        dept_names = [_DEPT_LABELS.get(d, d) for d in departments]
        if dept_names:
            st.markdown(f"- Departments consulted: **{', '.join(dept_names)}**")
        complexity = classification.get("complexity", "")
        if complexity:
            st.markdown(f"- Complexity: **{complexity}**")
        summary = classification.get("summary", "")
        if summary:
            st.markdown(f"- Summary: {summary}")

        st.divider()

        # Individual department responses
        st.markdown("**Department Responses**")
        if department_results:
            for result in department_results:
                dept = result.get("department", "unknown")
                dept_label = _DEPT_LABELS.get(dept, dept)
                resolved = result.get("resolved", True)
                status_icon = "✅" if resolved else "⚠️"
                with st.container():
                    st.markdown(f"**{status_icon} {dept_label}**")
                    st.markdown(result.get("response", ""))
                    if not resolved and result.get("escalation"):
                        esc = result["escalation"]
                        target = _DEPT_LABELS.get(esc.get("target", ""), esc.get("target", ""))
                        st.caption(f"Escalated to: {target} — {esc.get('context', '')}")
        else:
            st.caption("No department responses recorded.")

        # Escalations
        if escalation_queue:
            st.divider()
            st.markdown("**Escalations Handled**")
            for esc in escalation_queue:
                target = _DEPT_LABELS.get(esc.get("target", ""), esc.get("target", ""))
                st.markdown(f"- Escalated to **{target}**: {esc.get('context', '')}")


def render() -> None:
    """Render the Multi-Department Support System interface."""
    st.header("Multi-Department Support System")
    st.caption("Multi-agent orchestration with parallel department routing and clarification flow")

    _init_state()

    # -- Reset button and stage indicator --
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset", key="p6_reset"):
            _reset_state()
            st.rerun()
    with col2:
        stage = st.session_state.get("p6_stage", "idle")
        stage_labels = {
            "idle": "Ready",
            "clarification": "Awaiting Clarification",
            "done": "Complete",
        }
        st.caption(f"Stage: **{stage_labels.get(stage, stage)}**")

    st.divider()

    # -- Render current stage --
    stage = st.session_state.get("p6_stage", "idle")

    if stage == "idle":
        _render_request_form()
    elif stage == "clarification":
        _render_clarification()
    elif stage == "done":
        _render_done()

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: Multi-Agent Orchestration")
