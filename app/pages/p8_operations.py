"""P8 — Autonomous Operations tab.

Three-column layout for submitting cross-department operations requests,
monitoring autonomous task cascading, reviewing approval gates, and
viewing cumulative platform metrics.
"""

import streamlit as st
from adapters import autonomous_ops
from components import doc_viewer, overview

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/08-autonomous-operations.md"

# Priority display labels
_PRIORITY_LABELS = ["low", "medium", "high"]

# Source display labels
_SOURCE_LABELS = ["ui", "api", "support_form", "scheduler"]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Initialize session state defaults for P8."""
    if "p8_stage" not in st.session_state:
        st.session_state["p8_stage"] = "idle"
        st.session_state["p8_thread_id"] = None
        st.session_state["p8_activity_log"] = []  # list of (label, content) tuples
        st.session_state["p8_approval_pending"] = None


def _reset_state() -> None:
    """Clear all p8_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p8_"):
            del st.session_state[key]


def _add_activity(label: str, content: str) -> None:
    """Append an entry to the activity log."""
    st.session_state["p8_activity_log"].append((label, content))


# ---------------------------------------------------------------------------
# Left column: Console (input + task queue + approval panel)
# ---------------------------------------------------------------------------

def _render_console() -> None:
    """Render the left console: request input, task queue, and approval panel."""
    st.subheader("Operations Console")

    stage = st.session_state.get("p8_stage", "idle")

    # -- Request input (only shown when idle) --
    if stage == "idle":
        _render_request_input()

    # -- Task queue (shown during processing and done stages) --
    if stage in ("processing", "done", "approval"):
        _render_task_queue()

    # -- Approval panel (shown when waiting for approval) --
    if stage == "approval":
        _render_approval_panel()


def _render_request_input() -> None:
    """Render the request submission form."""
    st.markdown("**New Request**")

    # Sample request selector
    samples = autonomous_ops.get_sample_requests()
    sample_labels = ["(custom)"] + [
        f"{s['text'][:60]}..." if len(s["text"]) > 60 else s["text"]
        for s in samples
    ]

    selected = st.selectbox("Quick-start with a sample:", sample_labels, key="p8_sample")

    if selected != "(custom)":
        idx = sample_labels.index(selected) - 1
        sample = samples[idx]
        request_text = sample["text"]
        metadata = sample["metadata"]

        st.text_area(
            "Request",
            value=request_text,
            height=100,
            disabled=True,
            key="p8_sample_display",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Priority: **{metadata.get('priority', 'medium')}**")
        with col2:
            st.caption(f"Pattern: **{sample.get('pattern', 'single')}**")

        # Show expected risk as a hint
        expected_risk = sample.get("expected_risk", "low")
        if expected_risk == "high":
            st.warning("This request may require approval (high-risk action).")

        can_submit = True
        submit_request_text = request_text
        submit_metadata = metadata
    else:
        # Custom form
        submit_request_text = st.text_area(
            "Request text",
            placeholder="Describe the operation you need performed...",
            height=100,
            key="p8_custom_text",
        )

        col1, col2 = st.columns(2)
        with col1:
            priority = st.selectbox("Priority", _PRIORITY_LABELS, index=1, key="p8_priority")
        with col2:
            source = st.selectbox("Source", _SOURCE_LABELS, key="p8_source")

        submit_metadata = {
            "user_id": "admin",
            "priority": priority,
            "source": source,
        }
        can_submit = bool(submit_request_text and submit_request_text.strip())

    st.divider()

    if st.button(
        "Submit Request",
        key="p8_submit",
        type="primary",
        disabled=not can_submit,
    ):
        thread_id = autonomous_ops.create_thread_id()
        st.session_state["p8_thread_id"] = thread_id
        st.session_state["p8_stage"] = "processing"
        _add_activity("Request", submit_request_text[:120] + ("..." if len(submit_request_text) > 120 else ""))

        with st.spinner("Dispatching autonomous agents..."):
            try:
                interrupt_val = autonomous_ops.start_request(
                    thread_id, submit_request_text, submit_metadata
                )
                if interrupt_val:
                    st.session_state["p8_stage"] = "approval"
                    st.session_state["p8_approval_pending"] = interrupt_val
                    _add_activity("Approval Required", interrupt_val.get("message", "High-risk action detected."))
                else:
                    st.session_state["p8_stage"] = "done"
                    _add_activity("Completed", "All departments responded successfully.")
                st.rerun()
            except RuntimeError as e:
                st.session_state["p8_stage"] = "idle"
                st.error(str(e))


def _render_task_queue() -> None:
    """Render the autonomous task queue panel."""
    thread_id = st.session_state.get("p8_thread_id")
    if not thread_id:
        return

    task_queue = autonomous_ops.get_task_queue(thread_id)

    st.markdown("**Task Queue**")
    if task_queue:
        for i, task in enumerate(task_queue):
            target = task.get("target_dept", "unknown")
            action = task.get("action", "")
            st.caption(f"{i + 1}. **{target}** — {action[:80]}")
    else:
        st.caption("No pending tasks.")


def _render_approval_panel() -> None:
    """Render the human-in-the-loop approval panel."""
    interrupt_val = st.session_state.get("p8_approval_pending")
    if not interrupt_val:
        return

    st.divider()
    st.markdown("**Approval Required**")
    st.warning(interrupt_val.get("message", "A high-risk action requires your approval."))

    # Show action details if available
    action_details = interrupt_val.get("action_details", {})
    if action_details:
        with st.expander("Action details", expanded=True):
            for k, v in action_details.items():
                st.markdown(f"**{k}:** {v}")
    elif "classification" in interrupt_val:
        clf = interrupt_val["classification"]
        with st.expander("Classification details", expanded=True):
            st.markdown(f"**Action type:** {clf.get('action_type', 'N/A')}")
            st.markdown(f"**Departments:** {', '.join(clf.get('departments', []))}")
            st.markdown(f"**Complexity:** {clf.get('complexity', 'N/A')}")

    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("Approve", key="p8_approve", type="primary"):
            _handle_approval({"action": "approve"})

    with col_reject:
        if st.button("Reject", key="p8_reject"):
            _handle_approval({"action": "reject", "reason": "Operator rejected the action."})


def _handle_approval(decision: dict) -> None:
    """Process an approval or rejection decision."""
    thread_id = st.session_state.get("p8_thread_id")
    action_label = "Approved" if decision["action"] == "approve" else "Rejected"
    _add_activity("Operator Decision", action_label)

    with st.spinner(f"Processing {decision['action']}..."):
        try:
            interrupt_val = autonomous_ops.resume_approval(thread_id, decision)
            if interrupt_val:
                st.session_state["p8_stage"] = "approval"
                st.session_state["p8_approval_pending"] = interrupt_val
                _add_activity("Approval Required", interrupt_val.get("message", "Another approval needed."))
            else:
                st.session_state["p8_stage"] = "done"
                st.session_state["p8_approval_pending"] = None
                _add_activity("Completed", "Request fully processed.")
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Center column: Activity feed and results
# ---------------------------------------------------------------------------

def _render_activity() -> None:
    """Render the center column: activity feed and final response."""
    st.subheader("Activity Feed")

    stage = st.session_state.get("p8_stage", "idle")
    activity_log = st.session_state.get("p8_activity_log", [])

    if not activity_log:
        st.caption("No activity yet. Submit a request to get started.")
        return

    # Activity log entries
    for label, content in activity_log:
        with st.container():
            st.markdown(f"**{label}**")
            st.caption(content)
            st.divider()

    # Final response when done
    if stage == "done":
        thread_id = st.session_state.get("p8_thread_id")
        if thread_id:
            state = autonomous_ops.get_state(thread_id)
            final_response = state.get("final_response", "")
            resolution_status = state.get("resolution_status", "")

            if final_response:
                st.success("Request Completed")
                st.markdown("**Final Response:**")
                st.text_area(
                    "Response",
                    value=final_response,
                    height=200,
                    disabled=True,
                    key="p8_final_response_display",
                    label_visibility="collapsed",
                )
                if resolution_status:
                    st.caption(f"Resolution: **{resolution_status}**")

            # Behind the scenes expander
            dept_results = state.get("department_results", [])
            completed_tasks = state.get("completed_tasks", [])
            classification = state.get("classification", {})

            with st.expander("Behind the scenes", expanded=False):
                if classification:
                    st.markdown("**Classification:**")
                    st.markdown(f"- Action type: `{classification.get('action_type', 'N/A')}`")
                    st.markdown(f"- Departments: `{', '.join(classification.get('departments', []))}`")
                    st.markdown(f"- Complexity: `{classification.get('complexity', 'N/A')}`")

                if dept_results:
                    st.markdown(f"**Department Results** ({len(dept_results)} departments):")
                    for result in dept_results:
                        dept_name = result.get("department", "unknown")
                        resolved = result.get("resolved", False)
                        status_icon = "✓" if resolved else "~"
                        st.markdown(f"- {status_icon} **{dept_name}**: {result.get('response', '')[:120]}")

                if completed_tasks:
                    st.markdown(f"**Completed Tasks** ({len(completed_tasks)}):")
                    for task in completed_tasks:
                        st.markdown(f"- {task.get('target_dept', '?')} — {task.get('action', '')[:80]}")


# ---------------------------------------------------------------------------
# Right column: Metrics dashboard
# ---------------------------------------------------------------------------

def _render_metrics() -> None:
    """Render the right column: metrics dashboard."""
    st.subheader("Platform Metrics")

    thread_id = st.session_state.get("p8_thread_id")

    if not thread_id:
        st.caption("Metrics will appear after the first request.")
        return

    metrics = autonomous_ops.get_metrics(thread_id)

    if not metrics:
        st.caption("No metrics recorded yet.")
    else:
        # Key metric cards
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Total Requests", metrics.get("total_requests", 0))
            st.metric("Students Onboarded", metrics.get("students_onboarded", 0))
            st.metric("Content Generated", metrics.get("content_generated", 0))
            st.metric("Support Requests", metrics.get("support_requests", 0))

        with col_b:
            st.metric("Tutors Assigned", metrics.get("tutors_assigned", 0))
            st.metric("Content Published", metrics.get("content_published", 0))
            st.metric("QA Reviews", metrics.get("qa_reviews", 0))
            st.metric("Support Resolved", metrics.get("support_resolved", 0))

        # Department invocations table
        dept_invocations = metrics.get("department_invocations", {})
        if dept_invocations:
            st.markdown("**Department Invocations:**")
            for dept, count in sorted(dept_invocations.items(), key=lambda x: -x[1]):
                st.caption(f"{dept}: **{count}**")

    st.divider()

    # Generate report button
    st.markdown("**Reporting**")
    if st.button("Generate Report", key="p8_report"):
        if thread_id:
            with st.spinner("Generating platform report..."):
                try:
                    report_thread_id = autonomous_ops.create_thread_id()
                    interrupt_val = autonomous_ops.start_request(
                        report_thread_id,
                        "How is the platform performing this week?",
                        {"user_id": "admin", "priority": "low", "source": "ui"},
                    )
                    if not interrupt_val:
                        state = autonomous_ops.get_state(report_thread_id)
                        report_text = state.get("final_response", "No report generated.")
                        st.text_area(
                            "Report",
                            value=report_text,
                            height=200,
                            disabled=True,
                            key="p8_report_output",
                            label_visibility="collapsed",
                        )
                    else:
                        st.warning("Report request requires approval.")
                except RuntimeError as e:
                    st.error(str(e))
        else:
            st.info("Submit a request first to enable reporting.")


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Autonomous Operations interface."""
    st.header("Autonomous Operations")
    st.caption("Cross-department orchestration with autonomous task cascading")

    overview.render(
        business_scenario=(
            "LinguaFlow's operations span six departments: Student Onboarding, Tutor Management, "
            "Content Pipeline, Quality Assurance, Support, and Reporting. A master orchestrator "
            "receives requests, classifies them, and dispatches to the right department agents — "
            "in parallel when multiple departments are involved. Agents can generate follow-up tasks "
            "for other departments, creating autonomous cascades (e.g., onboarding a student "
            "automatically triggers tutor matching). High-risk actions like publishing content or "
            "assigning tutors require human approval before proceeding."
        ),
        tech_flowchart="""
            digraph {
                rankdir=TB
                node [shape=box style="rounded,filled" fillcolor="#f0f4ff" fontname="Helvetica" fontsize=11]
                edge [fontname="Helvetica" fontsize=10]

                request [label="User\\nRequest" shape=note fillcolor="#fff3cd"]
                classifier [label="Request\\nClassifier"]
                risk [label="Risk\\nAssessor"]
                low [label="Low Risk?" shape=diamond fillcolor="#d4edda"]
                approval [label="interrupt()\\nApproval Gate" shape=octagon fillcolor="#f8d7da"]
                dispatch [label="Dispatch\\nDepartments"]

                subgraph cluster_departments {
                    label="Department Agents (DeepAgents + Send)"
                    style=dashed
                    onboard [label="Student\\nOnboarding"]
                    tutor [label="Tutor\\nManagement"]
                    content [label="Content\\nPipeline"]
                    qa [label="Quality\\nAssurance"]
                    support [label="Support"]
                    reporting [label="Reporting"]
                }

                aggregator [label="Result\\nAggregator"]
                queue [label="Task Queue\\nLoop" shape=diamond fillcolor="#ffeeba"]
                compose [label="Compose\\nOutput"]
                metrics [label="Reporting\\nSnapshot"]
                done [label="Final\\nResponse" shape=note fillcolor="#fff3cd"]
                langsmith [label="LangSmith\\nTracing" shape=ellipse fillcolor="#e8daef"]

                request -> classifier
                classifier -> risk
                risk -> low
                low -> dispatch [label="auto"]
                low -> approval [label="high risk"]
                approval -> dispatch [label="approved"]
                approval -> compose [label="rejected"]
                dispatch -> onboard
                dispatch -> tutor
                dispatch -> content
                dispatch -> qa
                dispatch -> support
                dispatch -> reporting
                onboard -> aggregator
                tutor -> aggregator
                content -> aggregator
                qa -> aggregator
                support -> aggregator
                reporting -> aggregator
                aggregator -> queue
                queue -> classifier [label="follow-ups"]
                queue -> compose [label="done"]
                compose -> metrics
                metrics -> done
                classifier -> langsmith [style=dashed]
            }
        """,
        key_prefix="p8",
    )

    _init_state()

    # -- Reset button and stage indicator --
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset", key="p8_reset"):
            _reset_state()
            st.rerun()
    with col2:
        stage = st.session_state.get("p8_stage", "idle")
        st.caption(f"Stage: **{stage}**")

    st.divider()

    # -- Three-column layout --
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_left:
        _render_console()

    with col_center:
        _render_activity()

    with col_right:
        _render_metrics()

    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: Autonomous Operations")
