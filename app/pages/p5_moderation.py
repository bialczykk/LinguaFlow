"""P5 — Content Moderation & QA System tab.

Form-based interface for a human-in-the-loop content pipeline.
Users select a content request, generate a draft, then act as
moderator at two interrupt points (draft review, final review).
"""

import streamlit as st
from adapters import content_moderation
from components import doc_viewer, overview

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/05-content-moderation-qa.md"

# Content type display labels
_TYPE_LABELS = {
    "grammar_explanation": "Grammar Explanation",
    "vocabulary_exercise": "Vocabulary Exercise",
    "reading_passage": "Reading Passage",
}

# Pipeline stage labels
_STAGE_LABELS = {
    "idle": "Ready",
    "draft_review": "Draft Review",
    "final_review": "Final Review",
    "done": "Complete",
}


def _reset_state() -> None:
    """Clear all p5_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p5_"):
            del st.session_state[key]


def _init_state() -> None:
    """Initialize session state defaults."""
    if "p5_stage" not in st.session_state:
        st.session_state["p5_stage"] = "idle"
        st.session_state["p5_thread_id"] = None
        st.session_state["p5_interrupt"] = None
        st.session_state["p5_log"] = []  # list of (label, content) tuples


def _add_log(label: str, content: str) -> None:
    """Append an entry to the pipeline log."""
    st.session_state["p5_log"].append((label, content))


def _render_request_form() -> None:
    """Render the content request form (idle stage)."""
    st.subheader("Content Request")

    # Sample request selector
    samples = content_moderation.get_sample_requests()
    sample_labels = ["(custom)"] + [
        f"{s['topic']} ({_TYPE_LABELS.get(s['content_type'], s['content_type'])}, {s['difficulty']})"
        for s in samples
    ]

    selected = st.selectbox("Quick-start with a sample:", sample_labels, key="p5_sample")

    if selected != "(custom)":
        # Use sample data directly — show as read-only info
        idx = sample_labels.index(selected) - 1
        sample = samples[idx]
        topic = sample["topic"]
        content_type = sample["content_type"]
        difficulty = sample["difficulty"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Topic:** {topic}")
        with col2:
            st.info(f"**Type:** {_TYPE_LABELS.get(content_type, content_type)}")
        with col3:
            st.info(f"**Level:** {difficulty}")
        can_generate = True
    else:
        # Custom form fields
        col1, col2 = st.columns(2)
        with col1:
            content_type = st.selectbox(
                "Content type",
                content_moderation.get_content_types(),
                format_func=lambda x: _TYPE_LABELS.get(x, x),
                key="p5_content_type",
            )
            difficulty = st.selectbox(
                "CEFR difficulty",
                content_moderation.get_cefr_levels(),
                key="p5_difficulty",
            )
        with col2:
            topic = st.text_input("Topic", key="p5_topic")
        can_generate = bool(topic)

    # Generate button
    if st.button("Generate Draft", key="p5_generate", type="primary", disabled=not can_generate):
        request = {
            "topic": topic,
            "content_type": content_type,
            "difficulty": difficulty,
        }
        st.session_state["p5_thread_id"] = content_moderation.create_thread_id()
        _add_log("Request", f"**{_TYPE_LABELS.get(content_type, content_type)}** on "
                 f"*{topic}* at **{difficulty}** level")

        with st.spinner("Generating draft content..."):
            try:
                interrupt_val = content_moderation.start_pipeline(
                    st.session_state["p5_thread_id"], request
                )
                if interrupt_val:
                    st.session_state["p5_stage"] = "draft_review"
                    st.session_state["p5_interrupt"] = interrupt_val
                    _add_log("Generated", f"Confidence: {interrupt_val.get('confidence', 'N/A')}")
                else:
                    st.session_state["p5_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_draft_review() -> None:
    """Render the draft review moderator interface."""
    st.subheader("Draft Review")
    interrupt_val = st.session_state["p5_interrupt"]

    # Show the draft content
    st.markdown("**Generated Content:**")
    st.text_area(
        "Draft",
        value=interrupt_val.get("content", ""),
        height=250,
        disabled=True,
        key="p5_draft_display",
        label_visibility="collapsed",
    )

    # Metadata
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{interrupt_val.get('confidence', 0):.0%}")
    with col2:
        st.metric("Revision Round", interrupt_val.get("revision_count", 0))

    st.divider()
    st.markdown("**Moderator Decision:**")

    # Action selector
    action = st.radio(
        "Action",
        ["approve", "edit", "reject"],
        format_func=lambda x: {"approve": "Approve", "edit": "Edit & Approve", "reject": "Reject with Feedback"}[x],
        horizontal=True,
        key="p5_draft_action",
    )

    # Conditional fields based on action
    edited_content = None
    feedback = ""

    if action == "edit":
        edited_content = st.text_area(
            "Edit the content:",
            value=interrupt_val.get("content", ""),
            height=250,
            key="p5_edited_content",
        )
    elif action == "reject":
        feedback = st.text_area(
            "Feedback for revision:",
            placeholder="Explain what needs to change...",
            key="p5_reject_feedback",
        )

    # Submit decision
    if st.button("Submit Decision", key="p5_draft_submit", type="primary"):
        decision = {"action": action}
        if action == "edit" and edited_content:
            decision["edited_content"] = edited_content
        if action == "reject":
            decision["feedback"] = feedback

        action_label = {"approve": "Approved", "edit": "Edited & Approved", "reject": "Rejected"}[action]
        _add_log("Draft Review", f"{action_label}" + (f" — {feedback}" if feedback else ""))

        with st.spinner("Processing decision..."):
            try:
                interrupt_val = content_moderation.resume_pipeline(
                    st.session_state["p5_thread_id"], decision
                )
                if interrupt_val:
                    # Next interrupt — could be draft_review again (after revise) or final_review
                    next_tasks = content_moderation.get_next_task(st.session_state["p5_thread_id"])
                    if "final_review" in next_tasks:
                        st.session_state["p5_stage"] = "final_review"
                    else:
                        st.session_state["p5_stage"] = "draft_review"
                    st.session_state["p5_interrupt"] = interrupt_val
                else:
                    st.session_state["p5_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_final_review() -> None:
    """Render the final review moderator interface."""
    st.subheader("Final Review")
    interrupt_val = st.session_state["p5_interrupt"]

    # Show the polished content
    st.markdown("**Polished Content (ready for publication):**")
    st.text_area(
        "Final",
        value=interrupt_val.get("content", ""),
        height=300,
        disabled=True,
        key="p5_final_display",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Final Decision:**")

    action = st.radio(
        "Action",
        ["approve", "reject"],
        format_func=lambda x: {"approve": "Publish", "reject": "Reject"}[x],
        horizontal=True,
        key="p5_final_action",
    )

    feedback = ""
    if action == "reject":
        feedback = st.text_area(
            "Reason for rejection:",
            key="p5_final_feedback",
        )

    if st.button("Submit Final Decision", key="p5_final_submit", type="primary"):
        decision = {"action": action}
        if feedback:
            decision["feedback"] = feedback

        _add_log("Final Review", "Published" if action == "approve" else f"Rejected — {feedback}")

        with st.spinner("Processing..."):
            try:
                content_moderation.resume_pipeline(
                    st.session_state["p5_thread_id"], decision
                )
                st.session_state["p5_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_done() -> None:
    """Render the completed pipeline summary."""
    st.subheader("Pipeline Complete")

    state = content_moderation.get_state(st.session_state["p5_thread_id"])

    if state.get("published"):
        st.success("Content published successfully!")
        st.markdown("**Final Content:**")
        st.text_area(
            "Published",
            value=state.get("polished_content", state.get("draft_content", "")),
            height=250,
            disabled=True,
            key="p5_published_display",
            label_visibility="collapsed",
        )
        metadata = state.get("publish_metadata", {})
        if metadata:
            st.caption(f"Review rounds: {metadata.get('review_rounds', 0)}")
    else:
        st.warning("Content was not published (rejected or max revisions reached).")
        if state.get("draft_content"):
            st.markdown("**Last Draft:**")
            st.text_area(
                "Last draft",
                value=state.get("draft_content", ""),
                height=200,
                disabled=True,
                key="p5_last_draft_display",
                label_visibility="collapsed",
            )


def render() -> None:
    """Render the Content Moderation & QA System interface."""
    st.header("Content Moderation & QA System")
    st.caption("Human-in-the-loop content pipeline with interrupt/resume")

    overview.render(
        business_scenario=(
            "The platform needs educational content (grammar explanations, exercises, "
            "reading passages) generated at scale, but every piece must pass human review "
            "before reaching students. Content moderators review AI-generated drafts, "
            "can edit or reject with feedback, and give final publication approval. "
            "This ensures quality control while dramatically speeding up content creation."
        ),
        tech_flowchart="""
            digraph {
                rankdir=LR
                node [shape=box style="rounded,filled" fillcolor="#f0f4ff" fontname="Helvetica" fontsize=11]
                edge [fontname="Helvetica" fontsize=10]

                request [label="Content\\nRequest" shape=note fillcolor="#fff3cd"]
                generate [label="Generate\\nDraft"]
                interrupt1 [label="interrupt()\\nDraft Review" shape=octagon fillcolor="#f8d7da"]
                moderator1 [label="Moderator\\nDecision" shape=diamond fillcolor="#ffeeba"]
                polish [label="Polish\\nContent"]
                interrupt2 [label="interrupt()\\nFinal Review" shape=octagon fillcolor="#f8d7da"]
                moderator2 [label="Publish?" shape=diamond fillcolor="#ffeeba"]
                publish [label="Published" shape=note fillcolor="#d4edda"]
                langsmith [label="LangSmith\\nTracing" shape=ellipse fillcolor="#e8daef"]

                request -> generate
                generate -> interrupt1
                interrupt1 -> moderator1 [label="Command\\n(resume)"]
                moderator1 -> generate [label="reject"]
                moderator1 -> polish [label="approve/edit"]
                polish -> interrupt2
                interrupt2 -> moderator2 [label="Command\\n(resume)"]
                moderator2 -> publish [label="approve"]
                moderator2 -> generate [label="reject"]
                generate -> langsmith [style=dashed]
            }
        """,
        key_prefix="p5",
    )

    _init_state()

    # -- Stage indicator and reset --
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset", key="p5_reset"):
            _reset_state()
            st.rerun()
    with col2:
        stage = st.session_state.get("p5_stage", "idle")
        stage_label = _STAGE_LABELS.get(stage, stage)
        st.caption(f"Stage: **{stage_label}**")

    st.divider()

    # -- Pipeline log (collapsible) --
    log = st.session_state.get("p5_log", [])
    if log:
        with st.expander("Pipeline Log", expanded=False):
            for label, content in log:
                st.markdown(f"**{label}:** {content}")

    # -- Render current stage --
    stage = st.session_state.get("p5_stage", "idle")

    if stage == "idle":
        _render_request_form()
    elif stage == "draft_review":
        _render_draft_review()
    elif stage == "final_review":
        _render_final_review()
    elif stage == "done":
        _render_done()

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: Human-in-the-Loop & Error Handling")
