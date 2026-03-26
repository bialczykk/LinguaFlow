"""P7 — Intelligent Curriculum Engine tab.

Hybrid UI: form for input, live progress view showing agent planning
and generated artifacts, with HITL approval gates at each stage.

DeepAgents concepts visible in the UI:
- Agent TodoList displayed in the activity panel
- Artifact preview after each generation step
- HITL review at each checkpoint (approve, revise, reject)
- Step progress tracker matching the LangGraph workflow
"""

import streamlit as st
from adapters import curriculum_engine
from components import doc_viewer

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/07-curriculum-engine.md"

# Step display labels for progress tracker
_STEP_LABELS = {
    "plan_curriculum": "Plan",
    "review_plan": "Review Plan",
    "generate_lesson": "Lesson",
    "review_lesson": "Review Lesson",
    "generate_exercises": "Exercises",
    "review_exercises": "Review Exercises",
    "generate_assessment": "Assessment",
    "review_assessment": "Review Assessment",
    "assemble_module": "Assemble",
    "done": "Done",
}

# Ordered step keys for progress bar
_PROGRESS_STEPS = [
    "plan_curriculum", "review_plan",
    "generate_lesson", "review_lesson",
    "generate_exercises", "review_exercises",
    "generate_assessment", "review_assessment",
    "assemble_module", "done",
]


def _reset_state() -> None:
    """Clear all p7_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p7_"):
            del st.session_state[key]


def _init_state() -> None:
    """Initialize session state defaults."""
    if "p7_stage" not in st.session_state:
        st.session_state["p7_stage"] = "idle"
        st.session_state["p7_thread_id"] = None
        st.session_state["p7_interrupt"] = None
        st.session_state["p7_log"] = []


def _add_log(label: str, content: str) -> None:
    """Append an entry to the pipeline log."""
    st.session_state["p7_log"].append((label, content))


def _render_progress_bar(current_step: str) -> None:
    """Render a horizontal step progress indicator."""
    if current_step not in _PROGRESS_STEPS:
        return

    current_idx = _PROGRESS_STEPS.index(current_step)
    total = len(_PROGRESS_STEPS)

    cols = st.columns(total)
    for i, (col, step_key) in enumerate(zip(cols, _PROGRESS_STEPS)):
        label = _STEP_LABELS.get(step_key, step_key)
        with col:
            if i < current_idx:
                st.markdown(f"~~{label}~~")
            elif i == current_idx:
                st.markdown(f"**:blue[{label}]**")
            else:
                st.markdown(f":gray[{label}]")


def _render_request_form() -> None:
    """Render the curriculum request form (idle stage)."""
    st.subheader("Curriculum Request")

    samples = curriculum_engine.get_sample_requests()
    sample_labels = ["(custom)"] + [
        f"{s['topic']} ({s['level']})"
        for s in samples
    ]

    selected = st.selectbox("Quick-start with a sample:", sample_labels, key="p7_sample")

    if selected != "(custom)":
        idx = sample_labels.index(selected) - 1
        sample = samples[idx]
        topic = sample["topic"]
        level = sample["level"]
        preferences = sample.get("preferences", {})
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Topic:** {topic}")
        with col2:
            st.info(f"**Level:** {level}")
        if preferences:
            st.info(f"**Preferences:** {', '.join(f'{k}: {v}' for k, v in preferences.items())}")
        can_generate = True
    else:
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Topic", key="p7_topic",
                                  placeholder="e.g., Business English for meetings")
        with col2:
            level = st.selectbox("CEFR Level", curriculum_engine.get_cefr_levels(), key="p7_level")

        teaching_style = st.selectbox(
            "Teaching Style (optional)",
            ["", "conversational", "formal", "interactive"],
            key="p7_style",
        )
        focus_input = st.text_input(
            "Focus Areas (optional, comma-separated)",
            key="p7_focus",
            placeholder="e.g., vocabulary, speaking",
        )
        preferences = {}
        if teaching_style:
            preferences["teaching_style"] = teaching_style
        if focus_input:
            preferences["focus_areas"] = [f.strip() for f in focus_input.split(",") if f.strip()]
        can_generate = bool(topic)

    if st.button("Generate Module", key="p7_generate", type="primary", disabled=not can_generate):
        request = {
            "topic": topic,
            "level": level,
            "preferences": preferences,
        }
        st.session_state["p7_thread_id"] = curriculum_engine.create_thread_id()
        _add_log("Request", f"**{topic}** at **{level}** level")

        with st.spinner("Planning curriculum..."):
            try:
                interrupt_val = curriculum_engine.start_pipeline(
                    st.session_state["p7_thread_id"], request
                )
                if interrupt_val:
                    st.session_state["p7_stage"] = interrupt_val.get("step", "review_plan")
                    st.session_state["p7_interrupt"] = interrupt_val
                    _add_log("Plan Created", "Awaiting review")
                else:
                    st.session_state["p7_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_plan_review() -> None:
    """Render the plan review interface."""
    st.subheader("Review Curriculum Plan")
    interrupt_val = st.session_state["p7_interrupt"]
    plan = interrupt_val.get("plan", {})

    st.markdown(f"### {plan.get('title', 'Curriculum Plan')}")
    st.markdown(plan.get("description", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Lesson Outline:**")
        st.markdown(plan.get("lesson_outline", ""))
    with col2:
        st.markdown("**Exercise Types:**")
        for ex_type in plan.get("exercise_types", []):
            st.markdown(f"- {ex_type}")

    st.markdown("**Assessment Approach:**")
    st.markdown(plan.get("assessment_approach", ""))

    st.divider()
    _render_review_controls("plan")


def _render_artifact_review(step: str) -> None:
    """Render the review interface for a generated artifact (lesson/exercises/assessment)."""
    artifact_name = step.replace("review_", "")
    st.subheader(f"Review {artifact_name.title()}")
    interrupt_val = st.session_state["p7_interrupt"]
    artifact = interrupt_val.get("artifact", {})

    content = artifact.get("content", "No content generated")
    st.markdown(content)

    todos = artifact.get("agent_todos", [])
    if todos:
        with st.expander("Agent Planning (TodoList)", expanded=False):
            for todo in todos:
                status = todo.get("status", "pending")
                st.markdown(f"- [{status}] {todo.get('content', '')}")

    st.divider()
    _render_review_controls(artifact_name)


def _render_review_controls(artifact_name: str) -> None:
    """Render approve/revise/reject buttons for a review step."""
    st.markdown("**Moderator Decision:**")

    action = st.radio(
        "Action",
        ["approve", "revise", "reject"] if artifact_name != "plan" else ["approve", "revise"],
        format_func=lambda x: {
            "approve": "Approve",
            "revise": "Request Revision",
            "reject": "Reject (skip this artifact)",
        }.get(x, x),
        horizontal=True,
        key=f"p7_{artifact_name}_action",
    )

    feedback = ""
    if action == "revise":
        feedback = st.text_area(
            "Feedback for revision:",
            placeholder="Explain what needs to change...",
            key=f"p7_{artifact_name}_feedback",
        )

    if st.button("Submit Decision", key=f"p7_{artifact_name}_submit", type="primary"):
        decision = {"action": action}
        if action == "revise":
            decision["feedback"] = feedback

        action_label = {"approve": "Approved", "revise": "Revision requested", "reject": "Rejected"}[action]
        _add_log(f"Review {artifact_name.title()}", f"{action_label}" + (f" — {feedback}" if feedback else ""))

        spinner_text = "Regenerating..." if action == "revise" else "Processing..."
        with st.spinner(spinner_text):
            try:
                interrupt_val = curriculum_engine.resume_pipeline(
                    st.session_state["p7_thread_id"], decision
                )
                if interrupt_val:
                    st.session_state["p7_stage"] = interrupt_val.get("step", "done")
                    st.session_state["p7_interrupt"] = interrupt_val
                else:
                    st.session_state["p7_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_done() -> None:
    """Render the completed module."""
    st.subheader("Module Complete")

    state = curriculum_engine.get_state(st.session_state["p7_thread_id"])
    assembled = state.get("assembled_module", "")

    if assembled:
        st.success("Curriculum module assembled successfully!")
        st.markdown(assembled)
    else:
        st.warning("No module was assembled.")


def render() -> None:
    """Render the Intelligent Curriculum Engine interface."""
    st.header("Intelligent Curriculum Engine")
    st.caption("DeepAgents-powered curriculum module generator with HITL approval")

    _init_state()

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset", key="p7_reset"):
            _reset_state()
            st.rerun()
    with col2:
        stage = st.session_state.get("p7_stage", "idle")
        stage_label = _STEP_LABELS.get(stage, stage.replace("_", " ").title())
        st.caption(f"Stage: **{stage_label}**")

    if stage != "idle":
        _render_progress_bar(stage)

    st.divider()

    log = st.session_state.get("p7_log", [])
    if log:
        with st.expander("Pipeline Log", expanded=False):
            for label, content in log:
                st.markdown(f"**{label}:** {content}")

    if stage == "idle":
        _render_request_form()
    elif stage == "review_plan":
        _render_plan_review()
    elif stage in ("review_lesson", "review_exercises", "review_assessment"):
        _render_artifact_review(stage)
    elif stage == "done":
        _render_done()
    else:
        st.info(f"Generating content ({stage})...")

    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: DeepAgents & Autonomous Agents")
