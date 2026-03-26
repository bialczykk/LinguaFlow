"""P2 — Lesson Plan Generator tab.

Two-phase interface: intake conversation, then plan generation.
"""

import streamlit as st
from adapters import lesson_planner
from components import doc_viewer

_DOC_PATH = "docs/02-lesson-plan-generator.md"


def _format_profile(profile) -> str:
    """Format a StudentProfile as readable markdown."""
    return (
        f"**Name:** {profile.name}  \n"
        f"**Level:** {profile.proficiency_level}  \n"
        f"**Lesson type:** {profile.lesson_type}  \n"
        f"**Goals:** {', '.join(profile.learning_goals)}  \n"
        f"**Topics:** {', '.join(profile.preferred_topics)}"
    )


def _format_plan(plan) -> str:
    """Format a LessonPlan as readable markdown."""
    lines = []
    lines.append(f"## {plan.title}")
    lines.append(f"**Level:** {plan.level} | **Type:** {plan.lesson_type} | "
                 f"**Duration:** {plan.estimated_duration_minutes} min")
    lines.append("")

    # Objectives
    lines.append("### 🎯 Objectives")
    for obj in plan.objectives:
        lines.append(f"- {obj}")
    lines.append("")

    # Warm-up
    lines.append("### 🔥 Warm-up")
    lines.append(plan.warm_up)
    lines.append("")

    # Main activities
    lines.append("### 📚 Main Activities")
    for act in plan.main_activities:
        lines.append(f"\n**{act.name}** ({act.duration_minutes} min)")
        lines.append(act.description)
        if act.materials:
            lines.append(f"*Materials: {', '.join(act.materials)}*")
    lines.append("")

    # Wrap-up
    lines.append("### 🏁 Wrap-up")
    lines.append(plan.wrap_up)
    lines.append("")

    # Homework
    lines.append("### 📝 Homework")
    lines.append(plan.homework)

    return "\n".join(lines)


def _handle_intake_message(user_input: str) -> str:
    """Process a message during the intake phase."""
    intake = st.session_state.get("p2_intake")
    if intake is None:
        return "Something went wrong — please reset and try again."
    try:
        response = lesson_planner.ask_intake(intake, user_input)
        # Check if intake is now complete
        if lesson_planner.is_intake_complete(intake):
            st.session_state["p2_intake_done"] = True
        return response
    except RuntimeError as e:
        return f"❌ {e}"


def render() -> None:
    """Render the Lesson Plan Generator interface."""
    st.header("Lesson Plan Generator")
    st.caption("Conversational intake followed by personalized lesson plan generation")

    # -- Sample profile selector (skip intake shortcut) --
    sample_labels = ["(use intake conversation)"] + [
        label for label, _ in lesson_planner.SAMPLE_PROFILES
    ]
    selected = st.selectbox(
        "Quick-start with a sample profile:",
        sample_labels,
        key="p2_sample_select",
    )

    if selected != "(use intake conversation)":
        idx = sample_labels.index(selected) - 1
        _, profile = lesson_planner.SAMPLE_PROFILES[idx]
        st.markdown(_format_profile(profile))
        if st.button("Generate lesson plan for this profile", key="p2_generate_sample"):
            st.session_state["p2_profile"] = profile
            st.session_state["p2_intake_done"] = True
            st.session_state["p2_skip_intake"] = True
            st.rerun()

    # -- Reset button --
    if st.button("🔄 Reset", key="p2_reset"):
        for key in [
            "p2_chat_history", "p2_intake", "p2_profile", "p2_plan",
            "p2_intake_done", "p2_skip_intake",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    # -- Phase 1: Intake conversation --
    if not st.session_state.get("p2_skip_intake", False):
        # Initialize intake
        if "p2_intake" not in st.session_state:
            st.session_state["p2_intake"] = lesson_planner.create_intake()
            st.session_state["p2_intake_done"] = False

        # Initialize chat history with a welcome message
        if "p2_chat_history" not in st.session_state:
            st.session_state["p2_chat_history"] = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi! I'm going to help create a personalized lesson plan. "
                        "Let's start — what's your name?"
                    ),
                }
            ]

        # Display messages
        for msg in st.session_state["p2_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input (only if intake not done)
        if not st.session_state.get("p2_intake_done", False):
            if user_input := st.chat_input("Answer the intake questions...", key="p2_chat"):
                st.session_state["p2_chat_history"].append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = _handle_intake_message(user_input)
                    st.markdown(response)
                st.session_state["p2_chat_history"].append(
                    {"role": "assistant", "content": response}
                )

    # -- Transition: Extract profile and offer generation --
    if st.session_state.get("p2_intake_done") and "p2_profile" not in st.session_state:
        intake = st.session_state.get("p2_intake")
        if intake is not None:
            try:
                profile = lesson_planner.extract_profile(intake)
                st.session_state["p2_profile"] = profile
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))

    # -- Phase 2: Profile review and plan generation --
    if "p2_profile" in st.session_state:
        st.divider()
        st.subheader("📋 Student Profile")
        st.markdown(_format_profile(st.session_state["p2_profile"]))

        if "p2_plan" not in st.session_state:
            if st.button("🚀 Generate Lesson Plan", key="p2_generate"):
                with st.spinner("Generating lesson plan (this may take a moment)..."):
                    try:
                        plan = lesson_planner.generate_plan(st.session_state["p2_profile"])
                        st.session_state["p2_plan"] = plan
                        st.rerun()
                    except RuntimeError as e:
                        st.error(str(e))

    # -- Display generated plan --
    if "p2_plan" in st.session_state:
        st.divider()
        st.markdown(_format_plan(st.session_state["p2_plan"]))

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: LangGraph StateGraph Fundamentals")
