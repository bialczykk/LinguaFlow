"""P4 — Tutor Matching & Scheduling Agent tab.

Chat-based interface for finding and booking tutors through a
multi-phase conversational agent with tool calling.
"""

import streamlit as st
from adapters import tutor_matching
from components import doc_viewer, overview

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/04-tutor-matching-agent.md"

# Phase labels for the status indicator
_PHASE_LABELS = {
    "gather": "Gathering preferences",
    "search": "Searching tutors",
    "present": "Presenting options",
    "book": "Booking session",
    "done": "Session complete",
}


def _handle_message(user_input: str) -> str:
    """Send a user message to the tutor matching agent."""
    thread_id = st.session_state.get("p4_thread_id")
    is_first = st.session_state.get("p4_is_first_turn", True)

    try:
        response = tutor_matching.send_message(thread_id, user_input, is_first_turn=is_first)
        st.session_state["p4_is_first_turn"] = False
        return response
    except RuntimeError as e:
        return f"❌ {e}"


def render() -> None:
    """Render the Tutor Matching Agent interface."""
    st.header("Tutor Matching & Scheduling Agent")
    st.caption("Find and book the right English tutor through conversation")

    overview.render(
        business_scenario=(
            "Students describe what they want to learn and their scheduling constraints "
            "through natural conversation. The agent searches the tutor database, presents "
            "matching options, and handles booking \u2014 all in one chat flow. This replaces "
            "the manual process of browsing tutor profiles, checking availability, and "
            "coordinating schedules."
        ),
        tech_flowchart="""
            digraph {
                rankdir=LR
                node [shape=box style="rounded,filled" fillcolor="#f0f4ff" fontname="Helvetica" fontsize=11]
                edge [fontname="Helvetica" fontsize=10]

                user [label="Student\\nMessage" shape=note fillcolor="#fff3cd"]
                agent [label="ReAct Agent\\n(tool-calling LLM)"]
                tools [label="Tools" shape=record fillcolor="#d4edda"
                       label="{Tools|search_tutors|check_availability|book_session}"]
                checkpointer [label="SQLite\\nCheckpointer" shape=cylinder fillcolor="#d6eaf8"]
                phase [label="Phase\\nTracking" fillcolor="#ffeeba"]
                response [label="Agent\\nResponse" shape=note fillcolor="#fff3cd"]
                langsmith [label="LangSmith\\nTracing" shape=ellipse fillcolor="#e8daef"]

                user -> agent
                agent -> tools [label="bind_tools()"]
                tools -> agent [label="results"]
                agent -> checkpointer [dir=both label="persist\\nstate"]
                agent -> phase [label="gather > search\\n> present > book"]
                agent -> response
                agent -> langsmith [style=dashed]
            }
        """,
        key_prefix="p4",
    )

    # -- Initialize session state --
    if "p4_thread_id" not in st.session_state:
        st.session_state["p4_thread_id"] = tutor_matching.create_thread_id()
        st.session_state["p4_is_first_turn"] = True

    # -- Sample scenario selector --
    scenarios = tutor_matching.get_sample_scenarios()
    scenario_labels = ["(type your own message below)"] + [s["label"] for s in scenarios]
    selected = st.selectbox(
        "Quick-start with a sample scenario:",
        scenario_labels,
        key="p4_scenario_select",
    )

    if selected != "(type your own message below)":
        scenario_idx = scenario_labels.index(selected) - 1
        scenario_msg = scenarios[scenario_idx]["message"]
        if st.button("Start this scenario", key="p4_start_scenario"):
            # Reset state for new scenario
            st.session_state["p4_thread_id"] = tutor_matching.create_thread_id()
            st.session_state["p4_is_first_turn"] = True
            st.session_state["p4_chat_history"] = [
                {"role": "user", "content": scenario_msg}
            ]
            response = _handle_message(scenario_msg)
            st.session_state["p4_chat_history"].append(
                {"role": "assistant", "content": response}
            )
            st.rerun()

    # -- Reset button and phase indicator --
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reset", key="p4_reset"):
            for key in list(st.session_state.keys()):
                if key.startswith("p4_"):
                    del st.session_state[key]
            st.rerun()
    with col2:
        phase = tutor_matching.get_phase(st.session_state.get("p4_thread_id", ""))
        phase_label = _PHASE_LABELS.get(phase, phase)
        st.caption(f"Phase: **{phase_label}**")

    st.divider()

    # -- Chat interface --
    if "p4_chat_history" not in st.session_state:
        st.session_state["p4_chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to LinguaFlow Tutor Matching! I can help you find "
                    "and book the perfect English tutor.\n\n"
                    "Tell me what you're looking for — what skills do you want to "
                    "work on? Do you have a preferred timezone or schedule?"
                ),
            }
        ]

    # Fixed-height scrollable chat container
    chat_box = st.container(height=500)

    # Display messages inside the container
    with chat_box:
        for msg in st.session_state["p4_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input below the container
    if user_input := st.chat_input(
        "Describe what kind of tutor you're looking for...",
        key="p4_chat",
    ):
        st.session_state["p4_chat_history"].append({"role": "user", "content": user_input})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = _handle_message(user_input)
                st.markdown(response)
        st.session_state["p4_chat_history"].append({"role": "assistant", "content": response})

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: Tool Use & Persistence")
