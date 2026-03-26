"""P1 — Grammar Correction Agent tab.

Chat-based interface for grammar analysis with follow-up conversation.
"""

import streamlit as st
from adapters import grammar_agent
from components import doc_viewer

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/01-grammar-correction-agent.md"


def _format_feedback_as_markdown(feedback) -> str:
    """Convert GrammarFeedback into a readable markdown string."""
    lines = []

    # CEFR level and summary
    p = feedback.proficiency
    lines.append(f"### 📊 Proficiency: {p.cefr_level}")
    lines.append(p.summary)
    lines.append("")

    # Strengths
    if p.strengths:
        lines.append("**Strengths:**")
        for s in p.strengths:
            lines.append(f"- {s}")
        lines.append("")

    # Areas to improve
    if p.areas_to_improve:
        lines.append("**Areas to improve:**")
        for a in p.areas_to_improve:
            lines.append(f"- {a}")
        lines.append("")

    # Grammar issues
    lines.append(f"### ✏️ Grammar Issues ({len(feedback.issues)} found)")
    for i, issue in enumerate(feedback.issues, 1):
        severity_icon = {"minor": "🟡", "moderate": "🟠", "major": "🔴"}.get(
            issue.severity, "⚪"
        )
        lines.append(
            f"\n**{i}. {severity_icon} {issue.error_category}** ({issue.severity})"
        )
        lines.append(f'- Original: *"{issue.original_text}"*')
        lines.append(f'- Corrected: *"{issue.corrected_text}"*')
        lines.append(f"- {issue.explanation}")

    # Corrected text
    lines.append("\n### ✅ Corrected Text")
    lines.append(feedback.corrected_full_text)

    return "\n".join(lines)


def _handle_message(user_input: str) -> str:
    """Process a user message — either analyze new text or handle follow-up."""
    # If no analysis has been done yet, treat the input as text to analyze
    if st.session_state.get("p1_feedback") is None:
        try:
            feedback = grammar_agent.run_analysis(user_input)
            st.session_state["p1_feedback"] = feedback
            st.session_state["p1_original_text"] = user_input
            st.session_state["p1_handler"] = grammar_agent.create_conversation(
                user_input, feedback
            )
            return _format_feedback_as_markdown(feedback)
        except RuntimeError as e:
            return f"❌ {e}"

    # Otherwise, treat as a follow-up question
    handler = st.session_state.get("p1_handler")
    if handler is None:
        return "Something went wrong — please reset and try again."
    try:
        return grammar_agent.ask_followup(handler, user_input)
    except RuntimeError as e:
        return f"❌ {e}"


def render() -> None:
    """Render the Grammar Correction Agent interface."""
    st.header("Grammar Correction Agent")
    st.caption("Analyze student writing for grammar issues and CEFR proficiency")

    # -- Sample text selector --
    samples = grammar_agent.get_sample_texts()
    sample_labels = ["(paste your own text below)"] + [s["label"] for s in samples]
    selected = st.selectbox(
        "Quick-start with a sample text:",
        sample_labels,
        key="p1_sample_select",
    )

    # If a sample was selected and it's different from last selection, auto-submit it
    if selected != "(paste your own text below)":
        sample_idx = sample_labels.index(selected) - 1
        sample_text = samples[sample_idx]["text"]
        if st.button("Analyze this sample", key="p1_analyze_sample"):
            # Reset state for new analysis
            st.session_state["p1_feedback"] = None
            st.session_state["p1_handler"] = None
            st.session_state["p1_chat_history"] = [
                {"role": "user", "content": sample_text}
            ]
            response = _handle_message(sample_text)
            st.session_state["p1_chat_history"].append(
                {"role": "assistant", "content": response}
            )
            st.rerun()

    # -- Reset button --
    if st.button("🔄 Reset", key="p1_reset"):
        for key in ["p1_chat_history", "p1_feedback", "p1_handler", "p1_original_text"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    # -- Chat interface --
    # Initialize history
    if "p1_chat_history" not in st.session_state:
        st.session_state["p1_chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! Paste a piece of student writing and I'll analyze it "
                    "for grammar issues and assess the CEFR proficiency level.\n\n"
                    "After the analysis, you can ask follow-up questions about any "
                    "of the issues found."
                ),
            }
        ]

    # Display messages
    for msg in st.session_state["p1_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Paste student writing or ask a follow-up question...", key="p1_chat"):
        st.session_state["p1_chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = _handle_message(user_input)
            st.markdown(response)
        st.session_state["p1_chat_history"].append({"role": "assistant", "content": response})

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: LangChain Fundamentals")
