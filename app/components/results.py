"""Reusable components for displaying structured results."""

import streamlit as st


def score_card(
    label: str,
    score: int,
    max_score: int = 5,
    feedback: str = "",
    evidence: list[str] | None = None,
) -> None:
    """Render a score card with a label, score bar, feedback, and evidence.

    Used by P3 assessment to display per-dimension criteria scores.
    """
    st.markdown(f"**{label}**")
    # Visual score indicator
    filled = "🟢" * score
    empty = "⚪" * (max_score - score)
    st.markdown(f"{filled}{empty} **{score}/{max_score}**")
    if feedback:
        st.markdown(feedback)
    if evidence:
        with st.expander("Evidence", expanded=False):
            for item in evidence:
                st.markdown(f"- _{item}_")
    st.divider()


def badge(label: str, value: str, color: str = "blue") -> None:
    """Render a colored badge with a label and value.

    Args:
        label: Small text above the value.
        value: The main display value (e.g., "B1").
        color: CSS color name for the badge background.
    """
    st.markdown(
        f'<div style="background-color:{color};color:white;padding:8px 16px;'
        f'border-radius:8px;display:inline-block;margin-bottom:8px;">'
        f'<small>{label}</small><br><strong style="font-size:1.4em;">{value}</strong>'
        f"</div>",
        unsafe_allow_html=True,
    )


def bullet_list(title: str, items: list[str]) -> None:
    """Render a titled bullet list."""
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")
