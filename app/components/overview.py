"""Reusable overview toggle component.

Renders an expandable section at the top of each tab showing:
1. The business scenario the feature solves
2. A technical flowchart of the LangGraph/DeepAgents/LangSmith implementation
"""

import streamlit as st


def render(
    business_scenario: str,
    tech_flowchart: str,
    *,
    key_prefix: str,
) -> None:
    """Render an expandable overview toggle.

    Args:
        business_scenario: Markdown text explaining the business problem.
        tech_flowchart: Graphviz DOT string for the technical flowchart.
        key_prefix: Session state key prefix (e.g. "p1") to avoid widget ID collisions.
    """
    with st.expander("How it works", expanded=False, icon=":material/info:"):
        # Business scenario
        st.markdown("**Business Scenario**")
        st.markdown(business_scenario)

        st.markdown("")

        # Technical flowchart
        st.markdown("**Technical Architecture**")
        st.graphviz_chart(tech_flowchart, use_container_width=True)
