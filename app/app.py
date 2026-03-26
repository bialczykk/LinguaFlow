"""LinguaFlow Learning Lab — interactive interface for all learning projects."""

import streamlit as st

# -- Page configuration (must be first Streamlit call) --
st.set_page_config(
    page_title="LinguaFlow Learning Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Hide the default sidebar navigation (we use tabs instead) --
st.html("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
</style>
""")

# -- Import page modules --
from pages import p1_grammar, p2_lesson, p3_assessment  # noqa: E402


def main() -> None:
    """Render the app with one tab per project."""
    st.title("🎓 LinguaFlow Learning Lab")
    st.caption("Interactive interface for LangGraph ecosystem learning projects")

    # -- Tab bar --
    tab1, tab2, tab3 = st.tabs([
        "✏️ Grammar Agent",
        "📋 Lesson Planner",
        "📊 Assessment Pipeline",
    ])

    with tab1:
        p1_grammar.render()

    with tab2:
        p2_lesson.render()

    with tab3:
        p3_assessment.render()


if __name__ == "__main__":
    main()
