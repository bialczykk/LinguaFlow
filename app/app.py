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
from pages import p1_grammar, p2_lesson, p3_assessment, p4_tutor, p5_moderation, p6_support, p7_curriculum, p8_operations  # noqa: E402


def main() -> None:
    """Render the app with one tab per project."""
    st.title("🎓 LinguaFlow Learning Lab")
    st.caption("Interactive interface for LangGraph ecosystem learning projects")

    # -- Tab bar --
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "✏️ Grammar Agent",
        "📋 Lesson Planner",
        "📊 Assessment Pipeline",
        "🤝 Tutor Matching",
        "🛡️ Content Moderation",
        "🎯 Support System",
        "🧠 Curriculum Engine",
        "🚀 Autonomous Ops",
    ])

    with tab1:
        p1_grammar.render()

    with tab2:
        p2_lesson.render()

    with tab3:
        p3_assessment.render()

    with tab4:
        p4_tutor.render()

    with tab5:
        p5_moderation.render()

    with tab6:
        p6_support.render()

    with tab7:
        p7_curriculum.render()

    with tab8:
        p8_operations.render()


if __name__ == "__main__":
    main()
