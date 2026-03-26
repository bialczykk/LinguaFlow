# LinguaFlow Learning Lab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app with tabs for projects 1-3, each with interactive UI and collapsible documentation with sidebar TOC.

**Architecture:** Streamlit app in `app/` at repo root. Thin adapter modules bridge to project code via `sys.path` manipulation. Reusable components for chat, doc viewing, and result display. Each project tab is a separate page module.

**Tech Stack:** Streamlit, Python, python-dotenv. All project dependencies already in shared `.venv`.

**Spec:** `docs/superpowers/specs/2026-03-26-linguaflow-app-design.md`

---

## File Structure

```
app/
├── app.py                  # Entry point — page config, tab routing
├── requirements.txt        # streamlit, python-dotenv
├── adapters/
│   ├── __init__.py
│   ├── grammar_agent.py    # P1 wrapper
│   ├── lesson_planner.py   # P2 wrapper
│   └── assessment.py       # P3 wrapper
├── components/
│   ├── __init__.py
│   ├── chat.py             # Chat UI component
│   ├── doc_viewer.py       # Markdown + sidebar TOC component
│   └── results.py          # Structured result cards
└── pages/
    ├── __init__.py
    ├── p1_grammar.py       # Grammar Correction Agent tab
    ├── p2_lesson.py        # Lesson Plan Generator tab
    └── p3_assessment.py    # Student Assessment Pipeline tab
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `app/requirements.txt`
- Create: `app/app.py`
- Create: `app/adapters/__init__.py`
- Create: `app/components/__init__.py`
- Create: `app/pages/__init__.py`

- [ ] **Step 1: Create directory structure and requirements.txt**

```bash
mkdir -p app/adapters app/components app/pages
```

Write `app/requirements.txt`:
```
streamlit
python-dotenv
```

- [ ] **Step 2: Create empty `__init__.py` files**

Write empty files:
- `app/adapters/__init__.py`
- `app/components/__init__.py`
- `app/pages/__init__.py`

- [ ] **Step 3: Create the Streamlit entry point**

Write `app/app.py`:
```python
"""LinguaFlow Learning Lab — interactive interface for all learning projects."""

import streamlit as st

# -- Page configuration (must be first Streamlit call) --
st.set_page_config(
    page_title="LinguaFlow Learning Lab",
    page_icon="🎓",
    layout="wide",
)

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
```

- [ ] **Step 4: Create placeholder page modules**

Write `app/pages/p1_grammar.py`:
```python
"""P1 — Grammar Correction Agent tab."""

import streamlit as st


def render() -> None:
    """Render the Grammar Correction Agent interface."""
    st.header("Grammar Correction Agent")
    st.info("Coming soon — this tab will provide grammar analysis and follow-up chat.")
```

Write `app/pages/p2_lesson.py`:
```python
"""P2 — Lesson Plan Generator tab."""

import streamlit as st


def render() -> None:
    """Render the Lesson Plan Generator interface."""
    st.header("Lesson Plan Generator")
    st.info("Coming soon — this tab will provide intake conversation and lesson plan generation.")
```

Write `app/pages/p3_assessment.py`:
```python
"""P3 — Student Assessment Pipeline tab."""

import streamlit as st


def render() -> None:
    """Render the Student Assessment Pipeline interface."""
    st.header("Student Assessment Pipeline")
    st.info("Coming soon — this tab will provide CEFR writing assessment.")
```

- [ ] **Step 5: Verify the app runs**

Run: `cd app && streamlit run app.py --server.headless true`

Expected: App launches with three tabs showing placeholder messages. Kill the server after verifying.

- [ ] **Step 6: Commit**

```bash
git add app/
git commit -m "feat(app): scaffold Streamlit app with tab routing and placeholder pages"
```

---

### Task 2: Documentation Viewer Component

**Files:**
- Create: `app/components/doc_viewer.py`

- [ ] **Step 1: Write the doc viewer component**

Write `app/components/doc_viewer.py`:
```python
"""Markdown documentation viewer with sidebar table of contents."""

import re
from pathlib import Path

import streamlit as st


def _parse_headings(markdown_text: str) -> list[dict]:
    """Extract ## and ### headings from markdown for the TOC.

    Returns a list of dicts with keys: level (2 or 3), text, anchor.
    """
    headings = []
    for match in re.finditer(r"^(#{2,3})\s+(.+)$", markdown_text, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        # Build a URL-friendly anchor from the heading text
        anchor = re.sub(r"[^\w\s-]", "", text.lower())
        anchor = re.sub(r"[\s]+", "-", anchor)
        headings.append({"level": level, "text": text, "anchor": anchor})
    return headings


def _inject_anchors(markdown_text: str) -> str:
    """Inject HTML anchor tags before each ## and ### heading.

    Streamlit's st.markdown doesn't auto-generate IDs for headings,
    so we insert <a> tags manually to enable TOC linking.
    """

    def _replace_heading(match: re.Match) -> str:
        hashes = match.group(1)
        text = match.group(2).strip()
        anchor = re.sub(r"[^\w\s-]", "", text.lower())
        anchor = re.sub(r"[\s]+", "-", anchor)
        return f'<a id="{anchor}"></a>\n\n{hashes} {text}'

    return re.sub(r"^(#{2,3})\s+(.+)$", _replace_heading, markdown_text, flags=re.MULTILINE)


def render(doc_path: str, title: str = "Documentation") -> None:
    """Render a markdown doc inside an expander with a sidebar TOC.

    Args:
        doc_path: Path to the markdown file (relative to repo root or absolute).
        title: Label shown on the expander.
    """
    path = Path(doc_path)
    if not path.exists():
        st.warning(f"Documentation not found: {doc_path}")
        return

    markdown_text = path.read_text(encoding="utf-8")
    headings = _parse_headings(markdown_text)
    enriched_markdown = _inject_anchors(markdown_text)

    with st.expander(f"📚 {title}", expanded=False):
        toc_col, doc_col = st.columns([1, 3], gap="medium")

        # -- Left column: Table of Contents --
        with toc_col:
            st.markdown("#### Contents")
            for h in headings:
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;" if h["level"] == 3 else ""
                st.markdown(
                    f'{indent}<a href="#{h["anchor"]}" target="_self">{h["text"]}</a>',
                    unsafe_allow_html=True,
                )

        # -- Right column: Full document --
        with doc_col:
            st.markdown(enriched_markdown, unsafe_allow_html=True)
```

- [ ] **Step 2: Smoke-test with P1 docs**

Temporarily edit `app/pages/p1_grammar.py` to add at the end of `render()`:
```python
from components import doc_viewer
doc_viewer.render(
    "projects/01-grammar-correction-agent/../../docs/01-grammar-correction-agent.md",
    title="Documentation: LangChain Fundamentals",
)
```

Run: `cd app && streamlit run app.py --server.headless true`

Expected: P1 tab shows a collapsible "Documentation: LangChain Fundamentals" expander. Expanding it shows the TOC on the left, full doc on the right. Kill server after verifying, then revert the smoke-test edit.

- [ ] **Step 3: Commit**

```bash
git add app/components/doc_viewer.py
git commit -m "feat(app): add documentation viewer component with sidebar TOC"
```

---

### Task 3: Chat UI Component

**Files:**
- Create: `app/components/chat.py`

- [ ] **Step 1: Write the chat component**

Write `app/components/chat.py`:
```python
"""Reusable chat interface component for conversational projects."""

from typing import Callable

import streamlit as st


def render(
    *,
    history_key: str,
    on_user_message: Callable[[str], str],
    placeholder: str = "Type a message...",
    intro_message: str | None = None,
) -> None:
    """Render a chat interface with message history.

    Args:
        history_key: Session state key for this chat's message history.
            The history is a list of dicts: {"role": "user"|"assistant", "content": str}
        on_user_message: Callback that takes the user's message string and returns
            the assistant's response string. Called when the user submits a message.
        placeholder: Placeholder text for the chat input.
        intro_message: Optional welcome message shown as the first assistant message.
    """
    # -- Initialize history if needed --
    if history_key not in st.session_state:
        st.session_state[history_key] = []
        if intro_message:
            st.session_state[history_key].append(
                {"role": "assistant", "content": intro_message}
            )

    # -- Display existing messages --
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -- Chat input --
    if user_input := st.chat_input(placeholder):
        # Show user message immediately
        st.session_state[history_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = on_user_message(user_input)
            st.markdown(response)
        st.session_state[history_key].append({"role": "assistant", "content": response})
```

- [ ] **Step 2: Commit**

```bash
git add app/components/chat.py
git commit -m "feat(app): add reusable chat UI component"
```

---

### Task 4: Results Display Component

**Files:**
- Create: `app/components/results.py`

- [ ] **Step 1: Write the results component**

Write `app/components/results.py`:
```python
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
```

- [ ] **Step 2: Commit**

```bash
git add app/components/results.py
git commit -m "feat(app): add results display components (score cards, badges, bullet lists)"
```

---

### Task 5: P1 Adapter — Grammar Correction Agent

**Files:**
- Create: `app/adapters/grammar_agent.py`

- [ ] **Step 1: Write the P1 adapter**

Write `app/adapters/grammar_agent.py`:
```python
"""Adapter for Project 01 — Grammar Correction Agent.

Handles sys.path setup, environment loading, and wraps project functions
with error handling for use in the Streamlit app.
"""

import sys
from pathlib import Path

# -- Path setup: add P1 project directory to sys.path so its modules can be imported --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P1_DIR = _REPO_ROOT / "projects" / "01-grammar-correction-agent"
if str(_P1_DIR) not in sys.path:
    sys.path.insert(0, str(_P1_DIR))

# -- Load environment variables from repo root .env --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Import project modules (after path setup) --
from chains import analyze_grammar  # noqa: E402
from conversation import ConversationHandler  # noqa: E402
from data.sample_texts import SAMPLE_TEXTS  # noqa: E402
from models import GrammarFeedback  # noqa: E402


def get_sample_texts() -> list[dict[str, str]]:
    """Return the list of sample texts available for testing.

    Each dict has keys: "label" (display name) and "text" (student writing).
    """
    return SAMPLE_TEXTS


def run_analysis(student_text: str) -> GrammarFeedback:
    """Analyze student writing and return structured grammar feedback.

    Args:
        student_text: The student's writing to analyze.

    Returns:
        GrammarFeedback with issues, proficiency assessment, and corrected text.

    Raises:
        RuntimeError: If the analysis fails.
    """
    try:
        return analyze_grammar(student_text)
    except Exception as e:
        raise RuntimeError(f"Grammar analysis failed: {e}") from e


def create_conversation(
    original_text: str, feedback: GrammarFeedback
) -> ConversationHandler:
    """Create a new conversation handler for follow-up questions.

    Args:
        original_text: The student's original writing.
        feedback: The GrammarFeedback from run_analysis().

    Returns:
        A ConversationHandler ready to accept .ask() calls.
    """
    return ConversationHandler(original_text=original_text, feedback=feedback)


def ask_followup(handler: ConversationHandler, message: str) -> str:
    """Send a follow-up question and get the tutor's response.

    Args:
        handler: An active ConversationHandler.
        message: The user's follow-up question.

    Returns:
        The tutor's response string.

    Raises:
        RuntimeError: If the conversation call fails.
    """
    try:
        return handler.ask(message)
    except Exception as e:
        raise RuntimeError(f"Conversation failed: {e}") from e
```

- [ ] **Step 2: Verify imports work**

Run: `cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning" && python -c "from app.adapters.grammar_agent import get_sample_texts; print(len(get_sample_texts()), 'samples loaded')"`

Expected: `4 samples loaded`

- [ ] **Step 3: Commit**

```bash
git add app/adapters/grammar_agent.py
git commit -m "feat(app): add P1 grammar agent adapter"
```

---

### Task 6: P1 Page — Grammar Correction Agent Tab

**Files:**
- Modify: `app/pages/p1_grammar.py`

- [ ] **Step 1: Write the full P1 page**

Replace `app/pages/p1_grammar.py` with:
```python
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
```

- [ ] **Step 2: Verify P1 tab works**

Run: `cd app && streamlit run app.py --server.headless true`

Expected: P1 tab shows sample selector, chat with welcome message, and collapsible documentation. Kill server after verifying.

- [ ] **Step 3: Commit**

```bash
git add app/pages/p1_grammar.py
git commit -m "feat(app): implement P1 grammar agent tab with chat UI and docs"
```

---

### Task 7: P2 Adapter — Lesson Plan Generator

**Files:**
- Create: `app/adapters/lesson_planner.py`

- [ ] **Step 1: Write the P2 adapter**

Write `app/adapters/lesson_planner.py`:
```python
"""Adapter for Project 02 — Lesson Plan Generator.

Handles sys.path setup, environment loading, and wraps project functions
with error handling for use in the Streamlit app.
"""

import sys
from pathlib import Path

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P2_DIR = _REPO_ROOT / "projects" / "02-lesson-plan-generator"
if str(_P2_DIR) not in sys.path:
    sys.path.insert(0, str(_P2_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Import project modules --
from intake import IntakeConversation  # noqa: E402
from graph import build_graph  # noqa: E402
from models import LessonPlan, StudentProfile  # noqa: E402
from data.sample_profiles import (  # noqa: E402
    BEGINNER_CONVERSATION,
    INTERMEDIATE_CONVERSATION,
    BEGINNER_GRAMMAR,
    INTERMEDIATE_GRAMMAR,
    EXAM_PREP_INTERMEDIATE,
    EXAM_PREP_ADVANCED,
)

# Pre-built list of sample profiles for the UI
SAMPLE_PROFILES: list[tuple[str, StudentProfile]] = [
    ("Yuki — A2 Conversation", BEGINNER_CONVERSATION),
    ("Carlos — B1 Conversation", INTERMEDIATE_CONVERSATION),
    ("Fatima — A1 Grammar", BEGINNER_GRAMMAR),
    ("Hans — B2 Grammar", INTERMEDIATE_GRAMMAR),
    ("Mei — B2 Exam Prep", EXAM_PREP_INTERMEDIATE),
    ("Olga — C1 Exam Prep", EXAM_PREP_ADVANCED),
]


def create_intake() -> IntakeConversation:
    """Create a new intake conversation instance."""
    return IntakeConversation()


def ask_intake(intake: IntakeConversation, message: str) -> str:
    """Send a message in the intake conversation.

    Returns:
        The intake assistant's response.

    Raises:
        RuntimeError: If the intake call fails.
    """
    try:
        return intake.ask(message)
    except Exception as e:
        raise RuntimeError(f"Intake conversation failed: {e}") from e


def is_intake_complete(intake: IntakeConversation) -> bool:
    """Check whether the intake has gathered enough information."""
    return intake.is_complete()


def extract_profile(intake: IntakeConversation) -> StudentProfile:
    """Extract the student profile from a completed intake.

    Raises:
        RuntimeError: If extraction fails.
    """
    try:
        return intake.get_profile()
    except Exception as e:
        raise RuntimeError(f"Profile extraction failed: {e}") from e


def generate_plan(profile: StudentProfile) -> LessonPlan:
    """Run the lesson plan generation graph for a student profile.

    Args:
        profile: A complete StudentProfile.

    Returns:
        The generated LessonPlan.

    Raises:
        RuntimeError: If graph execution fails.
    """
    try:
        graph = build_graph()
        initial_state = {
            "student_profile": profile,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        }
        result = graph.invoke(initial_state, config={"tags": ["p2-lesson-plan-generator"]})
        return result["final_plan"]
    except Exception as e:
        raise RuntimeError(f"Lesson plan generation failed: {e}") from e
```

- [ ] **Step 2: Verify imports work**

Run: `cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning" && python -c "from app.adapters.lesson_planner import SAMPLE_PROFILES; print(len(SAMPLE_PROFILES), 'profiles loaded')"`

Expected: `6 profiles loaded`

- [ ] **Step 3: Commit**

```bash
git add app/adapters/lesson_planner.py
git commit -m "feat(app): add P2 lesson planner adapter"
```

---

### Task 8: P2 Page — Lesson Plan Generator Tab

**Files:**
- Modify: `app/pages/p2_lesson.py`

- [ ] **Step 1: Write the full P2 page**

Replace `app/pages/p2_lesson.py` with:
```python
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
```

- [ ] **Step 2: Verify P2 tab renders**

Run: `cd app && streamlit run app.py --server.headless true`

Expected: P2 tab shows sample profile selector, intake chat with welcome, and collapsible docs. Kill server after verifying.

- [ ] **Step 3: Commit**

```bash
git add app/pages/p2_lesson.py
git commit -m "feat(app): implement P2 lesson planner tab with intake chat and plan display"
```

---

### Task 9: P3 Adapter — Student Assessment Pipeline

**Files:**
- Create: `app/adapters/assessment.py`

- [ ] **Step 1: Write the P3 adapter**

Write `app/adapters/assessment.py`:
```python
"""Adapter for Project 03 — Student Assessment Pipeline.

Handles sys.path setup, environment loading, vector store caching,
and wraps project functions with error handling.
"""

import sys
from pathlib import Path

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P3_DIR = _REPO_ROOT / "projects" / "03-student-assessment-pipeline"
if str(_P3_DIR) not in sys.path:
    sys.path.insert(0, str(_P3_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Import project modules --
from ingestion import build_vector_store, get_vector_store, DEFAULT_PERSIST_DIR  # noqa: E402
from graph import build_graph  # noqa: E402
from models import Assessment  # noqa: E402
from data.sample_submissions import ALL_SUBMISSIONS  # noqa: E402

# -- Vector store persist directory (relative to P3 project dir) --
_PERSIST_DIR = str(_P3_DIR / "chroma_db")


def get_sample_submissions() -> list[dict[str, str]]:
    """Return sample submissions for the UI.

    Each dict has keys: submission_text, submission_context, student_level_hint.
    """
    return ALL_SUBMISSIONS


def ensure_vector_store():
    """Get or build the vector store, caching the result.

    Returns the Chroma vector store instance. Builds it on first call
    if the persisted store doesn't exist.

    Raises:
        RuntimeError: If vector store setup fails.
    """
    try:
        # Try loading existing store first
        import os
        if os.path.exists(_PERSIST_DIR) and os.listdir(_PERSIST_DIR):
            return get_vector_store(persist_directory=_PERSIST_DIR)
        else:
            # Need to build — change to P3 dir so relative paths in ingestion.py work
            original_dir = os.getcwd()
            os.chdir(str(_P3_DIR))
            try:
                return build_vector_store(persist_directory=_PERSIST_DIR)
            finally:
                os.chdir(original_dir)
    except Exception as e:
        raise RuntimeError(f"Vector store setup failed: {e}") from e


def run_assessment(
    submission_text: str,
    submission_context: str,
    student_level_hint: str = "",
) -> Assessment:
    """Run the full assessment pipeline on a student submission.

    Args:
        submission_text: The student's writing.
        submission_context: What they were asked to write.
        student_level_hint: Optional self-reported CEFR level.

    Returns:
        A complete Assessment object.

    Raises:
        RuntimeError: If the assessment pipeline fails.
    """
    try:
        vector_store = ensure_vector_store()
        graph = build_graph(vector_store)
        result = graph.invoke(
            {
                "submission_text": submission_text,
                "submission_context": submission_context,
                "student_level_hint": student_level_hint,
            },
            config={"tags": ["p3-student-assessment"]},
        )
        return result["final_assessment"]
    except Exception as e:
        raise RuntimeError(f"Assessment pipeline failed: {e}") from e
```

- [ ] **Step 2: Verify imports work**

Run: `cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning" && python -c "from app.adapters.assessment import get_sample_submissions; print(len(get_sample_submissions()), 'submissions loaded')"`

Expected: `3 submissions loaded`

- [ ] **Step 3: Commit**

```bash
git add app/adapters/assessment.py
git commit -m "feat(app): add P3 student assessment adapter"
```

---

### Task 10: P3 Page — Student Assessment Pipeline Tab

**Files:**
- Modify: `app/pages/p3_assessment.py`

- [ ] **Step 1: Write the full P3 page**

Replace `app/pages/p3_assessment.py` with:
```python
"""P3 — Student Assessment Pipeline tab.

Form-based interface for submitting writing and viewing CEFR assessment results.
"""

import streamlit as st
from adapters import assessment
from components import doc_viewer, results

_DOC_PATH = "docs/03-student-assessment-pipeline.md"

# CEFR level colors for the badge
_LEVEL_COLORS = {
    "A1": "#e74c3c",
    "A2": "#e67e22",
    "B1": "#f1c40f",
    "B2": "#2ecc71",
    "C1": "#3498db",
    "C2": "#9b59b6",
}


def render() -> None:
    """Render the Student Assessment Pipeline interface."""
    st.header("Student Assessment Pipeline")
    st.caption("Submit student writing for comprehensive CEFR-level assessment")

    # -- Sample submission selector --
    samples = assessment.get_sample_submissions()
    sample_labels = ["(enter your own text below)"] + [
        f"Sample {i + 1}: {s['submission_context'][:60]}..."
        for i, s in enumerate(samples)
    ]
    selected_idx = st.selectbox(
        "Quick-start with a sample submission:",
        range(len(sample_labels)),
        format_func=lambda i: sample_labels[i],
        key="p3_sample_select",
    )

    # -- Input form --
    if selected_idx > 0:
        sample = samples[selected_idx - 1]
        default_text = sample["submission_text"]
        default_context = sample["submission_context"]
        default_hint = sample["student_level_hint"]
    else:
        default_text = ""
        default_context = ""
        default_hint = ""

    submission_text = st.text_area(
        "Student writing:",
        value=default_text,
        height=200,
        key="p3_text_input",
        placeholder="Paste the student's writing here...",
    )

    col1, col2 = st.columns(2)
    with col1:
        submission_context = st.text_input(
            "Submission context (what they were asked to write):",
            value=default_context,
            key="p3_context_input",
        )
    with col2:
        level_options = ["", "A1", "A2", "B1", "B2", "C1", "C2"]
        hint_idx = level_options.index(default_hint) if default_hint in level_options else 0
        student_level_hint = st.selectbox(
            "Self-reported CEFR level (optional):",
            level_options,
            index=hint_idx,
            format_func=lambda x: "(none)" if x == "" else x,
            key="p3_hint_select",
        )

    # -- Action buttons --
    col_assess, col_reset = st.columns([1, 5])
    with col_assess:
        assess_clicked = st.button("📊 Assess", key="p3_assess", type="primary")
    with col_reset:
        if st.button("🔄 Reset", key="p3_reset"):
            st.session_state.pop("p3_result", None)
            st.rerun()

    # -- Run assessment --
    if assess_clicked:
        if not submission_text.strip():
            st.warning("Please enter some student writing to assess.")
        else:
            with st.spinner("Running assessment pipeline (this may take a moment)..."):
                try:
                    result = assessment.run_assessment(
                        submission_text=submission_text,
                        submission_context=submission_context,
                        student_level_hint=student_level_hint,
                    )
                    st.session_state["p3_result"] = result
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))

    # -- Display results --
    if "p3_result" in st.session_state:
        result = st.session_state["p3_result"]
        st.divider()

        # Overall level badge
        color = _LEVEL_COLORS.get(result.overall_level, "#95a5a6")
        results.badge("Overall CEFR Level", result.overall_level, color=color)

        if hasattr(result, "confidence"):
            st.caption(f"Confidence: {result.confidence}")

        st.markdown("")

        # Criteria scores
        st.subheader("📏 Criteria Scores")
        score_cols = st.columns(2)
        for i, criterion in enumerate(result.criteria_scores):
            with score_cols[i % 2]:
                results.score_card(
                    label=criterion.dimension,
                    score=criterion.score,
                    feedback=criterion.feedback,
                    evidence=criterion.evidence,
                )

        # Comparative summary
        if hasattr(result, "comparative_summary") and result.comparative_summary:
            st.subheader("📊 Comparative Analysis")
            st.markdown(result.comparative_summary)

        # Strengths, areas to improve, recommendations
        col_left, col_right = st.columns(2)
        with col_left:
            if result.strengths:
                results.bullet_list("💪 Strengths", result.strengths)
        with col_right:
            if result.areas_to_improve:
                results.bullet_list("🎯 Areas to Improve", result.areas_to_improve)

        if result.recommendations:
            st.markdown("")
            results.bullet_list("📋 Recommendations", result.recommendations)

    # -- Documentation --
    st.divider()
    doc_viewer.render(
        _DOC_PATH,
        title="Documentation: RAG, Vector Stores & Phased Retrieval",
    )
```

- [ ] **Step 2: Verify P3 tab renders**

Run: `cd app && streamlit run app.py --server.headless true`

Expected: P3 tab shows sample selector, text area, context input, CEFR dropdown, Assess/Reset buttons, and collapsible docs. Kill server after verifying.

- [ ] **Step 3: Commit**

```bash
git add app/pages/p3_assessment.py
git commit -m "feat(app): implement P3 assessment tab with form UI and result display"
```

---

### Task 11: Integration & Polish

**Files:**
- Modify: `app/app.py` (doc path resolution)
- Modify: `app/components/doc_viewer.py` (path resolution)
- Create or modify: `.gitignore` (add `.superpowers/`)

- [ ] **Step 1: Fix doc path resolution for running from `app/` directory**

The doc viewer needs to resolve paths relative to the repo root, not the CWD. Update `app/components/doc_viewer.py` — change the `render()` function's path resolution:

Replace the line:
```python
    path = Path(doc_path)
```
with:
```python
    # Resolve relative to repo root, not CWD
    path = Path(doc_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
```

- [ ] **Step 2: Add `.superpowers/` to `.gitignore`**

Append to `.gitignore`:
```
# Brainstorming visual companion
.superpowers/
```

- [ ] **Step 3: Full integration test**

Run: `cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning/app" && streamlit run app.py --server.headless true`

Verify all three tabs:
- P1: Sample selector works, chat renders, docs expand with TOC
- P2: Sample profiles listed, intake chat renders, docs expand with TOC
- P3: Form fields populate from sample, buttons render, docs expand with TOC

Kill server after verifying.

- [ ] **Step 4: Commit**

```bash
git add -A app/ .gitignore
git commit -m "feat(app): fix path resolution and finalize integration"
```

---

### Task 12: App README

**Files:**
- Create: `app/README.md`

- [ ] **Step 1: Write the app README**

Write `app/README.md`:
```markdown
# LinguaFlow Learning Lab

Interactive web interface for testing and exploring LinguaFlow learning projects.

## Quick Start

```bash
# From the repo root (with .venv activated)
cd app
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Tabs

| Tab | Project | Interface |
|-----|---------|-----------|
| Grammar Agent | P1 — Grammar Correction Agent | Chat: paste writing → get analysis → ask follow-ups |
| Lesson Planner | P2 — Lesson Plan Generator | Chat: intake conversation → generate plan |
| Assessment Pipeline | P3 — Student Assessment Pipeline | Form: submit writing → get CEFR assessment |

Each tab includes collapsible documentation with a sidebar table of contents.

## Architecture

- **`app.py`** — Entry point with tab routing
- **`adapters/`** — Thin wrappers that import from project modules
- **`components/`** — Reusable UI: chat, doc viewer, result cards
- **`pages/`** — One module per project tab

## Adding New Projects

1. Create an adapter in `adapters/` wrapping the project's key functions
2. Create a page in `pages/` defining the tab UI
3. Add the tab to `app.py`
```

- [ ] **Step 2: Commit**

```bash
git add app/README.md
git commit -m "docs(app): add README with quick start and architecture overview"
```
