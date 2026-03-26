# LinguaFlow Learning Lab — Interactive App Design Specification

**Date:** 2026-03-26
**Status:** Approved

---

## 1. Overview

A locally hosted Streamlit application that provides an interactive web interface for every project in the LinguaFlow learning path. Each project gets its own tab with a functional interface for testing the project's capabilities, plus collapsible educational documentation rendered from the existing `docs/` files.

The app lives in `app/` at the repo root, completely separate from the project directories. It imports project code through thin adapter modules.

**Initial scope:** Projects 1, 2, and 3. Future projects are added by following the established pattern (adapter + page + tab registration).

---

## 2. Tech Stack

- **Streamlit** — UI framework, light theme as default
- **Python** — same shared `.venv` at repo root as all projects
- **No additional frontend tooling** — no Node.js, no build step

### Dependencies

The app's own `requirements.txt`:

```
streamlit
python-dotenv
```

All project-specific dependencies (langchain, langgraph, chromadb, etc.) are already installed in the shared `.venv`.

### Run Command

```bash
streamlit run app/app.py
```

---

## 3. Directory Structure

```
app/
├── app.py                  # Streamlit entry point — page config, tab routing
├── requirements.txt        # App-specific dependencies
├── adapters/               # Thin wrappers per project
│   ├── __init__.py
│   ├── grammar_agent.py    # Wraps P1: analyze_grammar(), ConversationHandler
│   ├── lesson_planner.py   # Wraps P2: IntakeConversation, build_graph()
│   └── assessment.py       # Wraps P3: build_vector_store(), build_graph()
├── components/             # Reusable Streamlit UI pieces
│   ├── __init__.py
│   ├── chat.py             # Chat-style interface (used by P1, P2)
│   ├── doc_viewer.py       # Markdown renderer with sidebar TOC
│   └── results.py          # Structured result display (scores, tables, etc.)
└── pages/                  # One module per project tab
    ├── __init__.py
    ├── p1_grammar.py       # Grammar Correction Agent tab
    ├── p2_lesson.py        # Lesson Plan Generator tab
    └── p3_assessment.py    # Student Assessment Pipeline tab
```

---

## 4. Tab Interfaces

### 4.1 P1 — Grammar Correction Agent

**Interaction model:** Chat UI (`st.chat_message` / `st.chat_input`)

**Flow:**
1. User types or pastes student writing (or selects a sample text from a selectbox above the chat)
2. System returns structured feedback: list of grammar issues, CEFR proficiency assessment, full corrected text
3. User can ask follow-up questions about the feedback in the same chat
4. Typing `new: <text>` submits new writing for analysis

**Session state:**
- `p1_chat_history` — list of message dicts
- `p1_handler` — `ConversationHandler` instance
- `p1_feedback` — last `GrammarFeedback` result

**Adapter wraps:**
- `analyze_grammar(text: str) → GrammarFeedback` from `chains.py`
- `ConversationHandler` from `conversation.py` for follow-up turns

### 4.2 P2 — Lesson Plan Generator

**Interaction model:** Two-phase chat UI

**Phase 1 — Intake conversation:**
- Chat-driven Q&A using `IntakeConversation.ask()` / `is_complete()`
- Gathers student name, CEFR level, goals, topics, lesson type
- When intake is complete, shows the extracted `StudentProfile` and a "Generate Plan" button

**Phase 2 — Plan generation:**
- "Generate Plan" button triggers `build_graph().invoke()` with a spinner
- Final `LessonPlan` renders as structured output: title, objectives, activities table with timing, homework

**Session state:**
- `p2_chat_history` — list of message dicts
- `p2_intake` — `IntakeConversation` instance
- `p2_profile` — extracted `StudentProfile`
- `p2_plan` — generated `LessonPlan`

**Adapter wraps:**
- `IntakeConversation` from `intake.py`
- `build_graph()` from `graph.py`

### 4.3 P3 — Student Assessment Pipeline

**Interaction model:** Form-based (not conversational)

**Flow:**
1. User enters student writing in a text area, submission context in a text input, and optionally selects a CEFR level hint from a dropdown
2. Alternatively, selects a pre-loaded sample submission from a selectbox
3. Clicks "Assess" button — graph pipeline runs with a spinner
4. Results render as:
   - Overall CEFR level badge
   - Four criteria score cards (vocabulary, grammar, fluency, coherence) — each with score/5, feedback, evidence
   - Comparative summary
   - Strengths, areas to improve, recommendations lists

**Session state:**
- `p3_vector_store` — Chroma vector store (built once on first use, cached)
- `p3_result` — last `Assessment` result

**Adapter wraps:**
- `get_vector_store()` / `build_vector_store()` from `ingestion.py`
- `build_graph(vector_store)` from `graph.py`

### 4.4 All Tabs — Common Elements

- **Reset button** to clear session state and start fresh
- **Collapsible documentation section** at the bottom (see Section 5)
- **Error handling** — adapters catch exceptions and display user-friendly messages via `st.error()`

---

## 5. Documentation Viewer

Each tab includes a collapsible section (`st.expander`) at the bottom that renders the project's educational documentation from `docs/`.

**Layout when expanded — two-column split (25/75):**

- **Left column (25%):** Sidebar TOC auto-generated by parsing `##` and `###` headers from the markdown file. Each heading is a clickable anchor link.
- **Right column (75%):** Full markdown rendered with `st.markdown()`, with anchor IDs injected at each heading for TOC linking. Code blocks render with Streamlit's native syntax highlighting.

**Implementation:**
- A reusable `doc_viewer` component in `components/doc_viewer.py`
- Takes a file path to a markdown doc
- Parses headers automatically — no manual TOC maintenance
- Expander label shows the doc title (e.g., "📚 Documentation: LangChain Fundamentals")

**Doc mapping:**
| Tab | Doc file |
|-----|----------|
| P1 — Grammar Agent | `docs/01-grammar-correction-agent.md` |
| P2 — Lesson Planner | `docs/02-lesson-plan-generator.md` |
| P3 — Assessment | `docs/03-student-assessment-pipeline.md` |

---

## 6. Adapters

Each adapter module in `app/adapters/` is a thin wrapper that:

1. **Handles imports** — adds the relevant `projects/` directory to `sys.path` so project modules can be imported
2. **Wraps key functions** — exposes a clean interface for the page modules, hiding internal project structure
3. **Manages initialization** — handles one-time setup (e.g., P3's vector store build on first use)
4. **Loads environment** — reads `.env` from the project directory via `python-dotenv`
5. **Error isolation** — catches project-level exceptions and returns user-friendly error messages

### Adding Future Projects

When a new project is ready for the app:

1. Add an adapter in `app/adapters/` wrapping the project's key functions
2. Add a page module in `app/pages/` defining the tab UI
3. Register the new tab in `app/app.py`

No changes to existing tabs or adapters required.

---

## 7. State Management

Streamlit session state is namespaced per tab to avoid collisions:

| Namespace | Keys |
|-----------|------|
| P1 | `p1_chat_history`, `p1_handler`, `p1_feedback` |
| P2 | `p2_chat_history`, `p2_intake`, `p2_profile`, `p2_plan` |
| P3 | `p3_vector_store`, `p3_result` |

Each tab's reset button clears only its own namespace.

---

## 8. Constraints & Non-Goals

- **Local only** — no deployment, no auth, no production concerns
- **No custom theming beyond Streamlit defaults** — light theme, native components
- **No real-time streaming of LLM output** — show spinner during processing, render complete results (streaming can be added later if desired)
- **No database** — all state lives in Streamlit session state (ephemeral, resets on page reload)
- **Projects are not modified** — the app only imports from them, never changes project code
