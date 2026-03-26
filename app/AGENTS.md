# LinguaFlow Learning Lab — App Development Guide

This file is scoped to the `app/` directory. It guides Claude when adding new projects to the Streamlit interface or modifying existing tabs.

## Architecture

```
app/
├── app.py                  # Entry point — page config, tab routing
├── adapters/               # One thin wrapper per project
│   ├── _importer.py        # Module collision handler (CRITICAL — read below)
│   ├── grammar_agent.py    # P1
│   ├── lesson_planner.py   # P2
│   └── assessment.py       # P3
├── components/             # Shared UI components
│   ├── chat.py             # Chat interface (st.chat_message / st.chat_input)
│   ├── doc_viewer.py       # Markdown renderer with sidebar TOC
│   └── results.py          # Score cards, badges, bullet lists
└── pages/                  # One module per project tab
    ├── p1_grammar.py       # Chat-based grammar analysis
    ├── p2_lesson.py        # Two-phase: intake chat → plan generation
    └── p3_assessment.py    # Form-based CEFR assessment
```

## Running the App

```bash
cd app
source ../.venv/bin/activate
streamlit run app.py
```

## Adding a New Project — Step-by-Step Workflow

When a new learning project (e.g., P4) is ready and needs to be added to the app, follow this exact workflow:

### Phase 1: Understand the Project Interface

Before writing any code, explore the new project's modules to understand:
- **Entry points**: What functions/classes does `main.py` call? Those are what the adapter wraps.
- **Interaction model**: Is it conversational (chat UI) or one-shot (form UI)?
- **Pydantic models**: What structured output does it produce? This drives the results display.
- **Sample data**: Does it have sample inputs for quick testing?
- **Dependencies**: Does it need runtime initialization (like P3's vector store)?

### Phase 2: Create the Adapter

Create `app/adapters/<project_name>.py`. Follow this template exactly:

```python
"""Adapter for Project NN — <Project Name>.

Handles sys.path setup, environment loading, and wraps project functions
with error handling for use in the Streamlit app.
"""

import sys
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "<NN>-<project-directory-name>"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env")

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from <module> import <function>  # noqa: E402
# ... more imports ...
```

**Rules:**
- ALWAYS call `clear_project_modules()` immediately before the project imports. This prevents module name collisions (see Module Collision section below).
- Wrap every project function call in try/except, raising `RuntimeError` with a user-friendly message.
- Expose only what the page needs — keep the adapter thin.
- If the project has sample data, expose it via a getter function.

### Phase 3: Create the Page

Create `app/pages/p<N>_<name>.py`. The page must:

1. Import from `adapters.<adapter_name>` and `components.doc_viewer`
2. Define a `render()` function (this is called by `app.py`)
3. Choose the right interaction pattern:
   - **Conversational projects** → use `st.chat_message` / `st.chat_input` (see `p1_grammar.py` or `p2_lesson.py`)
   - **One-shot/form projects** → use `st.text_area` / `st.button` (see `p3_assessment.py`)
4. Namespace all session state keys with a prefix: `p<N>_` (e.g., `p4_chat_history`, `p4_result`)
5. Include a Reset button that clears all `p<N>_*` session state keys
6. End with the doc viewer:
   ```python
   doc_viewer.render("docs/<NN>-<project-name>.md", title="Documentation: <Topic>")
   ```

### Phase 4: Register the Tab

Edit `app/app.py`:

1. Add the import: `from pages import p<N>_<name>`
2. Add a tab to the `st.tabs()` list
3. Add `with tab<N>: p<N>_<name>.render()`

### Phase 5: Test with Playwright

This is mandatory. After implementation, test the app end-to-end using Playwright before considering the work done.

**Testing workflow:**

1. **Launch the app in background:**
   ```bash
   cd app && source ../.venv/bin/activate && streamlit run app.py --server.headless true --server.port 8503
   ```
   Use `run_in_background: true` on the Bash tool.

2. **Wait for startup**, then navigate with Playwright:
   ```
   browser_navigate → http://localhost:8503
   ```

3. **Check each tab loads without errors:**
   - `browser_snapshot` to verify the page rendered correctly
   - `browser_click` on each tab to switch between them
   - Verify all UI elements are present (inputs, buttons, doc expander)

4. **Test the new tab's functionality:**
   - Fill in inputs / send chat messages using Playwright
   - Verify results render correctly
   - Test the Reset button
   - Test sample data selection

5. **Check for errors:**
   - `browser_console_messages` with level "error" to catch any runtime errors
   - If errors are found, fix them and re-test

6. **Iterate until clean.** Do not mark the work as done until all tabs load and the new tab's core flow works end-to-end.

## Module Collision — CRITICAL

All projects share module names (`models`, `graph`, `nodes`, `prompts`, `data`, etc.). Python's `sys.modules` cache means the first project imported "wins" — subsequent projects get the wrong module.

**The fix:** `adapters/_importer.py` provides `clear_project_modules()`. Every adapter MUST call it immediately before its project imports. This clears the conflicting names from `sys.modules`. Previously loaded adapters are unaffected because they hold direct references to their imported objects.

**If you add a new project that introduces a new shared module name** (one not already in the `_CONFLICTING` set in `_importer.py`), add it to the set. Check by looking at the project's module filenames and comparing against other projects.

## Reusable Components

Use existing components before creating new ones:

- **`components/chat.py`** — Chat UI with message history, configurable callback. Used by P1, P2.
- **`components/doc_viewer.py`** — Renders markdown with sidebar TOC inside an expander. Used by all tabs.
- **`components/results.py`** — `score_card()`, `badge()`, `bullet_list()`. Used by P3.

If a new project needs a display pattern that doesn't exist, add it to `components/results.py` or create a new component file — not inline in the page module.

## Session State Conventions

- Prefix all keys with `p<N>_` to avoid collisions between tabs
- Common patterns:
  - `p<N>_chat_history` — list of `{"role": "user"|"assistant", "content": str}`
  - `p<N>_result` — the project's main output object
  - `p<N>_handler` / `p<N>_intake` — stateful conversation objects
- Reset button must clear ALL keys for its tab
