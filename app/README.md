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
