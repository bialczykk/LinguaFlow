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

## Deploy (GitHub + Streamlit Community Cloud)

### 1. Push this repo to GitHub

Create an empty repository on GitHub (no README/license if you already have them locally), then:

```bash
cd "/path/to/LangGraph Learning"
git remote add origin https://github.com/<you>/<repo>.git
git add -A
git commit -m "Your message"
git push -u origin main
```

If `origin` already exists, use `git remote set-url origin ...` instead of `add`.

### 2. Secrets on Streamlit Cloud

Adapters call `ensure_repo_env()` (`adapters/_env.py`), which loads the repo-root `.env` locally and, when deployed, copies [Streamlit secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management) into `os.environ` so `ANTHROPIC_API_KEY` and other vars match what LangChain expects.

In **Streamlit Community Cloud** → your app → **Settings** → **Secrets**, paste TOML with the same keys as your `.env`, for example:

```toml
ANTHROPIC_API_KEY = "paste-your-anthropic-key-here"
```

Optional LangSmith:

```toml
LANGCHAIN_TRACING_V2 = "true"
LANGSMITH_API_KEY = "paste-langsmith-key-if-used"
LANGSMITH_PROJECT = "your-project"
```

See `app/.streamlit/secrets.toml.example` for a template. Do not commit real `secrets.toml`; Cloud stores secrets only in the dashboard.

### 3. Cloud app settings

- **Main file:** `app/app.py`
- **Python package file:** `requirements.txt` at the **repository root** (includes Streamlit + all tab dependencies)

First deploy can take several minutes while `sentence-transformers` / Chroma dependencies install.
