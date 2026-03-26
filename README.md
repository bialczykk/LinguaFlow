# LinguaFlow — LangGraph Learning Lab

Educational monorepo for learning the **LangGraph** ecosystem (LangChain, LangGraph, LangSmith, DeepAgents) through hands-on projects. The domain is **LinguaFlow**, a fictional English tutoring platform—each project is motivated by a realistic feature without production-style overhead.

## What’s inside

| Path | Purpose |
|------|---------|
| `resources/LEARNING_PATH.md` | **Start here** — ordered list of projects, concepts, and difficulty |
| `projects/` | Eight self-contained subprojects (`01-…` through `08-…`), each with its own `requirements.txt` and runnable code |
| `docs/` | Long-form educational notes per project |
| `app/` | **Streamlit UI** — one tab per project for interactive demos ([`app/README.md`](app/README.md)) |
| `requirements.txt` | Root dependency set for running the **full Streamlit app** (e.g. Streamlit Community Cloud) |

## Quick start (local)

```bash
cd "/path/to/LangGraph Learning"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a **repo-root** `.env` (never commit it) with at least:

- `ANTHROPIC_API_KEY` — required for LLM calls  
- Optional: `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2`, etc.

Run the web lab:

```bash
cd app && streamlit run app.py
```

Run a single project’s tests (example):

```bash
cd projects/01-grammar-correction-agent
pip install -r requirements.txt
pytest
```

## Deploying the Streamlit app

See **[`app/README.md`](app/README.md)** — GitHub, **Streamlit Community Cloud** (`app/app.py`, root `requirements.txt`), and **Secrets** (paste TOML in the Cloud dashboard; do not commit real keys).

## Conventions

- **Model:** project code uses `ChatAnthropic(model="claude-haiku-4-5-20251001")` unless noted otherwise.  
- **Isolation:** each `projects/NN-…` folder is self-contained (venv-friendly).  
- **Docs:** every learning project should have a matching file under `docs/`.

## Security

- **Never commit `.env`** or real API keys. `.env` is listed in `.gitignore`.  
- If GitHub **push protection** blocks a push, remove secrets from **all commits** in history (not only the latest tree), rotate any exposed keys, then push again.  
- If keys were ever pushed, revoke them in the Anthropic and LangSmith consoles and issue new ones.

## License / use

Learning and experimentation—this is not a production product.
