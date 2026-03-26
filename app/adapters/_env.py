"""Repo environment: root ``.env`` locally; Streamlit Cloud secrets → ``os.environ``.

LangChain reads standard env vars (e.g. ``ANTHROPIC_API_KEY``). On Streamlit Community
Cloud, secrets are TOML in the dashboard; we mirror them into ``os.environ`` after
``load_dotenv`` so local and deployed runs match.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_env() -> None:
    """Load repo ``.env``, then apply ``st.secrets`` when running under Streamlit."""
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")

    try:
        import streamlit as st
    except ImportError:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    def _apply_nested(prefix: str, data: object) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}_{k}" if prefix else str(k)
                if isinstance(v, dict):
                    _apply_nested(key, v)
                else:
                    os.environ[key] = str(v)
        elif prefix:
            os.environ[prefix] = str(data)

    try:
        for key in secrets:
            val = secrets[key]
            if isinstance(val, dict):
                _apply_nested(str(key), val)
            else:
                os.environ[str(key)] = str(val)
    except Exception:
        pass
