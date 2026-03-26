"""Adapter for Project 03 — Student Assessment Pipeline.

Handles sys.path setup, environment loading, vector store caching,
and wraps project functions with error handling.
"""

import sys
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_P3_DIR = _REPO_ROOT / "projects" / "03-student-assessment-pipeline"
if str(_P3_DIR) not in sys.path:
    sys.path.insert(0, str(_P3_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")

# -- Clear cached modules from other projects, then import P3 modules --
clear_project_modules()
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
