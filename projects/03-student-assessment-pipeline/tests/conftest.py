"""Shared test fixtures for the Student Assessment Pipeline.

Provides a pre-populated Chroma vector store fixture that multiple
test modules can use without re-ingesting documents each time.
"""

import pytest
from ingestion import build_vector_store


@pytest.fixture(scope="session")
def vector_store(tmp_path_factory):
    """Build a Chroma vector store in a temp directory for the test session.

    Uses session scope so the (slow) embedding step only runs once
    across all tests.
    """
    persist_dir = str(tmp_path_factory.mktemp("chroma_test"))
    store = build_vector_store(persist_directory=persist_dir)
    return store
