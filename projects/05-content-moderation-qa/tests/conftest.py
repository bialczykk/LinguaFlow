"""Shared pytest fixtures for the Content Moderation tests."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from graph import build_graph


@pytest.fixture
def graph_with_memory():
    """Compiled graph with InMemorySaver for interrupt/resume tests."""
    checkpointer = InMemorySaver()
    return build_graph(checkpointer=checkpointer)


@pytest.fixture
def sample_initial_state():
    """Initial state for a grammar explanation request."""
    return {
        "content_request": {
            "topic": "Present Perfect Tense",
            "content_type": "grammar_explanation",
            "difficulty": "B1",
        },
        "draft_content": "",
        "generation_confidence": 0.0,
        "draft_decision": {},
        "revision_count": 0,
        "polished_content": "",
        "final_decision": {},
        "published": False,
        "publish_metadata": None,
    }
