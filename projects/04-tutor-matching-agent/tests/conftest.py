"""Shared pytest fixtures for the Tutor Matching Agent tests."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from graph import build_graph


@pytest.fixture
def graph_with_memory():
    """Compiled graph with InMemorySaver checkpointer for persistence tests."""
    checkpointer = InMemorySaver()
    return build_graph(checkpointer=checkpointer)
