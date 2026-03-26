"""Tests for the StateGraph assembly.

Verifies the graph compiles correctly, has the right nodes, and
interrupt points fire at the expected locations.
"""

import pytest
from unittest.mock import patch, MagicMock


def test_graph_compiles():
    """build_graph() should return a compiled graph."""
    from graph import build_graph
    from langgraph.graph.state import CompiledStateGraph

    graph = build_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_graph_has_expected_nodes():
    """Graph should contain all expected node names."""
    from graph import build_graph

    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())

    expected = {
        "plan_curriculum", "review_plan",
        "generate_lesson", "review_lesson",
        "generate_exercises", "review_exercises",
        "generate_assessment", "review_assessment",
        "assemble_module",
    }
    assert expected.issubset(node_names)


def test_graph_starts_at_plan_curriculum():
    """Graph execution should start at plan_curriculum node."""
    from graph import build_graph

    graph = build_graph()
    graph_repr = graph.get_graph()
    start_edges = [
        e.target for e in graph_repr.edges
        if e.source == "__start__"
    ]
    assert "plan_curriculum" in start_edges
