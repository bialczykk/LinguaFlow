"""StateGraph assembly for the Student Assessment Pipeline.

Wires together the 5 node functions into a sequential graph:
retrieve_standards → criteria_scoring → retrieve_samples →
comparative_analysis → synthesize

LangGraph concepts demonstrated:
- StateGraph construction with TypedDict state schema
- Sequential edges (add_edge) for a linear pipeline
- functools.partial to inject the vector store into retrieval nodes
- Graph compilation and invocation

The key architectural insight: this is a linear graph, but the adaptive
behavior comes from criteria_scoring producing a preliminary_level that
retrieve_samples uses as a metadata filter. The graph structure is simple;
the intelligence is in how nodes use state from previous nodes.
"""

from functools import partial

from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from models import AssessmentState
from nodes import (
    retrieve_standards_node,
    criteria_scoring_node,
    retrieve_samples_node,
    comparative_analysis_node,
    synthesize_node,
)


def build_graph(vector_store: Chroma):
    """Build and compile the assessment StateGraph.

    The vector store is injected into retrieval nodes via functools.partial,
    so the graph nodes have a clean (state) -> dict signature while still
    accessing the store.

    Args:
        vector_store: Pre-populated Chroma instance with rubrics, standards,
                      and sample essays.

    Returns:
        Compiled LangGraph graph ready for .invoke() or .stream().
    """
    # Bind the vector store to retrieval nodes using partial
    # This way, the graph nodes have the signature (state) -> dict
    # that LangGraph expects, while still accessing the vector store
    retrieve_standards = partial(retrieve_standards_node, vector_store=vector_store)
    retrieve_samples = partial(retrieve_samples_node, vector_store=vector_store)

    # Build the graph
    graph = (
        StateGraph(AssessmentState)
        .add_node("retrieve_standards", retrieve_standards)
        .add_node("criteria_scoring", criteria_scoring_node)
        .add_node("retrieve_samples", retrieve_samples)
        .add_node("comparative_analysis", comparative_analysis_node)
        .add_node("synthesize", synthesize_node)
        # Wire the sequential flow
        .add_edge(START, "retrieve_standards")
        .add_edge("retrieve_standards", "criteria_scoring")
        .add_edge("criteria_scoring", "retrieve_samples")
        .add_edge("retrieve_samples", "comparative_analysis")
        .add_edge("comparative_analysis", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )

    return graph
