"""End-to-end interrupt/resume workflow tests.

These tests exercise the full graph with real interrupt/resume cycles.
Each test hits the LLM (Anthropic API).
"""

import pytest
from langgraph.types import Command

from graph import build_graph


@pytest.mark.integration
class TestHappyPath:
    """Generate → approve draft → approve final → published."""

    def test_full_approve_flow(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "happy-1"}, "tags": ["p5-content-moderation"]}

        # Step 1: invoke — should generate content and hit draft_review interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result
        interrupt_payload = result["__interrupt__"][0].value
        assert "content" in interrupt_payload
        assert len(interrupt_payload["content"]) > 50

        # Step 2: approve draft — should hit final_review interrupt
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert "__interrupt__" in result
        final_payload = result["__interrupt__"][0].value
        assert "content" in final_payload

        # Step 3: approve final — should publish
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True
        assert result.get("publish_metadata") is not None


@pytest.mark.integration
class TestEditPath:
    """Generate → edit draft → approve final → published."""

    def test_edit_replaces_content(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "edit-1"}, "tags": ["p5-content-moderation"]}

        # Generate and hit first interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Edit the draft
        edited_text = "This is the moderator's edited version of the content."
        result = graph_with_memory.invoke(
            Command(resume={
                "action": "edit",
                "edited_content": edited_text,
            }),
            config=config,
        )
        # Should hit final review with polished version of the edited content
        assert "__interrupt__" in result

        # Approve final
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True


@pytest.mark.integration
class TestRejectAndRevise:
    """Generate → reject → revise → approve → approve → published."""

    def test_reject_loops_back_to_revision(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "reject-1"}, "tags": ["p5-content-moderation"]}

        # Generate and hit first interrupt
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Reject with feedback — should revise and hit draft_review again
        result = graph_with_memory.invoke(
            Command(resume={
                "action": "reject",
                "feedback": "Too advanced for B1 level. Simplify the language.",
            }),
            config=config,
        )
        # Should hit draft_review interrupt again with revised content
        assert "__interrupt__" in result
        revised_payload = result["__interrupt__"][0].value
        assert revised_payload["revision_count"] == 1

        # Approve the revision
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        # Should hit final review
        assert "__interrupt__" in result

        # Approve final
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert result.get("published") is True


@pytest.mark.integration
class TestMaxRevisions:
    """Reject twice → graph ends without publishing."""

    def test_max_revisions_ends_graph(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "maxrev-1"}, "tags": ["p5-content-moderation"]}

        # Generate
        result = graph_with_memory.invoke(sample_initial_state, config=config)
        assert "__interrupt__" in result

        # Reject #1
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Needs work"}),
            config=config,
        )
        assert "__interrupt__" in result  # revision 1, back to draft_review

        # Reject #2
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Still not good"}),
            config=config,
        )
        assert "__interrupt__" in result  # revision 2, back to draft_review

        # Reject #3 — should hit max revisions and end
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Giving up"}),
            config=config,
        )
        # Graph should have ended — no interrupt, not published
        assert result.get("published", False) is False


@pytest.mark.integration
class TestFinalRejection:
    """Approve draft → reject final → not published."""

    def test_final_rejection_kills_content(self, graph_with_memory, sample_initial_state):
        config = {"configurable": {"thread_id": "finalrej-1"}, "tags": ["p5-content-moderation"]}

        # Generate
        result = graph_with_memory.invoke(sample_initial_state, config=config)

        # Approve draft
        result = graph_with_memory.invoke(
            Command(resume={"action": "approve"}), config=config
        )
        assert "__interrupt__" in result  # final review

        # Reject final
        result = graph_with_memory.invoke(
            Command(resume={"action": "reject", "feedback": "Not publication ready"}),
            config=config,
        )
        assert result.get("published", False) is False


class TestGraphStructure:
    """Verify graph wiring without LLM calls."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        for expected in ["generate", "draft_review", "revise", "polish", "final_review", "publish"]:
            assert expected in node_names, f"Missing node: {expected}"
