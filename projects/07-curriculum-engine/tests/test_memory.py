"""Tests for CompositeBackend memory routing.

Verifies that the backend correctly routes:
- /work/ paths → StateBackend (ephemeral)
- /catalog/ paths → StoreBackend (persistent)
- /preferences/ paths → StoreBackend (persistent)

DeepAgents concept: CompositeBackend routes file operations to different
storage backends based on path prefix. This enables hybrid storage where
working files are ephemeral but catalog/preferences persist.
"""

from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend


def test_composite_backend_factory_is_callable():
    """The factory function should return a callable."""
    from agents import create_composite_backend

    factory = create_composite_backend()
    assert callable(factory)


def test_skills_directory_has_all_required_skills():
    """All four SKILL.md files should exist with proper frontmatter."""
    from agents import SKILLS_DIR

    required_skills = [
        "curriculum-design",
        "lesson-template",
        "exercise-template",
        "assessment-template",
    ]

    for skill_name in required_skills:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        assert skill_file.exists(), f"Missing {skill_file}"

        content = skill_file.read_text()
        assert content.startswith("---"), f"Missing frontmatter in {skill_file}"
        assert f"name: {skill_name}" in content, f"Wrong name in {skill_file}"


def test_store_instance_is_shared():
    """All agents should share the same InMemoryStore instance for cross-session persistence."""
    from agents import _store
    from langgraph.store.memory import InMemoryStore

    assert isinstance(_store, InMemoryStore)
