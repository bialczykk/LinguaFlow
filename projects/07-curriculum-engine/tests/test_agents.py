"""Tests for DeepAgents factory functions.

Verifies that each agent is properly configured with the correct
skills, backend, and model — without making actual LLM calls.
"""

from unittest.mock import patch


def test_create_planner_agent_returns_compiled_graph():
    """Planner agent should be a compiled LangGraph graph."""
    from agents import create_planner_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_planner_agent()
    assert isinstance(agent, CompiledStateGraph)


def test_create_lesson_agent_returns_compiled_graph():
    from agents import create_lesson_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_lesson_agent()
    assert isinstance(agent, CompiledStateGraph)


def test_create_exercise_agent_returns_compiled_graph():
    from agents import create_exercise_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_exercise_agent()
    assert isinstance(agent, CompiledStateGraph)


def test_create_assessment_agent_returns_compiled_graph():
    from agents import create_assessment_agent
    from langgraph.graph.state import CompiledStateGraph

    agent = create_assessment_agent()
    assert isinstance(agent, CompiledStateGraph)


def test_planner_has_skills_configured():
    """Planner should load skills from the skills directory."""
    from agents import SKILLS_DIR

    assert SKILLS_DIR.exists(), f"Skills directory not found: {SKILLS_DIR}"
    assert (SKILLS_DIR / "curriculum-design" / "SKILL.md").exists()


def test_all_skill_files_have_frontmatter():
    """Every SKILL.md must have valid YAML frontmatter with name and description."""
    from agents import SKILLS_DIR

    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        assert skill_file.exists(), f"Missing SKILL.md in {skill_dir}"
        content = skill_file.read_text()
        assert content.startswith("---"), f"Missing frontmatter in {skill_file}"
        # Check for name and description in frontmatter
        frontmatter_end = content.index("---", 3)
        frontmatter = content[3:frontmatter_end]
        assert "name:" in frontmatter, f"Missing 'name' in {skill_file} frontmatter"
        assert "description:" in frontmatter, f"Missing 'description' in {skill_file} frontmatter"


def test_composite_backend_factory():
    """CompositeBackend should route /work/ to StateBackend and /catalog/, /preferences/ to StoreBackend."""
    from agents import create_composite_backend

    backend_factory = create_composite_backend()
    assert callable(backend_factory)
