# agents.py
"""DeepAgents factory functions for the Curriculum Engine.

Creates four specialized agents using create_deep_agent():
- Planner: plans the curriculum structure using TodoList
- Lesson Writer: generates lesson content
- Exercise Creator: creates practice exercises
- Assessment Builder: builds graded assessments

Each agent loads SKILL.md files for domain knowledge and output format
guidance. The CompositeBackend routes working files to ephemeral storage
and catalog/preferences to persistent storage.

DeepAgents concepts demonstrated:
- create_deep_agent() with skills, backend, and system_prompt
- CompositeBackend for hybrid ephemeral/persistent storage
- StateBackend (ephemeral) vs StoreBackend (persistent)
- Skills loading via SkillsMiddleware (automatic from skills= parameter)
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend
from langchain_anthropic import ChatAnthropic
from langgraph.store.memory import InMemoryStore

from prompts import (
    ASSESSMENT_BUILDER_PROMPT,
    EXERCISE_CREATOR_PROMPT,
    LESSON_WRITER_PROMPT,
    PLANNER_PROMPT,
)

# -- Paths --
_PROJECT_DIR = Path(__file__).resolve().parent
SKILLS_DIR = _PROJECT_DIR / "skills"

# -- Model: always use cheapest Anthropic model --
_MODEL = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# -- Shared store for persistent memory across threads --
_store = InMemoryStore()

# -- LangSmith tags --
_TAGS = ["p7-curriculum-engine"]


def create_composite_backend():
    """Create a CompositeBackend factory that routes paths to the right storage.

    - /work/ → StateBackend (ephemeral, thread-scoped drafts)
    - /catalog/ → StoreBackend (persistent, cross-session module catalog)
    - /preferences/ → StoreBackend (persistent, cross-session user preferences)
    - Default → StateBackend (ephemeral)

    DeepAgents concept: CompositeBackend lets a single agent use multiple
    storage strategies. Ephemeral files disappear when the thread ends,
    while persistent files survive across sessions.

    Returns a factory callable: (runtime) -> CompositeBackend.
    """
    def factory(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/catalog/": StoreBackend(runtime),
                "/preferences/": StoreBackend(runtime),
            },
        )
    return factory


def create_planner_agent():
    """Create the curriculum planning agent.

    Uses TodoList to break down the planning task and writes the
    curriculum plan to /work/plan.json. Loads the curriculum-design
    skill for CEFR levels and Bloom's taxonomy guidance.
    """
    return create_deep_agent(
        name="curriculum-planner",
        model=_MODEL,
        system_prompt=PLANNER_PROMPT,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_lesson_agent():
    """Create the lesson writer agent.

    Reads the plan from /work/plan.json, loads curriculum-design and
    lesson-template skills, and writes the lesson to /work/lesson.md.
    """
    return create_deep_agent(
        name="lesson-writer",
        model=_MODEL,
        system_prompt=LESSON_WRITER_PROMPT,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_exercise_agent():
    """Create the exercise creator agent.

    Reads the plan and lesson, loads curriculum-design and exercise-template
    skills, and writes exercises to /work/exercises.md.
    """
    return create_deep_agent(
        name="exercise-creator",
        model=_MODEL,
        system_prompt=EXERCISE_CREATOR_PROMPT,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )


def create_assessment_agent():
    """Create the assessment builder agent.

    Reads the plan and lesson, loads curriculum-design and assessment-template
    skills, and writes the assessment to /work/assessment.md.
    """
    return create_deep_agent(
        name="assessment-builder",
        model=_MODEL,
        system_prompt=ASSESSMENT_BUILDER_PROMPT,
        skills=[str(SKILLS_DIR) + "/"],
        backend=create_composite_backend(),
        store=_store,
    )
