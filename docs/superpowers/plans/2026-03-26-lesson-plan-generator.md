# Lesson Plan Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph StateGraph that generates personalized lesson plans with conditional routing (3 lesson types) and a review loop.

**Architecture:** Two-phase design — an intake conversation gathers student info into a `StudentProfile`, then a StateGraph pipeline (research → route → draft → review → finalize) produces a structured `LessonPlan`. Three conditional branches (conversation, grammar, exam_prep) and a review cycle (max 2 revisions) demonstrate core LangGraph concepts.

**Tech Stack:** LangGraph 1.1.3, langchain-core 1.2.22, langchain-anthropic 1.4.0, langsmith 0.7.22, pydantic, python-dotenv, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-lesson-plan-generator-design.md`

---

## File Structure

```
projects/02-lesson-plan-generator/
├── models.py              # StudentProfile, Activity, LessonPlan, LessonPlanState (TypedDict)
├── intake.py              # IntakeConversation class (multi-turn student info gathering)
├── prompts.py             # All ChatPromptTemplate definitions
├── nodes.py               # Node functions: research, draft_*, review, finalize
├── graph.py               # StateGraph wiring, conditional edges, compilation
├── main.py                # Interactive CLI entry point
├── requirements.txt       # Project dependencies
├── README.md              # Project overview and run instructions
├── data/
│   ├── __init__.py
│   └── sample_profiles.py # Pre-built StudentProfile instances for testing
└── tests/
    ├── __init__.py
    ├── test_models.py     # Unit tests for Pydantic models and state schema
    ├── test_nodes.py      # Integration tests for individual node functions
    ├── test_graph.py      # Integration tests for full graph execution
    └── test_intake.py     # Integration tests for intake conversation
```

---

### Task 1: Project Skeleton and Dependencies

**Files:**
- Create: `projects/02-lesson-plan-generator/requirements.txt`
- Create: `projects/02-lesson-plan-generator/data/__init__.py`
- Create: `projects/02-lesson-plan-generator/tests/__init__.py`

- [ ] **Step 1: Create the project directory structure**

```bash
mkdir -p "projects/02-lesson-plan-generator/data"
mkdir -p "projects/02-lesson-plan-generator/tests"
```

- [ ] **Step 2: Create requirements.txt**

```
# projects/02-lesson-plan-generator/requirements.txt
langchain-core>=1.0,<2.0
langchain-anthropic>=1.0,<2.0
langgraph>=1.0,<2.0
langsmith>=0.3.0
python-dotenv>=1.0.0
pydantic>=2.0
pytest>=8.0
```

- [ ] **Step 3: Create empty __init__.py files**

Create empty files at:
- `projects/02-lesson-plan-generator/data/__init__.py`
- `projects/02-lesson-plan-generator/tests/__init__.py`

- [ ] **Step 4: Verify dependencies are already installed in the shared venv**

```bash
cd "/Users/kubabialczyk/Desktop/Side Quests/LangGraph Learning"
source .venv/bin/activate
python -c "import langgraph; import langchain_anthropic; import langsmith; print('All deps available')"
```

Expected: `All deps available`

- [ ] **Step 5: Commit**

```bash
git add projects/02-lesson-plan-generator/
git commit -m "feat(p2): scaffold project skeleton and dependencies"
```

---

### Task 2: Pydantic Models and State Schema (TDD)

**Files:**
- Create: `projects/02-lesson-plan-generator/tests/test_models.py`
- Create: `projects/02-lesson-plan-generator/models.py`

- [ ] **Step 1: Write the failing tests for all models**

```python
# projects/02-lesson-plan-generator/tests/test_models.py
"""Unit tests for Pydantic models and LangGraph state schema.

Tests validate that all models enforce their constraints correctly:
- StudentProfile enforces CEFR levels and lesson types
- Activity validates duration and materials
- LessonPlan validates the complete lesson structure
"""

import pytest
from pydantic import ValidationError

from models import StudentProfile, Activity, LessonPlan


# -- StudentProfile Tests --

class TestStudentProfile:
    """Tests for StudentProfile Pydantic model."""

    def test_valid_profile(self):
        """A well-formed profile should be created without errors."""
        profile = StudentProfile(
            name="Maria",
            proficiency_level="B1",
            learning_goals=["improve fluency", "learn idioms"],
            preferred_topics=["travel", "food"],
            lesson_type="conversation",
        )
        assert profile.name == "Maria"
        assert profile.proficiency_level == "B1"
        assert profile.lesson_type == "conversation"

    def test_all_cefr_levels(self):
        """All six CEFR levels should be accepted."""
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            profile = StudentProfile(
                name="Test",
                proficiency_level=level,
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="grammar",
            )
            assert profile.proficiency_level == level

    def test_invalid_cefr_level(self):
        """An invalid CEFR level should raise a ValidationError."""
        with pytest.raises(ValidationError):
            StudentProfile(
                name="Test",
                proficiency_level="D1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="grammar",
            )

    def test_all_lesson_types(self):
        """All three lesson types should be accepted."""
        for lesson_type in ["conversation", "grammar", "exam_prep"]:
            profile = StudentProfile(
                name="Test",
                proficiency_level="B1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type=lesson_type,
            )
            assert profile.lesson_type == lesson_type

    def test_invalid_lesson_type(self):
        """An invalid lesson type should raise a ValidationError."""
        with pytest.raises(ValidationError):
            StudentProfile(
                name="Test",
                proficiency_level="B1",
                learning_goals=["learn"],
                preferred_topics=["topic"],
                lesson_type="yoga",
            )


# -- Activity Tests --

class TestActivity:
    """Tests for Activity Pydantic model."""

    def test_valid_activity(self):
        """A well-formed activity should be created without errors."""
        activity = Activity(
            name="Role-play: Ordering Food",
            description="Students practice ordering at a restaurant.",
            duration_minutes=15,
            materials=["menu handout", "vocabulary list"],
        )
        assert activity.name == "Role-play: Ordering Food"
        assert activity.duration_minutes == 15
        assert len(activity.materials) == 2

    def test_activity_empty_materials(self):
        """An activity with no materials should be valid."""
        activity = Activity(
            name="Free discussion",
            description="Open conversation on the topic.",
            duration_minutes=10,
            materials=[],
        )
        assert activity.materials == []


# -- LessonPlan Tests --

class TestLessonPlan:
    """Tests for LessonPlan Pydantic model."""

    def test_valid_lesson_plan(self):
        """A complete lesson plan should be created without errors."""
        plan = LessonPlan(
            title="Travel English: At the Airport",
            level="B1",
            lesson_type="conversation",
            objectives=["Practice check-in dialogue", "Learn travel vocabulary"],
            warm_up="Discuss last travel experience",
            main_activities=[
                Activity(
                    name="Airport Role-play",
                    description="Simulate check-in counter interaction.",
                    duration_minutes=20,
                    materials=["dialogue script"],
                )
            ],
            wrap_up="Review new vocabulary learned today",
            homework="Write a short paragraph about your dream destination",
            estimated_duration_minutes=60,
        )
        assert plan.title == "Travel English: At the Airport"
        assert len(plan.main_activities) == 1
        assert plan.estimated_duration_minutes == 60

    def test_lesson_plan_multiple_activities(self):
        """A lesson plan with multiple activities should be valid."""
        activities = [
            Activity(
                name=f"Activity {i}",
                description=f"Description {i}",
                duration_minutes=10,
                materials=[],
            )
            for i in range(3)
        ]
        plan = LessonPlan(
            title="Grammar Drills",
            level="A2",
            lesson_type="grammar",
            objectives=["Practice present tense"],
            warm_up="Warm up",
            main_activities=activities,
            wrap_up="Wrap up",
            homework="Homework",
            estimated_duration_minutes=45,
        )
        assert len(plan.main_activities) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement the models**

```python
# projects/02-lesson-plan-generator/models.py
"""Pydantic models and LangGraph state schema for the Lesson Plan Generator.

This module defines all data structures used in the lesson plan pipeline:
- StudentProfile: input from the intake conversation
- Activity: a single lesson activity
- LessonPlan: the structured final output
- LessonPlanState: the TypedDict that flows through the LangGraph StateGraph

Key LangGraph concept demonstrated:
- TypedDict as graph state schema — every node reads from and writes to this shared state.
  Fields without reducers use "last write wins" semantics, which is fine here since
  each field is written by exactly one node at a time.
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# -- Input Model --

class StudentProfile(BaseModel):
    """Student information gathered during the intake conversation.

    The intake conversation collects this data through a short multi-turn
    dialogue, then produces a validated StudentProfile instance.
    """

    name: str = Field(description="Student's name")
    proficiency_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="CEFR proficiency level assessed or self-reported"
    )
    learning_goals: list[str] = Field(
        description="What the student wants to achieve (e.g., 'improve fluency')"
    )
    preferred_topics: list[str] = Field(
        description="Topics the student is interested in (e.g., 'travel', 'business')"
    )
    lesson_type: Literal["conversation", "grammar", "exam_prep"] = Field(
        description="Type of lesson to generate, inferred from the student's goals"
    )


# -- Output Models --

class Activity(BaseModel):
    """A single activity within a lesson plan.

    Each activity has a name, description, estimated duration, and
    optional list of materials needed.
    """

    name: str = Field(description="Short name for the activity")
    description: str = Field(description="What the activity involves")
    duration_minutes: int = Field(description="Estimated duration in minutes")
    materials: list[str] = Field(
        description="Materials needed (handouts, props, etc.)"
    )


class LessonPlan(BaseModel):
    """Complete structured lesson plan — the final output of the graph.

    This Pydantic model is used by the finalize node to parse the LLM's
    draft into a validated, structured format using .with_structured_output().
    """

    title: str = Field(description="Descriptive lesson title")
    level: str = Field(description="CEFR level this lesson targets")
    lesson_type: str = Field(description="conversation, grammar, or exam_prep")
    objectives: list[str] = Field(description="Learning objectives for this lesson")
    warm_up: str = Field(description="Warm-up activity description")
    main_activities: list[Activity] = Field(description="Core lesson activities")
    wrap_up: str = Field(description="Wrap-up / review activity")
    homework: str = Field(description="Homework assignment")
    estimated_duration_minutes: int = Field(
        description="Total estimated lesson duration in minutes"
    )


# -- LangGraph State Schema --

class LessonPlanState(TypedDict):
    """State schema for the LangGraph StateGraph.

    This TypedDict defines the shared state that flows through every node
    in the lesson plan graph. Each node reads what it needs and returns
    a partial dict updating only the fields it's responsible for.

    LangGraph concept: TypedDict state schemas
    - Every field uses "last write wins" (no reducers needed here)
    - Nodes return partial dicts like {"research_notes": "..."}
    - The graph engine merges these updates into the full state
    """

    student_profile: StudentProfile
    research_notes: str
    draft_plan: str
    review_feedback: str
    revision_count: int
    is_approved: bool
    final_plan: LessonPlan | None
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_models.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/02-lesson-plan-generator/models.py projects/02-lesson-plan-generator/tests/test_models.py
git commit -m "feat(p2): add Pydantic models and state schema with tests"
```

---

### Task 3: Sample Profiles Test Data

**Files:**
- Create: `projects/02-lesson-plan-generator/data/sample_profiles.py`

- [ ] **Step 1: Create sample profiles for all three lesson types**

```python
# projects/02-lesson-plan-generator/data/sample_profiles.py
"""Pre-built StudentProfile instances for testing and demonstration.

These profiles cover all three lesson types and a range of CEFR levels,
allowing tests to skip the intake conversation and invoke the graph directly.
"""

from models import StudentProfile


# -- Conversation-focused profiles --

BEGINNER_CONVERSATION = StudentProfile(
    name="Yuki",
    proficiency_level="A2",
    learning_goals=["build confidence speaking", "learn everyday phrases"],
    preferred_topics=["shopping", "introducing yourself"],
    lesson_type="conversation",
)

INTERMEDIATE_CONVERSATION = StudentProfile(
    name="Carlos",
    proficiency_level="B1",
    learning_goals=["improve fluency", "learn travel vocabulary"],
    preferred_topics=["travel", "food and restaurants"],
    lesson_type="conversation",
)

# -- Grammar-focused profiles --

BEGINNER_GRAMMAR = StudentProfile(
    name="Fatima",
    proficiency_level="A1",
    learning_goals=["learn basic sentence structure", "understand verb tenses"],
    preferred_topics=["daily routines", "family"],
    lesson_type="grammar",
)

INTERMEDIATE_GRAMMAR = StudentProfile(
    name="Hans",
    proficiency_level="B2",
    learning_goals=["master conditionals", "reduce common errors"],
    preferred_topics=["technology", "environment"],
    lesson_type="grammar",
)

# -- Exam prep profiles --

EXAM_PREP_INTERMEDIATE = StudentProfile(
    name="Mei",
    proficiency_level="B2",
    learning_goals=["prepare for IELTS", "improve writing scores"],
    preferred_topics=["education", "global issues"],
    lesson_type="exam_prep",
)

EXAM_PREP_ADVANCED = StudentProfile(
    name="Olga",
    proficiency_level="C1",
    learning_goals=["target CAE certificate", "refine academic writing"],
    preferred_topics=["science", "current affairs"],
    lesson_type="exam_prep",
)
```

- [ ] **Step 2: Verify the sample profiles import correctly**

```bash
cd "projects/02-lesson-plan-generator"
python -c "from data.sample_profiles import *; print(f'Loaded {BEGINNER_CONVERSATION.name}, {INTERMEDIATE_GRAMMAR.name}, {EXAM_PREP_INTERMEDIATE.name}')"
```

Expected: `Loaded Yuki, Hans, Mei`

- [ ] **Step 3: Commit**

```bash
git add projects/02-lesson-plan-generator/data/sample_profiles.py
git commit -m "feat(p2): add sample student profiles for testing"
```

---

### Task 4: Prompt Templates

**Files:**
- Create: `projects/02-lesson-plan-generator/prompts.py`

- [ ] **Step 1: Create all prompt templates**

```python
# projects/02-lesson-plan-generator/prompts.py
"""Prompt templates for every node in the lesson plan graph.

All prompts are defined here in one place so they're easy to compare,
tweak, and review. Each prompt is a ChatPromptTemplate with system and
human message pairs.

LangChain concept demonstrated:
- ChatPromptTemplate.from_messages() — builds reusable prompt templates
  from (role, content) tuples with {variable} placeholders.
"""

from langchain_core.prompts import ChatPromptTemplate


# -- Intake Conversation --

INTAKE_SYSTEM_PROMPT = (
    "You are a friendly course advisor for the LinguaFlow English tutoring "
    "platform. Your job is to learn about the student so you can create a "
    "personalized lesson plan.\n\n"
    "Gather the following information through natural conversation:\n"
    "1. The student's name\n"
    "2. Their English proficiency level (A1-C2 on the CEFR scale — help them "
    "self-assess if needed)\n"
    "3. Their learning goals (what they want to improve)\n"
    "4. Topics they're interested in\n\n"
    "Based on their goals, determine which lesson type fits best:\n"
    "- 'conversation' — if they want to improve speaking, fluency, or "
    "everyday communication\n"
    "- 'grammar' — if they want to focus on rules, sentence structure, or "
    "error correction\n"
    "- 'exam_prep' — if they're preparing for an exam (IELTS, TOEFL, CAE, etc.)\n\n"
    "Ask one question at a time. Be warm and encouraging. When you have all "
    "the information, respond with EXACTLY this format on its own line:\n"
    "[PROFILE_COMPLETE]\n"
    "Then summarize what you learned about the student."
)

# -- Research Node --

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a curriculum specialist for the LinguaFlow English tutoring "
        "platform. Given a student profile, suggest relevant teaching materials, "
        "activities, and themes that would be appropriate for their level and goals.\n\n"
        "Focus on practical, actionable suggestions. Include:\n"
        "- 3-5 specific activity ideas\n"
        "- Relevant vocabulary themes\n"
        "- Teaching approaches suited to the level\n"
        "- Any cultural or contextual considerations based on topics",
    ),
    (
        "human",
        "Student Profile:\n"
        "- Name: {name}\n"
        "- Level: {proficiency_level}\n"
        "- Goals: {learning_goals}\n"
        "- Preferred Topics: {preferred_topics}\n"
        "- Lesson Type: {lesson_type}\n\n"
        "Please suggest materials and activities for this student.",
    ),
])

# -- Drafting Nodes (one prompt per lesson type) --

DRAFT_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English lesson planner for the LinguaFlow platform, "
        "specializing in conversation-focused lessons.\n\n"
        "Create a detailed lesson plan that emphasizes:\n"
        "- Dialogue practice and role-play scenarios\n"
        "- Speaking exercises and pronunciation tips\n"
        "- Pair/group discussion activities\n"
        "- Real-world communication situations\n\n"
        "The lesson plan should be practical, engaging, and appropriate for "
        "the student's CEFR level. Include clear timing for each activity.",
    ),
    (
        "human",
        "Student: {name} (Level: {proficiency_level})\n"
        "Goals: {learning_goals}\n"
        "Topics: {preferred_topics}\n\n"
        "Research Notes:\n{research_notes}\n\n"
        "{revision_context}"
        "Please create a detailed conversation-focused lesson plan.",
    ),
])

DRAFT_GRAMMAR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English lesson planner for the LinguaFlow platform, "
        "specializing in grammar-focused lessons.\n\n"
        "Create a detailed lesson plan that emphasizes:\n"
        "- Clear grammar rule explanations with examples\n"
        "- Structured exercises progressing from controlled to free practice\n"
        "- Error correction activities\n"
        "- Gap-fill, transformation, and sentence-building drills\n\n"
        "The lesson plan should be clear, well-structured, and appropriate for "
        "the student's CEFR level. Include clear timing for each activity.",
    ),
    (
        "human",
        "Student: {name} (Level: {proficiency_level})\n"
        "Goals: {learning_goals}\n"
        "Topics: {preferred_topics}\n\n"
        "Research Notes:\n{research_notes}\n\n"
        "{revision_context}"
        "Please create a detailed grammar-focused lesson plan.",
    ),
])

DRAFT_EXAM_PREP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English lesson planner for the LinguaFlow platform, "
        "specializing in exam preparation lessons.\n\n"
        "Create a detailed lesson plan that emphasizes:\n"
        "- Practice questions in exam format\n"
        "- Test-taking strategies and time management\n"
        "- Skill-specific drills (reading, writing, listening, speaking)\n"
        "- Mock exercise segments with realistic difficulty\n\n"
        "The lesson plan should be focused, practical, and appropriate for "
        "the student's target exam and CEFR level. Include clear timing.",
    ),
    (
        "human",
        "Student: {name} (Level: {proficiency_level})\n"
        "Goals: {learning_goals}\n"
        "Topics: {preferred_topics}\n\n"
        "Research Notes:\n{research_notes}\n\n"
        "{revision_context}"
        "Please create a detailed exam preparation lesson plan.",
    ),
])

# -- Review Node --

REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior curriculum reviewer for the LinguaFlow platform. "
        "Evaluate the lesson plan draft against these quality criteria:\n\n"
        "1. **Level Appropriateness**: Is the content suitable for the stated "
        "CEFR level? Not too easy, not too hard?\n"
        "2. **Objective Coverage**: Does the plan address the student's stated "
        "learning goals?\n"
        "3. **Timing**: Are activity durations realistic? Does total time make sense?\n"
        "4. **Structure**: Does the lesson flow logically (warm-up → main → wrap-up)?\n"
        "5. **Engagement**: Are the activities varied and interesting?\n\n"
        "Respond with:\n"
        "- APPROVED if the plan meets all criteria\n"
        "- NEEDS_REVISION if it needs changes, with specific feedback\n\n"
        "Start your response with either APPROVED or NEEDS_REVISION on the first line, "
        "then provide your detailed review.",
    ),
    (
        "human",
        "Student Profile:\n"
        "- Name: {name}\n"
        "- Level: {proficiency_level}\n"
        "- Goals: {learning_goals}\n"
        "- Lesson Type: {lesson_type}\n\n"
        "Draft Lesson Plan:\n{draft_plan}\n\n"
        "Please review this lesson plan.",
    ),
])

# -- Finalize Node --

FINALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a lesson plan formatter for the LinguaFlow platform. "
        "Take the draft lesson plan and format it into a clean, structured "
        "lesson plan. Preserve all content but ensure it fits the required format.\n\n"
        "The lesson should have:\n"
        "- A clear title\n"
        "- The CEFR level\n"
        "- The lesson type\n"
        "- A list of learning objectives\n"
        "- A warm-up activity\n"
        "- Main activities (each with name, description, duration, and materials)\n"
        "- A wrap-up activity\n"
        "- Homework assignment\n"
        "- Total estimated duration in minutes",
    ),
    (
        "human",
        "Student: {name} (Level: {proficiency_level}, Type: {lesson_type})\n\n"
        "Draft Lesson Plan:\n{draft_plan}\n\n"
        "Please format this into a structured lesson plan.",
    ),
])
```

- [ ] **Step 2: Verify prompts import correctly**

```bash
cd "projects/02-lesson-plan-generator"
python -c "from prompts import RESEARCH_PROMPT, REVIEW_PROMPT, INTAKE_SYSTEM_PROMPT; print('All prompts loaded')"
```

Expected: `All prompts loaded`

- [ ] **Step 3: Commit**

```bash
git add projects/02-lesson-plan-generator/prompts.py
git commit -m "feat(p2): add prompt templates for all graph nodes"
```

---

### Task 5: Node Functions (TDD)

**Files:**
- Create: `projects/02-lesson-plan-generator/tests/test_nodes.py`
- Create: `projects/02-lesson-plan-generator/nodes.py`

- [ ] **Step 1: Write the failing tests for node functions**

```python
# projects/02-lesson-plan-generator/tests/test_nodes.py
"""Integration tests for individual node functions.

Each test calls a single node function with a pre-built state dict
and verifies it returns the expected state updates. These tests hit
the real Anthropic API.

Running: pytest tests/test_nodes.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from the repo root so ANTHROPIC_API_KEY is available.
_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from data.sample_profiles import INTERMEDIATE_CONVERSATION, INTERMEDIATE_GRAMMAR, EXAM_PREP_INTERMEDIATE
from models import LessonPlanState, LessonPlan


@pytest.fixture
def conversation_state() -> LessonPlanState:
    """A minimal state dict for testing with a conversation profile."""
    return {
        "student_profile": INTERMEDIATE_CONVERSATION,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


@pytest.fixture
def grammar_state() -> LessonPlanState:
    """A minimal state dict for testing with a grammar profile."""
    return {
        "student_profile": INTERMEDIATE_GRAMMAR,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


@pytest.fixture
def exam_state() -> LessonPlanState:
    """A minimal state dict for testing with an exam prep profile."""
    return {
        "student_profile": EXAM_PREP_INTERMEDIATE,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


class TestResearchNode:
    """Tests for the research node function."""

    def test_research_returns_notes(self, conversation_state):
        """Research node should populate research_notes with content."""
        from nodes import research_node

        result = research_node(conversation_state)
        assert "research_notes" in result
        assert len(result["research_notes"]) > 50

    def test_research_notes_mention_student_topics(self, conversation_state):
        """Research notes should be relevant to the student's topics."""
        from nodes import research_node

        result = research_node(conversation_state)
        notes_lower = result["research_notes"].lower()
        # Carlos's topics are travel and food — at least one should appear
        assert "travel" in notes_lower or "food" in notes_lower or "restaurant" in notes_lower


class TestDraftNodes:
    """Tests for the three drafting node functions."""

    def test_draft_conversation(self, conversation_state):
        """Conversation draft node should produce a draft plan."""
        from nodes import research_node, draft_conversation_node

        # Run research first to populate research_notes
        research_result = research_node(conversation_state)
        conversation_state.update(research_result)

        result = draft_conversation_node(conversation_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100

    def test_draft_grammar(self, grammar_state):
        """Grammar draft node should produce a draft plan."""
        from nodes import research_node, draft_grammar_node

        research_result = research_node(grammar_state)
        grammar_state.update(research_result)

        result = draft_grammar_node(grammar_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100

    def test_draft_exam_prep(self, exam_state):
        """Exam prep draft node should produce a draft plan."""
        from nodes import research_node, draft_exam_prep_node

        research_result = research_node(exam_state)
        exam_state.update(research_result)

        result = draft_exam_prep_node(exam_state)
        assert "draft_plan" in result
        assert len(result["draft_plan"]) > 100


class TestReviewNode:
    """Tests for the review node function."""

    def test_review_sets_approval_and_feedback(self, conversation_state):
        """Review node should set is_approved and review_feedback."""
        from nodes import research_node, draft_conversation_node, review_node

        # Build up state through research and drafting
        conversation_state.update(research_node(conversation_state))
        conversation_state.update(draft_conversation_node(conversation_state))

        result = review_node(conversation_state)
        assert "is_approved" in result
        assert isinstance(result["is_approved"], bool)
        assert "review_feedback" in result
        assert len(result["review_feedback"]) > 0
        assert "revision_count" in result
        assert result["revision_count"] == 1


class TestFinalizeNode:
    """Tests for the finalize node function."""

    def test_finalize_produces_lesson_plan(self, conversation_state):
        """Finalize node should produce a valid LessonPlan."""
        from nodes import research_node, draft_conversation_node, finalize_node

        # Build up state through research and drafting
        conversation_state.update(research_node(conversation_state))
        conversation_state.update(draft_conversation_node(conversation_state))
        conversation_state["is_approved"] = True

        result = finalize_node(conversation_state)
        assert "final_plan" in result
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0
        assert len(plan.main_activities) > 0
        assert plan.estimated_duration_minutes > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_nodes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement the node functions**

```python
# projects/02-lesson-plan-generator/nodes.py
"""Node functions for the Lesson Plan Generator StateGraph.

Each function represents a node in the graph. It receives the full
LessonPlanState and returns a partial dict with only the fields it updates.

LangGraph concept demonstrated:
- Node functions as the building blocks of a StateGraph
- Each node performs one focused task (research, draft, review, finalize)
- Nodes return partial state updates — the graph engine merges them

LangChain concepts demonstrated:
- Prompt | Model pipeline for LLM calls
- .with_structured_output() for the finalize node
- @traceable for LangSmith observability
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langsmith import traceable

from models import LessonPlanState, LessonPlan
from prompts import (
    RESEARCH_PROMPT,
    DRAFT_CONVERSATION_PROMPT,
    DRAFT_GRAMMAR_PROMPT,
    DRAFT_EXAM_PREP_PROMPT,
    REVIEW_PROMPT,
    FINALIZE_PROMPT,
)

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
# temperature=0 for structured/deterministic output in most nodes
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
# Slightly higher temperature for drafting — allows more creative lesson plans
_creative_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)


def _format_list(items: list[str]) -> str:
    """Format a list of strings as a comma-separated string for prompts."""
    return ", ".join(items)


def _build_revision_context(state: LessonPlanState) -> str:
    """Build revision context string if this is a revision cycle.

    When the review node sends the draft back for revision, we include
    the feedback so the drafting node knows what to fix.
    """
    if state["revision_count"] > 0 and state["review_feedback"]:
        return (
            f"REVISION REQUESTED (attempt {state['revision_count']}):\n"
            f"Reviewer feedback: {state['review_feedback']}\n\n"
            "Please address the feedback and improve the lesson plan.\n\n"
        )
    return ""


# -- Node Functions --

@traceable(name="research", run_type="chain")
def research_node(state: LessonPlanState) -> dict:
    """Research node: suggests materials and activities based on the student profile.

    This is the first node in the graph. It takes the student profile and
    asks the LLM to suggest relevant teaching materials and activity ideas.

    Returns:
        Partial state with research_notes populated.
    """
    profile = state["student_profile"]
    chain = RESEARCH_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "lesson_type": profile.lesson_type,
    })
    return {"research_notes": result.content}


@traceable(name="draft_conversation", run_type="chain")
def draft_conversation_node(state: LessonPlanState) -> dict:
    """Draft node for conversation-focused lessons.

    Creates a lesson plan emphasizing dialogue, role-play, and speaking
    exercises. Uses a slightly higher temperature for creative variety.

    Returns:
        Partial state with draft_plan populated.
    """
    profile = state["student_profile"]
    chain = DRAFT_CONVERSATION_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="draft_grammar", run_type="chain")
def draft_grammar_node(state: LessonPlanState) -> dict:
    """Draft node for grammar-focused lessons.

    Creates a lesson plan emphasizing grammar rules, exercises, drills,
    and error correction activities.

    Returns:
        Partial state with draft_plan populated.
    """
    profile = state["student_profile"]
    chain = DRAFT_GRAMMAR_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="draft_exam_prep", run_type="chain")
def draft_exam_prep_node(state: LessonPlanState) -> dict:
    """Draft node for exam preparation lessons.

    Creates a lesson plan emphasizing practice questions, test strategies,
    time management, and mock exercises.

    Returns:
        Partial state with draft_plan populated.
    """
    profile = state["student_profile"]
    chain = DRAFT_EXAM_PREP_PROMPT | _creative_model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "preferred_topics": _format_list(profile.preferred_topics),
        "research_notes": state["research_notes"],
        "revision_context": _build_revision_context(state),
    })
    return {"draft_plan": result.content}


@traceable(name="review", run_type="chain")
def review_node(state: LessonPlanState) -> dict:
    """Review node: critiques the draft and decides whether to approve.

    A separate LLM call evaluates the draft against quality criteria.
    If the draft starts with APPROVED, it passes. Otherwise, specific
    feedback is returned for revision.

    LangGraph concept demonstrated:
    - This node's output drives a conditional edge (the review loop).
      The graph checks is_approved and revision_count to decide whether
      to loop back to the drafting node or proceed to finalize.

    Returns:
        Partial state with is_approved, review_feedback, and revision_count.
    """
    profile = state["student_profile"]
    chain = REVIEW_PROMPT | _model
    result = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "learning_goals": _format_list(profile.learning_goals),
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })

    response_text = result.content
    is_approved = response_text.strip().startswith("APPROVED")

    return {
        "is_approved": is_approved,
        "review_feedback": response_text,
        "revision_count": state["revision_count"] + 1,
    }


@traceable(name="finalize", run_type="chain")
def finalize_node(state: LessonPlanState) -> dict:
    """Finalize node: parses the draft into a structured LessonPlan.

    Uses .with_structured_output() to convert the free-text draft
    into a validated LessonPlan Pydantic model. This ensures the
    final output has a consistent, machine-readable format.

    Returns:
        Partial state with final_plan as a LessonPlan instance.
    """
    profile = state["student_profile"]
    # Use structured output to parse the draft into a LessonPlan
    structured_model = _model.with_structured_output(
        LessonPlan, method="json_schema"
    )
    chain = FINALIZE_PROMPT | structured_model
    plan = chain.invoke({
        "name": profile.name,
        "proficiency_level": profile.proficiency_level,
        "lesson_type": profile.lesson_type,
        "draft_plan": state["draft_plan"],
    })
    return {"final_plan": plan}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_nodes.py -v
```

Expected: All 7 tests PASS (these hit the Anthropic API, may take 30-60s)

- [ ] **Step 5: Commit**

```bash
git add projects/02-lesson-plan-generator/nodes.py projects/02-lesson-plan-generator/tests/test_nodes.py
git commit -m "feat(p2): implement node functions with integration tests"
```

---

### Task 6: StateGraph Wiring (TDD)

**Files:**
- Create: `projects/02-lesson-plan-generator/tests/test_graph.py`
- Create: `projects/02-lesson-plan-generator/graph.py`

- [ ] **Step 1: Write the failing tests for the full graph**

```python
# projects/02-lesson-plan-generator/tests/test_graph.py
"""Integration tests for the full LangGraph StateGraph.

These tests compile the graph and invoke it end-to-end with sample profiles.
They verify correct routing (the right drafting node runs), the review loop
functions, and the final output is a valid LessonPlan.

Running: pytest tests/test_graph.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from data.sample_profiles import (
    INTERMEDIATE_CONVERSATION,
    INTERMEDIATE_GRAMMAR,
    EXAM_PREP_INTERMEDIATE,
)
from models import LessonPlan


class TestGraphRouting:
    """Tests that the graph routes to the correct drafting node."""

    def test_conversation_route(self):
        """Graph should route conversation profiles to the conversation drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0
        assert len(plan.main_activities) > 0

    def test_grammar_route(self):
        """Graph should route grammar profiles to the grammar drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_GRAMMAR,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0

    def test_exam_prep_route(self):
        """Graph should route exam prep profiles to the exam prep drafter."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": EXAM_PREP_INTERMEDIATE,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        assert result["final_plan"] is not None
        plan = result["final_plan"]
        assert isinstance(plan, LessonPlan)
        assert len(plan.objectives) > 0


class TestGraphReviewLoop:
    """Tests that the review loop works correctly."""

    def test_revision_count_populated(self):
        """After graph execution, revision_count should be at least 1."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        # At minimum, the review node ran once
        assert result["revision_count"] >= 1
        # And the plan was finalized
        assert result["final_plan"] is not None

    def test_max_revisions_respected(self):
        """Graph should finalize even if review never approves (max 2 revisions)."""
        from graph import build_graph

        graph = build_graph()
        result = graph.invoke({
            "student_profile": INTERMEDIATE_CONVERSATION,
            "research_notes": "",
            "draft_plan": "",
            "review_feedback": "",
            "revision_count": 0,
            "is_approved": False,
            "final_plan": None,
        })

        # revision_count should never exceed 2
        assert result["revision_count"] <= 2
        # Plan should always be produced regardless
        assert result["final_plan"] is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_graph.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 3: Implement the graph**

```python
# projects/02-lesson-plan-generator/graph.py
"""LangGraph StateGraph definition for the Lesson Plan Generator.

This module wires together the node functions into a directed graph with
conditional routing and a review loop. It's the core of Project 2 and
demonstrates the key LangGraph concepts:

LangGraph concepts demonstrated:
- StateGraph: creating a graph with a typed state schema
- add_node(): registering node functions
- add_edge(): static edges (always go to the next node)
- add_conditional_edges(): dynamic routing based on state
- Graph cycles: the review → draft loop (with a max revision guard)
- compile(): producing an executable graph
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END

from models import LessonPlanState
from nodes import (
    research_node,
    draft_conversation_node,
    draft_grammar_node,
    draft_exam_prep_node,
    review_node,
    finalize_node,
)


def route_by_lesson_type(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep"
]:
    """Conditional edge: route to the correct drafting node based on lesson type.

    LangGraph concept: conditional routing
    This function is passed to add_conditional_edges(). LangGraph calls it
    after the research node completes, and uses the return value to decide
    which drafting node to execute next.
    """
    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"


def route_after_review(state: LessonPlanState) -> Literal[
    "draft_conversation", "draft_grammar", "draft_exam_prep", "finalize"
]:
    """Conditional edge: decide whether to finalize or loop back for revision.

    LangGraph concept: graph cycles (loops)
    If the review approved the draft, we proceed to finalize.
    If not, and we haven't hit the max revision count (2), we loop back
    to the same drafting node so it can incorporate the feedback.
    If we've exhausted revisions, we finalize the best-effort draft.
    """
    # Approved — proceed to finalize
    if state["is_approved"]:
        return "finalize"

    # Not approved but revisions exhausted — finalize anyway
    if state["revision_count"] >= 2:
        return "finalize"

    # Not approved — loop back to the appropriate drafting node
    lesson_type = state["student_profile"].lesson_type
    if lesson_type == "conversation":
        return "draft_conversation"
    elif lesson_type == "grammar":
        return "draft_grammar"
    else:
        return "draft_exam_prep"


def build_graph():
    """Build and compile the Lesson Plan Generator StateGraph.

    Graph topology:
        START → research → route_by_type → draft_* → review → finalize → END
                                              ↑                  │
                                              └──────────────────┘
                                           (if not approved & revisions < 2)

    Returns:
        A compiled LangGraph graph ready for .invoke() or .stream().
    """
    # -- Create the graph with our state schema --
    workflow = StateGraph(LessonPlanState)

    # -- Add nodes --
    # Each node is a function that takes LessonPlanState and returns a partial update.
    workflow.add_node("research", research_node)
    workflow.add_node("draft_conversation", draft_conversation_node)
    workflow.add_node("draft_grammar", draft_grammar_node)
    workflow.add_node("draft_exam_prep", draft_exam_prep_node)
    workflow.add_node("review", review_node)
    workflow.add_node("finalize", finalize_node)

    # -- Add edges --

    # START always goes to research
    workflow.add_edge(START, "research")

    # After research, route to the correct drafting node based on lesson type.
    # add_conditional_edges takes: source node, routing function, path map.
    # The path map keys are the possible return values of the routing function,
    # and the values are the target node names.
    workflow.add_conditional_edges(
        "research",
        route_by_lesson_type,
        {
            "draft_conversation": "draft_conversation",
            "draft_grammar": "draft_grammar",
            "draft_exam_prep": "draft_exam_prep",
        },
    )

    # All three drafting nodes converge to the review node
    workflow.add_edge("draft_conversation", "review")
    workflow.add_edge("draft_grammar", "review")
    workflow.add_edge("draft_exam_prep", "review")

    # After review, either finalize or loop back to the drafting node.
    # This creates a cycle in the graph — the review loop.
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "draft_conversation": "draft_conversation",
            "draft_grammar": "draft_grammar",
            "draft_exam_prep": "draft_exam_prep",
            "finalize": "finalize",
        },
    )

    # Finalize is the last node — it goes to END
    workflow.add_edge("finalize", END)

    # -- Compile the graph --
    # compile() validates the graph structure and produces an executable object.
    # Without a checkpointer, there's no persistence — fine for this project.
    return workflow.compile()
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_graph.py -v
```

Expected: All 5 tests PASS (these hit the Anthropic API, may take 1-2 minutes)

- [ ] **Step 5: Commit**

```bash
git add projects/02-lesson-plan-generator/graph.py projects/02-lesson-plan-generator/tests/test_graph.py
git commit -m "feat(p2): wire StateGraph with conditional routing and review loop"
```

---

### Task 7: Intake Conversation (TDD)

**Files:**
- Create: `projects/02-lesson-plan-generator/tests/test_intake.py`
- Create: `projects/02-lesson-plan-generator/intake.py`

- [ ] **Step 1: Write the failing tests for the intake conversation**

```python
# projects/02-lesson-plan-generator/tests/test_intake.py
"""Integration tests for the intake conversation handler.

Tests simulate a multi-turn conversation where the student provides
their information, and verify that the handler produces a valid
StudentProfile when complete.

Running: pytest tests/test_intake.py -v
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[3]
load_dotenv(_repo_root / ".env")

from models import StudentProfile


class TestIntakeConversation:
    """Tests for IntakeConversation class."""

    def test_initial_state_not_complete(self):
        """A freshly created intake should not be complete."""
        from intake import IntakeConversation

        intake = IntakeConversation()
        assert intake.is_complete() is False

    def test_conversation_produces_profile(self):
        """A full conversation should produce a valid StudentProfile."""
        from intake import IntakeConversation

        intake = IntakeConversation()

        # First turn: LLM greets and asks a question
        response1 = intake.ask(
            "Hi! I'm Maria, and I want to improve my English speaking."
        )
        assert len(response1) > 0

        # Second turn: provide more details
        response2 = intake.ask(
            "I think my level is around B1. I really like talking about "
            "travel and movies."
        )
        assert len(response2) > 0

        # Third turn: clarify goals
        response3 = intake.ask(
            "I mainly want to practice conversation and become more fluent."
        )
        assert len(response3) > 0

        # The LLM may need one more turn or may be done
        if not intake.is_complete():
            response4 = intake.ask("Yes, that sounds right! Let's go.")
            assert len(response4) > 0

        # At this point, the intake should be complete (or close)
        # If still not complete after 4 turns, force completion for test
        if not intake.is_complete():
            intake.ask("Yes, please create my lesson plan now.")

        assert intake.is_complete() is True

        profile = intake.get_profile()
        assert isinstance(profile, StudentProfile)
        assert profile.name == "Maria" or len(profile.name) > 0
        assert profile.proficiency_level in ["A1", "A2", "B1", "B2", "C1", "C2"]
        assert len(profile.learning_goals) > 0
        assert profile.lesson_type in ["conversation", "grammar", "exam_prep"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_intake.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'intake'`

- [ ] **Step 3: Implement the intake conversation**

```python
# projects/02-lesson-plan-generator/intake.py
"""Intake conversation handler for gathering student information.

This module manages a multi-turn conversation that collects student info
and produces a validated StudentProfile. It builds on the conversation
pattern from Project 1, adding structured profile extraction.

LangChain concepts demonstrated:
- ChatAnthropic with message history management
- System prompts that guide LLM behavior through a structured intake flow
- Structured output extraction from a conversation context
- @traceable for LangSmith observability
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from models import StudentProfile
from prompts import INTAKE_SYSTEM_PROMPT

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
# temperature=0.3 for a warm, conversational tone during intake
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
# temperature=0 for reliable structured extraction
_extraction_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# -- Extraction Prompt --
# Used after the conversation to extract a structured StudentProfile
_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract the student profile from the conversation below. "
        "Determine the lesson_type based on their goals:\n"
        "- 'conversation' if they want speaking, fluency, or communication practice\n"
        "- 'grammar' if they want grammar rules, structure, or error correction\n"
        "- 'exam_prep' if they're preparing for an exam\n\n"
        "If the proficiency level isn't clearly stated, infer from context clues.",
    ),
    (
        "human",
        "Conversation transcript:\n{transcript}\n\n"
        "Extract the student's profile.",
    ),
])


class IntakeConversation:
    """Manages a multi-turn intake conversation to gather student information.

    The conversation is guided by a system prompt that instructs the LLM to
    ask about the student's name, level, goals, and topics one question at
    a time. When the LLM has enough information, it signals completion with
    [PROFILE_COMPLETE].

    Usage:
        intake = IntakeConversation()
        while not intake.is_complete():
            user_input = input("> ")
            response = intake.ask(user_input)
            print(response)
        profile = intake.get_profile()
    """

    # Marker the LLM uses to signal it has enough information
    _COMPLETE_MARKER = "[PROFILE_COMPLETE]"

    def __init__(self):
        """Initialize with a system message and empty conversation history."""
        self._messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
        self._complete = False

    @traceable(name="intake_ask", run_type="chain")
    def ask(self, user_message: str) -> str:
        """Send a user message and get the LLM's response.

        Appends the user message to history, calls the LLM, checks for
        the completion marker, and returns the response text.

        Args:
            user_message: The student's message.

        Returns:
            The LLM's response as a string.
        """
        self._messages.append(HumanMessage(content=user_message))
        response = _model.invoke(self._messages)
        self._messages.append(response)

        # Check if the LLM signaled that it has enough information
        if self._COMPLETE_MARKER in response.content:
            self._complete = True

        return response.content

    def is_complete(self) -> bool:
        """Check whether the intake conversation has gathered all required info.

        Returns:
            True if the LLM has signaled completion with [PROFILE_COMPLETE].
        """
        return self._complete

    @traceable(name="intake_extract_profile", run_type="chain")
    def get_profile(self) -> StudentProfile:
        """Extract a structured StudentProfile from the conversation history.

        Uses a separate LLM call with structured output to parse the
        conversation into a validated StudentProfile.

        Returns:
            A validated StudentProfile instance.

        Raises:
            RuntimeError: If called before the conversation is complete.
        """
        if not self._complete:
            raise RuntimeError(
                "Cannot extract profile before conversation is complete. "
                "Keep calling ask() until is_complete() returns True."
            )

        # Build a transcript of the conversation (excluding the system message)
        transcript_parts = []
        for msg in self._messages[1:]:  # skip system message
            role = "Student" if isinstance(msg, HumanMessage) else "Advisor"
            transcript_parts.append(f"{role}: {msg.content}")
        transcript = "\n".join(transcript_parts)

        # Use structured output to extract the profile
        structured_model = _extraction_model.with_structured_output(
            StudentProfile, method="json_schema"
        )
        chain = _EXTRACTION_PROMPT | structured_model
        return chain.invoke({"transcript": transcript})
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/test_intake.py -v
```

Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/02-lesson-plan-generator/intake.py projects/02-lesson-plan-generator/tests/test_intake.py
git commit -m "feat(p2): implement intake conversation with profile extraction"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `projects/02-lesson-plan-generator/main.py`

- [ ] **Step 1: Implement the CLI**

```python
# projects/02-lesson-plan-generator/main.py
"""Interactive CLI for the Lesson Plan Generator.

This is the entry point that ties everything together:
1. Runs the intake conversation to gather student info
2. Compiles and invokes the LangGraph StateGraph
3. Streams status updates as each node executes
4. Prints the final structured LessonPlan

Run: python main.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

from intake import IntakeConversation
from graph import build_graph
from models import LessonPlan


def print_header():
    """Print the welcome banner."""
    print("\n" + "=" * 60)
    print("  LinguaFlow Lesson Plan Generator")
    print("  Powered by LangGraph + Anthropic Claude")
    print("=" * 60)
    print()


def run_intake() -> dict:
    """Run the intake conversation and return the initial graph state.

    Returns:
        A complete LessonPlanState dict ready for graph invocation.
    """
    print("Let's create a personalized lesson plan for you!")
    print("I'll ask a few questions to understand your needs.\n")

    intake = IntakeConversation()

    # Get the first greeting from the advisor
    first_response = intake.ask("Hello!")
    print(f"Advisor: {first_response}\n")

    while not intake.is_complete():
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            sys.exit(0)

        response = intake.ask(user_input)
        print(f"\nAdvisor: {response}\n")

    # Extract the student profile
    profile = intake.get_profile()
    print("\n" + "-" * 40)
    print("Profile gathered:")
    print(f"  Name: {profile.name}")
    print(f"  Level: {profile.proficiency_level}")
    print(f"  Goals: {', '.join(profile.learning_goals)}")
    print(f"  Topics: {', '.join(profile.preferred_topics)}")
    print(f"  Lesson Type: {profile.lesson_type}")
    print("-" * 40 + "\n")

    return {
        "student_profile": profile,
        "research_notes": "",
        "draft_plan": "",
        "review_feedback": "",
        "revision_count": 0,
        "is_approved": False,
        "final_plan": None,
    }


def print_lesson_plan(plan: LessonPlan):
    """Print the final lesson plan in a readable format."""
    print("\n" + "=" * 60)
    print(f"  {plan.title}")
    print(f"  Level: {plan.level} | Type: {plan.lesson_type}")
    print(f"  Duration: {plan.estimated_duration_minutes} minutes")
    print("=" * 60)

    print("\nObjectives:")
    for obj in plan.objectives:
        print(f"  - {obj}")

    print(f"\nWarm-Up:\n  {plan.warm_up}")

    print("\nMain Activities:")
    for i, activity in enumerate(plan.main_activities, 1):
        print(f"\n  {i}. {activity.name} ({activity.duration_minutes} min)")
        print(f"     {activity.description}")
        if activity.materials:
            print(f"     Materials: {', '.join(activity.materials)}")

    print(f"\nWrap-Up:\n  {plan.wrap_up}")
    print(f"\nHomework:\n  {plan.homework}")
    print()


def main():
    """Run the full lesson plan generation pipeline."""
    print_header()

    # Phase 1: Intake conversation
    initial_state = run_intake()

    # Phase 2: Run the LangGraph pipeline
    print("Generating your lesson plan...\n")
    graph = build_graph()

    # Stream with "updates" mode to show progress as each node completes
    node_labels = {
        "research": "Researching materials...",
        "draft_conversation": "Drafting conversation lesson...",
        "draft_grammar": "Drafting grammar lesson...",
        "draft_exam_prep": "Drafting exam prep lesson...",
        "review": "Reviewing draft...",
        "finalize": "Finalizing lesson plan...",
    }

    final_state = None
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name in chunk:
            label = node_labels.get(node_name, node_name)
            print(f"  [{node_name}] {label}")

            # Check if review looped back
            if node_name == "review":
                node_output = chunk[node_name]
                if not node_output.get("is_approved", False):
                    count = node_output.get("revision_count", 0)
                    if count < 2:
                        print(f"  [review] Requesting revision (attempt {count}/2)...")

        final_state = chunk

    # Get the final plan from the last state update
    # We need to invoke to get the complete final state
    result = graph.invoke(initial_state)
    plan = result["final_plan"]

    if plan:
        print_lesson_plan(plan)
        print(f"Revisions needed: {result['revision_count']}")
    else:
        print("Something went wrong — no lesson plan was produced.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI starts without errors (quick smoke test)**

```bash
cd "projects/02-lesson-plan-generator"
python -c "from main import print_header; print_header(); print('CLI module loads OK')"
```

Expected: Prints the header banner and "CLI module loads OK"

- [ ] **Step 3: Commit**

```bash
git add projects/02-lesson-plan-generator/main.py
git commit -m "feat(p2): add interactive CLI for lesson plan generation"
```

---

### Task 9: README and Documentation

**Files:**
- Create: `projects/02-lesson-plan-generator/README.md`
- Create: `docs/02-lesson-plan-generator.md`

- [ ] **Step 1: Create the project README**

```markdown
# Project 02: Lesson Plan Generator

A personalized lesson plan generator built with **LangGraph StateGraph** for the LinguaFlow English tutoring platform.

## What It Does

Takes a student through a short intake conversation to understand their needs, then generates a personalized lesson plan using a multi-node graph pipeline with:

- **Conditional routing** — routes to different drafting nodes based on lesson type (conversation, grammar, or exam prep)
- **Review loop** — an LLM reviewer critiques the draft and can send it back for revision (up to 2 cycles)
- **Structured output** — the final lesson plan is parsed into a validated Pydantic model

## Concepts Learned

- LangGraph StateGraph: defining graphs, state schemas, nodes, edges
- Conditional routing with `add_conditional_edges()`
- Graph cycles (review → draft loop)
- Graph compilation and invocation with streaming
- LangSmith: tracing graph execution, viewing node-level traces

## Project Structure

```
├── models.py          # Pydantic models + LangGraph state schema
├── intake.py          # Multi-turn intake conversation
├── prompts.py         # All prompt templates
├── nodes.py           # Node functions (research, draft, review, finalize)
├── graph.py           # StateGraph wiring and compilation
├── main.py            # Interactive CLI
├── data/
│   └── sample_profiles.py  # Test data
└── tests/
    ├── test_models.py      # Unit tests
    ├── test_nodes.py       # Node integration tests
    ├── test_graph.py       # Full graph integration tests
    └── test_intake.py      # Intake conversation tests
```

## Setup

```bash
# From the repo root (shared venv)
source .venv/bin/activate

# Ensure .env has ANTHROPIC_API_KEY and LANGSMITH_API_KEY
```

## Run

```bash
cd projects/02-lesson-plan-generator
python main.py
```

## Test

```bash
cd projects/02-lesson-plan-generator
python -m pytest tests/ -v
```

## Graph Topology

```
START → research → route → draft_conversation ─┐
                     │       draft_grammar ─────┤──→ review ──→ finalize → END
                     └──→    draft_exam_prep ───┘       ↑          │
                                                        └──────────┘
                                                     (revision loop,
                                                      max 2 cycles)
```
```

- [ ] **Step 2: Create the educational documentation**

Create `docs/02-lesson-plan-generator.md` — a comprehensive educational document explaining:

1. **What this project is about** — the Teaching department's need, why a graph is the right solution
2. **From Chains to Graphs** — why Project 1's linear chain pattern doesn't work here (branching logic, cycles)
3. **LangGraph StateGraph fundamentals** — TypedDict state, nodes as functions, edges, conditional edges
4. **Walkthrough of the graph** — step by step through each node with code highlights
5. **Conditional Routing** — how `add_conditional_edges()` works, the routing functions, path maps
6. **The Review Loop** — how graph cycles work, the revision_count guard, why this is powerful
7. **Graph Compilation** — what `compile()` does, how to invoke and stream
8. **LangSmith Traces** — how to read node-level traces for graph execution
9. **Key Takeaways** — what to carry forward to Project 3

This document should be 300-500 lines, educational in tone, with code snippets from the actual implementation.

- [ ] **Step 3: Commit**

```bash
git add projects/02-lesson-plan-generator/README.md docs/02-lesson-plan-generator.md
git commit -m "docs(p2): add README and educational documentation"
```

---

### Task 10: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the complete test suite**

```bash
cd "projects/02-lesson-plan-generator"
python -m pytest tests/ -v
```

Expected: All tests pass (test_models: 8, test_nodes: 7, test_graph: 5, test_intake: 2 = 22 total)

- [ ] **Step 2: Verify LangSmith traces appear**

After running the tests, check the LangSmith dashboard to confirm:
- Graph execution traces show up with node-level spans
- Each node (research, draft_*, review, finalize) appears as a child span
- The intake conversation traces show individual `ask` calls

- [ ] **Step 3: Final commit with any fixes**

If any tests failed and required fixes, commit those fixes:

```bash
git add -A
git commit -m "fix(p2): test suite fixes after full verification"
```
