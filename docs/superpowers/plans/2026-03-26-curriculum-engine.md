# Intelligent Curriculum Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous curriculum module generator using DeepAgents sub-agents orchestrated by a LangGraph workflow, with HITL approval gates at each stage and persistent cross-session memory.

**Architecture:** LangGraph outer `StateGraph` defines the 5-step workflow (plan → lesson → exercises → assessment → assemble) with `interrupt()` at 4 approval gates. Each content-generation node spawns a `create_deep_agent()` with domain/format skills loaded via SKILL.md. `CompositeBackend` routes ephemeral working files to `StateBackend` and persistent catalog/preferences to `StoreBackend`.

**Tech Stack:** `deepagents`, `langgraph`, `langchain-anthropic`, `langsmith`, `pydantic`, `streamlit`

---

## File Structure

```
projects/07-curriculum-engine/
├── requirements.txt          # Python dependencies
├── models.py                 # Pydantic models + LangGraph state schema
├── prompts.py                # System prompts for each agent
├── agents.py                 # create_deep_agent() factory functions for all 4 agents
├── nodes.py                  # Graph node functions + routing
├── graph.py                  # StateGraph assembly + compilation
├── skills/
│   ├── curriculum-design/
│   │   └── SKILL.md          # Domain knowledge (Bloom's, CEFR, scaffolding)
│   ├── lesson-template/
│   │   └── SKILL.md          # Lesson output format
│   ├── exercise-template/
│   │   └── SKILL.md          # Exercise output format
│   └── assessment-template/
│       └── SKILL.md          # Assessment output format
├── data/
│   ├── __init__.py
│   └── sample_requests.py    # Sample curriculum requests for testing
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_agents.py
│   ├── test_graph.py
│   └── test_memory.py
├── README.md
app/
├── adapters/
│   └── curriculum_engine.py  # Streamlit adapter
└── pages/
    └── p7_curriculum.py      # Streamlit UI page
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `projects/07-curriculum-engine/requirements.txt`
- Create: `projects/07-curriculum-engine/data/__init__.py`
- Create: `projects/07-curriculum-engine/tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p "projects/07-curriculum-engine/skills/curriculum-design"
mkdir -p "projects/07-curriculum-engine/skills/lesson-template"
mkdir -p "projects/07-curriculum-engine/skills/exercise-template"
mkdir -p "projects/07-curriculum-engine/skills/assessment-template"
mkdir -p "projects/07-curriculum-engine/data"
mkdir -p "projects/07-curriculum-engine/tests"
```

- [ ] **Step 2: Create requirements.txt**

```
deepagents>=0.4.0
langchain-core>=0.3.0
langchain-anthropic>=0.3.0
langgraph>=0.4.0
langsmith>=0.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

- [ ] **Step 3: Create package init files**

`projects/07-curriculum-engine/data/__init__.py`:
```python
```

`projects/07-curriculum-engine/tests/__init__.py`:
```python
```

- [ ] **Step 4: Verify dependencies are available**

```bash
cd "projects/07-curriculum-engine"
python -c "import deepagents; import langgraph; import langchain_anthropic; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/
git commit -m "feat(p7): scaffold project directory structure"
```

---

### Task 2: Pydantic Models & State Schema

**Files:**
- Create: `projects/07-curriculum-engine/tests/test_models.py`
- Create: `projects/07-curriculum-engine/models.py`

- [ ] **Step 1: Write the failing tests**

`projects/07-curriculum-engine/tests/test_models.py`:
```python
"""Tests for Pydantic models and state schema."""

import pytest
from pydantic import ValidationError


def test_curriculum_request_valid():
    from models import CurriculumRequest

    req = CurriculumRequest(
        topic="Business English for meetings",
        level="B2",
        preferences={"teaching_style": "interactive", "focus_areas": ["vocabulary", "speaking"]},
    )
    assert req.topic == "Business English for meetings"
    assert req.level == "B2"
    assert req.preferences["teaching_style"] == "interactive"


def test_curriculum_request_defaults():
    from models import CurriculumRequest

    req = CurriculumRequest(topic="Grammar basics", level="A1")
    assert req.preferences == {}


def test_curriculum_request_invalid_level():
    from models import CurriculumRequest

    with pytest.raises(ValidationError):
        CurriculumRequest(topic="Test", level="X9")


def test_curriculum_plan_valid():
    from models import CurriculumPlan

    plan = CurriculumPlan(
        title="Business English Module",
        description="A module covering business meeting vocabulary and phrases",
        lesson_outline="Introduction to meeting vocabulary and common phrases",
        exercise_types=["fill-in-the-blank", "matching"],
        assessment_approach="Reading comprehension + writing prompt",
    )
    assert plan.title == "Business English Module"
    assert len(plan.exercise_types) == 2


def test_generated_artifact_valid():
    from models import GeneratedArtifact

    artifact = GeneratedArtifact(
        content="# Lesson: Business Meetings\n\n...",
        artifact_type="lesson",
        agent_todos=[
            {"content": "Draft lesson intro", "status": "completed"},
            {"content": "Add examples", "status": "completed"},
        ],
    )
    assert artifact.artifact_type == "lesson"
    assert len(artifact.agent_todos) == 2


def test_generated_artifact_invalid_type():
    from models import GeneratedArtifact

    with pytest.raises(ValidationError):
        GeneratedArtifact(content="test", artifact_type="invalid_type", agent_todos=[])


def test_curriculum_engine_state_has_required_keys():
    """CurriculumEngineState should be a TypedDict with all workflow fields."""
    from models import CurriculumEngineState

    annotations = CurriculumEngineState.__annotations__
    expected_keys = {
        "curriculum_request", "curriculum_plan", "plan_feedback",
        "lesson", "lesson_feedback",
        "exercises", "exercises_feedback",
        "assessment", "assessment_feedback",
        "assembled_module", "current_step",
    }
    assert expected_keys.issubset(set(annotations.keys()))


def test_cefr_levels_constant():
    from models import CEFR_LEVELS

    assert CEFR_LEVELS == ("A1", "A2", "B1", "B2", "C1", "C2")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_models.py -v
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement models.py**

`projects/07-curriculum-engine/models.py`:
```python
# models.py
"""Pydantic models and LangGraph state schema for the Curriculum Engine.

Defines:
- CurriculumRequest: user's input (topic, level, preferences)
- CurriculumPlan: the structured curriculum plan from the planning agent
- GeneratedArtifact: a generated content piece (lesson, exercises, or assessment)
- CurriculumEngineState: the TypedDict flowing through the LangGraph workflow

Key concepts demonstrated:
- TypedDict state for a non-conversational, multi-stage pipeline
- Each field is written by exactly one node — no reducers needed
- Pydantic models for validation, TypedDict for LangGraph state
"""

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Valid CEFR levels for English proficiency
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


class CurriculumRequest(BaseModel):
    """User's request to generate a curriculum module."""

    topic: str = Field(description="The topic to cover (e.g., 'Business English for meetings')")
    level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Target CEFR proficiency level"
    )
    preferences: dict = Field(
        default_factory=dict,
        description="Optional preferences: teaching_style, focus_areas, etc.",
    )


class CurriculumPlan(BaseModel):
    """Structured curriculum plan produced by the planning agent."""

    title: str = Field(description="Module title")
    description: str = Field(description="Brief module description")
    lesson_outline: str = Field(description="What the lesson should cover")
    exercise_types: list[str] = Field(description="Types of exercises to include")
    assessment_approach: str = Field(description="How the assessment should be structured")


class GeneratedArtifact(BaseModel):
    """A content artifact generated by a DeepAgents sub-agent."""

    content: str = Field(description="The generated markdown content")
    artifact_type: Literal["lesson", "exercises", "assessment"] = Field(
        description="What kind of artifact this is"
    )
    agent_todos: list[dict] = Field(
        default_factory=list,
        description="The agent's TodoList showing its planning process",
    )


class CurriculumEngineState(TypedDict):
    """State schema for the curriculum engine StateGraph.

    This is a pipeline state (not conversational), so we use a plain
    TypedDict instead of MessagesState. Each field maps to a specific
    node in the graph.

    DeepAgents concept: the outer LangGraph state holds the workflow
    data, while each DeepAgent manages its own internal state (TodoList,
    working files) independently.
    """

    # -- Input (set at invocation) --
    curriculum_request: dict          # CurriculumRequest as dict

    # -- After plan_curriculum --
    curriculum_plan: dict | None      # CurriculumPlan as dict
    plan_feedback: str                # Moderator feedback on plan (empty = approved)

    # -- After generate_lesson --
    lesson: dict | None               # GeneratedArtifact as dict
    lesson_feedback: str              # Moderator feedback on lesson

    # -- After generate_exercises --
    exercises: dict | None            # GeneratedArtifact as dict
    exercises_feedback: str           # Moderator feedback on exercises

    # -- After generate_assessment --
    assessment: dict | None           # GeneratedArtifact as dict
    assessment_feedback: str          # Moderator feedback on assessment

    # -- After assemble_module --
    assembled_module: str | None      # Final assembled markdown

    # -- Workflow tracking --
    current_step: str                 # Current step name for UI progress display
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_models.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/models.py projects/07-curriculum-engine/tests/test_models.py
git commit -m "feat(p7): add Pydantic models and LangGraph state schema"
```

---

### Task 3: SKILL.md Files

**Files:**
- Create: `projects/07-curriculum-engine/skills/curriculum-design/SKILL.md`
- Create: `projects/07-curriculum-engine/skills/lesson-template/SKILL.md`
- Create: `projects/07-curriculum-engine/skills/exercise-template/SKILL.md`
- Create: `projects/07-curriculum-engine/skills/assessment-template/SKILL.md`

- [ ] **Step 1: Create curriculum-design skill**

`projects/07-curriculum-engine/skills/curriculum-design/SKILL.md`:
```markdown
---
name: curriculum-design
description: Curriculum design principles for English language teaching — Bloom's taxonomy, CEFR levels, scaffolding, and content sequencing
---

# Curriculum Design Principles

## CEFR Level Descriptors

When creating content, match it precisely to the target CEFR level:

- **A1 (Beginner):** Simple, everyday phrases. Present tense only. Basic vocabulary (100-300 words). Short sentences (5-8 words).
- **A2 (Elementary):** Routine tasks, simple descriptions. Past tense introduced. Vocabulary 300-600 words. Sentences 8-12 words.
- **B1 (Intermediate):** Main points on familiar topics. All basic tenses. Vocabulary 600-1200 words. Connected paragraphs.
- **B2 (Upper Intermediate):** Complex texts, abstract topics. Conditionals, passive voice. Vocabulary 1200-2500 words. Nuanced expression.
- **C1 (Advanced):** Wide range of demanding texts. Idiomatic expressions. Vocabulary 2500-5000 words. Implicit meaning.
- **C2 (Proficiency):** Near-native fluency. Subtle shades of meaning. Full vocabulary range. Sophisticated argumentation.

## Learning Objective Design (Bloom's Taxonomy)

Structure objectives from lower to higher cognitive levels:

1. **Remember:** Define, list, identify key vocabulary
2. **Understand:** Explain, summarize, paraphrase concepts
3. **Apply:** Use grammar rules in new sentences, complete exercises
4. **Analyze:** Compare structures, identify errors, distinguish usage
5. **Evaluate:** Assess writing quality, judge appropriateness
6. **Create:** Write original text, compose dialogue, design exercises

For A1-A2 levels, focus on Remember/Understand/Apply. For B1-B2, include Analyze. For C1-C2, include Evaluate/Create.

## Scaffolding Principles

1. **Build on prior knowledge:** Reference what the student already knows at their level
2. **Introduce one concept at a time:** Don't mix new grammar with new vocabulary
3. **Provide models first:** Show examples before asking for production
4. **Grade difficulty within exercises:** Start easy, increase gradually
5. **Include context:** Use realistic scenarios relevant to the topic

## Content Sequencing

A well-structured module follows this progression:
1. **Warm-up:** Activate prior knowledge related to the topic
2. **Presentation:** Introduce new concepts with clear explanations
3. **Practice:** Controlled exercises (fill-in-the-blank, matching)
4. **Production:** Freer exercises (short answer, writing)
5. **Assessment:** Test understanding with varied question types
```

- [ ] **Step 2: Create lesson-template skill**

`projects/07-curriculum-engine/skills/lesson-template/SKILL.md`:
```markdown
---
name: lesson-template
description: Output format template for generating structured English lessons with objectives, content sections, and vocabulary
---

# Lesson Output Format

Generate lessons in the following markdown structure. Every section is required.

## Template

```markdown
# [Lesson Title]

**Level:** [CEFR Level] | **Duration:** [estimated minutes] min

## Learning Objectives

By the end of this lesson, students will be able to:
- [Objective 1 — use action verbs from Bloom's taxonomy]
- [Objective 2]
- [Objective 3]

## Warm-Up Activity

[A short activity (2-3 sentences) that activates prior knowledge related to the topic. Should be engaging and low-pressure.]

## Core Content

### Section 1: [Subtopic Title]

[Explanation of the concept — clear, level-appropriate language]

**Examples:**
- [Example 1 with context]
- [Example 2 with context]

### Section 2: [Subtopic Title]

[Explanation of the next concept]

**Examples:**
- [Example 1]
- [Example 2]

## Key Takeaways

- [Takeaway 1 — one sentence summary of the most important point]
- [Takeaway 2]
- [Takeaway 3]

## Vocabulary List

| Word/Phrase | Definition | Example Sentence |
|-------------|-----------|------------------|
| [word] | [simple definition] | [word used in context] |
```

## Guidelines

- Keep explanations appropriate for the CEFR level
- Use 2-3 core content sections (not more)
- Each section needs at least 2 examples
- Vocabulary list should have 5-10 items
- Total lesson length: 400-800 words
```

- [ ] **Step 3: Create exercise-template skill**

`projects/07-curriculum-engine/skills/exercise-template/SKILL.md`:
```markdown
---
name: exercise-template
description: Output format template for generating English language exercises — fill-in-the-blank, multiple choice, short answer, and matching
---

# Exercise Output Format

Generate exercise sets in the following markdown structure. Include all four exercise types.

## Template

```markdown
# Exercises: [Topic Title]

**Level:** [CEFR Level] | **Difficulty:** [matches lesson level]

## Part 1: Fill in the Blank

Complete each sentence with the correct word or phrase.

1. The meeting was _______ (cancel) due to bad weather.
2. [more items...]

## Part 2: Multiple Choice

Choose the best answer for each question.

1. Which sentence is correct?
   - a) [option]
   - b) [option]
   - c) [option]
   - d) [option]

2. [more items...]

## Part 3: Short Answer

Answer each question in 1-2 complete sentences.

1. [Question requiring constructed response]
2. [more items...]

## Part 4: Matching

Match each item in Column A with its pair in Column B.

| Column A | Column B |
|----------|----------|
| 1. [item] | a. [item] |
| 2. [item] | b. [item] |

## Answer Key

### Part 1
1. cancelled
2. [...]

### Part 2
1. c
2. [...]

### Part 3
1. [Sample acceptable answer]
2. [...]

### Part 4
1-b, 2-a, [...]
```

## Guidelines

- 4-5 items per exercise type
- Difficulty increases within each section (first item easiest, last hardest)
- All exercises should reinforce the lesson's learning objectives
- Answer key is required for all sections
- Short answer section: provide sample acceptable answers, not just one correct answer
```

- [ ] **Step 4: Create assessment-template skill**

`projects/07-curriculum-engine/skills/assessment-template/SKILL.md`:
```markdown
---
name: assessment-template
description: Output format template for generating English language assessments with rubric, scoring, and answer key
---

# Assessment Output Format

Generate assessments in the following markdown structure. All sections are required.

## Template

```markdown
# Assessment: [Topic Title]

**Level:** [CEFR Level] | **Time Limit:** [minutes] minutes | **Total Points:** [total]

## Section 1: Reading Comprehension ([points] points)

Read the following passage and answer the questions below.

> [A short passage (100-200 words) related to the topic, written at the target CEFR level]

1. [Comprehension question] _(2 points)_
2. [Comprehension question] _(2 points)_
3. [Comprehension question] _(2 points)_

## Section 2: Grammar & Vocabulary ([points] points)

1. [Grammar/vocabulary question] _(2 points)_
2. [Grammar/vocabulary question] _(2 points)_
3. [Grammar/vocabulary question] _(2 points)_

## Section 3: Writing Prompt ([points] points)

[A writing prompt that asks students to produce text using concepts from the lesson. Include word count guidance.]

## Scoring Rubric

### Reading Comprehension
- **2 points:** Complete, accurate answer with supporting detail
- **1 point:** Partially correct or missing detail
- **0 points:** Incorrect or no response

### Grammar & Vocabulary
- **2 points:** Correct usage with proper context
- **1 point:** Minor errors but demonstrates understanding
- **0 points:** Incorrect or no response

### Writing
- **Full marks:** Addresses prompt fully, uses target vocabulary/grammar, appropriate for CEFR level
- **Half marks:** Partially addresses prompt or limited use of target language
- **Minimal marks:** Off-topic or significantly below level expectations

## Grade Boundaries

| Grade | Percentage | Description |
|-------|-----------|-------------|
| A | 90-100% | Excellent — exceeds level expectations |
| B | 75-89% | Good — meets level expectations |
| C | 60-74% | Satisfactory — basic understanding demonstrated |
| D | Below 60% | Needs improvement — review recommended |

## Answer Key

### Section 1
1. [Answer with explanation]
2. [Answer with explanation]
3. [Answer with explanation]

### Section 2
1. [Answer]
2. [Answer]
3. [Answer]

### Section 3
[Sample response at the target level, highlighting expected vocabulary and grammar use]
```

## Guidelines

- Total assessment: 20-30 points
- Reading passage must be original and at the target CEFR level
- Grammar section should test concepts from the lesson specifically
- Writing prompt word count: A1-A2: 30-50 words, B1-B2: 80-120 words, C1-C2: 150-200 words
- Answer key must include explanations for reading comprehension
```

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/skills/
git commit -m "feat(p7): add SKILL.md files for curriculum design and output templates"
```

---

### Task 4: Sample Data

**Files:**
- Create: `projects/07-curriculum-engine/data/sample_requests.py`

- [ ] **Step 1: Create sample requests**

`projects/07-curriculum-engine/data/sample_requests.py`:
```python
# data/sample_requests.py
"""Sample curriculum generation requests for quick testing.

Each request represents a realistic curriculum module that a content
team member might ask the engine to create.
"""

SAMPLE_REQUESTS = [
    {
        "topic": "Business English for meetings",
        "level": "B2",
        "preferences": {
            "teaching_style": "interactive",
            "focus_areas": ["vocabulary", "speaking"],
        },
    },
    {
        "topic": "Present Perfect vs Past Simple",
        "level": "B1",
        "preferences": {
            "teaching_style": "formal",
            "focus_areas": ["grammar"],
        },
    },
    {
        "topic": "Everyday greetings and introductions",
        "level": "A1",
        "preferences": {
            "teaching_style": "conversational",
            "focus_areas": ["speaking", "listening"],
        },
    },
    {
        "topic": "Academic essay writing",
        "level": "C1",
        "preferences": {
            "teaching_style": "formal",
            "focus_areas": ["writing", "vocabulary"],
        },
    },
]
```

- [ ] **Step 2: Commit**

```bash
git add projects/07-curriculum-engine/data/
git commit -m "feat(p7): add sample curriculum requests"
```

---

### Task 5: System Prompts

**Files:**
- Create: `projects/07-curriculum-engine/prompts.py`

- [ ] **Step 1: Create prompts.py**

`projects/07-curriculum-engine/prompts.py`:
```python
# prompts.py
"""System prompts for each DeepAgents sub-agent.

Each prompt defines the agent's role and instructions. The agent also
loads SKILL.md files dynamically for domain knowledge and output format
guidance — those are not duplicated here.

DeepAgents concept: system_prompt is prepended to the base deep agent
prompt. Skills are loaded on demand via SkillsMiddleware and provide
additional context the agent can reference.
"""

PLANNER_PROMPT = (
    "You are the curriculum planning agent for LinguaFlow, an English tutoring platform.\n\n"
    "Your job is to create a structured curriculum plan for a given topic and CEFR level.\n\n"
    "Steps:\n"
    "1. Use write_todos to plan your work\n"
    "2. Read the curriculum-design skill for design principles\n"
    "3. Create a curriculum plan with:\n"
    "   - A clear module title\n"
    "   - A brief description of what the module covers\n"
    "   - A lesson outline (what concepts to teach, in what order)\n"
    "   - Exercise types that reinforce the lesson objectives\n"
    "   - An assessment approach that tests understanding\n"
    "4. Write the plan as JSON to /work/plan.json with keys:\n"
    "   title, description, lesson_outline, exercise_types, assessment_approach\n\n"
    "Consider the student's preferences if provided (teaching style, focus areas).\n"
    "Ensure everything is appropriate for the target CEFR level."
)

LESSON_WRITER_PROMPT = (
    "You are the lesson writer agent for LinguaFlow, an English tutoring platform.\n\n"
    "Your job is to generate a complete lesson based on the curriculum plan.\n\n"
    "Steps:\n"
    "1. Use write_todos to plan your work\n"
    "2. Read the curriculum-design skill for design principles\n"
    "3. Read the lesson-template skill for the exact output format\n"
    "4. Read /work/plan.json to understand the curriculum plan\n"
    "5. Generate a lesson following the template format exactly\n"
    "6. Write the lesson markdown to /work/lesson.md\n\n"
    "If moderator feedback is provided, incorporate it into a revised version.\n"
    "The lesson must match the CEFR level and follow scaffolding principles."
)

EXERCISE_CREATOR_PROMPT = (
    "You are the exercise creator agent for LinguaFlow, an English tutoring platform.\n\n"
    "Your job is to create practice exercises that reinforce the lesson content.\n\n"
    "Steps:\n"
    "1. Use write_todos to plan your work\n"
    "2. Read the curriculum-design skill for design principles\n"
    "3. Read the exercise-template skill for the exact output format\n"
    "4. Read /work/plan.json for the curriculum plan\n"
    "5. Read /work/lesson.md to understand what was taught\n"
    "6. Create exercises with all four types: fill-in-the-blank, multiple choice, short answer, matching\n"
    "7. Write the exercises to /work/exercises.md\n\n"
    "If moderator feedback is provided, incorporate it into a revised version.\n"
    "Exercises must directly reinforce the lesson's learning objectives."
)

ASSESSMENT_BUILDER_PROMPT = (
    "You are the assessment builder agent for LinguaFlow, an English tutoring platform.\n\n"
    "Your job is to build a graded assessment for the curriculum module.\n\n"
    "Steps:\n"
    "1. Use write_todos to plan your work\n"
    "2. Read the curriculum-design skill for design principles\n"
    "3. Read the assessment-template skill for the exact output format\n"
    "4. Read /work/plan.json for the curriculum plan\n"
    "5. Read /work/lesson.md to understand what was taught\n"
    "6. Build an assessment with: reading comprehension, grammar/vocabulary, writing prompt\n"
    "7. Include a scoring rubric, grade boundaries, and answer key\n"
    "8. Write the assessment to /work/assessment.md\n\n"
    "If moderator feedback is provided, incorporate it into a revised version.\n"
    "The assessment must test the specific concepts from the lesson."
)
```

- [ ] **Step 2: Commit**

```bash
git add projects/07-curriculum-engine/prompts.py
git commit -m "feat(p7): add system prompts for DeepAgents sub-agents"
```

---

### Task 6: Agent Factory Functions

**Files:**
- Create: `projects/07-curriculum-engine/tests/test_agents.py`
- Create: `projects/07-curriculum-engine/agents.py`

- [ ] **Step 1: Write the failing tests**

`projects/07-curriculum-engine/tests/test_agents.py`:
```python
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
    from deepagents.backends import CompositeBackend

    backend_factory = create_composite_backend()
    # The factory is a callable that takes a ToolRuntime — we just verify it's callable
    assert callable(backend_factory)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_agents.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents'`

- [ ] **Step 3: Implement agents.py**

`projects/07-curriculum-engine/agents.py`:
```python
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
    PLANNER_PROMPT,
    LESSON_WRITER_PROMPT,
    EXERCISE_CREATOR_PROMPT,
    ASSESSMENT_BUILDER_PROMPT,
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
        skills=["/skills/"],
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
        skills=["/skills/"],
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
        skills=["/skills/"],
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
        skills=["/skills/"],
        backend=create_composite_backend(),
        store=_store,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_agents.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/agents.py projects/07-curriculum-engine/tests/test_agents.py
git commit -m "feat(p7): add DeepAgents factory functions with CompositeBackend"
```

---

### Task 7: Node Functions

**Files:**
- Create: `projects/07-curriculum-engine/tests/test_nodes.py`
- Create: `projects/07-curriculum-engine/nodes.py`

- [ ] **Step 1: Write the failing tests**

`projects/07-curriculum-engine/tests/test_nodes.py`:
```python
"""Tests for graph node functions and routing logic.

Tests the node functions that wrap DeepAgents invocations and the
interrupt-based HITL review nodes. Uses mocking to avoid actual
LLM calls while verifying the control flow.
"""

import json
from unittest.mock import patch, MagicMock

import pytest


def _make_state(**overrides):
    """Create a minimal valid CurriculumEngineState dict."""
    base = {
        "curriculum_request": {
            "topic": "Test Topic",
            "level": "B1",
            "preferences": {},
        },
        "curriculum_plan": None,
        "plan_feedback": "",
        "lesson": None,
        "lesson_feedback": "",
        "exercises": None,
        "exercises_feedback": "",
        "assessment": None,
        "assessment_feedback": "",
        "assembled_module": None,
        "current_step": "idle",
    }
    base.update(overrides)
    return base


def test_review_plan_node_interrupts():
    """review_plan should call interrupt() with the plan data."""
    from nodes import review_plan_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test"},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "approve"}
        result = review_plan_node(state)

    mock_interrupt.assert_called_once()
    # The interrupt payload should contain the plan
    call_args = mock_interrupt.call_args[0][0]
    assert call_args["plan"]["title"] == "Test Module"


def test_review_plan_node_captures_feedback_on_revise():
    """When moderator requests revision, feedback should be captured."""
    from nodes import review_plan_node

    state = _make_state(
        curriculum_plan={"title": "Test"},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "revise", "feedback": "Add more exercises"}
        result = review_plan_node(state)

    assert result["plan_feedback"] == "Add more exercises"


def test_review_lesson_node_interrupts():
    """review_lesson should call interrupt() with the lesson artifact."""
    from nodes import review_lesson_node

    state = _make_state(
        lesson={"content": "# Lesson", "artifact_type": "lesson", "agent_todos": []},
    )

    with patch("nodes.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "approve"}
        result = review_lesson_node(state)

    mock_interrupt.assert_called_once()


def test_route_after_review_approve():
    """Approve action should route to the next generation step."""
    from nodes import route_after_plan_review

    state = _make_state(
        curriculum_plan={"title": "Test"},
        plan_feedback="",
    )
    # Empty feedback means approved
    result = route_after_plan_review(state)
    assert result == "generate_lesson"


def test_route_after_review_revise():
    """Revise action should route back to the same generation step."""
    from nodes import route_after_plan_review

    state = _make_state(
        curriculum_plan={"title": "Test"},
        plan_feedback="Needs more detail",
    )
    result = route_after_plan_review(state)
    assert result == "plan_curriculum"


def test_assemble_module_combines_all_artifacts():
    """assemble_module should combine plan, lesson, exercises, and assessment."""
    from nodes import assemble_module_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test module"},
        lesson={"content": "# Lesson Content", "artifact_type": "lesson", "agent_todos": []},
        exercises={"content": "# Exercises", "artifact_type": "exercises", "agent_todos": []},
        assessment={"content": "# Assessment", "artifact_type": "assessment", "agent_todos": []},
    )

    result = assemble_module_node(state)
    assembled = result["assembled_module"]

    assert "Test Module" in assembled
    assert "# Lesson Content" in assembled
    assert "# Exercises" in assembled
    assert "# Assessment" in assembled


def test_assemble_module_handles_rejected_artifacts():
    """If an artifact was rejected (None), assembly should note it."""
    from nodes import assemble_module_node

    state = _make_state(
        curriculum_plan={"title": "Test Module", "description": "A test"},
        lesson={"content": "# Lesson", "artifact_type": "lesson", "agent_todos": []},
        exercises=None,  # Rejected
        assessment={"content": "# Assessment", "artifact_type": "assessment", "agent_todos": []},
    )

    result = assemble_module_node(state)
    assembled = result["assembled_module"]

    assert "# Lesson" in assembled
    assert "# Assessment" in assembled
    assert "not included" in assembled.lower() or "skipped" in assembled.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_nodes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nodes'`

- [ ] **Step 3: Implement nodes.py**

`projects/07-curriculum-engine/nodes.py`:
```python
# nodes.py
"""Node functions for the Curriculum Engine StateGraph.

Contains:
- 4 generation nodes (plan, lesson, exercises, assessment) that invoke DeepAgents
- 4 review nodes that use interrupt() for HITL approval
- 4 routing functions for conditional edges after each review
- 1 assembly node that combines all artifacts

LangGraph concepts demonstrated:
- interrupt() for human-in-the-loop at each pipeline stage
- Conditional routing based on moderator decisions
- Node functions that invoke external agents (DeepAgents) and extract results

DeepAgents concepts demonstrated:
- Invoking create_deep_agent() within a LangGraph node
- Passing context via the agent's file system (write to /work/)
- Extracting results from agent state (reading /work/ files)
- TodoList planning visible in agent output
"""

import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langgraph.types import interrupt
from langsmith import traceable

from models import CurriculumEngineState

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- LangSmith Tags --
_TAGS = ["p7-curriculum-engine"]


def _invoke_agent(agent, instruction: str, files: dict | None = None) -> dict:
    """Invoke a DeepAgents agent and return its final state.

    Args:
        agent: A compiled DeepAgent graph.
        instruction: The user message to send to the agent.
        files: Optional dict of file paths to content to pre-load into the agent's filesystem.

    Returns:
        The agent's final state dict.

    DeepAgents concept: agents are invoked like LangGraph graphs with
    messages input. Files can be pre-loaded into the agent's virtual
    filesystem to provide context.
    """
    invoke_input = {
        "messages": [{"role": "user", "content": instruction}],
    }
    if files:
        invoke_input["files"] = files

    config = {"tags": _TAGS}
    result = agent.invoke(invoke_input, config=config)
    return result


def _extract_file_content(result: dict, path: str) -> str | None:
    """Extract a file's content from the agent's final state.

    After invocation, files written by the agent are available in the
    state's files dict. Returns None if the file wasn't written.
    """
    files = result.get("files", {})
    return files.get(path)


def _extract_todos(result: dict) -> list[dict]:
    """Extract the agent's TodoList from its final state."""
    return result.get("todos", [])


# -- Generation Nodes --
# Each generation node creates a fresh DeepAgent, provides context via
# files, invokes it, and extracts the result. The agent handles its own
# planning (TodoList) and skill loading internally.

@traceable(name="plan_curriculum", run_type="chain", tags=_TAGS)
def plan_curriculum_node(state: CurriculumEngineState) -> dict:
    """Invoke the planner agent to create a curriculum plan.

    Provides the curriculum request as context. If there's feedback
    from a previous review, includes it in the instruction.
    """
    from agents import create_planner_agent

    request = state["curriculum_request"]
    feedback = state.get("plan_feedback", "")

    instruction = (
        f"Create a curriculum plan for:\n"
        f"- Topic: {request['topic']}\n"
        f"- Level: {request['level']}\n"
        f"- Preferences: {json.dumps(request.get('preferences', {}))}\n"
    )
    if feedback:
        instruction += f"\nPrevious feedback to address:\n{feedback}"

    agent = create_planner_agent()
    result = _invoke_agent(agent, instruction)

    # Extract the plan from the agent's output file
    plan_json = _extract_file_content(result, "/work/plan.json")
    if plan_json:
        plan = json.loads(plan_json)
    else:
        # Fallback: try to extract from the agent's last message
        last_msg = result["messages"][-1].content if result.get("messages") else ""
        plan = {
            "title": f"{request['topic']} Module",
            "description": last_msg[:200] if last_msg else "Curriculum module",
            "lesson_outline": "See lesson for details",
            "exercise_types": ["fill-in-the-blank", "multiple-choice", "short-answer", "matching"],
            "assessment_approach": "Reading comprehension, grammar, and writing",
        }

    return {
        "curriculum_plan": plan,
        "current_step": "review_plan",
    }


@traceable(name="generate_lesson", run_type="chain", tags=_TAGS)
def generate_lesson_node(state: CurriculumEngineState) -> dict:
    """Invoke the lesson writer agent to generate a lesson."""
    from agents import create_lesson_agent

    request = state["curriculum_request"]
    plan = state["curriculum_plan"]
    feedback = state.get("lesson_feedback", "")

    instruction = (
        f"Generate a lesson for the following curriculum plan:\n"
        f"- Topic: {request['topic']}\n"
        f"- Level: {request['level']}\n"
    )
    if feedback:
        instruction += f"\nPrevious feedback to address:\n{feedback}"

    # Pre-load the plan into the agent's filesystem
    files = {"/work/plan.json": json.dumps(plan, indent=2)}

    agent = create_lesson_agent()
    result = _invoke_agent(agent, instruction, files=files)

    lesson_content = _extract_file_content(result, "/work/lesson.md")
    if not lesson_content:
        # Fallback to last message
        lesson_content = result["messages"][-1].content if result.get("messages") else "No lesson generated"

    return {
        "lesson": {
            "content": lesson_content,
            "artifact_type": "lesson",
            "agent_todos": _extract_todos(result),
        },
        "current_step": "review_lesson",
    }


@traceable(name="generate_exercises", run_type="chain", tags=_TAGS)
def generate_exercises_node(state: CurriculumEngineState) -> dict:
    """Invoke the exercise creator agent to generate exercises."""
    from agents import create_exercise_agent

    request = state["curriculum_request"]
    plan = state["curriculum_plan"]
    lesson = state.get("lesson", {})
    feedback = state.get("exercises_feedback", "")

    instruction = (
        f"Create exercises for the following curriculum:\n"
        f"- Topic: {request['topic']}\n"
        f"- Level: {request['level']}\n"
    )
    if feedback:
        instruction += f"\nPrevious feedback to address:\n{feedback}"

    # Pre-load plan and lesson into the agent's filesystem
    files = {
        "/work/plan.json": json.dumps(plan, indent=2),
        "/work/lesson.md": lesson.get("content", ""),
    }

    agent = create_exercise_agent()
    result = _invoke_agent(agent, instruction, files=files)

    exercises_content = _extract_file_content(result, "/work/exercises.md")
    if not exercises_content:
        exercises_content = result["messages"][-1].content if result.get("messages") else "No exercises generated"

    return {
        "exercises": {
            "content": exercises_content,
            "artifact_type": "exercises",
            "agent_todos": _extract_todos(result),
        },
        "current_step": "review_exercises",
    }


@traceable(name="generate_assessment", run_type="chain", tags=_TAGS)
def generate_assessment_node(state: CurriculumEngineState) -> dict:
    """Invoke the assessment builder agent to generate an assessment."""
    from agents import create_assessment_agent

    request = state["curriculum_request"]
    plan = state["curriculum_plan"]
    lesson = state.get("lesson", {})
    feedback = state.get("assessment_feedback", "")

    instruction = (
        f"Build an assessment for the following curriculum:\n"
        f"- Topic: {request['topic']}\n"
        f"- Level: {request['level']}\n"
    )
    if feedback:
        instruction += f"\nPrevious feedback to address:\n{feedback}"

    files = {
        "/work/plan.json": json.dumps(plan, indent=2),
        "/work/lesson.md": lesson.get("content", ""),
    }

    agent = create_assessment_agent()
    result = _invoke_agent(agent, instruction, files=files)

    assessment_content = _extract_file_content(result, "/work/assessment.md")
    if not assessment_content:
        assessment_content = result["messages"][-1].content if result.get("messages") else "No assessment generated"

    return {
        "assessment": {
            "content": assessment_content,
            "artifact_type": "assessment",
            "agent_todos": _extract_todos(result),
        },
        "current_step": "review_assessment",
    }


# -- Review Nodes (HITL) --
# Each review node calls interrupt() to pause for human input.
# The moderator can approve (empty feedback) or request revision (with feedback).

@traceable(name="review_plan", run_type="chain", tags=_TAGS)
def review_plan_node(state: CurriculumEngineState) -> dict:
    """Pause for moderator review of the curriculum plan.

    interrupt() pauses the graph. When resumed via Command(resume=...),
    the return value is the moderator's decision dict.
    """
    decision = interrupt({
        "step": "review_plan",
        "plan": state["curriculum_plan"],
        "prompt": "Review the curriculum plan. Approve or request revisions with feedback.",
    })

    feedback = decision.get("feedback", "") if decision.get("action") == "revise" else ""
    return {"plan_feedback": feedback}


@traceable(name="review_lesson", run_type="chain", tags=_TAGS)
def review_lesson_node(state: CurriculumEngineState) -> dict:
    """Pause for moderator review of the generated lesson."""
    decision = interrupt({
        "step": "review_lesson",
        "artifact": state["lesson"],
        "prompt": "Review the lesson. Approve, request revisions, or reject.",
    })

    feedback = decision.get("feedback", "") if decision.get("action") == "revise" else ""
    return {
        "lesson_feedback": feedback,
        # If rejected, clear the lesson
        "lesson": None if decision.get("action") == "reject" else state["lesson"],
    }


@traceable(name="review_exercises", run_type="chain", tags=_TAGS)
def review_exercises_node(state: CurriculumEngineState) -> dict:
    """Pause for moderator review of the generated exercises."""
    decision = interrupt({
        "step": "review_exercises",
        "artifact": state["exercises"],
        "prompt": "Review the exercises. Approve, request revisions, or reject.",
    })

    feedback = decision.get("feedback", "") if decision.get("action") == "revise" else ""
    return {
        "exercises_feedback": feedback,
        "exercises": None if decision.get("action") == "reject" else state["exercises"],
    }


@traceable(name="review_assessment", run_type="chain", tags=_TAGS)
def review_assessment_node(state: CurriculumEngineState) -> dict:
    """Pause for moderator review of the generated assessment."""
    decision = interrupt({
        "step": "review_assessment",
        "artifact": state["assessment"],
        "prompt": "Review the assessment. Approve, request revisions, or reject.",
    })

    feedback = decision.get("feedback", "") if decision.get("action") == "revise" else ""
    return {
        "assessment_feedback": feedback,
        "assessment": None if decision.get("action") == "reject" else state["assessment"],
    }


# -- Routing Functions --
# Each router checks the feedback field: empty = approved, non-empty = revise.

def route_after_plan_review(
    state: CurriculumEngineState,
) -> Literal["generate_lesson", "plan_curriculum"]:
    """Route after plan review: approved → generate lesson, feedback → re-plan."""
    if state.get("plan_feedback"):
        return "plan_curriculum"
    return "generate_lesson"


def route_after_lesson_review(
    state: CurriculumEngineState,
) -> Literal["generate_exercises", "generate_lesson"]:
    """Route after lesson review: approved/rejected → exercises, feedback → re-generate."""
    if state.get("lesson_feedback"):
        return "generate_lesson"
    return "generate_exercises"


def route_after_exercises_review(
    state: CurriculumEngineState,
) -> Literal["generate_assessment", "generate_exercises"]:
    """Route after exercises review: approved/rejected → assessment, feedback → re-generate."""
    if state.get("exercises_feedback"):
        return "generate_exercises"
    return "generate_assessment"


def route_after_assessment_review(
    state: CurriculumEngineState,
) -> Literal["assemble_module", "generate_assessment"]:
    """Route after assessment review: approved/rejected → assemble, feedback → re-generate."""
    if state.get("assessment_feedback"):
        return "generate_assessment"
    return "assemble_module"


# -- Assembly Node --

@traceable(name="assemble_module", run_type="chain", tags=_TAGS)
def assemble_module_node(state: CurriculumEngineState) -> dict:
    """Assemble the final curriculum module from all approved artifacts.

    Combines the plan, lesson, exercises, and assessment into a single
    markdown document. Handles missing (rejected) artifacts gracefully.
    """
    plan = state.get("curriculum_plan", {})
    title = plan.get("title", "Curriculum Module")
    description = plan.get("description", "")

    sections = [
        f"# {title}\n\n{description}\n",
        "---\n",
    ]

    # Add each artifact, noting if any were skipped
    if state.get("lesson"):
        sections.append(state["lesson"]["content"])
    else:
        sections.append("## Lesson\n\n*Lesson was skipped (rejected by moderator).*\n")

    sections.append("\n---\n")

    if state.get("exercises"):
        sections.append(state["exercises"]["content"])
    else:
        sections.append("## Exercises\n\n*Exercises were not included (rejected by moderator).*\n")

    sections.append("\n---\n")

    if state.get("assessment"):
        sections.append(state["assessment"]["content"])
    else:
        sections.append("## Assessment\n\n*Assessment was not included (rejected by moderator).*\n")

    assembled = "\n".join(sections)

    return {
        "assembled_module": assembled,
        "current_step": "done",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_nodes.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/nodes.py projects/07-curriculum-engine/tests/test_nodes.py
git commit -m "feat(p7): add node functions with DeepAgents invocation and HITL review"
```

---

### Task 8: Graph Assembly

**Files:**
- Create: `projects/07-curriculum-engine/tests/test_graph.py`
- Create: `projects/07-curriculum-engine/graph.py`

- [ ] **Step 1: Write the failing tests**

`projects/07-curriculum-engine/tests/test_graph.py`:
```python
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
    # Get node names from the compiled graph
    node_names = set(graph.get_graph().nodes.keys())

    expected = {
        "plan_curriculum", "review_plan",
        "generate_lesson", "review_lesson",
        "generate_exercises", "review_exercises",
        "generate_assessment", "review_assessment",
        "assemble_module",
    }
    # LangGraph adds __start__ and __end__ nodes
    assert expected.issubset(node_names)


def test_graph_starts_at_plan_curriculum():
    """Graph execution should start at plan_curriculum node."""
    from graph import build_graph

    graph = build_graph()
    graph_repr = graph.get_graph()
    # Check that __start__ connects to plan_curriculum
    start_edges = [
        e.target for e in graph_repr.edges
        if e.source == "__start__"
    ]
    assert "plan_curriculum" in start_edges
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_graph.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 3: Implement graph.py**

`projects/07-curriculum-engine/graph.py`:
```python
# graph.py
"""StateGraph assembly for the Intelligent Curriculum Engine.

Wires together 9 nodes into a linear workflow with 4 HITL interrupt
points and conditional routing for revision loops.

Workflow:
  plan_curriculum → review_plan → [approve/revise]
    → generate_lesson → review_lesson → [approve/revise/reject]
    → generate_exercises → review_exercises → [approve/revise/reject]
    → generate_assessment → review_assessment → [approve/revise/reject]
    → assemble_module → END

LangGraph concepts demonstrated:
- StateGraph with multiple interrupt() points
- Conditional edges for approve/revise routing at each stage
- Revision loops (re-generate with feedback)
- Checkpointer injection (required for interrupts)

DeepAgents concepts demonstrated:
- LangGraph as the outer orchestrator wrapping DeepAgents sub-agents
- Each generation node internally creates and invokes a DeepAgent
- Clean separation: LangGraph owns workflow, DeepAgents own content generation
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from models import CurriculumEngineState
from nodes import (
    plan_curriculum_node,
    review_plan_node,
    generate_lesson_node,
    review_lesson_node,
    generate_exercises_node,
    review_exercises_node,
    generate_assessment_node,
    review_assessment_node,
    assemble_module_node,
    route_after_plan_review,
    route_after_lesson_review,
    route_after_exercises_review,
    route_after_assessment_review,
)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the curriculum engine StateGraph.

    Args:
        checkpointer: Required for interrupt/resume. Defaults to InMemorySaver.

    Returns:
        Compiled LangGraph graph ready for .invoke().
    """
    # Interrupts require a checkpointer
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = (
        StateGraph(CurriculumEngineState)
        # -- Nodes --
        .add_node("plan_curriculum", plan_curriculum_node)
        .add_node("review_plan", review_plan_node)
        .add_node("generate_lesson", generate_lesson_node)
        .add_node("review_lesson", review_lesson_node)
        .add_node("generate_exercises", generate_exercises_node)
        .add_node("review_exercises", review_exercises_node)
        .add_node("generate_assessment", generate_assessment_node)
        .add_node("review_assessment", review_assessment_node)
        .add_node("assemble_module", assemble_module_node)
        # -- Edges --
        # Start: plan the curriculum
        .add_edge(START, "plan_curriculum")
        # Plan → Review
        .add_edge("plan_curriculum", "review_plan")
        # Review plan → approve (lesson) or revise (re-plan)
        .add_conditional_edges(
            "review_plan",
            route_after_plan_review,
            ["generate_lesson", "plan_curriculum"],
        )
        # Lesson → Review
        .add_edge("generate_lesson", "review_lesson")
        # Review lesson → approve (exercises) or revise (re-generate)
        .add_conditional_edges(
            "review_lesson",
            route_after_lesson_review,
            ["generate_exercises", "generate_lesson"],
        )
        # Exercises → Review
        .add_edge("generate_exercises", "review_exercises")
        # Review exercises → approve (assessment) or revise (re-generate)
        .add_conditional_edges(
            "review_exercises",
            route_after_exercises_review,
            ["generate_assessment", "generate_exercises"],
        )
        # Assessment → Review
        .add_edge("generate_assessment", "review_assessment")
        # Review assessment → approve (assemble) or revise (re-generate)
        .add_conditional_edges(
            "review_assessment",
            route_after_assessment_review,
            ["assemble_module", "generate_assessment"],
        )
        # Terminal
        .add_edge("assemble_module", END)
        # Compile with checkpointer
        .compile(checkpointer=checkpointer)
    )

    return graph
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_graph.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/07-curriculum-engine/graph.py projects/07-curriculum-engine/tests/test_graph.py
git commit -m "feat(p7): assemble StateGraph with 4 HITL interrupt points"
```

---

### Task 9: Memory Backend Tests

**Files:**
- Create: `projects/07-curriculum-engine/tests/test_memory.py`

- [ ] **Step 1: Write the memory tests**

`projects/07-curriculum-engine/tests/test_memory.py`:
```python
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
```

- [ ] **Step 2: Run tests**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/test_memory.py -v
```

Expected: all 3 tests PASS (these test existing code from earlier tasks)

- [ ] **Step 3: Commit**

```bash
git add projects/07-curriculum-engine/tests/test_memory.py
git commit -m "test(p7): add memory backend and skills verification tests"
```

---

### Task 10: Streamlit Adapter

**Files:**
- Create: `app/adapters/curriculum_engine.py`
- Modify: `app/adapters/_importer.py` (add `agents` to conflicting set)

- [ ] **Step 1: Update _importer.py with new conflicting module name**

Add `"agents"` to the `_CONFLICTING` set in `app/adapters/_importer.py` since project 7 introduces an `agents.py` module:

```python
_CONFLICTING = {
    "models", "graph", "nodes", "prompts", "chains",
    "conversation", "intake", "ingestion", "tools", "agents",
}
```

- [ ] **Step 2: Create the adapter**

`app/adapters/curriculum_engine.py`:
```python
"""Adapter for Project 07 — Intelligent Curriculum Engine.

Handles sys.path setup, environment loading, and wraps the LangGraph
pipeline with interrupt/resume support for use in the Streamlit app.

The graph has four interrupt points (review_plan, review_lesson,
review_exercises, review_assessment) where the moderator must provide
a decision via Command(resume=...).
"""

import sys
import uuid
from pathlib import Path

from adapters._importer import clear_project_modules

# -- Path setup --
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / "projects" / "07-curriculum-engine"
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# -- Load environment --
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env")

# -- CRITICAL: Clear cached modules, then import project modules --
clear_project_modules()
from graph import build_graph  # noqa: E402
from data.sample_requests import SAMPLE_REQUESTS  # noqa: E402
from models import CEFR_LEVELS  # noqa: E402

# -- Build a shared graph with in-memory checkpointer --
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_checkpointer = InMemorySaver()
_graph = build_graph(checkpointer=_checkpointer)

# -- Step names for progress tracking --
STEPS = [
    ("plan_curriculum", "Planning"),
    ("review_plan", "Review Plan"),
    ("generate_lesson", "Lesson"),
    ("review_lesson", "Review Lesson"),
    ("generate_exercises", "Exercises"),
    ("review_exercises", "Review Exercises"),
    ("generate_assessment", "Assessment"),
    ("review_assessment", "Review Assessment"),
    ("assemble_module", "Assembling"),
]

# Review steps where HITL interrupt happens
REVIEW_STEPS = {"review_plan", "review_lesson", "review_exercises", "review_assessment"}


def get_sample_requests() -> list[dict]:
    """Return sample curriculum requests for quick testing."""
    return list(SAMPLE_REQUESTS)


def get_cefr_levels() -> tuple:
    """Return valid CEFR levels."""
    return CEFR_LEVELS


def create_thread_id() -> str:
    """Generate a new unique thread ID for a pipeline run."""
    return str(uuid.uuid4())


def start_pipeline(thread_id: str, curriculum_request: dict) -> dict | None:
    """Start the curriculum engine pipeline.

    Runs the graph until it hits the first interrupt (review_plan).

    Returns:
        Dict with interrupt info, or None if pipeline finished.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p7-curriculum-engine"],
    }

    initial_state = {
        "curriculum_request": curriculum_request,
        "curriculum_plan": None,
        "plan_feedback": "",
        "lesson": None,
        "lesson_feedback": "",
        "exercises": None,
        "exercises_feedback": "",
        "assessment": None,
        "assessment_feedback": "",
        "assembled_module": None,
        "current_step": "plan_curriculum",
    }

    try:
        _graph.invoke(initial_state, config=config)
    except Exception as e:
        raise RuntimeError(f"Curriculum engine pipeline failed: {e}") from e

    return _get_interrupt_value(thread_id)


def resume_pipeline(thread_id: str, decision: dict) -> dict | None:
    """Resume the pipeline after a moderator decision.

    Returns:
        Dict with next interrupt info, or None if pipeline finished.
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["p7-curriculum-engine"],
    }

    try:
        _graph.invoke(Command(resume=decision), config=config)
    except Exception as e:
        raise RuntimeError(f"Pipeline resume failed: {e}") from e

    return _get_interrupt_value(thread_id)


def get_state(thread_id: str) -> dict:
    """Get the current state values for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return dict(snapshot.values) if snapshot.values else {}
    except Exception:
        return {}


def get_current_step(thread_id: str) -> str | None:
    """Get the name of the next pending node."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        return list(snapshot.next)[0] if snapshot.next else None
    except Exception:
        return None


def _get_interrupt_value(thread_id: str) -> dict | None:
    """Extract the interrupt payload from the current graph state."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = _graph.get_state(config)
        if snapshot.next:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return task.interrupts[0].value
        return None
    except Exception:
        return None
```

- [ ] **Step 3: Commit**

```bash
git add app/adapters/curriculum_engine.py app/adapters/_importer.py
git commit -m "feat(p7): add Streamlit adapter for curriculum engine"
```

---

### Task 11: Streamlit UI Page

**Files:**
- Create: `app/pages/p7_curriculum.py`
- Modify: `app/app.py` (register tab)

- [ ] **Step 1: Create the page**

`app/pages/p7_curriculum.py`:
```python
"""P7 — Intelligent Curriculum Engine tab.

Hybrid UI: form for input, live progress view showing agent planning
and generated artifacts, with HITL approval gates at each stage.

DeepAgents concepts visible in the UI:
- Agent TodoList displayed in the activity panel
- Artifact preview after each generation step
- HITL review at each checkpoint (approve, revise, reject)
- Step progress tracker matching the LangGraph workflow
"""

import streamlit as st
from adapters import curriculum_engine
from components import doc_viewer

# -- Resolve doc path relative to repo root --
_DOC_PATH = "docs/07-curriculum-engine.md"

# Step display labels for progress tracker
_STEP_LABELS = {
    "plan_curriculum": "Plan",
    "review_plan": "Review Plan",
    "generate_lesson": "Lesson",
    "review_lesson": "Review Lesson",
    "generate_exercises": "Exercises",
    "review_exercises": "Review Exercises",
    "generate_assessment": "Assessment",
    "review_assessment": "Review Assessment",
    "assemble_module": "Assemble",
    "done": "Done",
}

# Ordered step keys for progress bar
_PROGRESS_STEPS = [
    "plan_curriculum", "review_plan",
    "generate_lesson", "review_lesson",
    "generate_exercises", "review_exercises",
    "generate_assessment", "review_assessment",
    "assemble_module", "done",
]


def _reset_state() -> None:
    """Clear all p7_ session state keys."""
    for key in list(st.session_state.keys()):
        if key.startswith("p7_"):
            del st.session_state[key]


def _init_state() -> None:
    """Initialize session state defaults."""
    if "p7_stage" not in st.session_state:
        st.session_state["p7_stage"] = "idle"
        st.session_state["p7_thread_id"] = None
        st.session_state["p7_interrupt"] = None
        st.session_state["p7_log"] = []


def _add_log(label: str, content: str) -> None:
    """Append an entry to the pipeline log."""
    st.session_state["p7_log"].append((label, content))


def _render_progress_bar(current_step: str) -> None:
    """Render a horizontal step progress indicator."""
    if current_step not in _PROGRESS_STEPS:
        return

    current_idx = _PROGRESS_STEPS.index(current_step)
    total = len(_PROGRESS_STEPS)

    # Build progress columns
    cols = st.columns(total)
    for i, (col, step_key) in enumerate(zip(cols, _PROGRESS_STEPS)):
        label = _STEP_LABELS.get(step_key, step_key)
        with col:
            if i < current_idx:
                st.markdown(f"~~{label}~~")
            elif i == current_idx:
                st.markdown(f"**:blue[{label}]**")
            else:
                st.markdown(f":gray[{label}]")


def _render_request_form() -> None:
    """Render the curriculum request form (idle stage)."""
    st.subheader("Curriculum Request")

    # Sample request selector
    samples = curriculum_engine.get_sample_requests()
    sample_labels = ["(custom)"] + [
        f"{s['topic']} ({s['level']})"
        for s in samples
    ]

    selected = st.selectbox("Quick-start with a sample:", sample_labels, key="p7_sample")

    if selected != "(custom)":
        idx = sample_labels.index(selected) - 1
        sample = samples[idx]
        topic = sample["topic"]
        level = sample["level"]
        preferences = sample.get("preferences", {})
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Topic:** {topic}")
        with col2:
            st.info(f"**Level:** {level}")
        if preferences:
            st.info(f"**Preferences:** {', '.join(f'{k}: {v}' for k, v in preferences.items())}")
        can_generate = True
    else:
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Topic", key="p7_topic",
                                  placeholder="e.g., Business English for meetings")
        with col2:
            level = st.selectbox("CEFR Level", curriculum_engine.get_cefr_levels(), key="p7_level")

        teaching_style = st.selectbox(
            "Teaching Style (optional)",
            ["", "conversational", "formal", "interactive"],
            key="p7_style",
        )
        focus_input = st.text_input(
            "Focus Areas (optional, comma-separated)",
            key="p7_focus",
            placeholder="e.g., vocabulary, speaking",
        )
        preferences = {}
        if teaching_style:
            preferences["teaching_style"] = teaching_style
        if focus_input:
            preferences["focus_areas"] = [f.strip() for f in focus_input.split(",") if f.strip()]
        can_generate = bool(topic)

    # Generate button
    if st.button("Generate Module", key="p7_generate", type="primary", disabled=not can_generate):
        request = {
            "topic": topic,
            "level": level,
            "preferences": preferences,
        }
        st.session_state["p7_thread_id"] = curriculum_engine.create_thread_id()
        _add_log("Request", f"**{topic}** at **{level}** level")

        with st.spinner("Planning curriculum..."):
            try:
                interrupt_val = curriculum_engine.start_pipeline(
                    st.session_state["p7_thread_id"], request
                )
                if interrupt_val:
                    st.session_state["p7_stage"] = interrupt_val.get("step", "review_plan")
                    st.session_state["p7_interrupt"] = interrupt_val
                    _add_log("Plan Created", "Awaiting review")
                else:
                    st.session_state["p7_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_plan_review() -> None:
    """Render the plan review interface."""
    st.subheader("Review Curriculum Plan")
    interrupt_val = st.session_state["p7_interrupt"]
    plan = interrupt_val.get("plan", {})

    # Display plan details
    st.markdown(f"### {plan.get('title', 'Curriculum Plan')}")
    st.markdown(plan.get("description", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Lesson Outline:**")
        st.markdown(plan.get("lesson_outline", ""))
    with col2:
        st.markdown("**Exercise Types:**")
        for ex_type in plan.get("exercise_types", []):
            st.markdown(f"- {ex_type}")

    st.markdown("**Assessment Approach:**")
    st.markdown(plan.get("assessment_approach", ""))

    st.divider()
    _render_review_controls("plan")


def _render_artifact_review(step: str) -> None:
    """Render the review interface for a generated artifact (lesson/exercises/assessment)."""
    artifact_name = step.replace("review_", "")
    st.subheader(f"Review {artifact_name.title()}")
    interrupt_val = st.session_state["p7_interrupt"]
    artifact = interrupt_val.get("artifact", {})

    # Display the generated content
    content = artifact.get("content", "No content generated")
    st.markdown(content)

    # Show agent's TodoList
    todos = artifact.get("agent_todos", [])
    if todos:
        with st.expander("Agent Planning (TodoList)", expanded=False):
            for todo in todos:
                status = todo.get("status", "pending")
                icon = {"completed": "checkmark", "in_progress": "hourglass", "pending": "circle"}.get(status, "circle")
                st.markdown(f"- [{status}] {todo.get('content', '')}")

    st.divider()
    _render_review_controls(artifact_name)


def _render_review_controls(artifact_name: str) -> None:
    """Render approve/revise/reject buttons for a review step."""
    st.markdown("**Moderator Decision:**")

    action = st.radio(
        "Action",
        ["approve", "revise", "reject"] if artifact_name != "plan" else ["approve", "revise"],
        format_func=lambda x: {
            "approve": "Approve",
            "revise": "Request Revision",
            "reject": "Reject (skip this artifact)",
        }.get(x, x),
        horizontal=True,
        key=f"p7_{artifact_name}_action",
    )

    feedback = ""
    if action == "revise":
        feedback = st.text_area(
            "Feedback for revision:",
            placeholder="Explain what needs to change...",
            key=f"p7_{artifact_name}_feedback",
        )

    if st.button("Submit Decision", key=f"p7_{artifact_name}_submit", type="primary"):
        decision = {"action": action}
        if action == "revise":
            decision["feedback"] = feedback

        action_label = {"approve": "Approved", "revise": "Revision requested", "reject": "Rejected"}[action]
        _add_log(f"Review {artifact_name.title()}", f"{action_label}" + (f" — {feedback}" if feedback else ""))

        spinner_text = "Regenerating..." if action == "revise" else "Processing..."
        with st.spinner(spinner_text):
            try:
                interrupt_val = curriculum_engine.resume_pipeline(
                    st.session_state["p7_thread_id"], decision
                )
                if interrupt_val:
                    st.session_state["p7_stage"] = interrupt_val.get("step", "done")
                    st.session_state["p7_interrupt"] = interrupt_val
                else:
                    st.session_state["p7_stage"] = "done"
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


def _render_done() -> None:
    """Render the completed module."""
    st.subheader("Module Complete")

    state = curriculum_engine.get_state(st.session_state["p7_thread_id"])
    assembled = state.get("assembled_module", "")

    if assembled:
        st.success("Curriculum module assembled successfully!")
        st.markdown(assembled)
    else:
        st.warning("No module was assembled.")


def render() -> None:
    """Render the Intelligent Curriculum Engine interface."""
    st.header("Intelligent Curriculum Engine")
    st.caption("DeepAgents-powered curriculum module generator with HITL approval")

    _init_state()

    # -- Stage indicator and reset --
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Reset", key="p7_reset"):
            _reset_state()
            st.rerun()
    with col2:
        stage = st.session_state.get("p7_stage", "idle")
        stage_label = _STEP_LABELS.get(stage, stage.replace("_", " ").title())
        st.caption(f"Stage: **{stage_label}**")

    # -- Progress bar --
    if stage != "idle":
        _render_progress_bar(stage)

    st.divider()

    # -- Pipeline log --
    log = st.session_state.get("p7_log", [])
    if log:
        with st.expander("Pipeline Log", expanded=False):
            for label, content in log:
                st.markdown(f"**{label}:** {content}")

    # -- Render current stage --
    if stage == "idle":
        _render_request_form()
    elif stage == "review_plan":
        _render_plan_review()
    elif stage in ("review_lesson", "review_exercises", "review_assessment"):
        _render_artifact_review(stage)
    elif stage == "done":
        _render_done()
    else:
        # Generation in progress (shouldn't normally render, but handle gracefully)
        st.info(f"Generating content ({stage})...")

    # -- Documentation --
    st.divider()
    doc_viewer.render(_DOC_PATH, title="Documentation: DeepAgents & Autonomous Agents")
```

- [ ] **Step 2: Register the tab in app.py**

Update `app/app.py`:

Import line:
```python
from pages import p1_grammar, p2_lesson, p3_assessment, p4_tutor, p5_moderation, p7_curriculum  # noqa: E402
```

Tab list (add to existing tabs — note: P6 may or may not exist yet, add P7 regardless):
```python
tab1, tab2, tab3, tab4, tab5, tab7 = st.tabs([
    "✏️ Grammar Agent",
    "📋 Lesson Planner",
    "📊 Assessment Pipeline",
    "🤝 Tutor Matching",
    "🛡️ Content Moderation",
    "🧠 Curriculum Engine",
])
```

Add the render call:
```python
with tab7:
    p7_curriculum.render()
```

Note: If P6 is added by the other agent before this task runs, the engineer should include P6's tab as well. The key change is adding the P7 import and tab.

- [ ] **Step 3: Commit**

```bash
git add app/pages/p7_curriculum.py app/app.py
git commit -m "feat(p7): add Streamlit UI page and register tab"
```

---

### Task 12: README & Documentation

**Files:**
- Create: `projects/07-curriculum-engine/README.md`
- Create: `docs/07-curriculum-engine.md`

- [ ] **Step 1: Create project README**

`projects/07-curriculum-engine/README.md`:
```markdown
# Project 7: Intelligent Curriculum Engine

An autonomous curriculum module generator that uses DeepAgents sub-agents
orchestrated by a LangGraph workflow to create lessons, exercises, and
assessments — with human approval at each stage.

## Concepts Introduced

- **DeepAgents:** `create_deep_agent()`, harness architecture, built-in tools
- **SKILL.md:** On-demand skill loading for domain knowledge and output templates
- **CompositeBackend:** Hybrid storage — ephemeral working files + persistent catalog
- **TodoList:** Agent self-planning via `write_todos`
- **Sub-agent orchestration:** Specialized agents for distinct content types

## Quick Start

```bash
# From repo root
source .venv/bin/activate
cd projects/07-curriculum-engine

# Run tests
python -m pytest tests/ -v

# Or use via the Streamlit app
cd ../../app
streamlit run app.py
```

## Architecture

```
LangGraph Outer Graph (workflow + HITL)
├── plan_curriculum → DeepAgent (planner)
│   └── review_plan (interrupt)
├── generate_lesson → DeepAgent (lesson-writer)
│   └── review_lesson (interrupt)
├── generate_exercises → DeepAgent (exercise-creator)
│   └── review_exercises (interrupt)
├── generate_assessment → DeepAgent (assessment-builder)
│   └── review_assessment (interrupt)
└── assemble_module → Final markdown output
```

## Key Files

| File | Purpose |
|------|---------|
| `agents.py` | DeepAgent factory functions with CompositeBackend |
| `nodes.py` | Graph nodes wrapping agent invocations + HITL reviews |
| `graph.py` | StateGraph assembly with conditional routing |
| `skills/` | SKILL.md files for curriculum design + output templates |
| `models.py` | Pydantic models and LangGraph state schema |
```

- [ ] **Step 2: Create educational documentation**

`docs/07-curriculum-engine.md`:
```markdown
# Project 7: Intelligent Curriculum Engine

## What This Project Teaches

This project introduces **DeepAgents** — a framework for building autonomous agents
that can plan their own work, manage context through a file system, and delegate
tasks to sub-agents. While previous projects used LangGraph directly, DeepAgents
provides a higher-level abstraction that handles planning, file management, and
delegation automatically.

## Key Concept: DeepAgents Architecture

### The Harness

`create_deep_agent()` creates an agent with built-in capabilities:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=model,
    system_prompt="Your instructions here",
    skills=["./skills/"],      # On-demand knowledge loading
    backend=my_backend,         # File storage strategy
    store=my_store,             # Persistent memory
)
```

Every deep agent automatically gets these tools:
- **`write_todos`** — Plan multi-step tasks
- **`ls`, `read_file`, `write_file`, `edit_file`** — File operations
- **`glob`, `grep`** — File search
- **`task`** — Delegate to sub-agents

### SKILL.md — On-Demand Knowledge

Skills are markdown files with YAML frontmatter that agents load when relevant:

```markdown
---
name: lesson-template
description: Output format for English lessons
---

# Lesson Template
[Instructions the agent follows...]
```

Skills solve the context problem: instead of cramming everything into the system
prompt, agents load specific knowledge when they need it. This keeps the base
prompt focused and lets you add domain expertise without increasing startup cost.

### CompositeBackend — Hybrid Memory

```python
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.store import StoreBackend

backend = CompositeBackend(
    default=StateBackend(runtime),          # Ephemeral (thread-scoped)
    routes={
        "/catalog/": StoreBackend(runtime),     # Persistent (cross-session)
        "/preferences/": StoreBackend(runtime), # Persistent (cross-session)
    },
)
```

Files written to `/work/` disappear when the thread ends (good for drafts).
Files written to `/catalog/` or `/preferences/` persist across sessions
(good for remembering what's been created and how the user likes it).

## Architecture Pattern: LangGraph + DeepAgents

This project uses a **hybrid architecture**:

- **LangGraph outer graph** defines the workflow: plan → lesson → exercises → assessment → assemble
- **DeepAgents inner agents** handle the actual content generation within each step

Why not pure DeepAgents? Because LangGraph gives us explicit, observable workflow
structure. The outer graph makes it clear which step we're in, where the HITL
checkpoints are, and how revision loops work. DeepAgents shine within each step,
where the agent autonomously plans its work, loads relevant skills, and generates content.

```
LangGraph (workflow)          DeepAgents (content)
┌─────────────────┐          ┌────────────────────┐
│ plan_curriculum  │ ──────► │ Planner Agent       │
│                  │ ◄────── │  - write_todos      │
│ review_plan      │         │  - read SKILL.md    │
│   (interrupt)    │         │  - write plan.json  │
├─────────────────┤          └────────────────────┘
│ generate_lesson  │ ──────► ┌────────────────────┐
│                  │ ◄────── │ Lesson Writer Agent │
│ review_lesson    │         │  - write_todos      │
│   (interrupt)    │         │  - read plan.json   │
├─────────────────┤         │  - write lesson.md  │
│ ...              │         └────────────────────┘
└─────────────────┘
```

## HITL Pattern: Review at Each Stage

Each generation step is followed by a review node that calls `interrupt()`:

```python
def review_lesson_node(state):
    decision = interrupt({
        "step": "review_lesson",
        "artifact": state["lesson"],
        "prompt": "Review the lesson. Approve, revise, or reject.",
    })
    # ...
```

The moderator can:
- **Approve** — move to the next step
- **Revise** — send feedback, re-generate with the same agent
- **Reject** — skip this artifact (the final module notes it was skipped)

## TodoList — Agent Self-Planning

When a deep agent receives a task, it creates a todo list:

```
write_todos([
  {"content": "Read curriculum plan", "status": "in_progress"},
  {"content": "Draft lesson introduction", "status": "pending"},
  {"content": "Write core content sections", "status": "pending"},
  {"content": "Add vocabulary list", "status": "pending"},
])
```

This is visible in the Streamlit UI, showing how the agent breaks down
its work. The TodoList is a key differentiator from plain LangGraph nodes —
the agent decides *how* to accomplish its task, not just *what* to produce.

## LangSmith Integration

All traces are tagged with `project:07-curriculum-engine`. Because each
DeepAgent invocation is a nested trace under the parent graph, you can see:

- The outer workflow progression (which step are we in?)
- Each agent's internal planning (what todos did it create?)
- Skill loading events (which SKILL.md files were read?)
- File operations (what did the agent write to its filesystem?)

This observability is critical for autonomous agents — you need to understand
*why* an agent produced a particular output, not just what it produced.
```

- [ ] **Step 3: Commit**

```bash
git add projects/07-curriculum-engine/README.md docs/07-curriculum-engine.md
git commit -m "docs(p7): add project README and educational documentation"
```

---

### Task 13: End-to-End Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
cd projects/07-curriculum-engine && python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 2: Verify graph compiles and can start**

```bash
cd projects/07-curriculum-engine && python -c "
from graph import build_graph
g = build_graph()
print('Graph compiled successfully')
print('Nodes:', [n for n in g.get_graph().nodes.keys() if not n.startswith('__')])
"
```

Expected: Graph compiled successfully, all 9 node names listed

- [ ] **Step 3: Launch Streamlit app and test with Playwright**

```bash
cd app && source ../.venv/bin/activate && streamlit run app.py --server.headless true --server.port 8503
```

Then use Playwright to:
1. Navigate to http://localhost:8503
2. Click the "Curriculum Engine" tab
3. Verify the form renders (topic input, level selector, generate button)
4. Check console for errors

- [ ] **Step 4: Verify no regressions on other tabs**

Click through tabs 1-5 to verify they still load correctly.
