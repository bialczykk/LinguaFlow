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
        # The agent may return a dict directly or a JSON string
        plan = plan_json if isinstance(plan_json, dict) else json.loads(plan_json)
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

    files = {"/work/plan.json": json.dumps(plan, indent=2)}

    agent = create_lesson_agent()
    result = _invoke_agent(agent, instruction, files=files)

    lesson_content = _extract_file_content(result, "/work/lesson.md")
    if not lesson_content:
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

@traceable(name="review_plan", run_type="chain", tags=_TAGS)
def review_plan_node(state: CurriculumEngineState) -> dict:
    """Pause for moderator review of the curriculum plan."""
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
