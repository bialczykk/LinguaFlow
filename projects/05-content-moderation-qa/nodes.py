# nodes.py
"""Node functions for the Content Moderation & QA StateGraph.

Contains 6 node functions and 2 routing functions:
- generate_node: LLM generates lesson content
- draft_review_node: interrupt() for first moderator review
- revise_node: LLM revises content with moderator feedback
- polish_node: LLM does final cleanup
- final_review_node: interrupt() for final moderator review
- publish_node: marks content as published
- route_after_draft_review: conditional routing after draft review
- route_after_final_review: conditional routing after final review

LangGraph concepts demonstrated:
- interrupt() to pause execution for human input
- Command(resume=...) to provide the human's decision
- RetryPolicy for transient error handling on LLM nodes
- Revision loop with max revision guard

Human-in-the-loop concepts:
- Two interrupt points with different purposes
- draft_review: approve/edit/reject with revision loop
- final_review: approve/reject gate (no loop)
"""

import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.types import interrupt
from langsmith import traceable

from models import ContentModerationState
from prompts import GENERATE_PROMPT, REVISE_PROMPT, POLISH_PROMPT

# -- Environment Setup --
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Model Configuration --
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# -- LangSmith Tags --
_TAGS = ["p5-content-moderation"]

# -- Max revision rounds --
MAX_REVISIONS = 2


def _parse_json_response(text: str) -> dict:
    """Parse a JSON response from the LLM, handling markdown code fences.

    The LLM sometimes wraps JSON in ```json ... ``` blocks.
    This helper strips those before parsing.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


@traceable(name="generate", run_type="chain", tags=_TAGS)
def generate_node(state: ContentModerationState) -> dict:
    """Generate a lesson snippet from a content request.

    Uses the LLM to produce content matching the requested topic, type,
    and CEFR difficulty level. The LLM also self-assesses its confidence.

    LangGraph concept: a standard LLM node that produces content
    for downstream human review.
    """
    request = state["content_request"]
    chain = GENERATE_PROMPT | _model

    response = chain.invoke(
        {
            "topic": request["topic"],
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
        },
        config={"tags": _TAGS},
    )

    parsed = _parse_json_response(response.content)

    return {
        "draft_content": parsed.get("content", response.content),
        "generation_confidence": float(parsed.get("confidence", 0.5)),
    }


@traceable(name="draft_review", run_type="chain", tags=_TAGS)
def draft_review_node(state: ContentModerationState) -> dict:
    """First human review checkpoint — pause for moderator decision.

    Calls interrupt() with the draft content and metadata. The graph
    pauses here until the moderator resumes with Command(resume=...).

    The moderator's decision dict must have:
    - action: "approve" | "edit" | "reject"
    - feedback: str (optional, used for reject)
    - edited_content: str (optional, used for edit)

    Human-in-the-loop concept: interrupt() pauses the graph. When resumed,
    the return value of interrupt() is the moderator's decision.
    """
    decision = interrupt({
        "content": state["draft_content"],
        "confidence": state["generation_confidence"],
        "revision_count": state["revision_count"],
        "prompt": "Review this draft. Approve, edit, or reject with feedback.",
    })

    # If moderator edited, update draft_content so polish reads from it
    updates = {"draft_decision": decision}
    if decision.get("action") == "edit" and decision.get("edited_content"):
        updates["draft_content"] = decision["edited_content"]

    return updates


def route_after_draft_review(
    state: ContentModerationState,
) -> Literal["polish", "revise", "__end__"]:
    """Route based on the moderator's draft review decision.

    - approve/edit → polish (content is ready for cleanup)
    - reject + revisions remaining → revise (loop back)
    - reject + max revisions reached → END (give up)
    """
    action = state["draft_decision"].get("action", "reject")

    if action in ("approve", "edit"):
        return "polish"

    # Reject — check revision budget
    if state["revision_count"] >= MAX_REVISIONS:
        return "__end__"

    return "revise"


@traceable(name="revise", run_type="chain", tags=_TAGS)
def revise_node(state: ContentModerationState) -> dict:
    """Revise content based on moderator feedback.

    Takes the previous draft and the moderator's feedback, generates
    a new version. Increments revision_count.

    LangGraph concept: revision loop — this node feeds back into
    draft_review, creating a cycle in the graph.
    """
    request = state["content_request"]
    feedback = state["draft_decision"].get("feedback", "Please improve the content.")

    chain = REVISE_PROMPT | _model

    response = chain.invoke(
        {
            "topic": request["topic"],
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
            "previous_draft": state["draft_content"],
            "feedback": feedback,
        },
        config={"tags": _TAGS},
    )

    parsed = _parse_json_response(response.content)

    return {
        "draft_content": parsed.get("content", response.content),
        "generation_confidence": float(parsed.get("confidence", 0.5)),
        "revision_count": state["revision_count"] + 1,
    }


@traceable(name="polish", run_type="chain", tags=_TAGS)
def polish_node(state: ContentModerationState) -> dict:
    """Final formatting and cleanup pass on approved/edited content.

    Produces the polished version that goes to final review.
    """
    request = state["content_request"]
    chain = POLISH_PROMPT | _model

    response = chain.invoke(
        {
            "content_type": request["content_type"],
            "difficulty": request["difficulty"],
            "content": state["draft_content"],
        },
        config={"tags": _TAGS},
    )

    return {"polished_content": response.content}


@traceable(name="final_review", run_type="chain", tags=_TAGS)
def final_review_node(state: ContentModerationState) -> dict:
    """Second human review checkpoint — final publication gate.

    Calls interrupt() with the polished content. Moderator sees the
    final version and decides: approve or reject. No revision loop
    at this stage — reject means the content is killed.

    Human-in-the-loop concept: a simpler interrupt point (binary decision)
    compared to draft_review's three-way decision.
    """
    decision = interrupt({
        "content": state["polished_content"],
        "prompt": "Final review. Approve for publication or reject.",
    })

    return {"final_decision": decision}


def route_after_final_review(
    state: ContentModerationState,
) -> Literal["publish", "__end__"]:
    """Route based on the moderator's final review decision.

    - approve → publish
    - reject → END (content killed)
    """
    action = state["final_decision"].get("action", "reject")

    if action == "approve":
        return "publish"

    return "__end__"


@traceable(name="publish", run_type="chain", tags=_TAGS)
def publish_node(state: ContentModerationState) -> dict:
    """Mark content as published and record metadata.

    This is the terminal node for successfully reviewed content.
    In a real system, this would write to a database or CMS.
    """
    return {
        "published": True,
        "publish_metadata": {
            "moderator_notes": state.get("final_decision", {}).get("feedback", ""),
            "review_rounds": state.get("revision_count", 0),
        },
    }
