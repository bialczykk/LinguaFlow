"""Integration tests for the follow-up conversation handler.

Tests verify that the conversation handler can answer questions about
grammar feedback and maintains context across turns.
Requires a valid ANTHROPIC_API_KEY in the root .env file.
"""

import os
import pytest

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


def test_conversation_handler_answers_question():
    """The conversation handler returns a non-empty response to a
    follow-up question about grammar feedback."""
    from conversation import ConversationHandler
    from models import (
        GrammarFeedback,
        GrammarIssue,
        ProficiencyAssessment,
    )

    feedback = GrammarFeedback(
        issues=[
            GrammarIssue(
                original_text="He go",
                corrected_text="He goes",
                error_category="subject-verb agreement",
                explanation="Third person singular needs 's' on the verb.",
                severity="major",
            )
        ],
        proficiency=ProficiencyAssessment(
            cefr_level="A2",
            strengths=["Simple vocabulary"],
            areas_to_improve=["Verb conjugation"],
            summary="Beginner level.",
        ),
        corrected_full_text="He goes to school every day.",
    )

    handler = ConversationHandler(
        original_text="He go to school every day.",
        feedback=feedback,
    )

    response = handler.ask("Can you explain the subject-verb agreement rule more?")

    assert isinstance(response, str)
    assert len(response) > 20


def test_conversation_handler_remembers_context():
    """The conversation handler maintains message history so it can
    reference earlier turns in the conversation."""
    from conversation import ConversationHandler
    from models import (
        GrammarFeedback,
        GrammarIssue,
        ProficiencyAssessment,
    )

    feedback = GrammarFeedback(
        issues=[
            GrammarIssue(
                original_text="She have",
                corrected_text="She has",
                error_category="subject-verb agreement",
                explanation="'Have' becomes 'has' with third person singular.",
                severity="major",
            )
        ],
        proficiency=ProficiencyAssessment(
            cefr_level="A2",
            strengths=["Clear ideas"],
            areas_to_improve=["Verb forms"],
            summary="Beginner level.",
        ),
        corrected_full_text="She has many books.",
    )

    handler = ConversationHandler(
        original_text="She have many books.",
        feedback=feedback,
    )

    handler.ask("What was my biggest mistake?")
    response = handler.ask("Can you give me a practice sentence for that rule?")

    assert isinstance(response, str)
    assert len(response) > 10
