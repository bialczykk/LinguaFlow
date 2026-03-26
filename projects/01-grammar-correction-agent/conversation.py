"""Follow-up conversation handler for grammar feedback discussion.

After the analysis chain returns structured feedback, the student
may want to ask follow-up questions. This module manages that
conversation, maintaining message history and injecting the
original feedback as context.

Key LangChain concepts demonstrated:
- ChatAnthropic: reusing the model for conversational responses
- Message history: maintaining multi-turn conversation state
- @traceable: tracing conversation turns in LangSmith
"""

from pathlib import Path

from dotenv import load_dotenv

# Load API keys from the shared repo-root .env file, two levels up from this file
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable

from models import GrammarFeedback


class ConversationHandler:
    """Manages follow-up conversation about grammar feedback.

    Initialized with the original student text and the GrammarFeedback
    from the analysis chain. Maintains a running message history so the
    student can have a natural, multi-turn conversation about their
    grammar issues.

    The agent ends each response with a suggested next step to guide
    the student's learning.
    """

    def __init__(self, original_text: str, feedback: GrammarFeedback) -> None:
        # Reuse the same model as the analysis chain; lower temperature for
        # consistent, educational responses
        self._model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

        # Format each grammar issue into a readable numbered list so the
        # system prompt is clear and easy for the model to reference
        issues_text = "\n".join(
            f"  {i+1}. [{issue.severity.upper()}] '{issue.original_text}' → "
            f"'{issue.corrected_text}' ({issue.error_category}): {issue.explanation}"
            for i, issue in enumerate(feedback.issues)
        )

        # Flatten proficiency data into a short block for the system prompt
        proficiency_text = (
            f"CEFR Level: {feedback.proficiency.cefr_level}\n"
            f"Strengths: {', '.join(feedback.proficiency.strengths)}\n"
            f"Areas to improve: {', '.join(feedback.proficiency.areas_to_improve)}\n"
            f"Summary: {feedback.proficiency.summary}"
        )

        # The message history is a plain list that grows with each turn.
        # Starting with a SystemMessage injects persistent context (the full
        # analysis result) into every request without the student needing to
        # re-state it.
        self._messages: list = [
            SystemMessage(
                content=(
                    "You are a friendly, encouraging English grammar tutor for "
                    "the LinguaFlow platform. You've just analyzed a student's "
                    "writing and given them feedback. Now you're having a "
                    "follow-up conversation to help them understand the "
                    "feedback better.\n\n"
                    f"ORIGINAL STUDENT TEXT:\n{original_text}\n\n"
                    f"GRAMMAR ISSUES FOUND:\n{issues_text}\n\n"
                    f"PROFICIENCY ASSESSMENT:\n{proficiency_text}\n\n"
                    f"CORRECTED TEXT:\n{feedback.corrected_full_text}\n\n"
                    "INSTRUCTIONS:\n"
                    "- Answer the student's questions clearly and educationally\n"
                    "- Use simple language appropriate to their CEFR level\n"
                    "- Give examples when explaining grammar rules\n"
                    "- Be encouraging — celebrate what they do well\n"
                    "- At the end of EVERY response, suggest a next step the "
                    "student could take (e.g., 'Want me to explain another "
                    "rule?', 'Try rewriting the sentence and I'll check it', "
                    "'Would you like a practice exercise on this topic?')\n"
                    "- If the student submits new text for correction, let "
                    "them know they should type 'new:' followed by their text "
                    "to get a fresh analysis"
                )
            )
        ]

    @traceable(name="grammar_followup", run_type="chain", tags=["p1-grammar-correction"])
    def ask(self, user_message: str) -> str:
        """Send a follow-up message and return the tutor's reply.

        Each call appends the user message and the AI response to
        self._messages, so subsequent calls have full conversation history.
        This is the simplest form of LangChain message history — a plain
        list passed directly to the model on every invocation.
        """
        # Append the student's question to the running history
        self._messages.append(HumanMessage(content=user_message))

        # Invoke the model with the full history — this is what gives the
        # model memory of previous turns without any external storage
        response = self._model.invoke(self._messages)

        # Append the AI reply so the next call sees the full context
        self._messages.append(AIMessage(content=response.content))

        return response.content
