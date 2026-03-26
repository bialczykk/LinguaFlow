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
# temperature=0.3 gives warm, natural conversation responses with some creativity
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

# temperature=0 for extraction — we want deterministic, accurate structured output
_extraction_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# -- Extraction Prompt --
# This prompt is used AFTER the conversation to extract a structured profile.
# It receives the full transcript and maps it to the StudentProfile schema.
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

    The conversation is driven by the INTAKE_SYSTEM_PROMPT, which instructs
    the LLM to ask questions one at a time. When the LLM has gathered enough
    information, it outputs the [PROFILE_COMPLETE] marker. At that point,
    get_profile() can be called to extract a structured StudentProfile.

    Usage:
        intake = IntakeConversation()
        while not intake.is_complete():
            user_input = input("> ")
            response = intake.ask(user_input)
            print(response)
        profile = intake.get_profile()
    """

    # The marker the LLM outputs to signal it has all the information it needs.
    # Defined in INTAKE_SYSTEM_PROMPT — the LLM is instructed to emit exactly this.
    _COMPLETE_MARKER = "[PROFILE_COMPLETE]"

    def __init__(self):
        """Initialize with a system message and empty conversation history.

        The system message sets the LLM's role as a course advisor and
        defines what information to gather and how to signal completion.
        """
        # Message history starts with just the system prompt.
        # Each call to ask() appends a HumanMessage + AIMessage pair.
        self._messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
        self._complete = False

    @traceable(name="intake_ask", run_type="chain", tags=["p2-lesson-plan-generator"])
    def ask(self, user_message: str) -> str:
        """Send a user message and get the LLM's response.

        Appends the user message to history, invokes the model with the
        full history (so context is preserved), then appends the AI response.
        Checks for the completion marker in the response.

        Args:
            user_message: The student's input text.

        Returns:
            The LLM's response as a plain string.
        """
        # Append the new user message to the running history
        self._messages.append(HumanMessage(content=user_message))

        # Invoke the model with the full message history for context
        response = _model.invoke(self._messages)

        # Append the AI response to history for the next turn
        self._messages.append(response)

        # Check if the LLM has signalled it has all required information
        if self._COMPLETE_MARKER in response.content:
            self._complete = True

        return response.content

    def is_complete(self) -> bool:
        """Check whether the intake conversation has gathered all required info.

        Returns True once the LLM has emitted the [PROFILE_COMPLETE] marker,
        indicating it has collected name, level, goals, topics, and lesson type.
        """
        return self._complete

    @traceable(name="intake_extract_profile", run_type="chain", tags=["p2-lesson-plan-generator"])
    def get_profile(self) -> StudentProfile:
        """Extract a structured StudentProfile from the conversation history.

        Uses a separate extraction chain with structured output to parse the
        full conversation transcript into a validated Pydantic model.

        LangChain concept: with_structured_output()
        - Instructs the LLM to return JSON matching the StudentProfile schema
        - method="json_schema" uses the model's native JSON mode for reliability
        - The Pydantic model handles validation automatically

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

        # Build a readable transcript from message history (skip the system message)
        transcript_parts = []
        for msg in self._messages[1:]:
            role = "Student" if isinstance(msg, HumanMessage) else "Advisor"
            transcript_parts.append(f"{role}: {msg.content}")
        transcript = "\n".join(transcript_parts)

        # Chain: extraction prompt → model with structured output → StudentProfile
        # with_structured_output() wraps the model to return a parsed Pydantic object
        structured_model = _extraction_model.with_structured_output(
            StudentProfile, method="json_schema"
        )
        chain = _EXTRACTION_PROMPT | structured_model
        return chain.invoke({"transcript": transcript})
