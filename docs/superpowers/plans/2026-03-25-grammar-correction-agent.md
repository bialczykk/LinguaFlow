# Grammar Correction Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conversational grammar correction agent that analyzes student writing, returns structured feedback with CEFR proficiency assessment, and supports educational follow-up conversation.

**Architecture:** A LangChain chain using `ChatAnthropic` with `.with_structured_output()` for grammar analysis, plus a conversational follow-up handler that maintains message history. Both decorated with `@traceable` for LangSmith observability. Interactive CLI loop ties them together.

**Tech Stack:** `langchain-core`, `langchain-anthropic`, `langsmith`, `python-dotenv`, `pydantic`

---

## File Structure

```
projects/01-grammar-correction-agent/
  models.py                # Pydantic models: GrammarIssue, ProficiencyAssessment, GrammarFeedback
  chains.py                # Analysis chain: prompt + ChatAnthropic + structured output
  conversation.py          # Follow-up conversation handler with message history
  main.py                  # Interactive CLI entry point
  data/
    sample_texts.py        # Sample student writing at different CEFR levels
  tests/
    test_models.py         # Model validation tests
    test_chains.py         # Chain integration tests
    test_conversation.py   # Conversation handler tests
  requirements.txt         # Project dependencies
  README.md                # Overview and how to run
```

---

### Task 0: Set up shared virtual environment and project skeleton

**Files:**
- Create: `.venv` (at repo root)
- Create: `projects/01-grammar-correction-agent/requirements.txt`
- Create: `projects/01-grammar-correction-agent/data/__init__.py`
- Create: `projects/01-grammar-correction-agent/tests/__init__.py`

- [ ] **Step 1: Create shared virtual environment at repo root**

Run from repo root:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 2: Create requirements.txt**

Create `projects/01-grammar-correction-agent/requirements.txt`:
```
langchain>=1.0,<2.0
langchain-core>=1.0,<2.0
langchain-anthropic>=0.3.0
langsmith>=0.3.0
python-dotenv>=1.0.0
pydantic>=2.0
pytest>=8.0
```

- [ ] **Step 3: Install dependencies into shared venv**

```bash
source .venv/bin/activate
pip install -r projects/01-grammar-correction-agent/requirements.txt
```

- [ ] **Step 4: Create empty package files**

Create `projects/01-grammar-correction-agent/data/__init__.py` (empty file).
Create `projects/01-grammar-correction-agent/tests/__init__.py` (empty file).

- [ ] **Step 5: Verify the environment works**

```bash
source .venv/bin/activate
python -c "from langchain_anthropic import ChatAnthropic; from langsmith import traceable; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 6: Commit**

```bash
git add projects/01-grammar-correction-agent/requirements.txt projects/01-grammar-correction-agent/data/__init__.py projects/01-grammar-correction-agent/tests/__init__.py
git commit -m "feat(p1): scaffold project 1 skeleton and install dependencies"
```

---

### Task 1: Define Pydantic models

**Files:**
- Create: `projects/01-grammar-correction-agent/models.py`
- Create: `projects/01-grammar-correction-agent/tests/test_models.py`

- [ ] **Step 1: Write failing tests for Pydantic models**

Create `projects/01-grammar-correction-agent/tests/test_models.py`:
```python
"""Tests for grammar feedback Pydantic models."""

import pytest
from pydantic import ValidationError


def test_grammar_issue_valid():
    """GrammarIssue accepts valid data with all required fields."""
    from models import GrammarIssue

    issue = GrammarIssue(
        original_text="He go to school every day.",
        corrected_text="He goes to school every day.",
        error_category="subject-verb agreement",
        explanation="Third-person singular subjects require 's' on the verb in present simple.",
        severity="major",
    )
    assert issue.original_text == "He go to school every day."
    assert issue.severity == "major"


def test_grammar_issue_invalid_severity():
    """GrammarIssue rejects severity values outside the allowed literals."""
    from models import GrammarIssue

    with pytest.raises(ValidationError):
        GrammarIssue(
            original_text="test",
            corrected_text="test",
            error_category="test",
            explanation="test",
            severity="critical",  # not in Literal["minor", "moderate", "major"]
        )


def test_proficiency_assessment_valid():
    """ProficiencyAssessment accepts valid CEFR levels and lists."""
    from models import ProficiencyAssessment

    assessment = ProficiencyAssessment(
        cefr_level="B1",
        strengths=["Good vocabulary range", "Clear sentence structure"],
        areas_to_improve=["Article usage", "Verb tenses"],
        summary="Intermediate level with solid foundations.",
    )
    assert assessment.cefr_level == "B1"
    assert len(assessment.strengths) == 2


def test_proficiency_assessment_invalid_cefr():
    """ProficiencyAssessment rejects invalid CEFR levels."""
    from models import ProficiencyAssessment

    with pytest.raises(ValidationError):
        ProficiencyAssessment(
            cefr_level="D1",  # not a valid CEFR level
            strengths=[],
            areas_to_improve=[],
            summary="test",
        )


def test_grammar_feedback_valid():
    """GrammarFeedback composes issues and proficiency into a full response."""
    from models import GrammarIssue, GrammarFeedback, ProficiencyAssessment

    feedback = GrammarFeedback(
        issues=[
            GrammarIssue(
                original_text="He go",
                corrected_text="He goes",
                error_category="subject-verb agreement",
                explanation="Third person singular needs 's'.",
                severity="major",
            )
        ],
        proficiency=ProficiencyAssessment(
            cefr_level="A2",
            strengths=["Simple vocabulary"],
            areas_to_improve=["Verb conjugation"],
            summary="Beginner with basic communication ability.",
        ),
        corrected_full_text="He goes to school every day.",
    )
    assert len(feedback.issues) == 1
    assert feedback.proficiency.cefr_level == "A2"


def test_grammar_feedback_empty_issues():
    """GrammarFeedback allows an empty issues list (perfect text)."""
    from models import GrammarFeedback, ProficiencyAssessment

    feedback = GrammarFeedback(
        issues=[],
        proficiency=ProficiencyAssessment(
            cefr_level="C2",
            strengths=["Flawless grammar"],
            areas_to_improve=[],
            summary="Native-like proficiency.",
        ),
        corrected_full_text="This text is perfect.",
    )
    assert len(feedback.issues) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implement Pydantic models**

Create `projects/01-grammar-correction-agent/models.py`:
```python
"""Pydantic models for structured grammar feedback.

These models define the schema for the grammar correction agent's output.
The LLM is constrained to return data matching these schemas via
ChatAnthropic's .with_structured_output() method.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GrammarIssue(BaseModel):
    """A single grammar issue found in the student's writing.

    Each issue captures the original error, the correction, and an
    educational explanation so the student understands the rule.
    """

    original_text: str = Field(
        description="The problematic fragment from the student's writing"
    )
    corrected_text: str = Field(
        description="The corrected version of the fragment"
    )
    error_category: str = Field(
        description=(
            "Grammar category, e.g. 'subject-verb agreement', "
            "'tense', 'article usage', 'punctuation', 'word order'"
        )
    )
    explanation: str = Field(
        description=(
            "Educational explanation of why this is wrong and how "
            "the grammar rule works — written for a language learner"
        )
    )
    severity: Literal["minor", "moderate", "major"] = Field(
        description="How impactful the error is on comprehension"
    )


class ProficiencyAssessment(BaseModel):
    """Overall CEFR proficiency assessment based on the writing sample.

    Uses the Common European Framework of Reference (CEFR) scale,
    the international standard for describing language ability.
    """

    cefr_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] = Field(
        description="Assessed CEFR proficiency level"
    )
    strengths: list[str] = Field(
        description="What the student does well in their writing"
    )
    areas_to_improve: list[str] = Field(
        description="Key areas where the student should focus improvement"
    )
    summary: str = Field(
        description="Brief overall assessment of the student's writing level"
    )


class GrammarFeedback(BaseModel):
    """Complete grammar feedback for a student writing sample.

    This is the top-level model returned by the analysis chain.
    It contains individual grammar issues, an overall proficiency
    assessment, and the full corrected text.
    """

    issues: list[GrammarIssue] = Field(
        description="All grammar issues found in the writing sample"
    )
    proficiency: ProficiencyAssessment = Field(
        description="Overall CEFR proficiency assessment"
    )
    corrected_full_text: str = Field(
        description="The entire student submission with all corrections applied"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_models.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/01-grammar-correction-agent/models.py projects/01-grammar-correction-agent/tests/test_models.py
git commit -m "feat(p1): add Pydantic models for grammar feedback with CEFR assessment"
```

---

### Task 2: Build the analysis chain

**Files:**
- Create: `projects/01-grammar-correction-agent/chains.py`
- Create: `projects/01-grammar-correction-agent/tests/test_chains.py`

- [ ] **Step 1: Write integration test for the analysis chain**

Create `projects/01-grammar-correction-agent/tests/test_chains.py`:
```python
"""Integration tests for the grammar analysis chain.

These tests call the real Anthropic API and verify that the chain
returns properly structured GrammarFeedback. They require a valid
ANTHROPIC_API_KEY in the root .env file.
"""

import os
import sys
import pytest

# Load env vars from root .env so API keys are available
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


def test_analyze_grammar_returns_grammar_feedback():
    """The analysis chain returns a GrammarFeedback with at least one issue
    for text that contains obvious grammar errors."""
    from chains import analyze_grammar
    from models import GrammarFeedback

    # Text with deliberate errors: missing article, wrong verb form
    text = "He go to school every day. She have many book on the table."

    result = analyze_grammar(text)

    assert isinstance(result, GrammarFeedback)
    assert len(result.issues) > 0
    assert result.proficiency.cefr_level in ("A1", "A2", "B1", "B2", "C1", "C2")
    assert len(result.corrected_full_text) > 0


def test_analyze_grammar_handles_correct_text():
    """The analysis chain returns a valid GrammarFeedback even for
    well-written text (may have zero issues)."""
    from chains import analyze_grammar
    from models import GrammarFeedback

    text = "The quick brown fox jumps over the lazy dog."

    result = analyze_grammar(text)

    assert isinstance(result, GrammarFeedback)
    assert result.proficiency.cefr_level in ("A1", "A2", "B1", "B2", "C1", "C2")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_chains.py -v
```

Expected: `ModuleNotFoundError: No module named 'chains'`

- [ ] **Step 3: Implement the analysis chain**

Create `projects/01-grammar-correction-agent/chains.py`:
```python
"""Grammar analysis chain using ChatAnthropic with structured output.

This module contains the core analysis logic: a prompt template paired
with an Anthropic Claude model that returns structured GrammarFeedback.

Key LangChain concepts demonstrated:
- ChatPromptTemplate: building reusable prompt templates
- ChatAnthropic: configuring the Anthropic model
- .with_structured_output(): constraining LLM output to a Pydantic schema
- @traceable: making function calls visible in LangSmith
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from models import GrammarFeedback

# -- Prompt Template --
# This prompt instructs Claude to act as an English grammar teacher
# and return structured feedback matching our Pydantic schema.
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert English grammar teacher for the LinguaFlow "
                "tutoring platform. Your role is to analyze student writing "
                "and provide detailed, educational grammar feedback.\n\n"
                "For each grammar issue you find:\n"
                "- Identify the exact problematic text fragment\n"
                "- Provide the corrected version\n"
                "- Categorize the error type (e.g., subject-verb agreement, "
                "tense, article usage, punctuation, word order, spelling)\n"
                "- Write a clear, encouraging explanation of the grammar rule "
                "that a language learner can understand\n"
                "- Rate the severity: 'minor' (doesn't affect understanding), "
                "'moderate' (somewhat confusing), 'major' (changes meaning or "
                "is incomprehensible)\n\n"
                "Also assess the student's overall proficiency using the CEFR "
                "scale (A1-C2), noting specific strengths and areas to improve.\n\n"
                "Always be encouraging and educational in tone. The goal is to "
                "help the student learn, not to criticize."
            ),
        ),
        (
            "human",
            "Please analyze the following student writing:\n\n{student_text}",
        ),
    ]
)

# -- Model Configuration --
# ChatAnthropic is the LangChain wrapper for Anthropic's Claude models.
# temperature=0 gives deterministic output, which is good for consistent grading.
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# -- Structured Output Chain --
# .with_structured_output() constrains the model to return a GrammarFeedback
# Pydantic object. method="json_schema" uses Anthropic's native JSON schema
# support for reliable structured generation.
_analysis_chain = ANALYSIS_PROMPT | _model.with_structured_output(
    GrammarFeedback, method="json_schema"
)


@traceable(name="grammar_analysis", run_type="chain")
def analyze_grammar(student_text: str) -> GrammarFeedback:
    """Analyze student writing and return structured grammar feedback.

    This function is the main entry point for grammar analysis.
    It sends the student text through the analysis chain and returns
    a GrammarFeedback object containing issues, proficiency assessment,
    and the corrected full text.

    The @traceable decorator makes this function visible in LangSmith,
    so you can see the prompt, model response, and latency in the
    LangSmith dashboard.

    Args:
        student_text: The student's writing to analyze.

    Returns:
        GrammarFeedback with issues, proficiency assessment, and corrected text.
    """
    return _analysis_chain.invoke({"student_text": student_text})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_chains.py -v
```

Expected: Both tests PASS (requires valid `ANTHROPIC_API_KEY` in root `.env`)

- [ ] **Step 5: Commit**

```bash
git add projects/01-grammar-correction-agent/chains.py projects/01-grammar-correction-agent/tests/test_chains.py
git commit -m "feat(p1): add grammar analysis chain with structured output and LangSmith tracing"
```

---

### Task 3: Build the conversation handler

**Files:**
- Create: `projects/01-grammar-correction-agent/conversation.py`
- Create: `projects/01-grammar-correction-agent/tests/test_conversation.py`

- [ ] **Step 1: Write integration test for the conversation handler**

Create `projects/01-grammar-correction-agent/tests/test_conversation.py`:
```python
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

    # Create a sample feedback to use as context
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
    assert len(response) > 20  # Should be a substantive answer


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

    # First turn
    handler.ask("What was my biggest mistake?")

    # Second turn — should reference the conversation context
    response = handler.ask("Can you give me a practice sentence for that rule?")

    assert isinstance(response, str)
    assert len(response) > 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_conversation.py -v
```

Expected: `ModuleNotFoundError: No module named 'conversation'`

- [ ] **Step 3: Implement the conversation handler**

Create `projects/01-grammar-correction-agent/conversation.py`:
```python
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
        """Set up the conversation with the analysis context.

        Args:
            original_text: The student's original writing submission.
            feedback: The structured GrammarFeedback from the analysis chain.
        """
        self._model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

        # Format the feedback into a readable string for the system prompt
        issues_text = "\n".join(
            f"  {i+1}. [{issue.severity.upper()}] '{issue.original_text}' → "
            f"'{issue.corrected_text}' ({issue.error_category}): {issue.explanation}"
            for i, issue in enumerate(feedback.issues)
        )

        proficiency_text = (
            f"CEFR Level: {feedback.proficiency.cefr_level}\n"
            f"Strengths: {', '.join(feedback.proficiency.strengths)}\n"
            f"Areas to improve: {', '.join(feedback.proficiency.areas_to_improve)}\n"
            f"Summary: {feedback.proficiency.summary}"
        )

        # The system message gives the model full context about the
        # student's writing and the feedback that was given.
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

    @traceable(name="grammar_followup", run_type="chain")
    def ask(self, user_message: str) -> str:
        """Send a follow-up message and get the tutor's response.

        Appends the user message to history, calls the model with
        the full conversation, and appends the response to history.
        The @traceable decorator logs each conversation turn in LangSmith.

        Args:
            user_message: The student's follow-up question or message.

        Returns:
            The tutor's response as a string.
        """
        # Add the student's message to conversation history
        self._messages.append(HumanMessage(content=user_message))

        # Call the model with the full conversation history
        response = self._model.invoke(self._messages)

        # Add the model's response to history for future turns
        self._messages.append(AIMessage(content=response.content))

        return response.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/test_conversation.py -v
```

Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add projects/01-grammar-correction-agent/conversation.py projects/01-grammar-correction-agent/tests/test_conversation.py
git commit -m "feat(p1): add follow-up conversation handler with message history"
```

---

### Task 4: Add sample texts

**Files:**
- Create: `projects/01-grammar-correction-agent/data/sample_texts.py`

- [ ] **Step 1: Create sample student writing at different CEFR levels**

Create `projects/01-grammar-correction-agent/data/sample_texts.py`:
```python
"""Sample student writing at various CEFR levels for testing and demonstration.

Each sample is a dict with:
- label: short description including expected CEFR level
- text: the student's writing (with deliberate errors matching the level)

These are used in the interactive CLI to let users quickly try the agent
without typing their own text.
"""

SAMPLE_TEXTS = [
    {
        "label": "A2 (Elementary) — Daily routine description",
        "text": (
            "Every day I wake up at 7 clock and I go to the school. "
            "I have breakfast with my family, we eat breads and drink milk. "
            "After school I play with my friend in the park. "
            "Yesterday I go to the cinema and watch a very good film. "
            "I am like learning English because it is important for my future."
        ),
    },
    {
        "label": "B1 (Intermediate) — Opinion essay on technology",
        "text": (
            "In my opinion, technology have changed our lifes in many ways. "
            "Most of people today cannot imagine their daily routine without "
            "smartphones. I think that social media is both good and bad for "
            "the society. On one hand, it help us to stay connected with friends "
            "who lives far away. On the other hand, many peoples spend too much "
            "time scrolling and this affect their mental health. In conclusion, "
            "we should to use technology wisely and not let it control us."
        ),
    },
    {
        "label": "B2 (Upper-intermediate) — Formal email to a professor",
        "text": (
            "Dear Professor Smith,\n\n"
            "I am writing to you in regards of the assignment that was due last "
            "Friday. Unfortunately, I was not able to submit it on time due to "
            "an unexpected family emergency which happened me last week. I have "
            "already completed most of the work and I would be very grateful if "
            "you could extend the deadline for few more days. I assure you that "
            "the quality of my work will not be effected by this delay. I look "
            "forward to hear from you.\n\n"
            "Best regards,\nMaria"
        ),
    },
    {
        "label": "C1 (Advanced) — Academic paragraph on climate change",
        "text": (
            "The ramifications of climate change extend far beyond the environmental "
            "sphere, permeating into economic stability and social cohesion. While "
            "some argue that the transition to renewable energy sources will "
            "inevitably lead to job losses in traditional industries, this perspective "
            "fails to account the burgeoning green economy which has already began "
            "generating millions of positions worldwide. Furthermore, the cost of "
            "inaction — measured in terms of natural disasters, healthcare expenditures, "
            "and agricultural disruptions — far outweights the investment required for "
            "a comprehensive energy transition. It is therefore imperative that policy "
            "makers adopt a more holistic approach which takes into consideration both "
            "the immediate economic concerns and long-term sustainability objectives."
        ),
    },
]
```

- [ ] **Step 2: Commit**

```bash
git add projects/01-grammar-correction-agent/data/sample_texts.py
git commit -m "feat(p1): add sample student texts at A2, B1, B2, C1 CEFR levels"
```

---

### Task 5: Build the interactive CLI entry point

**Files:**
- Create: `projects/01-grammar-correction-agent/main.py`

- [ ] **Step 1: Implement the interactive CLI**

Create `projects/01-grammar-correction-agent/main.py`:
```python
"""Interactive CLI for the Grammar Correction Agent.

This is the entry point for the application. It:
1. Loads environment variables (.env) for API keys and LangSmith config
2. Lets the student pick a sample text or enter their own
3. Runs the grammar analysis chain and displays structured feedback
4. Enters a conversation loop for follow-up questions
5. Supports submitting new text (prefix with 'new:') or quitting

Run: python main.py
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables from the root .env file
# find_dotenv() walks up parent directories to locate it
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)

from chains import analyze_grammar
from conversation import ConversationHandler
from data.sample_texts import SAMPLE_TEXTS
from models import GrammarFeedback


def display_feedback(feedback: GrammarFeedback) -> None:
    """Display grammar feedback in a clean, readable format.

    Args:
        feedback: The structured GrammarFeedback from the analysis chain.
    """
    print("\n" + "=" * 60)
    print("  GRAMMAR ANALYSIS RESULTS")
    print("=" * 60)

    # -- Proficiency Assessment --
    p = feedback.proficiency
    print(f"\n  CEFR Level: {p.cefr_level}")
    print(f"  {p.summary}")
    print(f"\n  Strengths: {', '.join(p.strengths)}")
    print(f"  Improve:   {', '.join(p.areas_to_improve)}")

    # -- Grammar Issues --
    if feedback.issues:
        print(f"\n  ISSUES FOUND: {len(feedback.issues)}")
        print("-" * 60)
        for i, issue in enumerate(feedback.issues, 1):
            severity_icon = {"minor": ".", "moderate": "!", "major": "!!"}[
                issue.severity
            ]
            print(f"\n  {i}. [{severity_icon} {issue.severity.upper()}] {issue.error_category}")
            print(f"     Original:  \"{issue.original_text}\"")
            print(f"     Corrected: \"{issue.corrected_text}\"")
            print(f"     Why: {issue.explanation}")
    else:
        print("\n  No grammar issues found — great writing!")

    # -- Corrected Full Text --
    print("\n" + "-" * 60)
    print("  CORRECTED TEXT:")
    print(f"  {feedback.corrected_full_text}")
    print("=" * 60)


def get_student_text() -> str:
    """Prompt the student to enter text or pick a sample.

    Returns:
        The student's writing text to analyze.
    """
    print("\n  LINGUAFLOW GRAMMAR CORRECTION AGENT")
    print("=" * 60)
    print("\nChoose a sample text or enter your own:\n")

    for i, sample in enumerate(SAMPLE_TEXTS, 1):
        print(f"  {i}. {sample['label']}")

    print(f"  {len(SAMPLE_TEXTS) + 1}. Enter your own text")
    print()

    while True:
        choice = input("Your choice: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(SAMPLE_TEXTS):
                text = SAMPLE_TEXTS[idx - 1]["text"]
                print(f"\nSelected: {SAMPLE_TEXTS[idx - 1]['label']}")
                print(f"Text: {text[:80]}...")
                return text
            elif idx == len(SAMPLE_TEXTS) + 1:
                print("\nType or paste your text (press Enter twice to submit):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if lines:
                            break
                    else:
                        lines.append(line)
                return "\n".join(lines)
        print("Invalid choice, please try again.")


def conversation_loop(original_text: str, feedback: GrammarFeedback) -> str | None:
    """Run the follow-up conversation loop.

    The student can ask questions about the feedback, get practice
    exercises, or submit new text for analysis.

    Args:
        original_text: The student's original writing.
        feedback: The structured feedback from analysis.

    Returns:
        New text to analyze if the student submits new text with 'new:',
        or None if they choose to quit.
    """
    handler = ConversationHandler(original_text=original_text, feedback=feedback)

    print("\n  Follow-up Chat")
    print("-" * 60)
    print("  Ask me about your feedback, or:")
    print("  - Type 'new: <your text>' to analyze new writing")
    print("  - Type 'quit' or 'exit' to end the session")
    print("-" * 60)

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep practicing your English!")
            return None

        # Check if the student is submitting new text
        if user_input.lower().startswith("new:"):
            new_text = user_input[4:].strip()
            if new_text:
                return new_text
            else:
                print("Tutor: Please include your text after 'new:'. For example: new: I goes to school.")
                continue

        # Regular follow-up question
        print("\nTutor: ", end="")
        response = handler.ask(user_input)
        print(response)


def main() -> None:
    """Main application loop.

    Runs the analysis → conversation cycle. When the student submits
    new text via 'new:', the loop restarts with a fresh analysis.
    """
    text = get_student_text()

    while True:
        # Run grammar analysis
        print("\nAnalyzing your writing...")
        feedback = analyze_grammar(text)
        display_feedback(feedback)

        # Enter conversation loop
        new_text = conversation_loop(text, feedback)

        if new_text is None:
            # Student chose to quit
            break
        else:
            # Student submitted new text — loop back to analysis
            text = new_text
            print(f"\nNew text received! Analyzing...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manually test the CLI**

```bash
cd projects/01-grammar-correction-agent && python main.py
```

Pick sample text 1, verify feedback displays correctly, ask a follow-up question, then quit.

- [ ] **Step 3: Commit**

```bash
git add projects/01-grammar-correction-agent/main.py
git commit -m "feat(p1): add interactive CLI with sample texts and conversation loop"
```

---

### Task 6: Write README and educational documentation

**Files:**
- Create: `projects/01-grammar-correction-agent/README.md`
- Create: `docs/01-grammar-correction-agent.md`

- [ ] **Step 1: Write the project README**

Create `projects/01-grammar-correction-agent/README.md`:
```markdown
# Project 1: Grammar Correction Agent

**LinguaFlow — Student Success Department**

An AI-powered grammar correction agent that analyzes student writing, provides structured feedback with CEFR proficiency assessment, and supports follow-up conversation for deeper learning.

## What It Does

1. Takes a student's writing sample (choose from samples or enter your own)
2. Analyzes grammar and returns structured feedback:
   - Individual grammar issues with corrections and explanations
   - CEFR proficiency level assessment (A1-C2)
   - Strengths and areas to improve
   - Full corrected text
3. Lets you ask follow-up questions about the feedback
4. Submit new text anytime with `new: <your text>`

## Concepts Covered

- **LangChain chains**: prompt templates + model = chain
- **Structured output**: Pydantic models + `.with_structured_output()`
- **ChatAnthropic**: configuring Anthropic's Claude model
- **Conversation history**: maintaining multi-turn context
- **LangSmith tracing**: `@traceable` decorators for observability

## Setup

Ensure the shared virtual environment is activated:

```bash
# From the repo root
source .venv/bin/activate
pip install -r projects/01-grammar-correction-agent/requirements.txt
```

Ensure your root `.env` has:
```
ANTHROPIC_API_KEY=your-key-here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=linguaflow
```

## Run

```bash
cd projects/01-grammar-correction-agent
python main.py
```

## Test

```bash
cd projects/01-grammar-correction-agent
python -m pytest tests/ -v
```
```

- [ ] **Step 2: Write the educational documentation**

Create `docs/01-grammar-correction-agent.md` — a comprehensive educational document that explains every concept used in the project. This should teach the reader about LangChain chains, prompt templates, structured output, ChatAnthropic configuration, conversation history patterns, and LangSmith tracing. Include code snippets from the actual project files with explanations. Cover:

1. What is LangChain and why use it?
2. Prompt templates — `ChatPromptTemplate.from_messages()`
3. Model configuration — `ChatAnthropic` parameters
4. Structured output — Pydantic models + `.with_structured_output(method="json_schema")`
5. Chain composition — the `|` (pipe) operator
6. Conversation history — manual message list management
7. LangSmith tracing — `@traceable` decorator and what to look for in the dashboard

- [ ] **Step 3: Commit**

```bash
git add projects/01-grammar-correction-agent/README.md docs/01-grammar-correction-agent.md
git commit -m "docs(p1): add README and educational documentation for grammar correction agent"
```

---

### Task 7: Final integration test and cleanup

- [ ] **Step 1: Run the full test suite**

```bash
cd projects/01-grammar-correction-agent && python -m pytest tests/ -v
```

Expected: All tests PASS (6 model tests + 2 chain tests + 2 conversation tests)

- [ ] **Step 2: Run the CLI end-to-end**

```bash
cd projects/01-grammar-correction-agent && python main.py
```

Walk through: pick sample 2, review feedback, ask a follow-up question, submit new text with `new:`, review new feedback, then quit.

- [ ] **Step 3: Check LangSmith dashboard**

Open LangSmith at https://smith.langchain.com and verify:
- Traces appear for `grammar_analysis` and `grammar_followup`
- Each trace shows the prompt, model response, and latency
- Nested traces show the chain structure

- [ ] **Step 4: Final commit**

```bash
git add -A projects/01-grammar-correction-agent/
git commit -m "feat(p1): complete Grammar Correction Agent — Project 1 done"
```
