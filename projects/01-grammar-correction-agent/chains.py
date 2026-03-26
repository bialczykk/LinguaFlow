"""Grammar analysis chain using ChatAnthropic with structured output.

This module contains the core analysis logic: a prompt template paired
with an Anthropic Claude model that returns structured GrammarFeedback.

Key LangChain concepts demonstrated:
- ChatPromptTemplate: building reusable prompt templates
- ChatAnthropic: configuring the Anthropic model
- .with_structured_output(): constraining LLM output to a Pydantic schema
- @traceable: making function calls visible in LangSmith
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from models import GrammarFeedback

# -- Environment Setup --
# Load .env from the repo root (two levels up from this file's project folder).
# This must happen before instantiating ChatAnthropic so the API key is present.
_repo_root = Path(__file__).resolve().parents[2]
load_dotenv(_repo_root / ".env")

# -- Prompt Template --
# ChatPromptTemplate organizes system + human messages into a reusable template.
# The {student_text} placeholder is filled in at invocation time.
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
# temperature=0 ensures deterministic, consistent output — important for
# structured extraction where we need the model to reliably follow the schema.
_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# -- Structured Output Chain --
# The pipe operator (|) chains the prompt template to the model.
# .with_structured_output() wraps the model so it returns a validated
# GrammarFeedback Pydantic object instead of a raw text response.
# method="json_schema" instructs the model to use the JSON schema mode,
# which is the most reliable way to get structured output from Claude.
_analysis_chain = ANALYSIS_PROMPT | _model.with_structured_output(
    GrammarFeedback, method="json_schema"
)


@traceable(name="grammar_analysis", run_type="chain", tags=["p1-grammar-correction"])
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
