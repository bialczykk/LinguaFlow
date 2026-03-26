"""Prompt templates for every LLM node in the assessment graph.

All prompts are defined here in one place so they're easy to compare,
tweak, and review. Each prompt is a ChatPromptTemplate.

LangChain concept demonstrated:
- ChatPromptTemplate.from_messages() — reusable prompt templates
  with {variable} placeholders filled at invocation time.
"""

from langchain_core.prompts import ChatPromptTemplate

# -- Criteria Scoring Node --

CRITERIA_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English writing assessor for the LinguaFlow tutoring "
        "platform. You score student writing across four dimensions using CEFR "
        "standards as your reference.\n\n"
        "The four dimensions are:\n"
        "1. Grammar & Accuracy\n"
        "2. Vocabulary Range & Precision\n"
        "3. Coherence & Organization\n"
        "4. Task Achievement\n\n"
        "For each dimension:\n"
        "- Assign a score from 1 (lowest) to 5 (highest)\n"
        "- Cite specific evidence from the submission (direct quotes)\n"
        "- Provide specific, actionable feedback\n\n"
        "After scoring all dimensions, determine a preliminary CEFR level "
        "(A1, A2, B1, B2, C1, or C2) based on the aggregate scores and "
        "the standards provided.\n\n"
        "Use the following rubrics and CEFR standards as your reference:\n\n"
        "{retrieved_standards}",
    ),
    (
        "human",
        "Writing prompt given to the student:\n{submission_context}\n\n"
        "Student's submission:\n{submission_text}\n\n"
        "Please score this submission across all four dimensions and determine "
        "a preliminary CEFR level.",
    ),
])

# -- Comparative Analysis Node --

COMPARATIVE_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert English writing assessor for the LinguaFlow tutoring "
        "platform. You compare a student's submission against sample essays at "
        "similar CEFR levels.\n\n"
        "For each sample essay provided:\n"
        "- Note specific similarities between the submission and the sample\n"
        "- Note specific differences\n"
        "- Determine whether the submission is 'above', 'comparable' to, or "
        "'below' the sample in overall quality\n\n"
        "After comparing all samples, write a narrative summary that explains "
        "where the submission sits relative to the level. Use specific examples.\n\n"
        "The student was preliminarily scored at CEFR level {preliminary_level}.\n\n"
        "Sample essays for comparison:\n\n{retrieved_samples}",
    ),
    (
        "human",
        "Writing prompt given to the student:\n{submission_context}\n\n"
        "Student's submission:\n{submission_text}\n\n"
        "Please compare this submission against the sample essays and provide "
        "a detailed comparative analysis.",
    ),
])

# -- Synthesize Node --

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior assessor for the LinguaFlow tutoring platform. "
        "You produce final, comprehensive writing assessments by combining "
        "criteria-based scoring with comparative analysis.\n\n"
        "You have two sources of information:\n"
        "1. Criteria scores: per-dimension scores (1-5) with evidence and feedback\n"
        "2. Comparative analysis: how the submission compares to level-appropriate samples\n\n"
        "Produce a final assessment that:\n"
        "- Confirms or adjusts the preliminary CEFR level based on all evidence\n"
        "- Lists the student's key strengths\n"
        "- Lists specific areas for improvement\n"
        "- Provides actionable recommendations for what to study next\n"
        "- Rates your confidence in the assessment as 'high', 'medium', or 'low'\n\n"
        "Be encouraging but honest. The student should feel motivated to improve.",
    ),
    (
        "human",
        "Student's submission:\n{submission_text}\n\n"
        "Writing prompt:\n{submission_context}\n\n"
        "Criteria Scores:\n{criteria_scores}\n\n"
        "Comparative Analysis:\n{comparative_analysis}\n\n"
        "Please produce the final assessment.",
    ),
])
