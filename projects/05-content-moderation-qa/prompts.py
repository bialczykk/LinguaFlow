# prompts.py
"""Prompt templates for the content generation and revision nodes.

Three prompts:
- GENERATE_PROMPT: creates a lesson snippet from a content request
- REVISE_PROMPT: regenerates content incorporating moderator feedback
- POLISH_PROMPT: final formatting/cleanup pass on approved content

LangChain concept demonstrated:
- ChatPromptTemplate with {variable} placeholders
- Different prompts for different stages of the same content
"""

from langchain_core.prompts import ChatPromptTemplate

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow, an English tutoring platform. "
        "You produce high-quality lesson content for students.\n\n"
        "Generate a short lesson snippet (150-300 words) based on the request below. "
        "The content should be appropriate for the specified CEFR difficulty level.\n\n"
        "After generating the content, assess your own confidence in the quality "
        "on a scale from 0.0 to 1.0. Be honest — flag uncertainty.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate the lesson content.",
    ),
])

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow. A moderator has reviewed your "
        "previous draft and rejected it with feedback. Please revise the content "
        "based on their feedback.\n\n"
        "The revised content should still be 150-300 words and appropriate for "
        "the specified CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the revised lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Original Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Previous Draft:\n{previous_draft}\n\n"
        "Moderator Feedback:\n{feedback}\n\n"
        "Please revise the content.",
    ),
])

POLISH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an editor for LinguaFlow. Do a final formatting and cleanup pass "
        "on this lesson content. Fix any remaining grammar issues, improve clarity, "
        "and ensure consistent formatting. Keep the content at the same difficulty "
        "level and approximately the same length.\n\n"
        "Return ONLY the polished content text (no JSON wrapping).",
    ),
    (
        "human",
        "Content Type: {content_type}\n"
        "Difficulty Level: {difficulty}\n\n"
        "Content to polish:\n{content}\n\n"
        "Please produce the final polished version.",
    ),
])

# -- A/B Prompt Variants for evaluation --
# These are alternative generate prompts used by ab_comparison.py

GENERATE_PROMPT_STRUCTURED = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a content creator for LinguaFlow. Generate structured lesson content "
        "that follows a clear template:\n\n"
        "For grammar_explanation: Introduction → Rule → Examples → Common Mistakes\n"
        "For vocabulary_exercise: Word List → Definitions → Fill-in-the-Blank Exercises\n"
        "For reading_passage: Title → Passage → Comprehension Questions\n\n"
        "Keep content at 150-300 words, appropriate for the CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate the lesson content following the structured template.",
    ),
])

GENERATE_PROMPT_CREATIVE = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a creative content writer for LinguaFlow. Generate engaging lesson "
        "content that tells a story, uses humor, or creates a memorable scenario to "
        "teach the concept. Make the learning experience fun and memorable rather than "
        "formulaic.\n\n"
        "Keep content at 150-300 words, appropriate for the CEFR level.\n\n"
        "Return your response as JSON with two fields:\n"
        '- "content": the lesson snippet text\n'
        '- "confidence": your confidence score (0.0-1.0)',
    ),
    (
        "human",
        "Content Request:\n"
        "- Topic: {topic}\n"
        "- Type: {content_type}\n"
        "- Difficulty: {difficulty}\n\n"
        "Generate creative, engaging lesson content.",
    ),
])
