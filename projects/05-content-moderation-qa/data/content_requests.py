# data/content_requests.py
"""Sample content requests for testing and evaluation.

Each request specifies a topic, content type, and CEFR difficulty level.
Used by tests and by the LangSmith evaluation pipeline.

This is scaffolding — it provides realistic inputs for the graph.
"""

SAMPLE_REQUESTS = [
    {
        "topic": "Present Perfect Tense",
        "content_type": "grammar_explanation",
        "difficulty": "B1",
    },
    {
        "topic": "Food and Cooking Vocabulary",
        "content_type": "vocabulary_exercise",
        "difficulty": "A2",
    },
    {
        "topic": "Climate Change",
        "content_type": "reading_passage",
        "difficulty": "B2",
    },
    {
        "topic": "Daily Routines",
        "content_type": "grammar_explanation",
        "difficulty": "A1",
    },
    {
        "topic": "Business Email Etiquette",
        "content_type": "reading_passage",
        "difficulty": "C1",
    },
    {
        "topic": "Travel and Transportation",
        "content_type": "vocabulary_exercise",
        "difficulty": "A2",
    },
    {
        "topic": "Conditional Sentences (Type 2)",
        "content_type": "grammar_explanation",
        "difficulty": "B2",
    },
    {
        "topic": "Technology and Innovation",
        "content_type": "reading_passage",
        "difficulty": "C1",
    },
]
