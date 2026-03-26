# data/sample_requests.py
"""Sample curriculum generation requests for quick testing.

Each request represents a realistic curriculum module that a content
team member might ask the engine to create.
"""

SAMPLE_REQUESTS = [
    {
        "topic": "Business English for meetings",
        "level": "B2",
        "preferences": {
            "teaching_style": "interactive",
            "focus_areas": ["vocabulary", "speaking"],
        },
    },
    {
        "topic": "Present Perfect vs Past Simple",
        "level": "B1",
        "preferences": {
            "teaching_style": "formal",
            "focus_areas": ["grammar"],
        },
    },
    {
        "topic": "Everyday greetings and introductions",
        "level": "A1",
        "preferences": {
            "teaching_style": "conversational",
            "focus_areas": ["speaking", "listening"],
        },
    },
    {
        "topic": "Academic essay writing",
        "level": "C1",
        "preferences": {
            "teaching_style": "formal",
            "focus_areas": ["writing", "vocabulary"],
        },
    },
]
