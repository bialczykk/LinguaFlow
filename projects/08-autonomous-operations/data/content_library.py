"""Course catalog and student enrollment records for the content department.

COURSES provides the full content library with 12 courses across all CEFR levels.
ENROLLMENTS links students to courses with progress tracking.

Used by the content_pipeline department to recommend courses and track
completion, and by the reporting department for engagement metrics.
"""

COURSES = [
    {
        "course_id": "C001",
        "title": "Grammar Essentials",
        "type": "grammar",
        "level": "A2",
        "description": "Core grammar rules for elementary learners.",
    },
    {
        "course_id": "C002",
        "title": "Everyday Conversations",
        "type": "conversation",
        "level": "B1",
        "description": "Practical dialogue skills for intermediate speakers.",
    },
    {
        "course_id": "C003",
        "title": "Business English Fundamentals",
        "type": "business",
        "level": "B1",
        "description": "Professional communication and workplace vocabulary.",
    },
    {
        "course_id": "C004",
        "title": "Academic Writing",
        "type": "writing",
        "level": "C1",
        "description": "Essay structure, argumentation, and formal register.",
    },
    {
        "course_id": "C005",
        "title": "IELTS Preparation",
        "type": "exam_prep",
        "level": "B2",
        "description": "Comprehensive IELTS preparation across all four skills.",
    },
    {
        "course_id": "C006",
        "title": "Pronunciation Workshop",
        "type": "pronunciation",
        "level": "A2",
        "description": "Sound patterns, stress, and intonation practice.",
    },
    {
        "course_id": "C007",
        "title": "Idioms & Expressions",
        "type": "vocabulary",
        "level": "B2",
        "description": "Common English idioms, phrasal verbs, and collocations.",
    },
    {
        "course_id": "C008",
        "title": "Travel English",
        "type": "conversation",
        "level": "A2",
        "description": "Essential phrases and situations for travelling abroad.",
    },
    {
        "course_id": "C009",
        "title": "Advanced Grammar",
        "type": "grammar",
        "level": "C1",
        "description": "Complex structures, conditionals, and reported speech.",
    },
    {
        "course_id": "C010",
        "title": "News & Current Affairs",
        "type": "reading",
        "level": "B2",
        "description": "Reading comprehension through real news articles.",
    },
    {
        "course_id": "C011",
        "title": "TOEFL Preparation",
        "type": "exam_prep",
        "level": "B2",
        "description": "Targeted practice for all TOEFL sections.",
    },
    {
        "course_id": "C012",
        "title": "Creative Writing",
        "type": "writing",
        "level": "C1",
        "description": "Fiction, poetry, and narrative techniques in English.",
    },
]

# Enrollment records linking students to courses.
# progress_percent: 0–100, how far through the course the student is.
# status: "active" (in progress), "paused" (temporarily stopped), "enrolled" (not started)
ENROLLMENTS = [
    {"student_id": "S001", "course_id": "C003", "progress_percent": 60, "status": "active"},
    {"student_id": "S001", "course_id": "C002", "progress_percent": 85, "status": "active"},
    {"student_id": "S002", "course_id": "C001", "progress_percent": 45, "status": "active"},
    {"student_id": "S002", "course_id": "C006", "progress_percent": 20, "status": "paused"},
    {"student_id": "S003", "course_id": "C005", "progress_percent": 70, "status": "active"},
    {"student_id": "S003", "course_id": "C004", "progress_percent": 30, "status": "active"},
    {"student_id": "S004", "course_id": "C007", "progress_percent": 55, "status": "active"},
    {"student_id": "S004", "course_id": "C010", "progress_percent": 40, "status": "active"},
    {"student_id": "S004", "course_id": "C012", "progress_percent": 15, "status": "enrolled"},
]
