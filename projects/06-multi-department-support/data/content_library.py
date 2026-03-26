"""Mock content library and enrollment data for the content department.

Course catalog with topic, level, and type. Plus enrollment records
linking students to courses.
"""

COURSES = [
    {"course_id": "C001", "title": "Grammar Essentials", "level": "A2", "type": "grammar", "modules": 12, "description": "Core grammar rules for elementary learners."},
    {"course_id": "C002", "title": "Everyday Conversations", "level": "B1", "type": "conversation", "modules": 10, "description": "Practical dialogue skills for intermediate speakers."},
    {"course_id": "C003", "title": "Business English Fundamentals", "level": "B2", "type": "business", "modules": 8, "description": "Professional communication and workplace vocabulary."},
    {"course_id": "C004", "title": "Academic Writing", "level": "C1", "type": "writing", "modules": 6, "description": "Essay structure, argumentation, and formal register."},
    {"course_id": "C005", "title": "IELTS Preparation", "level": "B2", "type": "exam_prep", "modules": 15, "description": "Comprehensive IELTS preparation across all four skills."},
    {"course_id": "C006", "title": "Pronunciation Workshop", "level": "A2", "type": "pronunciation", "modules": 8, "description": "Sound patterns, stress, and intonation practice."},
    {"course_id": "C007", "title": "Idioms & Expressions", "level": "B2", "type": "vocabulary", "modules": 10, "description": "Common English idioms, phrasal verbs, and collocations."},
    {"course_id": "C008", "title": "Travel English", "level": "A2", "type": "conversation", "modules": 6, "description": "Essential phrases and situations for travelling abroad."},
    {"course_id": "C009", "title": "Advanced Grammar", "level": "C1", "type": "grammar", "modules": 10, "description": "Complex structures, conditionals, and reported speech."},
    {"course_id": "C010", "title": "News & Current Affairs", "level": "B2", "type": "reading", "modules": 12, "description": "Reading comprehension through real news articles."},
    {"course_id": "C011", "title": "TOEFL Preparation", "level": "B2", "type": "exam_prep", "modules": 14, "description": "Targeted practice for all TOEFL sections."},
    {"course_id": "C012", "title": "Creative Writing", "level": "C1", "type": "writing", "modules": 8, "description": "Fiction, poetry, and narrative techniques in English."},
]

ENROLLMENTS = [
    {"student_id": "S001", "course_id": "C001", "progress_percent": 75, "status": "active"},
    {"student_id": "S001", "course_id": "C002", "progress_percent": 30, "status": "active"},
    {"student_id": "S002", "course_id": "C003", "progress_percent": 50, "status": "active"},
    {"student_id": "S002", "course_id": "C005", "progress_percent": 10, "status": "paused"},
    {"student_id": "S003", "course_id": "C005", "progress_percent": 60, "status": "active"},
    {"student_id": "S004", "course_id": "C005", "progress_percent": 20, "status": "active"},
    {"student_id": "S004", "course_id": "C010", "progress_percent": 45, "status": "active"},
    {"student_id": "S005", "course_id": "C006", "progress_percent": 15, "status": "active"},
    {"student_id": "S005", "course_id": "C008", "progress_percent": 0, "status": "enrolled"},
]
