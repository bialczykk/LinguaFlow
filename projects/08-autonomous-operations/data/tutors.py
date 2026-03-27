"""Tutor profiles with specialties, availability, and capacity.

Reuses concepts from P4's tutor matching but with availability slots
and max student capacity for the assignment workflow.

Each tutor has cefr_levels they can teach and current_students vs max_students
to determine if they have capacity for new assignments.
"""

TUTORS = [
    {
        "tutor_id": "T001",
        "name": "Sarah Johnson",
        "specialties": ["business English", "presentation skills", "email writing"],
        "cefr_levels": ["B1", "B2", "C1"],
        "availability": ["Monday 09:00", "Monday 14:00", "Wednesday 10:00", "Friday 09:00"],
        "rating": 4.8,
        "max_students": 8,
        "current_students": 5,
    },
    {
        "tutor_id": "T002",
        "name": "James Wilson",
        "specialties": ["general English", "grammar", "pronunciation"],
        "cefr_levels": ["A2", "B1"],
        "availability": ["Tuesday 10:00", "Tuesday 15:00", "Thursday 10:00"],
        "rating": 4.6,
        "max_students": 10,
        "current_students": 7,
    },
    {
        "tutor_id": "T003",
        "name": "Emma Davis",
        "specialties": ["IELTS preparation", "academic writing", "reading comprehension"],
        "cefr_levels": ["B2", "C1"],
        "availability": ["Monday 11:00", "Wednesday 14:00", "Friday 11:00"],
        "rating": 4.9,
        "max_students": 6,
        "current_students": 4,
    },
    {
        "tutor_id": "T004",
        "name": "David Brown",
        "specialties": ["conversation practice", "idioms", "cultural context"],
        "cefr_levels": ["B2", "C1"],
        "availability": ["Tuesday 09:00", "Thursday 14:00", "Friday 15:00"],
        "rating": 4.7,
        "max_students": 8,
        "current_students": 6,
    },
    {
        "tutor_id": "T005",
        "name": "Lisa Chen",
        "specialties": ["business English", "vocabulary", "meeting skills"],
        "cefr_levels": ["B1", "B2"],
        "availability": ["Monday 15:00", "Wednesday 09:00", "Thursday 11:00"],
        "rating": 4.5,
        "max_students": 10,
        "current_students": 8,
    },
    {
        "tutor_id": "T006",
        "name": "Michael Park",
        "specialties": ["general English", "travel English", "beginner support"],
        "cefr_levels": ["A2", "B1"],
        "availability": ["Monday 10:00", "Tuesday 14:00", "Wednesday 15:00", "Friday 10:00"],
        "rating": 4.4,
        "max_students": 12,
        "current_students": 9,
    },
    {
        "tutor_id": "T007",
        "name": "Anna Kowalski",
        "specialties": ["grammar improvement", "writing skills", "IELTS preparation"],
        "cefr_levels": ["A2", "B1", "B2"],
        "availability": ["Tuesday 11:00", "Thursday 09:00", "Friday 14:00"],
        "rating": 4.8,
        "max_students": 8,
        "current_students": 3,
    },
    {
        "tutor_id": "T008",
        "name": "Carlos Mendez",
        "specialties": ["advanced conversation", "debate", "public speaking"],
        "cefr_levels": ["C1"],
        "availability": ["Wednesday 11:00", "Thursday 15:00"],
        "rating": 4.9,
        "max_students": 5,
        "current_students": 4,
    },
]
