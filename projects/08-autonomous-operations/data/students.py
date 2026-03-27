"""Student profiles for onboarding and cross-department operations.

Mix of new students (no study plan yet — for onboarding demos) and
existing students (enrolled, with history — for support/scheduling).

S001–S004 are active enrolled students with cefr_level, study plans, and history.
S005–S006 are new (unonboarded) students — triggers the onboarding workflow.
"""

STUDENTS = [
    {
        "student_id": "S001",
        "name": "Alice Chen",
        "email": "alice.chen@email.com",
        "cefr_level": "B1",
        "goals": ["business English", "presentation skills"],
        "enrollment_date": "2025-09-15",
        "status": "active",
    },
    {
        "student_id": "S002",
        "name": "Marco Rossi",
        "email": "marco.rossi@email.com",
        "cefr_level": "A2",
        "goals": ["general English", "grammar improvement"],
        "enrollment_date": "2025-11-01",
        "status": "active",
    },
    {
        "student_id": "S003",
        "name": "Yuki Tanaka",
        "email": "yuki.tanaka@email.com",
        "cefr_level": "B2",
        "goals": ["IELTS preparation", "academic writing"],
        "enrollment_date": "2026-01-10",
        "status": "active",
    },
    {
        "student_id": "S004",
        "name": "Priya Sharma",
        "email": "priya.sharma@email.com",
        "cefr_level": "C1",
        "goals": ["advanced conversation", "idioms"],
        "enrollment_date": "2025-06-20",
        "status": "active",
    },
    {
        "student_id": "S005",
        "name": "Lars Eriksson",
        "email": "lars.eriksson@email.com",
        "cefr_level": None,
        "goals": ["travel English"],
        "enrollment_date": None,
        "status": "new",  # Not yet onboarded — for onboarding demo
    },
    {
        "student_id": "S006",
        "name": "Maria Silva",
        "email": "maria.silva@email.com",
        "cefr_level": None,
        "goals": ["business English", "email writing"],
        "enrollment_date": None,
        "status": "new",  # Not yet onboarded — for onboarding demo
    },
]
