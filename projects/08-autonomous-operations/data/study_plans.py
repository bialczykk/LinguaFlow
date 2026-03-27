"""Pre-built study plans linked to existing students.

New students (S005, S006) have no study plans yet — those are
created during onboarding via the student_onboarding department agent.

Each plan links a student to a tutor and defines the focus areas
and weekly commitment.
"""

STUDY_PLANS = [
    {
        "plan_id": "SP-001",
        "student_id": "S001",
        "tutor_id": "T001",
        "level": "B1",
        "focus_areas": ["business English", "presentation skills"],
        "weekly_hours": 3,
        "created_date": "2025-09-20",
        "status": "active",
    },
    {
        "plan_id": "SP-002",
        "student_id": "S002",
        "tutor_id": "T002",
        "level": "A2",
        "focus_areas": ["general English", "grammar improvement"],
        "weekly_hours": 4,
        "created_date": "2025-11-05",
        "status": "active",
    },
    {
        "plan_id": "SP-003",
        "student_id": "S003",
        "tutor_id": "T003",
        "level": "B2",
        "focus_areas": ["IELTS preparation", "academic writing"],
        "weekly_hours": 5,
        "created_date": "2026-01-15",
        "status": "active",
    },
    {
        "plan_id": "SP-004",
        "student_id": "S004",
        "tutor_id": "T004",
        "level": "C1",
        "focus_areas": ["advanced conversation", "idioms"],
        "weekly_hours": 2,
        "created_date": "2025-07-01",
        "status": "active",
    },
]
