"""Mock lesson schedule data for the scheduling department.

Dates are generated dynamically relative to today so demos always show
upcoming lessons. Student and tutor IDs are consistent with other modules.
"""

from datetime import date, timedelta

_today = date.today()


def _day(offset: int) -> str:
    """Return ISO date string offset from today."""
    return (_today + timedelta(days=offset)).strftime("%Y-%m-%d")


LESSONS = [
    {
        "lesson_id": "L001",
        "student_id": "S001",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Fundamentals",
        "date": _day(-6),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L002",
        "student_id": "S001",
        "tutor_name": "Bob Chen",
        "subject": "Conversation Practice",
        "date": _day(-4),
        "time": "14:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L003",
        "student_id": "S002",
        "tutor_name": "Carol Davis",
        "subject": "Business English",
        "date": _day(1),
        "time": "09:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L004",
        "student_id": "S002",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Review",
        "date": _day(-8),
        "time": "11:00",
        "duration_minutes": 60,
        "status": "cancelled",
    },
    {
        "lesson_id": "L005",
        "student_id": "S003",
        "tutor_name": "Diana Evans",
        "subject": "IELTS Exam Prep",
        "date": _day(-3),
        "time": "15:00",
        "duration_minutes": 60,
        "status": "completed",
    },
    {
        "lesson_id": "L006",
        "student_id": "S004",
        "tutor_name": "Eve Foster",
        "subject": "IELTS Writing",
        "date": _day(2),
        "time": "16:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L007",
        "student_id": "S005",
        "tutor_name": "Frank Garcia",
        "subject": "Vocabulary Building",
        "date": _day(3),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
    {
        "lesson_id": "L008",
        "student_id": "S001",
        "tutor_name": "Alice Smith",
        "subject": "Grammar Advanced",
        "date": _day(4),
        "time": "10:00",
        "duration_minutes": 60,
        "status": "scheduled",
    },
]
