"""Mock lesson schedule data for the scheduling and support departments.

Dates are generated dynamically relative to today so demos always show
realistic upcoming and past lessons. Student and tutor IDs are consistent
with students.py and tutors.py.

Status values:
- "completed": lesson already happened
- "scheduled": upcoming lesson
- "cancelled": lesson was cancelled (may trigger refund)
"""

from datetime import date, timedelta

_today = date.today()


def _day(offset: int) -> str:
    """Return ISO date string offset from today (negative = past, positive = future)."""
    return (_today + timedelta(days=offset)).strftime("%Y-%m-%d")


LESSONS = [
    {
        "lesson_id": "L001",
        "student_id": "S001",
        "tutor_id": "T001",
        "subject": "Business English: Presentations",
        "date": _day(-6),
        "time": "09:00",
        "status": "completed",
    },
    {
        "lesson_id": "L002",
        "student_id": "S001",
        "tutor_id": "T001",
        "subject": "Email Writing Workshop",
        "date": _day(-4),
        "time": "14:00",
        "status": "completed",
    },
    {
        "lesson_id": "L003",
        "student_id": "S002",
        "tutor_id": "T002",
        "subject": "Grammar: Past Tenses",
        "date": _day(1),
        "time": "10:00",
        "status": "scheduled",
    },
    {
        "lesson_id": "L004",
        "student_id": "S002",
        "tutor_id": "T002",
        "subject": "Grammar Review",
        "date": _day(-8),
        "time": "10:00",
        "status": "cancelled",
    },
    {
        "lesson_id": "L005",
        "student_id": "S003",
        "tutor_id": "T003",
        "subject": "IELTS Reading Practice",
        "date": _day(-3),
        "time": "14:00",
        "status": "completed",
    },
    {
        "lesson_id": "L006",
        "student_id": "S004",
        "tutor_id": "T004",
        "subject": "Advanced Conversation: Idioms",
        "date": _day(-5),
        "time": "15:00",
        "status": "completed",
    },
    {
        "lesson_id": "L007",
        "student_id": "S003",
        "tutor_id": "T003",
        "subject": "IELTS Writing Task 2",
        "date": _day(2),
        "time": "14:00",
        "status": "scheduled",
    },
    {
        "lesson_id": "L008",
        "student_id": "S001",
        "tutor_id": "T001",
        "subject": "Presentation Skills: Q&A Handling",
        "date": _day(3),
        "time": "09:00",
        "status": "scheduled",
    },
    {
        "lesson_id": "L009",
        "student_id": "S004",
        "tutor_id": "T004",
        "subject": "Idioms in Business Context",
        "date": _day(0),
        "time": "15:00",
        "status": "scheduled",
    },
    {
        "lesson_id": "L010",
        "student_id": "S002",
        "tutor_id": "T002",
        "subject": "Pronunciation: Word Stress",
        "date": _day(5),
        "time": "10:00",
        "status": "scheduled",
    },
]
