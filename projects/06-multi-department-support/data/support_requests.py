"""Sample support requests covering all routing patterns.

Each request includes the text, sender metadata, and expected routing
for use in tests and the Streamlit demo.
"""

SAMPLE_REQUESTS = [
    {
        "text": "I can't log in to my account. I've tried resetting my password but the reset email never arrives.",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "high"},
        "expected_departments": ["tech_support"],
        "pattern": "single",
    },
    {
        "text": "Can I get a refund for the grammar lesson I had last Tuesday? The tutor didn't show up.",
        "metadata": {"sender_type": "student", "student_id": "S001", "priority": "medium"},
        "expected_departments": ["billing"],
        "pattern": "single",
    },
    {
        "text": "I need to reschedule my Business English lesson that's coming up tomorrow. Can we move it to next week?",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "medium"},
        "expected_departments": ["scheduling"],
        "pattern": "single",
    },
    {
        "text": "What B2 materials do you have for business English? I'm looking for something focused on presentations and meetings.",
        "metadata": {"sender_type": "student", "student_id": "S003", "priority": "low"},
        "expected_departments": ["content"],
        "pattern": "single",
    },
    {
        "text": "I want to cancel my Friday vocabulary lesson and get a refund for it.",
        "metadata": {"sender_type": "student", "student_id": "S005", "priority": "medium"},
        "expected_departments": ["billing", "scheduling"],
        "pattern": "parallel",
    },
    {
        "text": "I was charged twice for my IELTS prep lesson and now I can't access my lesson recordings from that session.",
        "metadata": {"sender_type": "student", "student_id": "S003", "priority": "high"},
        "expected_departments": ["billing", "tech_support"],
        "pattern": "parallel",
    },
    {
        "text": "I want to change tutors.",
        "metadata": {"sender_type": "student", "student_id": "S004", "priority": "low"},
        "expected_departments": [],
        "pattern": "clarification",
    },
    {
        "text": "My lesson was cancelled but I still got charged, and now I can't book a new one because the system shows an error.",
        "metadata": {"sender_type": "student", "student_id": "S002", "priority": "high"},
        "expected_departments": ["billing", "scheduling", "tech_support"],
        "pattern": "parallel",
    },
]
