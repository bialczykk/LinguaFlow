"""Sample requests covering all routing patterns.

Each request includes:
- text: the natural language request
- metadata: user_id, priority, source (mimics real request metadata)
- expected_departments: which department(s) should handle this
- expected_follow_ups: autonomous cascade departments triggered by the first
- pattern: routing pattern ("single", "parallel", "cascade")
- expected_risk: "low" or "high" (for HITL approval gate testing)

Used in tests, evaluation datasets (LangSmith), and the Streamlit demo
to demonstrate different orchestration patterns.

Routing patterns:
- single: one department handles the request end-to-end
- parallel: multiple departments handle simultaneously (Send)
- cascade: first department triggers follow-up tasks for another
"""

SAMPLE_REQUESTS = [
    {
        "text": "Onboard new student Maria Silva, she's interested in business English and email writing.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["student_onboarding"],
        "expected_follow_ups": ["tutor_management"],
        "pattern": "cascade",
        "expected_risk": "low",
    },
    {
        "text": "Generate a B2 reading passage about climate change and publish it.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["content_pipeline"],
        "expected_follow_ups": ["quality_assurance"],
        "pattern": "cascade",
        "expected_risk": "high",  # publish_content is a high-risk action
    },
    {
        "text": "I was charged twice for my IELTS prep lesson and now I can't access my lesson recordings.",
        "metadata": {"user_id": "student", "priority": "high", "source": "support_form"},
        "expected_departments": ["support"],
        "expected_follow_ups": [],
        "pattern": "parallel",  # billing + tech support handled within support dept
        "expected_risk": "low",
    },
    {
        "text": "How is the platform performing this week?",
        "metadata": {"user_id": "admin", "priority": "low", "source": "ui"},
        "expected_departments": ["reporting"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
    {
        "text": "Review all recently published content for quality.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["quality_assurance"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
    {
        "text": "Find a tutor for Lars Eriksson — he needs help with travel English at beginner level.",
        "metadata": {"user_id": "admin", "priority": "medium", "source": "ui"},
        "expected_departments": ["tutor_management"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "high",  # assign_tutor is a high-risk action
    },
    {
        "text": "Cancel my Thursday lesson and refund me.",
        "metadata": {"user_id": "student", "priority": "medium", "source": "support_form"},
        "expected_departments": ["support"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "high",  # refund is a high-risk action
    },
    {
        "text": "Check if student S001 is happy with their progress and review her tutor's performance.",
        "metadata": {"user_id": "admin", "priority": "low", "source": "ui"},
        "expected_departments": ["quality_assurance"],
        "expected_follow_ups": [],
        "pattern": "single",
        "expected_risk": "low",
    },
]
