"""Department tool functions for the multi-department support system.

Each department has 2 tools that sub-agents can call to look up information
or perform actions. Tools use @tool for LangChain integration and @traceable
for LangSmith observability.

Tools are stateless functions that operate on mock data. They return structured
data on success or a descriptive error string on failure.
"""

from langchain_core.tools import tool
from langsmith import traceable

from data.invoices import INVOICES
from data.system_status import SERVICES
from data.accounts import ACCOUNTS
from data.lessons import LESSONS
from data.content_library import COURSES, ENROLLMENTS

_TAGS = ["p6-multi-department-support"]


# --- Billing tools ---

@tool
@traceable(name="lookup_invoice", run_type="tool", tags=_TAGS)
def lookup_invoice(student_id: str) -> list[dict]:
    """Look up all invoices for a student by their student ID.

    Returns a list of invoice records with id, amount, status, date, and description.
    Returns an empty list if no invoices are found.
    """
    return [inv for inv in INVOICES if inv["student_id"] == student_id]


@tool
@traceable(name="check_refund_status", run_type="tool", tags=_TAGS)
def check_refund_status(invoice_id: str) -> dict | str:
    """Check the refund status of a specific invoice.

    Returns the invoice record if found, or an error message if not found.
    """
    for inv in INVOICES:
        if inv["invoice_id"] == invoice_id:
            return inv
    return f"Invoice {invoice_id} not found."


# --- Tech support tools ---

@tool
@traceable(name="check_system_status", run_type="tool", tags=_TAGS)
def check_system_status(service: str) -> dict | str:
    """Check the current status of a platform service.

    Valid services: video_platform, chat_system, payment_gateway, content_cdn, auth_service.
    Returns service health info or an error if the service name is not found.
    """
    if service in SERVICES:
        return SERVICES[service]
    return f"Service '{service}' not found. Valid services: {', '.join(SERVICES.keys())}"


@tool
@traceable(name="lookup_user_account", run_type="tool", tags=_TAGS)
def lookup_user_account(email: str) -> dict | str:
    """Look up a student account by email address.

    Returns account details including plan, timezone, and known issues.
    Returns an error message if the account is not found.
    """
    for account in ACCOUNTS.values():
        if account["email"] == email:
            return account
    return f"Account not found for email '{email}'."


# --- Scheduling tools ---

@tool
@traceable(name="check_lesson_schedule", run_type="tool", tags=_TAGS)
def check_lesson_schedule(student_id: str) -> list[dict]:
    """Get all lessons (past and upcoming) for a student.

    Returns a list of lesson records with id, tutor, subject, date, time, and status.
    Returns an empty list if no lessons are found.
    """
    return [lesson for lesson in LESSONS if lesson["student_id"] == student_id]


@tool
@traceable(name="reschedule_lesson", run_type="tool", tags=_TAGS)
def reschedule_lesson(lesson_id: str, new_date: str) -> dict:
    """Reschedule a lesson to a new date.

    Returns a dict with success status and details. Only scheduled (not completed
    or cancelled) lessons can be rescheduled.
    """
    for lesson in LESSONS:
        if lesson["lesson_id"] == lesson_id:
            if lesson["status"] != "scheduled":
                return {
                    "success": False,
                    "reason": f"Cannot reschedule — lesson status is '{lesson['status']}'.",
                }
            return {
                "success": True,
                "lesson_id": lesson_id,
                "old_date": lesson["date"],
                "new_date": new_date,
                "message": f"Lesson {lesson_id} rescheduled to {new_date}.",
            }
    return {"success": False, "reason": f"Lesson {lesson_id} not found."}


# --- Content tools ---

@tool
@traceable(name="search_content_library", run_type="tool", tags=_TAGS)
def search_content_library(query: str, level: str | None = None) -> list[dict]:
    """Search the course catalog by keyword and optional CEFR level.

    The query is matched against course title, type, and description (case-insensitive).
    Level filters by exact CEFR level (A2, B1, B2, C1).
    Returns matching courses or an empty list.
    """
    query_lower = query.lower()
    results = []
    for course in COURSES:
        text = f"{course['title']} {course['type']} {course['description']}".lower()
        if query_lower in text:
            if level and course["level"] != level:
                continue
            results.append(course)
    return results


@tool
@traceable(name="check_enrollment", run_type="tool", tags=_TAGS)
def check_enrollment(student_id: str) -> list[dict]:
    """Check which courses a student is enrolled in.

    Returns enrollment records with course_id, progress, and status.
    Returns an empty list if the student has no enrollments.
    """
    return [e for e in ENROLLMENTS if e["student_id"] == student_id]


# --- Tool groups for binding to sub-agent LLMs ---

BILLING_TOOLS = [lookup_invoice, check_refund_status]
TECH_SUPPORT_TOOLS = [check_system_status, lookup_user_account]
SCHEDULING_TOOLS = [check_lesson_schedule, reschedule_lesson]
CONTENT_TOOLS = [search_content_library, check_enrollment]
