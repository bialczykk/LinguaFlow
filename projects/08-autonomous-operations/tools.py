"""Department tools for LinguaFlow Autonomous Operations (Project 08).

17 tool functions across 6 departments, each decorated with:
  - @tool (langchain_core.tools) — makes the function callable by LangGraph agents
  - @traceable (langsmith) — logs every call to LangSmith with the project tag

All tools operate on mock data imported from data/ — no real LLM calls are made here.
The tools are the "actions" that department agents can perform; they are the leaf nodes
in the LangGraph tool-calling pattern.

Departments:
  1. Student Onboarding (2 tools)
  2. Tutor Management (3 tools)
  3. Content Pipeline (3 tools)
  4. Quality Assurance (3 tools)
  5. Support (4 tools)
  6. Reporting (2 tools)
"""

import time
from typing import Optional

from langchain_core.tools import tool
from langsmith import traceable

# ── Mock data imports ──────────────────────────────────────────────────────────
from data.students import STUDENTS
from data.tutors import TUTORS
from data.content_drafts import CONTENT_DRAFTS
from data.qa_records import QA_RECORDS
from data.invoices import INVOICES
from data.lessons import LESSONS
from data.system_status import SERVICES
from data.content_library import ENROLLMENTS
from data.metrics_seed import METRICS_SEED

# LangSmith tag applied to every tool trace for easy filtering in the dashboard.
_TAGS = ["p8-autonomous-operations"]

# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 1 — Student Onboarding
# ═══════════════════════════════════════════════════════════════════════════════

# CEFR level → recommended weekly study hours.
# Higher levels require less intense catch-up but more nuanced practice.
_WEEKLY_HOURS_BY_LEVEL = {
    "A1": 6,
    "A2": 5,
    "B1": 4,
    "B2": 3,
    "C1": 2,
    "C2": 2,
}

# Goals that signal the student likely starts at A1/A2 (beginner).
_BEGINNER_GOALS = {"travel english", "general english", "beginner"}


@tool
@traceable(tags=_TAGS)
def assess_student(student_id: str) -> dict | str:
    """Look up a student and return their profile with a level assessment.

    - If the student is found and has a cefr_level, the full profile is returned.
    - If the student is "new" (cefr_level is None), a suggested_assessment is added
      based on their stated goals to guide the onboarding agent.
    - If the student does not exist, an error string is returned.

    Args:
        student_id: The student's unique identifier (e.g. "S001").
    """
    student = next((s for s in STUDENTS if s["student_id"] == student_id), None)
    if student is None:
        return f"Student {student_id} not found."

    profile = dict(student)  # shallow copy so we don't mutate the source data

    if profile["cefr_level"] is None:
        # New student — suggest a placement level based on goals
        goals_lower = {g.lower() for g in (profile.get("goals") or [])}
        if goals_lower & _BEGINNER_GOALS:
            suggested = "A2"
        else:
            suggested = "B1"  # safe default for unknown goals
        profile["suggested_assessment"] = (
            f"Student has no CEFR level. Based on goals {profile['goals']}, "
            f"recommend placement test starting at {suggested}."
        )

    return profile


@tool
@traceable(tags=_TAGS)
def create_study_plan(student_id: str, level: str, goals: list[str]) -> dict:
    """Create a study plan for a student and return it as a dict.

    The plan_id is generated from the student_id and a timestamp to ensure
    uniqueness across calls. weekly_hours is derived from the student's level
    (beginners need more intensive study than advanced learners).

    Args:
        student_id: The student's unique identifier.
        level: CEFR level string (e.g. "B1").
        goals: List of learning goals that become the focus areas.
    """
    plan_id = f"SP-NEW-{student_id}-{int(time.time())}"
    weekly_hours = _WEEKLY_HOURS_BY_LEVEL.get(level.upper(), 3)

    return {
        "plan_id": plan_id,
        "student_id": student_id,
        "level": level,
        "focus_areas": goals,
        "weekly_hours": weekly_hours,
        "status": "active",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 2 — Tutor Management
# ═══════════════════════════════════════════════════════════════════════════════


@tool
@traceable(tags=_TAGS)
def search_tutors(specialty: str, level: Optional[str] = None) -> list[dict]:
    """Search for tutors by specialty (and optionally by CEFR level).

    The specialty match is a case-insensitive substring check against each
    tutor's specialties list. If level is provided, only tutors who cover that
    level are returned.

    Args:
        specialty: Keyword to search for in tutor specialties (e.g. "grammar").
        level: Optional CEFR level to filter by (e.g. "B1").
    """
    specialty_lower = specialty.lower()

    matches = [
        tutor for tutor in TUTORS
        if any(specialty_lower in s.lower() for s in tutor["specialties"])
    ]

    if level:
        matches = [t for t in matches if level in t["cefr_levels"]]

    return matches


@tool
@traceable(tags=_TAGS)
def check_availability(tutor_id: str) -> dict | str:
    """Return a tutor's availability slots and current capacity.

    Returns a dict with:
      - availability: list of open time slots
      - current_students: number of students currently assigned
      - max_students: maximum capacity
      - has_capacity: convenience boolean

    Returns an error string if the tutor is not found.

    Args:
        tutor_id: The tutor's unique identifier (e.g. "T001").
    """
    tutor = next((t for t in TUTORS if t["tutor_id"] == tutor_id), None)
    if tutor is None:
        return f"Tutor {tutor_id} not found."

    return {
        "tutor_id": tutor_id,
        "name": tutor["name"],
        "availability": tutor["availability"],
        "current_students": tutor["current_students"],
        "max_students": tutor["max_students"],
        "has_capacity": tutor["current_students"] < tutor["max_students"],
    }


@tool
@traceable(tags=_TAGS)
def assign_tutor(student_id: str, tutor_id: str) -> dict:
    """Attempt to assign a tutor to a student.

    Checks that the tutor exists and has available capacity
    (current_students < max_students) before confirming the assignment.

    Returns a dict with:
      - success: True/False
      - student_id, tutor_id (on success)
      - reason: explanation string (on failure)

    Args:
        student_id: The student's unique identifier.
        tutor_id: The tutor's unique identifier.
    """
    tutor = next((t for t in TUTORS if t["tutor_id"] == tutor_id), None)
    if tutor is None:
        return {"success": False, "reason": f"Tutor {tutor_id} not found."}

    if tutor["current_students"] >= tutor["max_students"]:
        return {
            "success": False,
            "reason": (
                f"Tutor {tutor['name']} is at full capacity "
                f"({tutor['current_students']}/{tutor['max_students']} students)."
            ),
        }

    return {
        "success": True,
        "student_id": student_id,
        "tutor_id": tutor_id,
        "tutor_name": tutor["name"],
        "message": f"Successfully assigned {tutor['name']} to student {student_id}.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 3 — Content Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


@tool
@traceable(tags=_TAGS)
def generate_content(topic: str, content_type: str, level: str) -> dict:
    """Generate a new content draft record (mock — no LLM call).

    In a real system this would invoke an LLM to write the content. Here we
    create a draft metadata record so the pipeline can track it and route it
    through review and publishing stages.

    The content_id is unique per call using a timestamp suffix.

    Args:
        topic: Subject matter for the content (e.g. "Past perfect tense").
        content_type: Category of content (e.g. "grammar_explanation", "vocabulary_exercise").
        level: CEFR level this content targets (e.g. "B1").
    """
    content_id = f"CD-NEW-{int(time.time())}"
    title = f"{topic} — {level} {content_type.replace('_', ' ').title()}"

    return {
        "content_id": content_id,
        "title": title,
        "type": content_type,
        "level": level,
        "status": "draft",
        "author": "content_agent",
    }


@tool
@traceable(tags=_TAGS)
def submit_for_review(content_id: str) -> dict:
    """Submit a content draft for QA review.

    Looks up the content in CONTENT_DRAFTS and conceptually transitions it
    to "in_review" status. Returns a success dict if found, failure dict if not.

    Args:
        content_id: The content item's unique identifier (e.g. "CD-004").
    """
    draft = next((c for c in CONTENT_DRAFTS if c["content_id"] == content_id), None)
    if draft is None:
        return {
            "success": False,
            "content_id": content_id,
            "reason": f"Content {content_id} not found in drafts.",
        }

    return {
        "success": True,
        "content_id": content_id,
        "title": draft["title"],
        "previous_status": draft["status"],
        "new_status": "in_review",
        "message": f"Content '{draft['title']}' submitted for QA review.",
    }


@tool
@traceable(tags=_TAGS)
def publish_content(content_id: str) -> dict:
    """Publish a content item if it has passed QA.

    Only content with qa_status="passed" can be published. Returns a success
    dict if publishable, or a failure dict with a reason if not.

    Args:
        content_id: The content item's unique identifier (e.g. "CD-001").
    """
    draft = next((c for c in CONTENT_DRAFTS if c["content_id"] == content_id), None)
    if draft is None:
        return {
            "success": False,
            "content_id": content_id,
            "reason": f"Content {content_id} not found.",
        }

    if draft.get("qa_status") != "passed":
        return {
            "success": False,
            "content_id": content_id,
            "reason": (
                f"Content cannot be published. "
                f"QA status is '{draft.get('qa_status', 'unknown')}' — must be 'passed'."
            ),
        }

    return {
        "success": True,
        "content_id": content_id,
        "title": draft["title"],
        "previous_status": draft["status"],
        "new_status": "published",
        "message": f"Content '{draft['title']}' is now published.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 4 — Quality Assurance
# ═══════════════════════════════════════════════════════════════════════════════


@tool
@traceable(tags=_TAGS)
def review_content(content_id: str) -> dict:
    """Return a review summary for a content item.

    Looks up the content in CONTENT_DRAFTS and joins it with any matching
    QA_RECORDS entry. The qa_record field is either a full record dict or
    the string "no review found" when none exists.

    Returns an error dict if the content_id is not found at all.

    Args:
        content_id: The content item's unique identifier (e.g. "CD-001").
    """
    draft = next((c for c in CONTENT_DRAFTS if c["content_id"] == content_id), None)
    if draft is None:
        return {"error": f"Content {content_id} not found.", "content_id": content_id}

    qa_record = next((r for r in QA_RECORDS if r["content_id"] == content_id), None)

    return {
        "content_id": content_id,
        "title": draft["title"],
        "level": draft["level"],
        "status": draft["status"],
        "qa_status": draft.get("qa_status"),
        "qa_record": qa_record if qa_record is not None else "no review found",
    }


@tool
@traceable(tags=_TAGS)
def flag_issue(department: str, issue: str) -> dict:
    """Create a flag record for an issue in a given department.

    Flags are used to surface problems that need human or agent attention.
    A unique flag_id is generated using a timestamp.

    Args:
        department: The department raising the issue (e.g. "content_pipeline").
        issue: A description of the problem (e.g. "Missing examples in B2 grammar exercise").
    """
    flag_id = f"FLAG-{int(time.time())}"

    return {
        "flag_id": flag_id,
        "department": department,
        "issue": issue,
        "status": "open",
        "message": f"Issue flagged in {department}: {issue}",
    }


@tool
@traceable(tags=_TAGS)
def check_satisfaction(student_id: str) -> dict:
    """Return a mock satisfaction score for a student based on their lesson history.

    Completed lessons are used as a proxy for satisfaction — the score is a
    simple heuristic (0.0–1.0). In a real system this would query survey results.

    Returns an error dict if the student is not found.

    Args:
        student_id: The student's unique identifier (e.g. "S001").
    """
    student = next((s for s in STUDENTS if s["student_id"] == student_id), None)
    if student is None:
        return {"error": f"Student {student_id} not found.", "student_id": student_id}

    completed = [l for l in LESSONS if l["student_id"] == student_id and l["status"] == "completed"]
    count = len(completed)

    # Mock score: starts at 0.7 baseline, +0.05 per completed lesson, capped at 1.0
    score = min(1.0, 0.7 + count * 0.05)

    summary = (
        f"Student has completed {count} lesson(s). "
        f"Estimated satisfaction score: {score:.2f}/1.0."
    )

    return {
        "student_id": student_id,
        "name": student["name"],
        "completed_lessons": count,
        "satisfaction_score": score,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 5 — Support
# ═══════════════════════════════════════════════════════════════════════════════


@tool
@traceable(tags=_TAGS)
def lookup_invoice(student_id: str) -> list[dict]:
    """Find all invoices for a student.

    Returns a list of invoice dicts (may be empty if no billing history exists).

    Args:
        student_id: The student's unique identifier (e.g. "S001").
    """
    return [inv for inv in INVOICES if inv["student_id"] == student_id]


@tool
@traceable(tags=_TAGS)
def check_schedule(student_id: str) -> list[dict]:
    """Find all lessons for a student.

    Returns a list of lesson dicts (may be empty for new students).

    Args:
        student_id: The student's unique identifier (e.g. "S001").
    """
    return [lesson for lesson in LESSONS if lesson["student_id"] == student_id]


@tool
@traceable(tags=_TAGS)
def check_system_status(service: str) -> dict:
    """Return the health status for a platform service.

    Looks up the service in SERVICES. Returns the health dict if found,
    or an error dict if the service name is not recognised.

    Args:
        service: Service name key (e.g. "video_platform", "chat_system").
    """
    health = SERVICES.get(service)
    if health is None:
        return {
            "error": f"Service '{service}' not found.",
            "available_services": list(SERVICES.keys()),
        }
    return health


@tool
@traceable(tags=_TAGS)
def check_enrollment(student_id: str) -> list[dict]:
    """Find all course enrollments for a student.

    Returns a list of enrollment dicts (may be empty for new students).

    Args:
        student_id: The student's unique identifier (e.g. "S001").
    """
    return [e for e in ENROLLMENTS if e["student_id"] == student_id]


# ═══════════════════════════════════════════════════════════════════════════════
# DEPARTMENT 6 — Reporting
# ═══════════════════════════════════════════════════════════════════════════════

# Keys relevant to each department — used to filter METRICS_SEED.
_DEPARTMENT_METRIC_KEYS = {
    "student_onboarding": ["students_onboarded"],
    "tutor_management": ["tutors_assigned"],
    "content_pipeline": ["content_generated", "content_published"],
    "quality_assurance": ["qa_reviews", "qa_flags"],
    "support": ["support_requests", "support_resolved"],
    "reporting": ["total_requests", "department_invocations"],
}


@tool
@traceable(tags=_TAGS)
def aggregate_metrics(department: str, period: str) -> dict:
    """Return aggregated platform metrics, optionally filtered by department.

    When department is "all", the full METRICS_SEED dict is returned.
    Otherwise, only the metric keys relevant to the specified department are included.

    Args:
        department: Department name or "all" (e.g. "content_pipeline").
        period: Reporting period label (e.g. "this_week") — stored for context.
    """
    if department == "all":
        return {**METRICS_SEED, "period": period}

    relevant_keys = _DEPARTMENT_METRIC_KEYS.get(department, list(METRICS_SEED.keys()))
    filtered = {k: METRICS_SEED[k] for k in relevant_keys if k in METRICS_SEED}
    filtered["department"] = department
    filtered["period"] = period
    return filtered


@tool
@traceable(tags=_TAGS)
def get_department_state(department: str) -> dict:
    """Return a live summary of state for a given department based on mock data.

    Each department has a meaningful state snapshot derived from the data modules:
      - student_onboarding: counts of active vs new students
      - tutor_management: total tutors, capacity utilisation
      - content_pipeline: content item counts per status
      - quality_assurance: QA record counts and pass/fail breakdown
      - support: invoice and lesson counts
      - reporting: metrics store summary

    Unknown departments return a generic dict.

    Args:
        department: Department identifier (e.g. "content_pipeline").
    """
    if department == "student_onboarding":
        active = sum(1 for s in STUDENTS if s["status"] == "active")
        new = sum(1 for s in STUDENTS if s["status"] == "new")
        return {
            "department": department,
            "total_students": len(STUDENTS),
            "active_students": active,
            "new_students": new,
        }

    if department == "tutor_management":
        total = len(TUTORS)
        total_slots = sum(t["max_students"] for t in TUTORS)
        used_slots = sum(t["current_students"] for t in TUTORS)
        return {
            "department": department,
            "total_tutors": total,
            "total_capacity": total_slots,
            "used_capacity": used_slots,
            "available_slots": total_slots - used_slots,
        }

    if department == "content_pipeline":
        from collections import Counter
        status_counts = Counter(c["status"] for c in CONTENT_DRAFTS)
        return {
            "department": department,
            "total_items": len(CONTENT_DRAFTS),
            "drafts": status_counts.get("draft", 0),
            "in_review": status_counts.get("in_review", 0),
            "published": status_counts.get("published", 0),
            "flagged": status_counts.get("flagged", 0),
        }

    if department == "quality_assurance":
        passed = sum(1 for r in QA_RECORDS if r["result"] == "pass")
        failed = sum(1 for r in QA_RECORDS if r["result"] == "fail")
        return {
            "department": department,
            "total_reviews": len(QA_RECORDS),
            "passed": passed,
            "failed": failed,
        }

    if department == "support":
        return {
            "department": department,
            "total_invoices": len(INVOICES),
            "total_lessons": len(LESSONS),
            "total_enrollments": len(ENROLLMENTS),
        }

    if department == "reporting":
        return {
            "department": department,
            "metrics_keys": list(METRICS_SEED.keys()),
            "total_requests": METRICS_SEED.get("total_requests", 0),
        }

    # Fallback for unrecognised departments
    return {
        "department": department,
        "message": f"No specific state handler for department '{department}'.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool group exports — consumed by department agents in agents.py
# ═══════════════════════════════════════════════════════════════════════════════

STUDENT_ONBOARDING_TOOLS = [assess_student, create_study_plan]
TUTOR_MANAGEMENT_TOOLS = [search_tutors, check_availability, assign_tutor]
CONTENT_PIPELINE_TOOLS = [generate_content, submit_for_review, publish_content]
QA_TOOLS = [review_content, flag_issue, check_satisfaction]
SUPPORT_TOOLS = [lookup_invoice, check_schedule, check_system_status, check_enrollment]
REPORTING_TOOLS = [aggregate_metrics, get_department_state]
