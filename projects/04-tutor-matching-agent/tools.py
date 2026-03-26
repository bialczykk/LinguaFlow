"""LangGraph tool functions for the Tutor Matching Agent.

Defines three tools that simulate external API calls:
- search_tutors: query the tutor database by criteria
- check_availability: look up a tutor's calendar slots
- book_session: reserve a session and get a confirmation

LangGraph concepts demonstrated:
- @tool decorator: turns a plain function into a LangGraph-compatible tool
- Tool functions are bound to the model via model.bind_tools()
- The prebuilt ToolNode executes these when the LLM requests them

These tools operate on the mock data in data/tutors.py and data/calendar.py.
In a real application, they would call REST APIs.
"""

import uuid

from langchain_core.tools import tool
from langsmith import traceable

from data.tutors import TUTORS
from data.calendar import SCHEDULES

_TAGS = ["p4-tutor-matching"]


@tool
@traceable(name="search_tutors", run_type="tool", tags=_TAGS)
def search_tutors(specialization: str, timezone: str | None = None, availability: str | None = None) -> list[dict]:
    """Search the tutor database by specialization, timezone, and availability.

    Args:
        specialization: Required. One of: grammar, conversation, business_english, exam_prep.
        timezone: Optional. Filter by tutor's timezone (e.g., 'Europe/London').
        availability: Optional. Preferred date in YYYY-MM-DD format to check availability.

    Returns:
        List of matching tutor profiles with all fields.
    """
    results = []
    for tutor in TUTORS:
        # Filter by required specialization
        if specialization not in tutor["specializations"]:
            continue
        # Optionally filter by timezone
        if timezone and tutor["timezone"] != timezone:
            continue
        # Optionally filter by date availability — only include tutors with at least one free slot
        if availability:
            tutor_slots = SCHEDULES.get(tutor["tutor_id"], [])
            has_availability = any(s["date"] == availability and not s["booked"] for s in tutor_slots)
            if not has_availability:
                continue
        results.append(tutor)
    return results


@tool
@traceable(name="check_availability", run_type="tool", tags=_TAGS)
def check_availability(tutor_id: str, date: str) -> list[dict] | str:
    """Check a tutor's available time slots for a specific date.

    Args:
        tutor_id: The tutor's unique ID (e.g., 't1').
        date: The date to check in YYYY-MM-DD format.

    Returns:
        List of available time slots, or an error message if tutor not found.
    """
    # Return an error string if the tutor doesn't exist in the schedule data
    if tutor_id not in SCHEDULES:
        return f"Tutor {tutor_id} not found in the system."

    slots = SCHEDULES[tutor_id]
    # Return only unbooked slots that match the requested date
    available = [
        {"date": s["date"], "start_time": s["start_time"], "end_time": s["end_time"]}
        for s in slots if s["date"] == date and not s["booked"]
    ]
    return available


@tool
@traceable(name="book_session", run_type="tool", tags=_TAGS)
def book_session(tutor_id: str, date: str, time: str, student_name: str) -> dict | str:
    """Book a tutoring session with a specific tutor.

    Args:
        tutor_id: The tutor's unique ID (e.g., 't1').
        date: Session date in YYYY-MM-DD format.
        time: Session start time in HH:MM format.
        student_name: The student's name for the booking.

    Returns:
        Booking confirmation dict, or an error message if the slot is not available.
    """
    # Guard: tutor must exist in the scheduling system
    if tutor_id not in SCHEDULES:
        return f"Tutor {tutor_id} not found in the system."

    # Find the matching slot and attempt to book it
    for slot in SCHEDULES[tutor_id]:
        if slot["date"] == date and slot["start_time"] == time:
            if slot["booked"]:
                # Slot is already taken — inform the caller
                return f"Slot {time} on {date} is not available — already booked."
            # Mark the slot as booked in-memory (simulates a calendar API write)
            slot["booked"] = True
            # Look up the tutor's full name for the confirmation
            tutor_name = tutor_id
            for t in TUTORS:
                if t["tutor_id"] == tutor_id:
                    tutor_name = t["name"]
                    break
            # Return a confirmation record with a unique booking ID
            return {
                "confirmation_id": f"BK-{uuid.uuid4().hex[:8].upper()}",
                "tutor_name": tutor_name,
                "student_name": student_name,
                "date": date,
                "time": time,
                "duration_minutes": 60,
            }

    return f"No slot found for tutor {tutor_id} at {time} on {date}."
