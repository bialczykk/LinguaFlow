"""Tests for tool functions against mock data. No LLM calls."""
import pytest
from tools import search_tutors, check_availability, book_session
from data.calendar import SCHEDULES


class TestSearchTutors:
    def test_search_by_specialization(self):
        result = search_tutors.invoke({"specialization": "exam_prep"})
        assert len(result) > 0
        for tutor in result:
            assert "exam_prep" in tutor["specializations"]

    def test_search_by_specialization_and_timezone(self):
        result = search_tutors.invoke({"specialization": "grammar", "timezone": "Europe/London"})
        assert len(result) > 0
        for tutor in result:
            assert "grammar" in tutor["specializations"]
            assert tutor["timezone"] == "Europe/London"

    def test_search_no_matches(self):
        result = search_tutors.invoke({"specialization": "grammar", "timezone": "Antarctica/South_Pole"})
        assert result == []

    def test_search_returns_all_fields(self):
        result = search_tutors.invoke({"specialization": "conversation"})
        assert len(result) > 0
        tutor = result[0]
        for field in ["tutor_id", "name", "specializations", "timezone", "rating", "bio", "hourly_rate"]:
            assert field in tutor


class TestCheckAvailability:
    def test_available_slots_exist(self):
        result = check_availability.invoke({"tutor_id": "t1", "date": "2026-04-01"})
        assert len(result) > 0
        for slot in result:
            assert slot["date"] == "2026-04-01"

    def test_no_slots_on_missing_date(self):
        result = check_availability.invoke({"tutor_id": "t1", "date": "2026-12-25"})
        assert result == []

    def test_invalid_tutor_id(self):
        result = check_availability.invoke({"tutor_id": "t999", "date": "2026-04-01"})
        assert "not found" in result.lower() or result == []


class TestBookSession:
    def test_successful_booking(self):
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "09:00":
                slot["booked"] = False
        result = book_session.invoke({"tutor_id": "t1", "date": "2026-04-01", "time": "09:00", "student_name": "Test Student"})
        assert "confirmation_id" in result
        assert result["tutor_name"] == "Alice Smith"
        assert result["student_name"] == "Test Student"

    def test_booking_marks_slot_as_taken(self):
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "11:00":
                slot["booked"] = False
        book_session.invoke({"tutor_id": "t1", "date": "2026-04-01", "time": "11:00", "student_name": "Test Student"})
        for slot in SCHEDULES["t1"]:
            if slot["date"] == "2026-04-01" and slot["start_time"] == "11:00":
                assert slot["booked"] is True

    def test_booking_unavailable_slot(self):
        for slot in SCHEDULES.get("t1", []):
            if slot["date"] == "2026-04-01" and slot["start_time"] == "14:00":
                slot["booked"] = True
        result = book_session.invoke({"tutor_id": "t1", "date": "2026-04-01", "time": "14:00", "student_name": "Test Student"})
        assert "not available" in str(result).lower() or "already booked" in str(result).lower()
