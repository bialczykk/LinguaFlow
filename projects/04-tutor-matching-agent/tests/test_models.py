"""Tests for Pydantic models and LangGraph state schema."""
import typing
import pytest
from models import Tutor, TimeSlot, BookingConfirmation, TutorMatchingState


class TestTutor:
    def test_valid_tutor(self):
        tutor = Tutor(tutor_id="t1", name="Alice Smith", specializations=["grammar", "exam_prep"],
                      timezone="Europe/London", rating=4.8, bio="10 years experience.", hourly_rate=35.0)
        assert tutor.tutor_id == "t1"
        assert "grammar" in tutor.specializations
        assert tutor.rating == 4.8

    def test_rating_bounds(self):
        with pytest.raises(Exception):
            Tutor(tutor_id="t1", name="X", specializations=["grammar"],
                  timezone="UTC", rating=5.5, bio="x", hourly_rate=10.0)


class TestTimeSlot:
    def test_valid_time_slot(self):
        slot = TimeSlot(date="2026-04-01", start_time="09:00", end_time="10:00")
        assert slot.date == "2026-04-01"
        assert slot.start_time == "09:00"


class TestBookingConfirmation:
    def test_valid_booking(self):
        booking = BookingConfirmation(confirmation_id="BK-001", tutor_name="Alice Smith",
                                     student_name="Bob", date="2026-04-01", time="09:00", duration_minutes=60)
        assert booking.confirmation_id == "BK-001"
        assert booking.duration_minutes == 60


class TestTutorMatchingState:
    def test_state_has_required_fields(self):
        hints = typing.get_type_hints(TutorMatchingState)
        assert "messages" in hints
        assert "phase" in hints
        assert "preferences" in hints
        assert "search_results" in hints
        assert "selected_tutor" in hints
        assert "booking_confirmation" in hints
