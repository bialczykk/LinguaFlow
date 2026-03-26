"""Tests for department tool functions against mock data."""

import pytest

from tools import (
    lookup_invoice,
    check_refund_status,
    check_system_status,
    lookup_user_account,
    check_lesson_schedule,
    reschedule_lesson,
    search_content_library,
    check_enrollment,
)


class TestBillingTools:
    def test_lookup_invoice_found(self):
        result = lookup_invoice.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(inv["student_id"] == "S001" for inv in result)

    def test_lookup_invoice_not_found(self):
        result = lookup_invoice.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_check_refund_status_found(self):
        result = check_refund_status.invoke({"invoice_id": "INV-004"})
        assert isinstance(result, dict)
        assert result["invoice_id"] == "INV-004"
        assert result["status"] == "refunded"

    def test_check_refund_status_not_found(self):
        result = check_refund_status.invoke({"invoice_id": "INV-999"})
        assert isinstance(result, str)
        assert "not found" in result.lower()


class TestTechSupportTools:
    def test_check_system_status_known(self):
        result = check_system_status.invoke({"service": "chat_system"})
        assert isinstance(result, dict)
        assert result["status"] == "degraded"

    def test_check_system_status_unknown(self):
        result = check_system_status.invoke({"service": "nonexistent"})
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_lookup_user_account_found(self):
        result = lookup_user_account.invoke({"email": "maria.garcia@email.com"})
        assert isinstance(result, dict)
        assert result["student_id"] == "S001"

    def test_lookup_user_account_not_found(self):
        result = lookup_user_account.invoke({"email": "nobody@email.com"})
        assert isinstance(result, str)
        assert "not found" in result.lower()


class TestSchedulingTools:
    def test_check_lesson_schedule(self):
        result = check_lesson_schedule.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(l["student_id"] == "S001" for l in result)

    def test_check_lesson_schedule_no_lessons(self):
        result = check_lesson_schedule.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_reschedule_lesson_success(self):
        result = reschedule_lesson.invoke({"lesson_id": "L003", "new_date": "2026-04-05"})
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["new_date"] == "2026-04-05"

    def test_reschedule_lesson_not_found(self):
        result = reschedule_lesson.invoke({"lesson_id": "L999", "new_date": "2026-04-05"})
        assert isinstance(result, dict)
        assert result["success"] is False


class TestContentTools:
    def test_search_content_library_with_level(self):
        result = search_content_library.invoke({"query": "business", "level": "B2"})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_search_content_library_no_level(self):
        result = search_content_library.invoke({"query": "grammar"})
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_search_content_library_no_results(self):
        result = search_content_library.invoke({"query": "quantum physics", "level": "C2"})
        assert isinstance(result, list)
        assert len(result) == 0

    def test_check_enrollment_found(self):
        result = check_enrollment.invoke({"student_id": "S001"})
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(e["student_id"] == "S001" for e in result)

    def test_check_enrollment_not_found(self):
        result = check_enrollment.invoke({"student_id": "S999"})
        assert isinstance(result, list)
        assert len(result) == 0
