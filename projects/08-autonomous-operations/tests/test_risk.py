"""Tests for risk assessment rules.

Tests are written first (TDD red phase) to verify that:
- HIGH_RISK_ACTIONS contains the expected departments and action names
- assess_risk() correctly returns "high" or "low" based on classification input
"""

import pytest

from risk import assess_risk, HIGH_RISK_ACTIONS


class TestHighRiskActions:
    """Verify the risk map covers all expected high-risk actions per department."""

    def test_publish_is_high_risk(self):
        """Publishing content is irreversible — must require human approval."""
        assert "publish_content" in HIGH_RISK_ACTIONS["content_pipeline"]

    def test_assign_tutor_is_high_risk(self):
        """Assigning a tutor affects a student's experience — requires approval."""
        assert "assign_tutor" in HIGH_RISK_ACTIONS["tutor_management"]

    def test_create_study_plan_is_high_risk(self):
        """Creating a study plan shapes a student's learning path — high stakes."""
        assert "create_study_plan" in HIGH_RISK_ACTIONS["student_onboarding"]

    def test_flag_issue_is_high_risk(self):
        """Flagging a QA issue can escalate to significant action — requires approval."""
        assert "flag_issue" in HIGH_RISK_ACTIONS["quality_assurance"]


class TestAssessRisk:
    """Verify risk assessment function produces correct tier labels."""

    def test_low_risk_lookup(self):
        """Data lookups should auto-execute without human approval."""
        classification = {
            "departments": ["support"],
            "action_type": "lookup",
        }
        assert assess_risk(classification) == "low"

    def test_high_risk_publish(self):
        """Publishing content should trigger human approval."""
        classification = {
            "departments": ["content_pipeline"],
            "action_type": "publish_content",
        }
        assert assess_risk(classification) == "high"

    def test_high_risk_assign_tutor(self):
        """Tutor assignment should trigger human approval."""
        classification = {
            "departments": ["tutor_management"],
            "action_type": "assign_tutor",
        }
        assert assess_risk(classification) == "high"

    def test_multi_dept_high_if_any_high(self):
        """If any department marks the action as high risk, the whole task is high risk."""
        classification = {
            "departments": ["support", "content_pipeline"],
            "action_type": "publish_content",
        }
        assert assess_risk(classification) == "high"

    def test_unknown_action_is_low(self):
        """Unknown departments and actions default to low risk (safe default)."""
        classification = {
            "departments": ["reporting"],
            "action_type": "aggregate_metrics",
        }
        assert assess_risk(classification) == "low"

    def test_follow_up_task_risk(self):
        """Follow-up tasks are also risk-assessed — is_follow_up flag doesn't bypass approval."""
        classification = {
            "departments": ["tutor_management"],
            "action_type": "assign_tutor",
            "is_follow_up": True,
        }
        assert assess_risk(classification) == "high"
