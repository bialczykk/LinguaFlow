---
name: reporting
description: KPI definitions, report structure, metric interpretation guidelines, and narrative summary rules for LinguaFlow operational reporting
---

# Reporting

## KPI Definitions

Understand what each metric represents before including it in a report:

- **student_retention_rate:** Percentage of students who continue their subscription into the next billing period. Formula: (active_students_end / active_students_start) × 100. A healthy rate is above 85%.
- **avg_sessions_per_week:** Average number of completed sessions per active student per week. Target: 1.5-2.5 sessions. Below 1.0 signals disengagement.
- **content_pass_rate:** Percentage of content submissions that pass QA review on the first attempt. Formula: (first_pass_approvals / total_submissions) × 100. Target: above 75%.
- **tutor_utilisation:** Percentage of a tutor's available capacity currently filled with students. Formula: (current_students / max_students) × 100. Healthy range: 70-90%.
- **avg_level_progression_weeks:** Average number of weeks for a student to advance one CEFR level. Lower is better, but below 8 weeks may indicate placement errors.
- **support_resolution_time_hours:** Average time in hours from ticket creation to resolution. Target: under 24 hours for billing/scheduling, under 4 hours for tech support.

## Report Template Structure

Every operational report must follow this structure:

1. **Report header:** report_id, period (start_date to end_date), generated_at, department(s) covered
2. **Executive summary:** 3-5 bullet points highlighting the most significant findings for the period
3. **KPI table:** Metric name, current value, target value, delta from previous period, status (on_track / at_risk / critical)
4. **Trend notes:** For any metric marked at_risk or critical, include a 1-2 sentence explanation of the likely cause
5. **Recommendations:** Up to 3 actionable recommendations based on the data, each linked to a specific KPI

## How to Interpret Metrics

- A drop in `student_retention_rate` of more than 5 points in a single period is a critical signal — cross-reference with `support_resolution_time_hours` and `avg_sessions_per_week` to identify root cause.
- A low `content_pass_rate` combined with a high `avg_level_progression_weeks` may indicate content difficulty is misaligned with student levels.
- High `tutor_utilisation` (above 90%) risks tutor burnout and should prompt a hiring recommendation.

## Narrative Summary Guidelines

The executive summary must be written in plain English, not jargon. Assume the reader is a non-technical stakeholder:

- Lead with the most important insight, not the most impressive number.
- Use comparisons to previous periods to provide context ("up 3 points from last month").
- Flag risks explicitly — do not bury a critical signal in positive framing.
- Keep the total narrative to under 150 words.
