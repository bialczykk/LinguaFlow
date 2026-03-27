"""Seed data for the MetricsStore.

Provides initial non-zero values so the reporting department has historical
data to work with even on first run. Matches the MetricsStore TypedDict shape
defined in models.py.

These values represent the state of the platform before the demo session:
- 4 students already onboarded (S001–S004)
- 4 tutors already assigned (one per active student)
- 5 content items generated, 2 published (CD-001, CD-002)
- 3 QA reviews completed (2 pass, 1 fail)
"""

METRICS_SEED = {
    "students_onboarded": 4,
    "tutors_assigned": 4,
    "content_generated": 5,
    "content_published": 2,
    "qa_reviews": 3,
    "qa_flags": 1,
    "support_requests": 0,
    "support_resolved": 0,
    "total_requests": 0,
    "department_invocations": {},
}
