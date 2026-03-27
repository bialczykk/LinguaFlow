"""QA review history for content items.

Links to content_drafts via content_id. Each review records the outcome,
reviewer notes, and any flags raised.

Used by the quality_assurance department to look up review history and by
the content_pipeline department to determine if content needs rework.

score: float 0.0–1.0, where 0.7+ is a pass threshold.
flags: optional list of issue tags, only present on failed reviews.
"""

QA_RECORDS = [
    {
        "review_id": "QA-001",
        "content_id": "CD-001",
        "reviewer": "qa_agent",
        "date": "2026-03-11",
        "result": "pass",
        "score": 0.92,
        "notes": "Clear explanations with good examples. Minor formatting suggestion applied.",
    },
    {
        "review_id": "QA-002",
        "content_id": "CD-002",
        "reviewer": "qa_agent",
        "date": "2026-03-13",
        "result": "pass",
        "score": 0.88,
        "notes": "Appropriate vocabulary level. Could use more context sentences, noted for v2.",
    },
    {
        "review_id": "QA-003",
        "content_id": "CD-005",
        "reviewer": "qa_agent",
        "date": "2026-03-16",
        "result": "fail",
        "score": 0.45,
        "notes": "Incorrect example in third conditional section. Mixed tenses in explanation.",
        "flags": ["accuracy_error", "needs_revision"],
    },
]
