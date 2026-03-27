"""Content items in various pipeline stages.

Used by the content_pipeline and quality_assurance departments.
Items progress through the pipeline: draft → in_review → published (or flagged).

Status values:
- "published": live and available to students
- "in_review": submitted for QA review
- "draft": being written, not yet submitted
- "flagged": failed QA — needs revision before republishing

qa_status tracks the QA outcome: "passed", "failed", "pending", or None (not submitted).
"""

CONTENT_DRAFTS = [
    {
        "content_id": "CD-001",
        "title": "Present Perfect Tense — When to Use It",
        "type": "grammar_explanation",
        "level": "B1",
        "status": "published",
        "author": "system",
        "created_date": "2026-03-10",
        "qa_status": "passed",
    },
    {
        "content_id": "CD-002",
        "title": "Business Email Vocabulary Builder",
        "type": "vocabulary_exercise",
        "level": "B2",
        "status": "published",
        "author": "system",
        "created_date": "2026-03-12",
        "qa_status": "passed",
    },
    {
        "content_id": "CD-003",
        "title": "IELTS Reading: Climate Change Passage",
        "type": "reading_passage",
        "level": "B2",
        "status": "in_review",
        "author": "system",
        "created_date": "2026-03-20",
        "qa_status": "pending",
    },
    {
        "content_id": "CD-004",
        "title": "Phrasal Verbs for Travel",
        "type": "vocabulary_exercise",
        "level": "A2",
        "status": "draft",
        "author": "system",
        "created_date": "2026-03-25",
        "qa_status": None,
    },
    {
        "content_id": "CD-005",
        "title": "Conditionals Deep Dive",
        "type": "grammar_explanation",
        "level": "B2",
        "status": "flagged",
        "author": "system",
        "created_date": "2026-03-15",
        "qa_status": "failed",
        "qa_notes": "Incorrect example in third conditional section. Mixed tenses in explanation.",
    },
]
