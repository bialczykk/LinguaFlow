"""Invoice records for existing students (S001–S004).

Covers various billing statuses — paid, pending, refunded, disputed —
to support billing support and refund workflow demos.

Note: S005 and S006 are new students with no billing history yet.
INV-005 and INV-006 represent a duplicate charge scenario for S003 (Yuki Tanaka).
"""

INVOICES = [
    {
        "invoice_id": "INV-001",
        "student_id": "S001",
        "amount": 45.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L001",
        "date": "2026-03-20",
        "description": "1-hour business English lesson with Sarah Johnson",
    },
    {
        "invoice_id": "INV-002",
        "student_id": "S001",
        "amount": 45.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L002",
        "date": "2026-03-22",
        "description": "1-hour presentation skills session with Sarah Johnson",
    },
    {
        "invoice_id": "INV-003",
        "student_id": "S002",
        "amount": 40.00,
        "currency": "USD",
        "status": "pending",
        "lesson_id": "L003",
        "date": "2026-03-25",
        "description": "1-hour grammar session with James Wilson",
    },
    {
        "invoice_id": "INV-004",
        "student_id": "S002",
        "amount": 40.00,
        "currency": "USD",
        "status": "refunded",
        "lesson_id": "L004",
        "date": "2026-03-18",
        "description": "Cancelled grammar lesson — full refund issued",
    },
    {
        "invoice_id": "INV-005",
        "student_id": "S003",
        "amount": 55.00,
        "currency": "USD",
        "status": "disputed",
        "lesson_id": "L005",
        "date": "2026-03-23",
        "description": "1-hour IELTS prep — student reports duplicate charge",
    },
    {
        "invoice_id": "INV-006",
        "student_id": "S003",
        "amount": 55.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L005",
        "date": "2026-03-23",
        "description": "1-hour IELTS prep — duplicate charge (same lesson, same day)",
    },
    {
        "invoice_id": "INV-007",
        "student_id": "S004",
        "amount": 60.00,
        "currency": "USD",
        "status": "paid",
        "lesson_id": "L006",
        "date": "2026-03-21",
        "description": "1-hour advanced conversation with David Brown",
    },
    {
        "invoice_id": "INV-008",
        "student_id": "S004",
        "amount": 60.00,
        "currency": "USD",
        "status": "pending",
        "lesson_id": "L009",
        "date": "2026-03-27",
        "description": "1-hour idioms and expressions session with David Brown",
    },
]
