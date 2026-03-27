"""Student account details for all students S001–S006.

Used by the support department for account lookups, tech issue diagnosis,
and new student onboarding checks.

Active students (S001–S004) have full account details and last_login history.
New students (S005–S006) have minimal account data — no login history yet.
"""

ACCOUNTS = {
    "S001": {
        "student_id": "S001",
        "email": "alice.chen@email.com",
        "plan_type": "premium",
        "timezone": "Asia/Shanghai",
        "last_login": "2026-03-26",
        "known_issues": [],
    },
    "S002": {
        "student_id": "S002",
        "email": "marco.rossi@email.com",
        "plan_type": "standard",
        "timezone": "Europe/Rome",
        "last_login": "2026-03-25",
        "known_issues": ["password_reset_pending"],
    },
    "S003": {
        "student_id": "S003",
        "email": "yuki.tanaka@email.com",
        "plan_type": "premium",
        "timezone": "Asia/Tokyo",
        "last_login": "2026-03-24",
        "known_issues": ["duplicate_charge_reported"],
    },
    "S004": {
        "student_id": "S004",
        "email": "priya.sharma@email.com",
        "plan_type": "premium",
        "timezone": "Asia/Kolkata",
        "last_login": "2026-03-20",
        "known_issues": [],
    },
    "S005": {
        "student_id": "S005",
        "email": "lars.eriksson@email.com",
        "plan_type": None,    # Not yet enrolled
        "timezone": "Europe/Stockholm",
        "last_login": None,   # Never logged in
        "known_issues": [],
    },
    "S006": {
        "student_id": "S006",
        "email": "maria.silva@email.com",
        "plan_type": None,    # Not yet enrolled
        "timezone": "America/Sao_Paulo",
        "last_login": None,   # Never logged in
        "known_issues": [],
    },
}
