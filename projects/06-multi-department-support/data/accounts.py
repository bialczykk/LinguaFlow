"""Mock student account data for tech support lookups.

Student IDs match across all data modules (invoices, lessons, enrollments).
"""

ACCOUNTS = {
    "S001": {
        "student_id": "S001",
        "name": "Maria Garcia",
        "email": "maria.garcia@email.com",
        "plan": "premium",
        "timezone": "Europe/Madrid",
        "joined": "2025-09-15",
        "last_login": "2026-03-26",
        "known_issues": [],
    },
    "S002": {
        "student_id": "S002",
        "name": "Kenji Tanaka",
        "email": "kenji.tanaka@email.com",
        "plan": "standard",
        "timezone": "Asia/Tokyo",
        "joined": "2025-11-01",
        "last_login": "2026-03-25",
        "known_issues": ["password_reset_pending"],
    },
    "S003": {
        "student_id": "S003",
        "name": "Olga Petrov",
        "email": "olga.petrov@email.com",
        "plan": "premium",
        "timezone": "Europe/Moscow",
        "joined": "2026-01-10",
        "last_login": "2026-03-24",
        "known_issues": ["duplicate_charge_reported"],
    },
    "S004": {
        "student_id": "S004",
        "name": "Carlos Silva",
        "email": "carlos.silva@email.com",
        "plan": "basic",
        "timezone": "America/Sao_Paulo",
        "joined": "2026-02-01",
        "last_login": "2026-03-20",
        "known_issues": ["browser_compatibility_chrome"],
    },
    "S005": {
        "student_id": "S005",
        "name": "Aisha Bello",
        "email": "aisha.bello@email.com",
        "plan": "standard",
        "timezone": "Africa/Lagos",
        "joined": "2026-03-01",
        "last_login": "2026-03-26",
        "known_issues": [],
    },
}
