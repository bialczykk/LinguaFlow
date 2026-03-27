"""Platform service health map for the support department.

SERVICES maps service names to health dicts. Used by the support department
to diagnose tech issues and inform students of known outages.

Most services are operational. chat_system is degraded — demonstrates
how the agent surfaces known issues when a student reports tech problems.
"""

SERVICES = {
    "video_platform": {
        "name": "Video Lesson Platform",
        "status": "operational",
        "uptime_percent": 99.9,
        "last_incident": "2026-03-10",
        "notes": "All video conferencing services running normally.",
    },
    "chat_system": {
        "name": "In-App Chat System",
        "status": "degraded",
        "uptime_percent": 95.2,
        "last_incident": "2026-03-26",
        "notes": "Intermittent message delivery delays. Engineering team investigating.",
    },
    "payment_gateway": {
        "name": "Payment Gateway",
        "status": "operational",
        "uptime_percent": 99.95,
        "last_incident": "2026-03-05",
        "notes": "All payment processing running normally.",
    },
    "content_cdn": {
        "name": "Content Delivery Network",
        "status": "operational",
        "uptime_percent": 99.8,
        "last_incident": "2026-03-15",
        "notes": "All learning materials loading normally.",
    },
    "auth_service": {
        "name": "Authentication Service",
        "status": "operational",
        "uptime_percent": 99.99,
        "last_incident": "2026-02-20",
        "notes": "Login and SSO services running normally.",
    },
}
