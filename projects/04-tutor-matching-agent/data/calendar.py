"""Mock calendar/scheduling data for the LinguaFlow tutor system.

Contains pre-built availability schedules for each tutor. The book_session
tool updates this data in-memory when a session is booked (marks slots as taken).

This is scaffolding — it simulates what a real calendar API would provide.
"""

SCHEDULES = {
    "t1": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "14:00", "end_time": "15:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "15:00", "end_time": "16:00", "booked": False},
    ],
    "t2": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-03", "start_time": "09:00", "end_time": "10:00", "booked": False},
    ],
    "t3": [
        {"date": "2026-04-01", "start_time": "07:00", "end_time": "08:00", "booked": False},
        {"date": "2026-04-01", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-02", "start_time": "07:00", "end_time": "08:00", "booked": False},
    ],
    "t4": [
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "16:00", "end_time": "17:00", "booked": False},
        {"date": "2026-04-02", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-03", "start_time": "14:00", "end_time": "15:00", "booked": False},
    ],
    "t5": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-02", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-02", "start_time": "14:00", "end_time": "15:00", "booked": False},
    ],
    "t6": [
        {"date": "2026-04-01", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-01", "start_time": "17:00", "end_time": "18:00", "booked": False},
        {"date": "2026-04-02", "start_time": "08:00", "end_time": "09:00", "booked": False},
        {"date": "2026-04-03", "start_time": "08:00", "end_time": "09:00", "booked": False},
    ],
    "t7": [
        {"date": "2026-04-01", "start_time": "12:00", "end_time": "13:00", "booked": False},
        {"date": "2026-04-01", "start_time": "15:00", "end_time": "16:00", "booked": False},
        {"date": "2026-04-02", "start_time": "12:00", "end_time": "13:00", "booked": False},
    ],
    "t8": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "16:00", "end_time": "17:00", "booked": False},
    ],
    "t9": [
        {"date": "2026-04-01", "start_time": "10:00", "end_time": "11:00", "booked": False},
        {"date": "2026-04-01", "start_time": "16:00", "end_time": "17:00", "booked": False},
        {"date": "2026-04-03", "start_time": "10:00", "end_time": "11:00", "booked": False},
    ],
    "t10": [
        {"date": "2026-04-01", "start_time": "11:00", "end_time": "12:00", "booked": False},
        {"date": "2026-04-01", "start_time": "14:00", "end_time": "15:00", "booked": False},
        {"date": "2026-04-02", "start_time": "11:00", "end_time": "12:00", "booked": False},
    ],
    "t11": [
        {"date": "2026-04-01", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-01", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-02", "start_time": "09:00", "end_time": "10:00", "booked": False},
        {"date": "2026-04-02", "start_time": "13:00", "end_time": "14:00", "booked": False},
        {"date": "2026-04-03", "start_time": "09:00", "end_time": "10:00", "booked": False},
    ],
    "t12": [
        {"date": "2026-04-01", "start_time": "07:00", "end_time": "08:00", "booked": False},
        {"date": "2026-04-01", "start_time": "18:00", "end_time": "19:00", "booked": False},
        {"date": "2026-04-02", "start_time": "07:00", "end_time": "08:00", "booked": False},
    ],
}
