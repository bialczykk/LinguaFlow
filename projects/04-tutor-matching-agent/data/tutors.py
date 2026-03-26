"""Mock tutor database for the LinguaFlow operations department.

Contains ~12 tutor profiles with varied specializations, timezones,
ratings, and hourly rates. This data simulates what a real tutor database
API would return.

This is scaffolding — it exists only to give the tools realistic data
to filter and return. The LangGraph concepts are in tools.py and nodes.py.
"""

TUTORS = [
    {"tutor_id": "t1", "name": "Alice Smith", "specializations": ["grammar", "exam_prep"], "timezone": "Europe/London", "rating": 4.9, "bio": "Cambridge-certified with 10 years of exam prep experience. Specializes in IELTS and FCE.", "hourly_rate": 45.0},
    {"tutor_id": "t2", "name": "Carlos Rivera", "specializations": ["conversation", "business_english"], "timezone": "America/New_York", "rating": 4.7, "bio": "Former corporate trainer. Makes business English practical and engaging.", "hourly_rate": 40.0},
    {"tutor_id": "t3", "name": "Yuki Tanaka", "specializations": ["grammar", "conversation"], "timezone": "Asia/Tokyo", "rating": 4.8, "bio": "Patient and methodical. Great at building confidence in beginners.", "hourly_rate": 35.0},
    {"tutor_id": "t4", "name": "Priya Patel", "specializations": ["exam_prep", "grammar"], "timezone": "Asia/Kolkata", "rating": 4.6, "bio": "IELTS examiner with insider knowledge of scoring criteria.", "hourly_rate": 38.0},
    {"tutor_id": "t5", "name": "Emma Johansson", "specializations": ["conversation", "grammar"], "timezone": "Europe/Stockholm", "rating": 4.5, "bio": "Focuses on natural speech patterns and everyday fluency.", "hourly_rate": 32.0},
    {"tutor_id": "t6", "name": "David Chen", "specializations": ["business_english", "exam_prep"], "timezone": "Asia/Shanghai", "rating": 4.9, "bio": "MBA graduate who teaches professional communication and presentation skills.", "hourly_rate": 50.0},
    {"tutor_id": "t7", "name": "Sarah O'Brien", "specializations": ["conversation", "grammar"], "timezone": "Europe/Dublin", "rating": 4.3, "bio": "Friendly and approachable. Loves helping students overcome speaking anxiety.", "hourly_rate": 30.0},
    {"tutor_id": "t8", "name": "Ahmed Hassan", "specializations": ["grammar", "business_english"], "timezone": "Africa/Cairo", "rating": 4.7, "bio": "Structured approach to grammar with real-world business applications.", "hourly_rate": 28.0},
    {"tutor_id": "t9", "name": "Maria Garcia", "specializations": ["exam_prep", "conversation"], "timezone": "Europe/Madrid", "rating": 4.4, "bio": "Bilingual examiner who understands the challenges of learning English as a second language.", "hourly_rate": 36.0},
    {"tutor_id": "t10", "name": "James Wilson", "specializations": ["business_english", "conversation"], "timezone": "America/Chicago", "rating": 4.6, "bio": "Former journalist. Teaches clear, concise professional writing and speaking.", "hourly_rate": 42.0},
    {"tutor_id": "t11", "name": "Lena Müller", "specializations": ["grammar", "exam_prep", "conversation"], "timezone": "Europe/Berlin", "rating": 4.8, "bio": "Polyglot and linguistics PhD. Makes grammar intuitive through pattern recognition.", "hourly_rate": 48.0},
    {"tutor_id": "t12", "name": "Kenji Nakamura", "specializations": ["conversation", "business_english"], "timezone": "Asia/Tokyo", "rating": 4.2, "bio": "Easygoing style focused on building vocabulary through conversation practice.", "hourly_rate": 25.0},
]
