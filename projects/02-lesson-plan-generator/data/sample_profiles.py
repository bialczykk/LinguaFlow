"""Pre-built StudentProfile instances for testing and demonstration.

These profiles cover all three lesson types and a range of CEFR levels,
allowing tests to skip the intake conversation and invoke the graph directly.
"""

from models import StudentProfile

BEGINNER_CONVERSATION = StudentProfile(
    name="Yuki",
    proficiency_level="A2",
    learning_goals=["build confidence speaking", "learn everyday phrases"],
    preferred_topics=["shopping", "introducing yourself"],
    lesson_type="conversation",
)

INTERMEDIATE_CONVERSATION = StudentProfile(
    name="Carlos",
    proficiency_level="B1",
    learning_goals=["improve fluency", "learn travel vocabulary"],
    preferred_topics=["travel", "food and restaurants"],
    lesson_type="conversation",
)

BEGINNER_GRAMMAR = StudentProfile(
    name="Fatima",
    proficiency_level="A1",
    learning_goals=["learn basic sentence structure", "understand verb tenses"],
    preferred_topics=["daily routines", "family"],
    lesson_type="grammar",
)

INTERMEDIATE_GRAMMAR = StudentProfile(
    name="Hans",
    proficiency_level="B2",
    learning_goals=["master conditionals", "reduce common errors"],
    preferred_topics=["technology", "environment"],
    lesson_type="grammar",
)

EXAM_PREP_INTERMEDIATE = StudentProfile(
    name="Mei",
    proficiency_level="B2",
    learning_goals=["prepare for IELTS", "improve writing scores"],
    preferred_topics=["education", "global issues"],
    lesson_type="exam_prep",
)

EXAM_PREP_ADVANCED = StudentProfile(
    name="Olga",
    proficiency_level="C1",
    learning_goals=["target CAE certificate", "refine academic writing"],
    preferred_topics=["science", "current affairs"],
    lesson_type="exam_prep",
)
