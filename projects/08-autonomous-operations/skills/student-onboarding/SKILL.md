---
name: student-onboarding
description: Student onboarding procedures — CEFR assessment, study plan creation, and follow-up orchestration for LinguaFlow
---

# Student Onboarding

## CEFR Assessment Criteria

Use the following rubric to place students into the correct starting level:

- **A2 (Elementary):** Can handle simple transactions and describe past events in basic terms. Uses present and past simple. Makes frequent grammar errors. Vocabulary under 600 words. Responds well to structured, formulaic tasks.
- **B1 (Intermediate):** Can discuss familiar topics with some fluency. Uses most basic tenses with reasonable accuracy. Occasional errors with complex structures. Vocabulary 600-1200 words. Can write short, connected texts.
- **B2 (Upper Intermediate):** Can understand main ideas of complex text. Uses a range of tenses and modal verbs with good accuracy. Vocabulary 1200-2500 words. Can argue a point with some sophistication.
- **C1 (Advanced):** Uses language flexibly and effectively for academic or professional purposes. Wide vocabulary range including idiomatic expressions. Rare errors, mostly in complex or idiomatic usage. Vocabulary 2500-5000 words.

When in doubt between two levels, default to the lower level — it is better to build confidence than to frustrate the student.

## Study Plan Template Structure

Every study plan must include the following fields:

- **focus_areas:** List of 2-4 skill areas to target (e.g., grammar, speaking, writing, vocabulary)
- **weekly_hours:** Recommended weekly study hours (A2: 5-7h, B1: 6-8h, B2: 7-10h, C1: 8-12h)
- **milestones:** 3 checkpoints — 30-day, 60-day, and 90-day — each with a measurable outcome
- **content_types:** Preferred exercise formats based on level (A2: vocabulary drills, structured grammar; B1+: reading passages, free writing)
- **target_level:** The CEFR level the student aims to reach within the plan duration

## Follow-up Rules

After creating a study plan, **always** generate a `tutor-management` follow-up action to match the student with an appropriate tutor. The follow-up payload must include:

- `student_id`
- `current_level` (assessed CEFR level)
- `target_level`
- `focus_areas` (from the study plan)

Never complete an onboarding without triggering tutor assignment — an unassigned student has no learning path.
