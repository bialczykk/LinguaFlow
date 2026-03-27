---
name: tutor-management
description: Tutor matching, scheduling, and assignment rules for LinguaFlow — ensures students are paired with the right tutor at the right time
---

# Tutor Management

## Matching Heuristics

When assigning a tutor to a student, evaluate candidates against these criteria in priority order:

1. **Specialty match:** Tutor's declared specialties must include at least one of the student's `focus_areas`. A tutor who specialises in grammar should not be assigned to a student focused on business speaking.
2. **Level match:** Tutor must be qualified to teach the student's current CEFR level. A tutor marked as qualified for B1-B2 should not be assigned to a C1 student.
3. **Availability:** Tutor must have open capacity slots within the student's preferred schedule window. Never assign a tutor already at maximum capacity.
4. **Rating:** Among equally qualified, available tutors, prefer those with a higher rating (scale 1.0-5.0). Minimum acceptable rating is 3.5.

If no tutor meets all four criteria, relax them in reverse order (first drop rating threshold to 3.0, then widen availability window, never relax level match or specialty match).

## Scheduling Rules

Before confirming any tutor assignment:

- Check the tutor's current `student_count` against their `max_students` capacity field.
- Verify no scheduling conflict exists in the tutor's calendar for the proposed session times.
- A tutor cannot be double-booked — two students cannot share the same session slot.

Session frequency defaults:
- **A2/B1:** 2 sessions per week
- **B2/C1:** 1-2 sessions per week (based on student preference)

## Assignment Confirmation Format

When confirming a tutor assignment, the output must include:

```
tutor_id: <id>
tutor_name: <name>
student_id: <id>
assigned_level: <CEFR level>
sessions_per_week: <number>
start_date: <ISO date>
```

Log the assignment event so the reporting department can track tutor utilisation and student progress.
