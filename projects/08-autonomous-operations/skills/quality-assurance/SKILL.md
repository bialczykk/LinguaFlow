---
name: quality-assurance
description: QA rubrics, quality thresholds, flag categories, and revision follow-up orchestration for LinguaFlow content review
---

# Quality Assurance

## QA Rubrics

Evaluate every piece of content across four dimensions, each scored 0.0-1.0:

- **Accuracy (0.0-1.0):** Are all grammar rules, vocabulary definitions, and factual claims correct? Score 1.0 for no errors, 0.5 for minor/fixable errors, 0.0 for fundamental errors that mislead the learner.
- **CEFR Alignment (0.0-1.0):** Does the language complexity, vocabulary range, and grammar scope match the declared target level? Check average sentence length, vocabulary frequency tier, and grammar structures used.
- **Engagement (0.0-1.0):** Is the content contextualised in a realistic scenario? Does it avoid purely abstract drills? Score 1.0 for strong real-world framing, 0.5 for partial context, 0.0 for decontextualised lists or meaningless filler sentences.
- **Formatting (0.0-1.0):** Does the content follow the expected structure for its type (see content-pipeline skill)? Are labels, headings, and question formats correct?

## Quality Thresholds

- **Overall score** = average of all four dimension scores.
- **Pass threshold:** Overall score >= 0.7 AND no individual dimension below 0.5.
- **Fail:** Overall score < 0.7 OR any single dimension below 0.5.

Content that passes moves to `published`. Content that fails moves to `revision_needed`.

## Flag Categories

When content fails, attach one or more flags explaining the issue:

- **accuracy_error:** A grammar rule is stated incorrectly, or an example sentence is ungrammatical without being labelled as such.
- **level_mismatch:** Vocabulary or grammar structures are significantly above or below the declared CEFR level.
- **needs_revision:** Content is structurally sound but requires targeted edits (e.g., improve engagement, fix a single example, reformat a section).

Each flag must include a brief human-readable note (1-2 sentences) describing what specifically needs fixing.

## Follow-up Rules

When content fails QA review, **always** generate a `content-pipeline` follow-up action requesting revision. The follow-up payload must include:

- `content_id`
- `flags` (list of flag categories with notes)
- `overall_score`
- `dimension_scores` (accuracy, cefr_alignment, engagement, formatting)

Never silently discard failed content. Every failure must produce a revision follow-up so the pipeline can track the item back to draft.
