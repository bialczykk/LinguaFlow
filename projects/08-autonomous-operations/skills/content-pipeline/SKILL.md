---
name: content-pipeline
description: Content creation standards, content types, pipeline stages, and review orchestration for LinguaFlow educational material
---

# Content Pipeline

## Content Standards

All content produced for LinguaFlow must meet three standards before publication:

- **CEFR Alignment:** Vocabulary, grammar structures, and sentence complexity must be appropriate for the declared target level. Do not mix level requirements within a single piece of content.
- **Accuracy:** Factual claims, grammar explanations, and example sentences must be correct. Any example that models incorrect English — even intentionally — must be clearly labelled as an error example.
- **Engagement:** Content must be relevant to learners' real-world needs. Dry, decontextualised exercises are discouraged. Use realistic scenarios, dialogues, or narratives as vehicles for practice.

## Content Types

The pipeline handles three primary content types, each with distinct expectations:

- **grammar_explanation:** A structured explanation of one grammar rule or pattern. Must include: rule statement, 2-3 positive examples, 1-2 negative examples (labelled), and a brief usage note. Target length: 150-300 words.
- **vocabulary_exercise:** A set of 8-12 items testing vocabulary in context. Must include a mix of gap-fill and matching tasks. Each word must appear in a sentence, never isolated.
- **reading_passage:** An authentic-style text of 150-400 words (scaled to CEFR level) followed by 4-6 comprehension questions. Questions must test inference, not just literal recall.

## Pipeline Stages

Content moves through the following stages in order:

1. **draft** — Initial creation. Content exists but has not been reviewed.
2. **in_review** — Submitted to QA for evaluation. No further edits until QA completes.
3. **published** — Approved by QA and available to students.
4. **revision_needed** — Returned by QA with flags. Must be revised before re-submitting.

## Follow-up Rules

When a piece of content transitions to `in_review`, **always** generate a `quality-assurance` follow-up action. The follow-up payload must include:

- `content_id`
- `content_type`
- `target_level`
- `submitted_by`

Do not mark content as published without a completed QA review. Skipping QA is not permitted under any circumstances.
