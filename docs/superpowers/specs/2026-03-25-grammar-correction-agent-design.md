# Project 1: Grammar Correction Agent — Design Spec

## Overview

A conversational grammar correction agent for LinguaFlow's Student Success department. Takes student writing samples, returns structured grammar feedback with CEFR proficiency assessment, and supports follow-up conversation where students can ask questions about the feedback or submit new text for analysis.

## Concepts Covered

- LangChain fundamentals: chains, prompt templates, model integration
- Structured output with Pydantic models
- Anthropic Claude model configuration (`langchain-anthropic`)
- Conversational follow-up with message history
- Basic LangSmith tracing with `@traceable` decorators

## Architecture

Two-phase interaction model:

1. **Analysis phase** — Student submits writing. A LangChain chain with structured output parses the text and returns a `GrammarFeedback` Pydantic model containing individual issues, a CEFR proficiency assessment, and the full corrected text.
2. **Conversation phase** — The student enters a loop where they can ask follow-up questions about the feedback, submit new text for analysis, or quit. The agent uses message history to maintain context and ends each response with a suggested next step.

## Data Models

### `GrammarIssue`
| Field | Type | Description |
|-------|------|-------------|
| `original_text` | `str` | The problematic fragment from the student's writing |
| `corrected_text` | `str` | The corrected version |
| `error_category` | `str` | Category (e.g., subject-verb agreement, tense, article usage, punctuation) |
| `explanation` | `str` | Educational explanation of why it's wrong and how the grammar rule works |
| `severity` | `Literal["minor", "moderate", "major"]` | How impactful the error is |

### `ProficiencyAssessment`
| Field | Type | Description |
|-------|------|-------------|
| `cefr_level` | `Literal["A1", "A2", "B1", "B2", "C1", "C2"]` | Assessed CEFR proficiency level |
| `strengths` | `list[str]` | What the student does well |
| `areas_to_improve` | `list[str]` | Key areas for improvement |
| `summary` | `str` | Brief overall assessment |

### `GrammarFeedback`
| Field | Type | Description |
|-------|------|-------------|
| `issues` | `list[GrammarIssue]` | All grammar issues found |
| `proficiency` | `ProficiencyAssessment` | Overall proficiency assessment |
| `corrected_full_text` | `str` | The entire submission with all corrections applied |

## Components

### Analysis Chain (`chains.py`)
- Prompt template instructing Claude to analyze student writing and return structured feedback
- Uses `ChatAnthropic` from `langchain-anthropic`
- Uses `.with_structured_output(GrammarFeedback)` for reliable Pydantic parsing
- Decorated with `@traceable(run_type="chain", name="grammar_analysis")` for LangSmith visibility

### Conversation Handler (`conversation.py`)
- Manages follow-up conversation after initial analysis
- Injects the original student text and `GrammarFeedback` into the conversation context
- Maintains message history (`ChatMessageHistory` or equivalent) for multi-turn dialogue
- Prompt guides the agent to:
  - Answer questions about specific corrections educationally
  - Detect when the student submits new text (triggers a fresh analysis)
  - End each response with a suggested next step
- Decorated with `@traceable(run_type="chain", name="grammar_followup")` for LangSmith visibility

### Entry Point (`main.py`)
- Loads environment variables from `.env`
- Runs an interactive loop:
  1. Prompts student for writing sample
  2. Runs analysis chain, displays formatted feedback
  3. Enters conversation loop (follow-up questions, new submissions, or quit)
- Handles graceful exit

### Sample Data (`data/sample_texts.py`)
- 3-4 sample student writing texts at different CEFR levels (A2, B1, B2, C1)
- Used for quick testing and demonstration
- Each sample has a brief label indicating expected level

## File Structure

```
projects/01-grammar-correction-agent/
  main.py                  # Entry point — interactive loop
  models.py                # Pydantic models (GrammarIssue, GrammarFeedback, etc.)
  chains.py                # Analysis chain with structured output
  conversation.py          # Follow-up conversation handler
  data/
    sample_texts.py        # Sample student writing at different CEFR levels
  requirements.txt         # Project dependencies
  README.md                # Overview and how to run
```

## Environment & Dependencies

**Shared `.env` at repo root** with:
- `ANTHROPIC_API_KEY`
- `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`

**Shared `.venv` at repo root.** All dependencies installed there.

**`requirements.txt`:**
- `langchain-core`
- `langchain-anthropic`
- `langsmith`
- `python-dotenv`
- `pydantic`

## LangSmith Integration

- Auto-tracing enabled via `LANGSMITH_TRACING=true` environment variable
- `@traceable` decorators on key functions with descriptive `run_name` parameters:
  - `grammar_analysis` — the structured output chain
  - `grammar_followup` — the conversation handler
- The educational doc (`docs/01-grammar-correction-agent.md`) will include a section on navigating the LangSmith dashboard, explaining what traces look like and what to inspect

## Interaction Flow

```
1. Student enters writing sample (or picks a sample text)
2. Analysis chain processes text → GrammarFeedback
3. Display: issues with corrections/explanations, CEFR level, corrected text
4. Conversation loop:
   a. Student types a message
   b. If "quit"/"exit" → end
   c. If looks like new text for analysis → run analysis chain again, display feedback
   d. Otherwise → conversation handler answers the question + suggests next step
   e. Back to (a)
```

## What This Project Does NOT Cover

- No LangGraph StateGraph (that's Project 2)
- No tool calling or external API integration (Project 4)
- No persistence across sessions (Project 4)
- No human-in-the-loop approval (Project 5)
- Conversation history is in-memory only, lost when the program exits
