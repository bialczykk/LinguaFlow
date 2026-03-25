# Project 1: Grammar Correction Agent

**LinguaFlow — Student Success Department**

An AI-powered grammar correction agent that analyzes student writing, provides structured feedback with CEFR proficiency assessment, and supports follow-up conversation for deeper learning.

## What It Does

1. Takes a student's writing sample (choose from samples or enter your own)
2. Analyzes grammar and returns structured feedback:
   - Individual grammar issues with corrections and explanations
   - CEFR proficiency level assessment (A1-C2)
   - Strengths and areas to improve
   - Full corrected text
3. Lets you ask follow-up questions about the feedback
4. Submit new text anytime with `new: <your text>`

## Concepts Covered

- **LangChain chains**: prompt templates + model = chain
- **Structured output**: Pydantic models + `.with_structured_output()`
- **ChatAnthropic**: configuring Anthropic's Claude model
- **Conversation history**: maintaining multi-turn context
- **LangSmith tracing**: `@traceable` decorators for observability

## Setup

Ensure the shared virtual environment is activated:

```bash
# From the repo root
source .venv/bin/activate
pip install -r projects/01-grammar-correction-agent/requirements.txt
```

Ensure your root `.env` has:
```
ANTHROPIC_API_KEY=your-key-here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=linguaflow
```

## Run

```bash
cd projects/01-grammar-correction-agent
python main.py
```

## Test

```bash
cd projects/01-grammar-correction-agent
python -m pytest tests/ -v
```
