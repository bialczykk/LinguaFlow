# Project 4: Tutor Matching & Scheduling Agent

A conversational LangGraph agent that helps students find and book the right English tutor on the LinguaFlow platform.

## What This Teaches

- **Tool calling**: `@tool`, `bind_tools()`, `ToolNode` from `langgraph.prebuilt`
- **MessagesState**: LangGraph's built-in message-based state, extended with custom fields
- **Checkpointers**: `InMemorySaver` vs `SqliteSaver` — pluggable persistence
- **Thread management**: `thread_id` for isolated conversations, resume from checkpoint
- **Agent loop pattern**: LLM → conditional edge → tool execution → back to LLM

## How It Works

The agent guides students through four phases:
1. **Gather** — asks about specialization, timezone, availability
2. **Present** — searches tutors and presents matches
3. **Book** — checks tutor availability and confirms a booking
4. **Done** — summarizes the booking

Unlike Projects 2-3 (deterministic node-per-step), this graph has just two nodes (agent + tools) with the LLM driving the flow through tool calls.

## Running

```bash
# New conversation (in-memory, lost on exit)
python main.py

# New conversation with durable persistence
python main.py --persist

# Resume a previous conversation
python main.py --persist --thread <thread-id>
```

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Just tool logic (no LLM)
python -m pytest tests/test_tools.py -v

# Integration tests (requires API key)
python -m pytest tests/ -v -m integration
```

## Project Structure

```
models.py    — State schema (extends MessagesState) and Pydantic models
prompts.py   — Phase-aware system prompts
tools.py     — @tool functions: search_tutors, check_availability, book_session
nodes.py     — agent_node, should_continue router, ToolNode factory
graph.py     — StateGraph wiring and compilation
main.py      — Interactive CLI with persistence options
data/        — Mock tutor database and calendar (scaffolding)
tests/       — Unit tests (tools) + integration tests (nodes, graph)
```
