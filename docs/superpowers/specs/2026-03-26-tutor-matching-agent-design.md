# Project 4: Tutor Matching & Scheduling Agent — Design Specification

## Overview

**Department:** Operations
**Difficulty:** Intermediate

The Operations department manages tutor-student matching and scheduling. They need a conversational agent that students can interact with to find the right tutor — factoring in specialization (grammar, conversation, business English, exam prep), timezone, and availability. The agent calls mock external APIs (tutor database, calendar system) and maintains conversation state so students can return later and continue the process.

This project introduces the canonical LangGraph agent pattern: `MessagesState` + `bind_tools()` + `ToolNode`, with checkpointer-based persistence for multi-session conversations. It contrasts with P2/P3's deterministic node-per-step approach by letting the **LLM drive the conversation flow** within a tool-calling loop.

## Concepts Introduced

- **Tool calling:** `@tool` decorator, `model.bind_tools()`, `ToolNode` from `langgraph.prebuilt`
- **MessagesState:** LangGraph's built-in message-based state with `operator.add` reducer, extended with custom fields
- **Checkpointers:** `InMemorySaver` for development, `SqliteSaver` for durable persistence — demonstrating the pluggable pattern
- **Thread management:** `thread_id` in config, separate conversation threads, resume from checkpoint
- **LangSmith:** `@traceable` on all nodes/tools, project-level tagging (`p4-tutor-matching`)

## Architecture: Multi-Phase Conversational Agent

The graph uses `MessagesState` extended with custom fields to track which phase the conversation is in. The LLM decides what to do next (ask questions, call tools, present results), and the graph provides the execution loop.

### Four Phases

1. **Gather Preferences** — the agent asks about specialization, timezone, availability, and any other preferences. No tools needed yet — pure conversation.
2. **Search Tutors** — once preferences are gathered, the LLM calls the `search_tutors` tool. Results are stored in state.
3. **Present & Refine** — the agent presents matching tutors and the student can ask questions, refine criteria (triggering another search), or pick a tutor.
4. **Confirm Booking** — the student selects a tutor and the agent calls `check_availability` and `book_session` tools to finalize.

### Graph Topology

```
START → agent_node → [has tool calls?]
                         ├── yes → tool_node → agent_node (loop)
                         └── no  → [phase check]
                                      ├── booking_confirmed → END
                                      └── else → END (yield, wait for next user message)
```

The core is the classic **agent loop** (LLM → tools → LLM), but the system prompt and state guide the LLM through phases. The agent node reads the current phase from state and adjusts its behavior accordingly. Phase transitions happen naturally as the LLM gathers enough info and decides to call tools.

The "wait for next user message" pattern works because with a checkpointer, the graph pauses at END and resumes when `invoke()` is called again with the same `thread_id` — the new user message appends to the existing messages list.

This is deliberately simpler than hard-coding phase transitions as separate nodes — it teaches the canonical agentic pattern where the LLM drives the flow and the graph provides the execution loop. The contrast with P2/P3's deterministic node-per-step approach is itself educational.

## State Schema

```python
class TutorMatchingState(MessagesState):
    """Extends MessagesState (messages: Annotated[list, operator.add]) with custom fields."""

    # Phase tracking — guides the system prompt
    phase: str                          # "gather", "search", "present", "book", "done"

    # Gathered student preferences (accumulated across conversation)
    preferences: dict                   # {"specialization": ..., "timezone": ..., "availability": ...}

    # Search results from tutor database
    search_results: list[dict]          # List of matching tutor records

    # Booking outcome
    selected_tutor: dict | None         # The tutor the student chose
    booking_confirmation: dict | None   # Final booking details
```

Key design choices:
- **`MessagesState` as base** — gives us `messages: Annotated[list, operator.add]` for free, the standard LangGraph pattern for conversational agents
- **`phase` field** — simple string, no reducer (last write wins). The agent node updates this based on conversation progress.
- **`preferences` as dict** — flexible, since students mention preferences in any order. The LLM extracts and accumulates them.
- **No reducers on custom fields** — each is written by exactly one logical step, consistent with P2/P3

## Tools (Mock APIs)

Three tools defined with `@tool` decorator, simulating external API calls:

### `search_tutors(specialization, timezone?, availability?)`
- Searches the mock tutor database by criteria
- Returns a list of matching tutor dicts (name, specialization, timezone, rating, bio, hourly rate)
- Filters by specialization (required), optionally narrows by timezone and availability
- The mock database lives in `data/tutors.py` — ~10-12 tutors with varied profiles

### `check_availability(tutor_id, date, time?)`
- Checks a specific tutor's calendar for open slots
- Returns available time slots for the requested date
- Mock calendar data in `data/calendar.py` — pre-built schedules for each tutor

### `book_session(tutor_id, date, time, student_name)`
- Books a session with the selected tutor
- Returns a booking confirmation (confirmation ID, tutor name, datetime, duration)
- Updates the mock calendar (marks the slot as taken)

All three are bound to the model via `bind_tools()` and executed by the prebuilt `ToolNode`. The tool functions themselves are simple — filter/lookup against in-memory data structures. They have realistic signatures and return types so the LLM learns to call them correctly.

The tools live in a `tools.py` module, separate from the mock data in `data/`. This keeps the separation clean — `tools.py` is LangGraph code, `data/` is scaffolding.

## Graph Nodes & Edges

### Nodes

**`agent_node`** — The conversational brain. Reads `phase` from state, constructs a phase-aware system prompt, invokes the model with bound tools. Returns the LLM response (which may contain tool calls or a plain message). After the LLM responds, the node infers the next phase from the conversation state using simple heuristics (e.g., if `search_tutors` was just called → `"present"`, if `book_session` returned a confirmation → `"done"`). Phase transitions are deterministic logic in the node, not LLM output — this keeps phase tracking reliable.

**`tool_node`** — Prebuilt `ToolNode(tools, handle_tool_errors=True)`. Executes whatever tool calls the LLM made and returns `ToolMessage` results. Error handling is built in — if a tool fails, the error is returned as a message so the LLM can recover.

### Edges

```
START → agent_node
agent_node → should_continue (conditional)
    - has tool_calls → tool_node
    - phase == "done" → END
    - else → END (yield, wait for next user message via checkpointer)
tool_node → agent_node (always loops back)
```

### `should_continue` routing function
- Checks if the last AI message has `tool_calls` → route to `tool_node`
- Checks if `phase == "done"` → route to `END`
- Otherwise → route to `END` (graph yields, waits for next user input via checkpointer)

## Checkpointer Design

Two-tier demonstration:

### Phase 1 — `InMemorySaver` (default)
The default when running the CLI. Shows the core concept — thread-based state persistence, multiple conversation turns, the graph pausing and resuming. Students see that closing the process loses everything.

### Phase 2 — `SqliteSaver` (--persist flag)
Enabled via a `--persist` CLI flag. Same graph, one-line swap at compile time. Students can:
1. Start a conversation, note the thread ID
2. Kill the process
3. Restart with `--persist --thread <id>` and continue exactly where they left off

The contrast between the two makes the checkpointer abstraction tangible:

```python
# In-memory (default) — state lost on restart
checkpointer = InMemorySaver()

# Durable (--persist) — state survives restart
checkpointer = SqliteSaver.from_conn_string("sqlite:///tutor_matching.db")

graph = build_graph().compile(checkpointer=checkpointer)
```

### Thread management in the CLI
- `--thread <id>` resumes an existing conversation
- Without `--thread`, generates a new UUID
- Thread ID is displayed at start so the user can note it for resumption

## Module Structure

```
projects/04-tutor-matching-agent/
├── models.py              # TutorMatchingState (extends MessagesState), Tutor, BookingConfirmation
├── prompts.py             # Phase-aware system prompts
├── tools.py               # @tool functions: search_tutors, check_availability, book_session
├── nodes.py               # agent_node, build tool_node, should_continue router
├── graph.py               # build_graph() — wires StateGraph, returns compiled graph
├── main.py                # CLI: --persist, --thread flags, conversation loop
├── data/
│   ├── __init__.py
│   ├── tutors.py          # Mock tutor database (~10-12 tutors)
│   └── calendar.py        # Mock calendar/availability data
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Shared fixtures (mock data, graph instances)
│   ├── test_models.py     # Pydantic model validation
│   ├── test_tools.py      # Tool functions against mock data (no LLM)
│   ├── test_nodes.py      # Node integration tests (hits LLM)
│   └── test_graph.py      # End-to-end: full conversation flows, persistence
├── README.md
└── requirements.txt
```

## Dependencies

```
langchain-core
langchain-anthropic
langgraph
langgraph-checkpoint-sqlite
langsmith
python-dotenv
pytest
```

## LangSmith Integration

- `@traceable` decorator on all node functions and tool functions
- All traces tagged with `["p4-tutor-matching"]`
- Config passed through: `{"tags": ["p4-tutor-matching"]}`

## Testing Strategy

1. **`test_models.py`** — Pydantic model validation, state schema structure
2. **`test_tools.py`** — Unit tests for each tool function against mock data. No LLM calls. Verifies filtering logic, availability checks, booking mechanics.
3. **`test_nodes.py`** — Integration tests for agent_node. Verifies the LLM receives the right system prompt per phase, makes appropriate tool calls.
4. **`test_graph.py`** — End-to-end conversation flows. Tests the full loop: user messages in → tool calls happen → booking confirmed. Also tests persistence: invoke with thread, get state, resume with same thread.
