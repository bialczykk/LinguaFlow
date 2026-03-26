# Project 4: Tutor Matching Agent

## Introduction

Projects 2 and 3 built deterministic pipelines: each node had a fixed job, edges were mostly unconditional, and the graph always ran the same steps in the same order. The LLM was a worker inside a larger orchestration plan that you, the developer, designed upfront.

Project 4 flips that model. Here, the LLM is the orchestrator. A single agent node receives the conversation, decides whether to call a tool (and which one), and then either loops back after getting tool results or hands control back to the user. The graph is minimal — two nodes and a conditional edge — and all the intelligence lives in the LLM's decisions about when to call what.

This project introduces four key LangGraph concepts:
- **Tool calling** — giving the LLM callable functions via `@tool` and `bind_tools()`
- **MessagesState** — the built-in state schema for conversational agents
- **The agent loop pattern** — the core `LLM → tools → LLM` cycle
- **Checkpointers and persistence** — saving and resuming conversation state across turns

---

## Tool Calling in LangGraph

Tool calling is how you give the LLM the ability to take actions. Instead of generating only text, the model can say "I want to call this function with these arguments."

### Defining tools with `@tool`

The `@tool` decorator from `langchain_core.tools` turns a plain Python function into a LangGraph-compatible tool. The key thing it does is extract the function's signature, type annotations, and docstring to build a schema the LLM can understand:

```python
from langchain_core.tools import tool

@tool
def search_tutors(specialization: str, timezone: str | None = None, availability: str | None = None) -> list[dict]:
    """Search the tutor database by specialization, timezone, and availability."""
    # ... implementation
```

The docstring becomes the tool's description. The type annotations become the parameter schema. This is what gets sent to the model so it knows what the tool does and how to call it.

Project 4 defines three tools in `tools.py`:
- `search_tutors` — queries the tutor database by specialization, timezone, and date
- `check_availability` — fetches open calendar slots for a specific tutor and date
- `book_session` — reserves a slot and returns a confirmation record

### Attaching tools to the model with `bind_tools()`

Defining a tool doesn't automatically make the LLM aware of it. You have to attach the tools to the model explicitly:

```python
_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
_tools = [search_tutors, check_availability, book_session]
_model_with_tools = _model.bind_tools(_tools)
```

`bind_tools()` serializes each tool's schema and includes it in every API call the model makes. The LLM now knows the names, descriptions, and parameter shapes of all three tools. When it decides a tool would help, it includes a `tool_calls` field in its response instead of — or alongside — text.

### Executing tools with `ToolNode`

When the LLM returns a response with `tool_calls`, something needs to actually run those functions. That's `ToolNode` from `langgraph.prebuilt`:

```python
from langgraph.prebuilt import ToolNode

def get_tool_node() -> ToolNode:
    return ToolNode(_tools, handle_tool_errors=True)
```

`ToolNode` inspects the `tool_calls` in the last `AIMessage`, finds the matching function by name, calls it with the provided arguments, and returns the result as a `ToolMessage`. All of this happens automatically — you don't write the dispatch logic yourself.

The `handle_tool_errors=True` flag means if a tool raises an exception, the error is caught and returned as a `ToolMessage` rather than crashing the graph. The LLM receives the error as context and can adapt — for example, by trying different arguments or telling the user something went wrong.

---

## MessagesState

For any conversational agent, you need a state schema where messages accumulate across turns. LangGraph provides `MessagesState` as a ready-made base for this:

```python
from langgraph.graph import MessagesState
```

Internally, `MessagesState` defines exactly one field:

```python
messages: Annotated[list[AnyMessage], operator.add]
```

The `Annotated` type with `operator.add` as the second argument is a **reducer**. Reducers tell LangGraph how to merge new state values with existing ones. `operator.add` on a list means "append" rather than "replace" — when a node returns `{"messages": [new_message]}`, that message is added to the end of the list, not substituted for the whole list. Without this reducer, every node return would wipe out the conversation history.

### Extending MessagesState with custom fields

You can subclass `MessagesState` to add domain-specific fields alongside the built-in messages list:

```python
class TutorMatchingState(MessagesState):
    phase: str                        # "gather", "present", "book", "done"
    preferences: dict                 # student preferences collected so far
    search_results: list[dict]        # results from the last search_tutors call
    selected_tutor: dict | None       # tutor the student chose
    booking_confirmation: dict | None # final booking details
```

The `messages` field and its reducer are inherited automatically. The custom fields use the default "last write wins" behavior — when a node returns `{"phase": "present"}`, it simply replaces whatever was there before. This is appropriate for these fields because only one node (`agent_node`) writes to them.

---

## The Agent Loop Pattern

The most important structural concept in Project 4 is the agent loop. Look at the graph definition in `graph.py`:

```python
graph = (
    StateGraph(TutorMatchingState)
    .add_node("agent_node", agent_node)
    .add_node("tool_node", tool_node)
    .add_edge(START, "agent_node")
    .add_conditional_edges("agent_node", should_continue, ["tool_node", "__end__"])
    .add_edge("tool_node", "agent_node")
    .compile(checkpointer=checkpointer)
)
```

There are only two nodes. The conditional edge after `agent_node` routes based on `should_continue`:

```python
def should_continue(state: TutorMatchingState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "__end__"
```

The logic is simple: if the LLM's last message contains tool calls, run the tools; otherwise, stop. After the tool node runs, control always returns to `agent_node`. This creates a loop:

```
START → agent_node → (tool calls?) → tool_node → agent_node → (done?) → END
                          ↑___________________________|
```

The loop continues until the LLM produces a response with no tool calls — meaning it has everything it needs to reply to the user.

### Contrast with P2/P3

In Projects 2 and 3, you designed the pipeline explicitly. Each node had a fixed job (gather requirements, generate content, validate output), and the graph structure determined the flow. The LLM was a component inside nodes you controlled.

Here, the LLM is the decision-maker. It decides when to search for tutors, when to check availability, and when to book. The graph doesn't encode that logic — it just provides the infrastructure to execute whatever the LLM decides. This is the fundamental shift from a **pipeline** to an **agent**.

---

## Checkpointers and Persistence

A checkpointer saves the full graph state after each "super-step" (each pass through a node). This is what makes multi-turn conversation possible: instead of losing state between user messages, the checkpointer stores it so the next invocation can pick up exactly where it left off.

### thread_id: conversation isolation

Each conversation gets a unique `thread_id` passed in the config:

```python
config = {"configurable": {"thread_id": thread_id}}
graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
```

The `thread_id` is the key LangGraph uses to look up and store state. Two different `thread_id`s are completely independent conversations. When you call `graph.invoke()` with the same `thread_id` on a second turn, the new message is appended to the existing state rather than starting fresh.

### InMemorySaver vs SqliteSaver

The project demonstrates the pluggable nature of checkpointers — the same graph works with either:

```python
# In-memory: fast, simple, lost when the process exits
checkpointer = InMemorySaver()

# SQLite: written to disk, survives restarts
checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
```

The checkpointer is injected at compile time via `build_graph(checkpointer=checkpointer)`. Nothing in the graph logic changes — only the storage backend does. This pluggability is by design: you can start with `InMemorySaver` during development and swap to a database-backed checkpointer in production without touching any node code.

---

## Phase-Based Prompting

The booking workflow has a natural sequence: gather preferences, search for tutors, present results, book a session. Rather than building separate graph nodes for each of these stages (as P2/P3 would), Project 4 handles all of them in a single `agent_node` — but guides the LLM's behavior with phase-specific system prompts.

### Tracking phase in state

The `phase` field in `TutorMatchingState` records where the conversation is. It starts as `"gather"` and advances through `"present"` and `"book"` to `"done"`.

### Deterministic phase transitions

Crucially, phase transitions are not decided by the LLM. They are inferred deterministically from which tool the LLM chose to call:

```python
if phase == "gather" and response.tool_calls:
    if "search_tutors" in [tc["name"] for tc in response.tool_calls]:
        new_phase = "present"

elif phase == "present" and response.tool_calls:
    if "check_availability" in ...:
        new_phase = "book"

elif phase == "book" and response.tool_calls:
    if "book_session" in ...:
        new_phase = "done"
```

The tool the LLM calls is the signal. Calling `search_tutors` implies gathering is done. Calling `check_availability` implies the student picked a tutor. This keeps state transitions predictable and auditable — a key advantage of combining LLM flexibility with explicit state management.

### Phase-specific system prompts

Each phase has its own system prompt in `prompts.py` that guides the LLM on what to do next. For example, the `"gather"` prompt says: ask for specialization, timezone, and availability, but don't call `search_tutors` until you have at least the specialization. The `"book"` prompt says: use `check_availability` first if you haven't, then finalize with `book_session`.

The system prompt changes, but the node code and graph structure don't. This is a lightweight way to give a single LLM node multi-phase awareness without multiplying graph complexity.
