# Project 01: Grammar Correction Agent

## A Deep Dive into LangChain Fundamentals

---

## 1. Introduction

### What We Built

This project builds an automated grammar correction agent for LinguaFlow's Student Success department. The agent takes a piece of student writing and does two things:

1. **Structured analysis** — identifies every grammar issue, categorizes it, explains the rule, and produces an overall CEFR proficiency assessment. All of this comes back as a clean, structured Python object, not a blob of text.
2. **Follow-up conversation** — once the analysis is done, the student can ask questions about any of the issues. The agent remembers the full conversation context, so follow-up questions like "Can you explain that tense error more?" work naturally.

### The Business Context

LinguaFlow's Student Success department handles hundreds of writing submissions per week. Before this agent, a human tutor would read each piece, write corrections by hand, and then field follow-up questions via email — slow and inconsistent. The grammar correction agent automates the first pass: instant, structured feedback every time, with a conversational interface for clarification.

This isn't just a demo. The design decisions here — structured output, conversation history, observability — are exactly what you'd build in a real product.

---

## 2. LangChain Fundamentals

### What is LangChain?

LangChain is a framework for building applications powered by large language models (LLMs). On its own, an LLM is just a function: text in, text out. LangChain wraps that function with everything a real application needs:

- **Prompt management** — reusable, parameterized prompt templates instead of f-strings scattered across your code
- **Model abstraction** — a uniform interface across providers (Anthropic, OpenAI, Google, etc.) so you can swap models without rewriting logic
- **Output parsing** — tools for extracting structured data from LLM responses
- **Composability** — a clean way to wire these pieces together into pipelines called *chains*

### The Core Abstraction: Chains

A chain is a sequence of steps where the output of one step becomes the input of the next. The simplest chain is:

```
prompt template → model → output
```

In LangChain, you express this with the pipe operator `|`:

```python
chain = prompt | model
result = chain.invoke({"variable": "value"})
```

Think of it like Unix pipes. Each component is a transformer. Data flows through left to right. This composability is what makes LangChain powerful — you can swap any component without touching the others.

---

## 3. Prompt Templates

### Why Templates Instead of f-strings?

You could write prompts like this:

```python
prompt = f"Analyze this student writing: {student_text}"
response = model.invoke(prompt)
```

It works, but it has problems. The prompt is buried in application code. You can't reuse it. You can't test it in isolation. And for chat models, you need to manage the message structure (system vs. human messages) yourself.

`ChatPromptTemplate` solves this by letting you define the prompt structure once, separately from the data that fills it in.

### `ChatPromptTemplate.from_messages()`

Here's the actual prompt from `chains.py`:

```python
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert English grammar teacher for the LinguaFlow "
                "tutoring platform. Your role is to analyze student writing "
                "and provide detailed, educational grammar feedback.\n\n"
                "For each grammar issue you find:\n"
                "- Identify the exact problematic text fragment\n"
                "- Provide the corrected version\n"
                "- Categorize the error type (e.g., subject-verb agreement, "
                "tense, article usage, punctuation, word order, spelling)\n"
                # ...
            ),
        ),
        (
            "human",
            "Please analyze the following student writing:\n\n{student_text}",
        ),
    ]
)
```

`from_messages()` takes a list of `(role, content)` tuples. The roles are:

- `"system"` — sets the model's persona, constraints, and behavior. This message is invisible to the end user but shapes every response.
- `"human"` — represents the user's input
- `"ai"` — represents a previous model response (used for conversation history)

### The `{variable}` Placeholder Syntax

Notice `{student_text}` in the human message. When you call `chain.invoke({"student_text": "The cat is sat on the mat."})`, LangChain replaces `{student_text}` with the actual value. It's the same syntax as Python's `str.format()`, just managed for you.

### Why System Messages Matter

The system message is where you encode expertise and constraints. Without it:

```
"Please analyze this student writing: I goes to school yesterday."
```

The model might respond casually, skip structure, or produce inconsistent output. With a well-crafted system message, you get a consistent, expert persona that follows your format every time. Think of the system message as the job description you give to a new employee — it tells them who they are, what they're doing, and how to behave.

---

## 4. Model Configuration

### `ChatAnthropic`

```python
from langchain_anthropic import ChatAnthropic

_model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
```

`ChatAnthropic` is a LangChain wrapper around Anthropic's API. The `langchain-anthropic` package provides it. Because it implements LangChain's `BaseChatModel` interface, it works with the same `|` composition and `.invoke()` call as any other model — swap it for `ChatOpenAI` and your chain logic stays identical.

### Key Parameters

**`model`** — specifies which Claude model to use. Different models trade off speed, capability, and cost. `claude-sonnet-4-5` is a strong general-purpose model well-suited for structured extraction tasks.

**`temperature`** — controls randomness in the output. The scale runs from 0 (deterministic) to 1 (highly varied). Two extremes:

| Temperature | Behavior | When to use |
|-------------|----------|-------------|
| `0` | Same input always produces the same output | Structured extraction, analysis |
| `0.7–1.0` | Creative, varied responses | Creative writing, brainstorming |

In `chains.py`, `temperature=0` is used for the structured grammar analysis. We need the model to reliably follow the JSON schema every time — creativity is the enemy here. In `conversation.py`, `temperature=0.3` gives the tutor a slightly more natural conversational style while still being consistent.

---

## 5. Structured Output with Pydantic

### The Problem: LLMs Return Text

When you send a prompt to an LLM, you get back a string. If you asked for "a list of grammar errors in JSON", you might get:

```
Here are the grammar errors I found:

```json
{"issues": [...]}
```

Note that the second sentence has a tense problem...
```

That's not valid JSON. It has prose around it. You'd need to write a parser — fragile, annoying, unreliable.

### Pydantic Models as Schemas

Pydantic is a Python library for data validation using type annotations. You define a class, and Pydantic enforces that instances have the right types and values. In the context of LLMs, Pydantic models serve as contracts — they describe exactly what shape the data should take.

Here's the model hierarchy from `models.py`:

```python
class GrammarIssue(BaseModel):
    original_text: str = Field(
        description="The problematic fragment from the student's writing"
    )
    corrected_text: str = Field(
        description="The corrected version of the fragment"
    )
    error_category: str = Field(
        description=(
            "Grammar category, e.g. 'subject-verb agreement', "
            "'tense', 'article usage', 'punctuation', 'word order'"
        )
    )
    explanation: str = Field(
        description=(
            "Educational explanation of why this is wrong and how "
            "the grammar rule works — written for a language learner"
        )
    )
    severity: Literal["minor", "moderate", "major"] = Field(
        description="How impactful the error is on comprehension"
    )
```

Three classes work together:

- **`GrammarIssue`** — one grammar error: what was wrong, what the fix is, what category it falls into, why it's wrong, and how bad it is
- **`ProficiencyAssessment`** — overall CEFR level, strengths, areas to improve, and a summary
- **`GrammarFeedback`** — the top-level container: a list of `GrammarIssue` objects, a `ProficiencyAssessment`, and the full corrected text

### `Field(description=...)` — More Than Just Documentation

Each field has a `description`. This is crucial. LangChain converts your Pydantic model into a JSON schema and sends it to the model. The `description` of each field becomes part of the schema — the model reads it to understand what to put there.

Compare these two field definitions:

```python
# Vague — the model has to guess
error_category: str

# Clear — the model knows exactly what format and examples are expected
error_category: str = Field(
    description="Grammar category, e.g. 'subject-verb agreement', 'tense', 'article usage', 'punctuation', 'word order'"
)
```

Good field descriptions are the difference between reliable structured output and garbage. Write them as if you're explaining the field to a junior developer who has never seen your data model.

### `Literal` for Enumerated Values

```python
severity: Literal["minor", "moderate", "major"]
```

`Literal` tells Pydantic — and through it, the LLM — that this field must be exactly one of these three values. The model cannot return `"low"` or `"significant"`. This kind of strict typing prevents drift in your data.

### `model.with_structured_output(Schema, method="json_schema")`

```python
_analysis_chain = ANALYSIS_PROMPT | _model.with_structured_output(
    GrammarFeedback, method="json_schema"
)
```

`with_structured_output()` wraps the model in a layer that:
1. Takes your Pydantic class and generates a JSON schema from it
2. Instructs the model to return output matching that schema
3. Parses the model's response and validates it against the schema
4. Returns a fully instantiated `GrammarFeedback` object

`method="json_schema"` uses Claude's native structured output mode, which is the most reliable approach. The model is directly constrained at the API level, not just asked nicely to return JSON.

The result: you call `chain.invoke(...)` and get back a real `GrammarFeedback` instance you can access with `feedback.issues[0].severity` — no string parsing, no try/except around `json.loads()`.

---

## 6. Chain Composition

### The Pipe Operator `|`

LangChain uses the `|` operator (Python's bitwise OR, overloaded here) to compose components into a pipeline. It's the same concept as Unix pipes — each step receives the output of the previous step.

```python
_analysis_chain = ANALYSIS_PROMPT | _model.with_structured_output(
    GrammarFeedback, method="json_schema"
)
```

This creates a chain with two steps. When you call `_analysis_chain.invoke({"student_text": "..."})`, here's what happens:

### Data Flow Step by Step

```
invoke({"student_text": "I goes to school yesterday."})
         │
         ▼
  ┌─────────────────────────┐
  │   ANALYSIS_PROMPT       │  Substitutes {student_text} into the template,
  │   (ChatPromptTemplate)  │  produces a list of ChatMessages
  └─────────────────────────┘
         │
         │  [SystemMessage("You are an expert..."), HumanMessage("Please analyze...")]
         ▼
  ┌────────────────────────────────────┐
  │  _model.with_structured_output()   │  Sends messages to Claude API,
  │  (ChatAnthropic + schema wrapper)  │  receives JSON, validates against schema
  └────────────────────────────────────┘
         │
         │  GrammarFeedback(issues=[...], proficiency=..., corrected_full_text="...")
         ▼
      Result
```

The elegance here is that `ANALYSIS_PROMPT` doesn't know anything about the model, and the model wrapper doesn't know anything about the prompt. They're decoupled components connected by a shared data contract (ChatMessages). This makes it easy to swap either piece independently.

---

## 7. Conversation History

### Why Conversation Context Matters

After the analysis, a student might ask: "Can you explain why 'I goes' is wrong?" or "What does B1 level mean?". For the model to answer intelligently, it needs to know what analysis it just produced. Without that context, every question would be a cold start.

The solution is simple: maintain a list of messages and send the entire history to the model on every turn.

### Message Types

LangChain provides three message classes for structured conversation history:

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
```

- **`SystemMessage`** — persistent instructions/context, always stays at the front of the list
- **`HumanMessage`** — a message from the user
- **`AIMessage`** — a message from the model (a previous response)

### The `ConversationHandler` Pattern

```python
class ConversationHandler:
    def __init__(self, original_text: str, feedback: GrammarFeedback) -> None:
        self._model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

        # Format analysis results into readable text for the system prompt
        issues_text = "\n".join(
            f"  {i+1}. [{issue.severity.upper()}] '{issue.original_text}' → "
            f"'{issue.corrected_text}' ({issue.error_category}): {issue.explanation}"
            for i, issue in enumerate(feedback.issues)
        )

        # Initialize message history with a rich system message
        self._messages: list = [
            SystemMessage(
                content=(
                    "You are a friendly, encouraging English grammar tutor...\n\n"
                    f"ORIGINAL STUDENT TEXT:\n{original_text}\n\n"
                    f"GRAMMAR ISSUES FOUND:\n{issues_text}\n\n"
                    # ...
                )
            )
        ]
```

The `__init__` method does something smart: it converts the structured `GrammarFeedback` object (a Pydantic model) back into readable text and embeds it into the system message. Now the model's full context is the analysis it just produced, even though the conversation starts fresh.

### How Messages Accumulate

```python
@traceable(name="grammar_followup", run_type="chain")
def ask(self, user_message: str) -> str:
    self._messages.append(HumanMessage(content=user_message))  # 1. Add user question
    response = self._model.invoke(self._messages)               # 2. Send full history
    self._messages.append(AIMessage(content=response.content))  # 3. Save AI reply
    return response.content
```

After three turns, `self._messages` looks like:

```
[
  SystemMessage("You are a grammar tutor. ORIGINAL TEXT: ... ISSUES: ..."),
  HumanMessage("Why is 'I goes' wrong?"),
  AIMessage("'I goes' is incorrect because..."),
  HumanMessage("Can you give me a practice exercise?"),
  AIMessage("Sure! Try these sentences..."),
  HumanMessage("What about articles?"),
]
```

On each call, the entire list goes to the model. The model sees the full conversation and can refer back to anything said earlier. This is the simplest form of memory in LangChain — no external storage, just a list growing in RAM.

### Why `temperature=0.3` for Conversation?

The analysis chain uses `temperature=0` because we need exact, schema-conforming output. But conversation is different — a slightly higher temperature produces more natural, varied phrasing, which feels less robotic in a tutoring context. `0.3` is a small nudge toward expressiveness while staying consistent.

---

## 8. LangSmith Tracing

### What is LangSmith?

LangSmith is Anthropic's (technically LangChain's) observability platform for LLM applications. When your chain or agent runs, LangSmith captures the full trace: what prompt was sent, what the model returned, how long each step took, how many tokens were used.

This isn't just nice to have — it's essential for debugging. When something goes wrong with an LLM application ("the structured output sometimes has an empty issues list"), you can't reproduce it easily by reading code. You need to see the actual prompts and responses that led to the bad output.

### Setup

Three environment variables control LangSmith:

```bash
LANGSMITH_TRACING=true          # Enable/disable tracing
LANGSMITH_API_KEY=ls-...        # Your LangSmith API key
LANGSMITH_PROJECT=linguaflow-01 # Groups traces by project in the dashboard
```

When `LANGSMITH_TRACING=true`, LangChain automatically traces all chain invocations. No code changes required for basic tracing.

### The `@traceable` Decorator

For functions that aren't themselves LangChain chains (but call chains internally), use the `@traceable` decorator:

```python
from langsmith import traceable

@traceable(name="grammar_analysis", run_type="chain")
def analyze_grammar(student_text: str) -> GrammarFeedback:
    return _analysis_chain.invoke({"student_text": student_text})
```

The parameters:

- **`name`** — the label this trace appears under in LangSmith. Choose something descriptive. `"grammar_analysis"` is clear; `"func1"` is not.
- **`run_type`** — the category of trace. Common values: `"chain"` (a sequence of LLM calls), `"tool"` (a function called by an agent), `"llm"` (a raw model call). This affects how LangSmith displays and groups the trace.

The `ConversationHandler.ask` method also uses `@traceable`:

```python
@traceable(name="grammar_followup", run_type="chain")
def ask(self, user_message: str) -> str:
    ...
```

### What to Look for in the LangSmith Dashboard

When you run the agent and visit `smith.langchain.com`:

**Trace Timeline**

Each run appears as a tree. The root node is your `@traceable` function (`grammar_analysis` or `grammar_followup`). Nested under it are the individual steps: the prompt template formatting, the model call, the output parsing. You can see the exact sequence of operations.

**Inputs and Outputs**

Click any node to see the raw input and output at that step. For the prompt template node, you'll see the formatted messages with the student text substituted in. For the model node, you'll see both the request payload and the raw JSON response from Claude.

**Latency**

Each node shows how long it took. If your chain is slow, you can pinpoint exactly which step is the bottleneck — is it the model API call, or is it slow post-processing?

**Token Usage**

The model call node shows prompt tokens, completion tokens, and total tokens. This is where you'll notice if your system message is surprisingly large, or if structured output is consuming more tokens than expected.

**Separate Traces**

Because `analyze_grammar` and `ask` are decorated with different `name` values, they appear as separate traces. You can filter by project (`linguaflow-01`) and see the grammar analysis traces separately from the follow-up conversation traces. Over time, this lets you compare runs, spot regressions, and understand how usage patterns evolve.

---

## 9. Key Takeaways

### What You Learned

This project introduced the core building blocks of every LangChain application:

| Concept | What It Does | Where to Look |
|---------|-------------|---------------|
| `ChatPromptTemplate` | Defines reusable, parameterized prompts | `chains.py` — `ANALYSIS_PROMPT` |
| `ChatAnthropic` | Wraps the Anthropic API with a uniform interface | `chains.py` and `conversation.py` |
| Pydantic models | Define the schema for structured LLM output | `models.py` |
| `with_structured_output()` | Constrains the model to return validated Pydantic objects | `chains.py` — `_analysis_chain` |
| Pipe operator `\|` | Composes prompt + model into a chain | `chains.py` |
| Message history | Maintains multi-turn conversation state | `conversation.py` — `ConversationHandler` |
| `@traceable` | Makes function calls visible in LangSmith | `chains.py` and `conversation.py` |
| LangSmith | Observability platform for debugging and monitoring | `.env` configuration |

### The Pattern You Should Recognize

Nearly every LangChain application follows this pattern:

```
1. Define a Pydantic schema for what you want back
2. Write a prompt template with clear field descriptions
3. Connect prompt | model.with_structured_output(Schema)
4. Wrap with @traceable for visibility
5. Manage conversation history as a list of messages
```

This pattern scales. You'll see it repeated, extended, and composed in every project that follows.

### What Comes Next: Project 2

Project 1 uses LangChain's linear chain: prompt → model → output. It works well for single-step tasks, but real applications need branching logic. What if the writing has no grammar errors — should the chain still produce a full analysis? What if we want to route students to different feedback templates based on their CEFR level?

**Project 2** introduces **LangGraph's `StateGraph`** — a way to model your application as a graph of nodes and edges, where routing decisions happen dynamically based on intermediate results. If Project 1 is a straight road, Project 2 is a road with intersections. The same LangChain building blocks (prompts, models, structured output) appear as nodes in the graph, but now they're orchestrated by a stateful workflow engine.

---

*This document covers Project 01 of the LinguaFlow learning path. Continue to `docs/02-lesson-plan-generator.md` when ready for LangGraph.*
