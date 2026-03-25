# LinguaFlow — LangGraph Ecosystem Learning Path

## Overview

This learning path teaches the LangGraph ecosystem (LangChain, LangGraph, LangSmith, DeepAgents) through 8 progressive projects set in the domain of **LinguaFlow** — a modern English tutoring platform connecting students with tutors at scale.

LinguaFlow operates across multiple departments: Teaching, Content, Student Success, Operations, and QA. Each project addresses a real business need from one or more departments, giving every implementation a realistic, motivated context.

## Ground Rules

1. **Focus on LangGraph ecosystem concepts.** Infrastructure, APIs, and supporting code exist only to serve the learning goals. Keep them low-effort (mock APIs, simple data stores, minimal scaffolding). Never let business context pollute the LangGraph implementation.
2. **Clean separation.** All infrastructure/mock code lives in clearly separated modules (e.g., `mock_apis/`, `data/`) so the LangGraph code reads cleanly on its own. The reader should always be able to identify what is LangGraph ecosystem code vs. supporting scaffolding.
3. **Educational clarity over production polish.** Every project must be well-commented and documented. Code should teach, not just work.
4. **Progressive complexity.** Each project introduces new concepts while reinforcing previous ones. Concepts introduced in earlier projects should be reused naturally in later ones.
5. **LangSmith from day one.** Basic tracing starts in Project 1 and deepens throughout. Project 5 is the dedicated deep-dive.

---

## Project 1: Grammar Correction Agent

**Department:** Student Success
**Difficulty:** Beginner

### Business Scenario
Student Success needs an agent that takes student writing samples and returns structured grammar feedback — corrections, explanations, and a proficiency assessment. The department currently does this manually and wants to automate initial feedback so tutors can focus on higher-level coaching.

### Concepts Introduced
- LangChain fundamentals: chains, prompt templates, model integration
- Structured output with Pydantic models
- Anthropic Claude model configuration
- Basic LangSmith tracing (observing chain execution)

### What You'll Learn
How to build a basic LangChain chain that takes input, processes it through a prompt + model, and returns structured output. This is the foundation everything else builds on.

---

## Project 2: Lesson Plan Generator

**Department:** Teaching
**Difficulty:** Beginner → Intermediate

### Business Scenario
The Teaching department needs an agent that generates personalized lesson plans. Given a student's proficiency level, learning goals, and preferred topics, the agent should research appropriate material, draft a lesson plan, review it for quality, and output a finalized plan. Some students need conversation-focused plans, others need grammar drills — the agent must route accordingly.

### Concepts Introduced
- LangGraph StateGraph: defining graphs, state schemas, nodes, edges
- Conditional routing (branching logic based on state)
- Graph compilation and invocation
- LangSmith: tracing graph execution, viewing node-level traces

### What You'll Learn
How to move from linear chains to stateful graphs. You'll see why graphs matter — the lesson plan workflow has genuine branching logic that a simple chain can't handle cleanly.

---

## Project 3: Student Assessment Pipeline

**Department:** Content
**Difficulty:** Intermediate

### Business Scenario
The Content department maintains a library of English learning materials — articles, grammar exercises, rubrics, and curriculum standards. They need a system that can assess student submissions by retrieving relevant materials from this library, comparing the submission against standards, and generating a detailed assessment with scores and recommendations.

### Concepts Introduced
- RAG pipeline: document loaders, text splitting, embeddings, vector stores
- Retrieval chains integrated into a LangGraph graph
- Combining retrieval with structured generation
- LangSmith: tracing retrieval quality, inspecting retrieved documents in traces

### What You'll Learn
How to build a RAG system and integrate it into a graph-based workflow. You'll understand document ingestion, embedding, retrieval, and how retrieval quality directly impacts agent output.

---

## Project 4: Tutor Matching & Scheduling Agent

**Department:** Operations
**Difficulty:** Intermediate

### Business Scenario
Operations manages tutor-student matching and scheduling. They need an agent that students can converse with to find the right tutor — factoring in availability, specialization (grammar, conversation, business English, exam prep), timezone, and student preferences. The agent calls external APIs (calendar system, tutor database) and maintains conversation state so students can return later and continue the process.

### Concepts Introduced
- Tool calling: defining and binding tools, tool execution within graphs
- External API integration (mock REST APIs for calendar and tutor database)
- LangGraph checkpointers: persisting state across sessions
- Conversation memory and thread management
- LangSmith: creating evaluation datasets, running evaluations on matching quality

### What You'll Learn
How agents interact with external systems through tools, and how LangGraph persistence lets conversations survive across sessions. This is where agents start feeling like real applications.

---

## Project 5: Content Moderation & QA System

**Department:** QA
**Difficulty:** Intermediate → Advanced

### Business Scenario
QA reviews all AI-generated content before publication. The system drafts lesson content, automatically flags items it's uncertain about, and pauses for human moderator review. Moderators can approve, edit, or reject content at each checkpoint. Rejected content goes back for regeneration with feedback. The system must handle errors gracefully and never publish unreviewed content.

### Concepts Introduced
- Human-in-the-loop: `interrupt()`, `Command(resume=...)`, approval workflows
- Middleware: HumanInTheLoopMiddleware, custom middleware with hooks
- 4-tier error handling strategy
- LangSmith deep dive: evaluation datasets, custom evaluators, A/B testing prompts, monitoring dashboards

### What You'll Learn
How to build agents that collaborate with humans — pausing, waiting, and resuming based on human decisions. This is also the LangSmith mastery project: you'll build custom evaluators, create test datasets, and set up monitoring.

---

## Project 6: Multi-Department Support System

**Department:** All departments
**Difficulty:** Advanced

### Business Scenario
LinguaFlow needs a unified support system. Incoming requests (from students, tutors, or internal staff) are received by a supervisor agent that understands the request, routes it to the right department agent (billing, tech support, lesson scheduling, content requests), and orchestrates the response. Agents can escalate to each other — e.g., a billing question about a cancelled lesson may need input from the scheduling agent. Some requests require parallel processing across departments.

### Concepts Introduced
- Supervisor agent pattern
- Specialized sub-agents with distinct capabilities
- Agent handoff and escalation
- Shared state across agents
- Parallel execution with `Send`
- LangSmith: cross-agent tracing, latency monitoring, cost tracking

### What You'll Learn
How to orchestrate multiple agents working together. You'll understand supervisor patterns, when to use sequential vs. parallel execution, and how to maintain coherent state across agent boundaries.

---

## Project 7: Intelligent Curriculum Engine

**Department:** Content
**Difficulty:** Advanced

### Business Scenario
The Content department wants an engine that can autonomously create entire curriculum modules. Given a topic and target level, it should plan the curriculum structure, generate individual lessons, create exercises, build assessments, and write teacher guides. The engine remembers what it's already created (persistent memory), can resume interrupted work, and delegates specialized tasks to sub-agents (one for exercises, one for assessments, etc.).

### Concepts Introduced
- DeepAgents: `create_deep_agent()`, harness architecture, SKILL.md format
- DeepAgents memory: StateBackend (ephemeral), StoreBackend (persistent), FilesystemMiddleware
- DeepAgents orchestration: SubAgentMiddleware, TodoList for task planning
- CompositeBackend for routing between memory backends
- LangSmith: monitoring autonomous agent behavior, setting up alerts

### What You'll Learn
How to build truly autonomous agents using DeepAgents. You'll understand the harness architecture, how agents plan and execute their own work, and how persistent memory lets agents build on previous output across sessions.

---

## Project 8: LinguaFlow Autonomous Operations (Capstone)

**Department:** All departments
**Difficulty:** Expert

### Business Scenario
The full LinguaFlow operations system. A master orchestrator manages autonomous agents across every department:
- **Student Onboarding:** Assesses new students, recommends tutors, creates initial study plans
- **Tutor Management:** Monitors tutor availability, handles scheduling conflicts, tracks performance
- **Content Pipeline:** Generates, reviews, and publishes learning materials with QA checkpoints
- **Quality Assurance:** Monitors content quality, student satisfaction, flags issues
- **Support:** Handles incoming requests with multi-agent routing and escalation
- **Reporting:** Aggregates data across departments, generates insights

Agents plan their own work using TodoList, delegate to sub-agents, request human approval for critical decisions (new tutor onboarding, content publication, refund processing), persist state across sessions, and operate semi-autonomously. LangSmith provides full observability with tracing, evaluations, and alerts.

### Concepts Introduced
- Everything from Projects 1-7 integrated into a cohesive system
- Cross-department agent coordination
- Autonomous task planning and execution at scale
- Advanced HITL patterns: tiered approval (auto-approve low-risk, human-approve high-risk)
- Full LangSmith observability: end-to-end tracing, automated evaluations, alerting, cost monitoring

### What You'll Learn
How all LangGraph ecosystem concepts work together in a production-like system. This is the culmination — you'll see how individual concepts compose into something greater than the sum of its parts.

---

## Concept Coverage Matrix

| Concept | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---|---|---|---|---|---|---|---|
| LangChain chains & prompts | **New** | Used | Used | Used | Used | Used | Used | Used |
| Structured output | **New** | Used | Used | Used | Used | Used | Used | Used |
| LangGraph StateGraph | | **New** | Used | Used | Used | Used | Used | Used |
| Conditional routing | | **New** | Used | Used | Used | Used | Used | Used |
| RAG pipeline | | | **New** | | | | Used | Used |
| Tool calling | | | | **New** | | Used | Used | Used |
| Checkpointers & persistence | | | | **New** | Used | Used | Used | Used |
| Human-in-the-loop | | | | | **New** | | Used | Used |
| Middleware | | | | | **New** | | | Used |
| Multi-agent / Supervisor | | | | | | **New** | | Used |
| Parallel execution (Send) | | | | | | **New** | | Used |
| DeepAgents core | | | | | | | **New** | Used |
| DeepAgents memory | | | | | | | **New** | Used |
| DeepAgents orchestration | | | | | | | **New** | Used |
| LangSmith tracing | **New** | Used | Used | Used | Used | Used | Used | Used |
| LangSmith evaluation | | | | **New** | **Deep** | Used | Used | Used |
| LangSmith monitoring | | | | | **New** | Used | Used | Used |
