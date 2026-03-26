# Project 03: Student Assessment Pipeline

## A Deep Dive into RAG, Vector Stores, and Phased Retrieval in LangGraph

---

## 1. Introduction

### What We Built

This project builds an automated student writing assessment pipeline for LinguaFlow's curriculum team. A student submits a piece of writing and receives a comprehensive CEFR-level assessment: scores across four dimensions (grammar, vocabulary, coherence, task achievement), a comparison against real sample essays at their level, and actionable recommendations.

The interesting part is *how* the assessment is produced. Instead of sending the submission directly to an LLM and asking "what level is this?", the pipeline first retrieves relevant rubrics and standards from a knowledge base, uses those to produce preliminary scores, and then — based on those preliminary scores — retrieves level-appropriate sample essays to ground the final assessment in concrete comparisons. This two-stage retrieval pattern is what makes the assessment credible rather than just opinionated.

### The Business Context

LinguaFlow's curriculum team needs consistent, evidence-based assessments they can trust. A raw LLM opinion ("this looks like B1") is hard to audit and drifts over time. By anchoring every assessment to specific rubric criteria and named sample essays, the pipeline produces results that can be reviewed, explained to students, and traced back to source documents.

---

## 2. What is RAG and Why Does It Matter?

### The Problem: LLMs Don't Know Your Data

Large language models like Claude are trained on vast amounts of text, but that training ends at a cutoff date and covers general knowledge — not your specific rubrics, your organization's standards, or your curated sample essays. If you ask an LLM to assess a student against LinguaFlow's proprietary CEFR rubric, it will do its best, but it's drawing from its training data about CEFR in general, not your specific criteria.

This isn't just a matter of outdated knowledge. Even for well-documented standards, your organization likely has its own interpretations, weightings, and examples. The LLM cannot know these from training alone.

### Why Not Just Fine-tune the Model?

Fine-tuning is one solution: train the model on your data so it internalizes your rubrics. But fine-tuning has significant drawbacks for this use case:

- **Cost and complexity** — fine-tuning requires large datasets, GPU resources, and expertise
- **Static** — if your rubrics change, you need to fine-tune again
- **Opaque** — the model absorbs your data in its weights; you can't easily inspect or trace *which* rubric drove a particular score
- **Overkill for retrieval** — your rubrics are a few thousand tokens; fine-tuning is for learning patterns from millions of examples

### The RAG Idea: Retrieve Then Generate

Retrieval-Augmented Generation (RAG) takes a different approach: instead of baking your data into the model, you fetch the relevant pieces at runtime and include them directly in the prompt.

The process looks like this:

```
Student submission arrives
         │
         ▼
   Retrieve relevant rubrics from knowledge base
         │
         ▼
   LLM sees: "Here are LinguaFlow's assessment criteria: [retrieved rubrics]
             Now score this submission: [student text]"
         │
         ▼
   Assessment grounded in your actual criteria
```

The LLM doesn't need to have memorized your rubrics. It just needs to read and apply them when they're placed in the prompt. This is much simpler to maintain, transparent to debug, and trivially updated when your documents change — you just re-ingest the new version into the knowledge base.

RAG is the right tool when your data is:
- Too large or specialized for fine-tuning
- Frequently updated
- Needs to be cited or audited
- Different per user, tenant, or context

---

## 3. The RAG Pipeline: Two Phases

RAG has two distinct phases that run at different times. Understanding this separation is key to understanding the code.

### Phase 1: Ingestion (One-Time Setup)

Ingestion runs once, or whenever your documents change. It processes your raw data and builds the searchable knowledge base. The steps are:

```
Raw documents (rubrics, standards, essays)
         │
         ▼  Load
   LangChain Document objects
         │
         ▼  Split
   Smaller chunks (1000 chars with overlap)
         │
         ▼  Embed
   Numeric vectors (768 dimensions each)
         │
         ▼  Store
   Chroma vector database on disk
```

In this project, ingestion is handled by `ingestion.py`. It's intentionally kept separate from the assessment graph — it's a setup concern, not part of the workflow that runs on every submission.

```python
# From ingestion.py
def build_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    all_documents = ALL_RUBRICS + ALL_STANDARDS + ALL_SAMPLE_ESSAYS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(all_documents)

    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )
    return vector_store
```

Each step here maps to a concept explained in depth below.

### Phase 2: Retrieval (Every Assessment Run)

Retrieval happens every time a student submits writing. Given a query (the student's text), the pipeline searches the pre-built knowledge base and returns the most relevant documents. The steps are:

```
Query text (student submission)
         │
         ▼  Embed (same model as ingestion)
   Query vector (768 dimensions)
         │
         ▼  Similarity search in Chroma
   Top-k most similar document vectors
         │
         ▼  Return
   Document objects with text + metadata
```

In this project, retrieval happens in two node functions in `nodes.py`. The first retrieves rubrics and standards; the second retrieves sample essays at the appropriate level. Both use the same mechanism — similarity search with metadata filters — but they query for different things at different points in the pipeline.

---

## 4. Text Splitting

### Why Splitting Is Necessary

You can't just embed entire documents and hope for the best. Consider a CEFR rubric document that covers all six levels (A1 through C2) across four assessment dimensions — that's a lot of text. If you embed the whole document as one vector, searches against it will return the whole rubric regardless of what level or dimension you actually need. The retrieval becomes imprecise.

Splitting solves this by dividing documents into smaller, focused chunks. Each chunk covers a narrower topic, so its vector represents that narrower meaning. When you search, you get the chunks that are genuinely relevant — not a large document that happens to contain relevant text somewhere inside it.

### How `RecursiveCharacterTextSplitter` Works

LangChain's `RecursiveCharacterTextSplitter` is the standard choice for most documents. It splits on a hierarchy of separators, trying the largest granularity first and falling back to smaller ones if the chunk is still too large.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
```

The splitting algorithm works like this:

1. Try to split on `"\n\n"` (paragraph breaks). If the resulting chunks are all under `chunk_size=1000` characters, done.
2. If a chunk is still too large, try splitting it on `"\n"` (line breaks).
3. If still too large, try `" "` (word boundaries).
4. If still too large, split on `""` (character level — last resort, breaks words).

This "recursive" approach preserves natural language boundaries whenever possible. Splitting at a paragraph break is better than splitting mid-sentence; splitting mid-sentence is better than splitting mid-word.

### The `chunk_overlap` Parameter

`chunk_overlap=200` means adjacent chunks share 200 characters of text. This is intentional. Consider a sentence that falls exactly at a chunk boundary:

```
... This student demonstrates strong vocabulary breadth | but limited range of
tense forms, suggesting a B1 ceiling for grammar ...
```

Without overlap, the first chunk ends at the split point and the second begins there. The idea connecting "vocabulary breadth" to "B1 ceiling" is split across two chunks that never share context. With 200-character overlap, both chunks contain the boundary text — the relationship is preserved.

Overlap increases storage size and retrieval noise slightly, but it's almost always worth it for retrieval quality.

### Choosing `chunk_size`

The right chunk size depends on your content and embedding model. A few guidelines:

| Chunk Size | Good For | Trade-off |
|-----------|----------|-----------|
| 256-512 | Short, atomic facts | Very precise retrieval; may fragment larger concepts |
| 1000-1500 | Rubric criteria, essay paragraphs | Good balance of precision and completeness |
| 2000+ | Whole sections or documents | Less precise retrieval; useful when context matters |

This project uses `chunk_size=1000` — enough to fit a complete rubric criterion description or a sample essay paragraph, without being so large that unrelated content from the same document appears in search results.

---

## 5. Embeddings

### What an Embedding Is

An embedding is a vector — a list of numbers — that represents the semantic meaning of a piece of text. The remarkable property of embedding models is that texts with similar meanings produce vectors that are close together in the vector space, even if they use different words.

For example:
- "The student uses a wide range of vocabulary" → `[0.12, -0.34, 0.78, ...]`
- "The writer demonstrates lexical diversity" → `[0.11, -0.31, 0.75, ...]`
- "The cat sat on the mat" → `[0.89, 0.22, -0.15, ...]`

The first two sentences have nearly identical vectors despite using different words. The third sentence is semantically unrelated and its vector is far away in the space. This is what makes vector search work: you don't search for keyword matches; you search for meaning matches.

### Why `sentence-transformers/all-mpnet-base-v2`

This project uses a local embedding model via HuggingFace rather than an API-based model:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
```

`all-mpnet-base-v2` is a strong general-purpose sentence embedding model that produces 768-dimensional vectors. It's a solid choice for this task because:

- **Semantic quality** — it's trained specifically on sentence-level semantic similarity, which is exactly what we need for comparing rubric descriptions to student writing
- **Local execution** — the model runs on your machine, not via an API. No network latency per embedding call, no API costs, and no data sent to a third party
- **No API key required** — removes one dependency for setup

### Local vs. API-Based Embeddings

| Approach | Pros | Cons |
|---------|------|------|
| Local (HuggingFace) | No cost, no latency, no data egress | Requires download (~420 MB), uses local CPU/GPU |
| API-based (e.g., OpenAI) | No local compute, easy to scale | API costs, latency, data sent externally |

The critical rule: **you must use the same embedding model for both ingestion and retrieval**. If you embed documents with `all-mpnet-base-v2` but then query with a different model, the vectors live in different spaces and similarity search will produce garbage. This is why the model name is a constant (`EMBEDDING_MODEL_NAME`) shared between `build_vector_store` and `get_vector_store`.

---

## 6. Vector Stores and Chroma

### What a Vector Store Does

A vector store is a database optimized for storing and searching vectors. The core operation is **nearest-neighbor search**: given a query vector, find the stored vectors that are closest to it (by cosine similarity or Euclidean distance). It returns the documents whose vectors are most similar to the query's vector — i.e., the documents most semantically similar to the query text.

Standard relational databases can't do this efficiently at scale. Vector stores use specialized indexing structures (like HNSW — Hierarchical Navigable Small World graphs) that allow approximate nearest-neighbor search in milliseconds, even across millions of vectors.

### Chroma

Chroma is an open-source vector store that is simple to use and runs locally without any external services. This makes it ideal for a learning project:

```python
# Ingestion: create and persist
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="linguaflow_assessment",
)

# Retrieval: load existing
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="linguaflow_assessment",
)
```

The `from_documents()` call handles embedding and storage in one step. When you call it, Chroma:
1. Runs each document through your embedding model to produce a vector
2. Stores the vector alongside the document text and metadata
3. Persists everything to the directory on disk so it survives between runs

### Metadata Filtering

Every document in Chroma can have a metadata dictionary attached. In this project, every document is tagged when ingested:

```python
# Example metadata on a rubric document
{"type": "rubric", "dimension": "grammar", "cefr_level": "B1"}

# Example metadata on a sample essay
{"type": "sample_essay", "cefr_level": "B2", "task": "opinion_essay"}
```

When you search, you can filter by metadata before doing similarity search. This is crucial — without filtering, a search for "grammar errors" might return sample essays that happen to contain grammar discussions, rather than the rubric documents you actually want.

The project uses Chroma's filter syntax in both retrieval nodes:

```python
# From retrieve_standards_node: only return rubric or standard documents
results = vector_store.similarity_search(
    query,
    k=10,
    filter={"$or": [{"type": "rubric"}, {"type": "standard"}]},
)

# From retrieve_samples_node: only return sample essays at specific levels
results = vector_store.similarity_search(
    query,
    k=5,
    filter={
        "$and": [
            {"type": "sample_essay"},
            {"cefr_level": {"$in": target_levels}},
        ]
    },
)
```

The `$or`, `$and`, and `$in` operators work like query operators in MongoDB or similar document databases. They let you combine metadata conditions. The combination of semantic search (the vector comparison) and metadata filtering is what makes retrieval precise: you get documents that are both *semantically relevant* and *structurally appropriate* for the query's purpose.

---

## 7. Phased Retrieval: The Key Architectural Pattern

### Single-Shot RAG vs. Phased Retrieval

Most introductory RAG examples do single-shot retrieval: retrieve documents once, then call the LLM. This is fine for simple Q&A. But assessment is a more nuanced task, and this project uses a more sophisticated pattern — **phased retrieval**, where results from an intermediate LLM call drive what gets retrieved next.

### The Two Phases in This Project

**Phase 1 — Standards retrieval**: At the start of the pipeline, the student's submission text is used to retrieve rubrics and CEFR level descriptors. This gives the LLM the scoring criteria it needs to evaluate the submission.

**LLM call — Criteria scoring**: Using the retrieved standards, the LLM scores the submission across four dimensions and assigns a preliminary CEFR level. This is an intermediate result, not the final answer.

**Phase 2 — Sample retrieval**: Now that we have a preliminary level (e.g., "B1"), we retrieve sample essays at that level *and adjacent levels* for contrast. The preliminary level from Phase 1 becomes the metadata filter in Phase 2.

```python
# From retrieve_samples_node
level = state["preliminary_level"]  # e.g., "B1" — came from criteria_scoring

level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
idx = level_order.index(level)
target_levels = [level]
if idx > 0:
    target_levels.append(level_order[idx - 1])  # B1 → also retrieve A2
if idx < len(level_order) - 1:
    target_levels.append(level_order[idx + 1])  # B1 → also retrieve B2

results = vector_store.similarity_search(
    query,
    k=5,
    filter={
        "$and": [
            {"type": "sample_essay"},
            {"cefr_level": {"$in": target_levels}},
        ]
    },
)
```

This is more interesting than retrieving all samples upfront for two reasons:

1. **Relevance** — there's no point comparing a A1-level submission against C2 sample essays. The comparison would be useless. By retrieving at the right level band, every comparison is meaningful.

2. **Efficiency** — level-filtering narrows the search space. You retrieve 5 well-targeted samples rather than 20 samples spanning all levels, most of which would be irrelevant.

### Why This Matters

Phased retrieval is a preview of a broader principle: in multi-step LLM applications, intermediate results are valuable signals. You don't have to plan the entire retrieval strategy upfront. You can let early stages produce structured intermediate outputs, and use those outputs to make smarter decisions about what to do next. This is one of the reasons LangGraph's stateful approach is so powerful — the state accumulates useful intermediate results that subsequent nodes can act on.

---

## 8. Integrating RAG into a LangGraph StateGraph

### The Graph Structure

This project's graph is linear — five nodes, wired sequentially:

```
START
  │
  ▼
retrieve_standards   ← Phase 1 retrieval
  │
  ▼
criteria_scoring     ← LLM scores the submission
  │
  ▼
retrieve_samples     ← Phase 2 retrieval (level-filtered)
  │
  ▼
comparative_analysis ← LLM compares submission to samples
  │
  ▼
synthesize           ← LLM produces final Assessment
  │
  ▼
END
```

The graph is simple, but the *intelligence* is in how state flows through it. Each node reads the fields it needs from `AssessmentState` and writes back the fields it produces:

```python
class AssessmentState(TypedDict):
    # Input
    submission_text: str
    submission_context: str
    student_level_hint: str

    # After retrieve_standards
    retrieved_standards: list[Document]

    # After criteria_scoring
    criteria_scores: CriteriaScores
    preliminary_level: str          # ← This field connects Phase 1 retrieval to Phase 2

    # After retrieve_samples
    retrieved_samples: list[Document]

    # After comparative_analysis
    comparative_analysis: ComparativeAnalysis

    # After synthesize
    final_assessment: Assessment
```

`preliminary_level` is the key connector: `criteria_scoring` writes it, `retrieve_samples` reads it. This single field is the bridge between the two retrieval phases.

### Using `functools.partial` to Inject the Vector Store

There's an elegant technical challenge here: LangGraph's node functions must have the signature `(state) -> dict`. But retrieval nodes need access to the vector store — an external object that isn't part of the state. How do you pass it in?

The solution is `functools.partial`:

```python
# From graph.py
from functools import partial

def build_graph(vector_store: Chroma):
    # Bind the vector store to retrieval nodes using partial
    # The resulting functions have signature (state) -> dict as LangGraph expects
    retrieve_standards = partial(retrieve_standards_node, vector_store=vector_store)
    retrieve_samples = partial(retrieve_samples_node, vector_store=vector_store)

    graph = (
        StateGraph(AssessmentState)
        .add_node("retrieve_standards", retrieve_standards)
        # ...
    )
```

`functools.partial` creates a new function by pre-filling some arguments of an existing function. `partial(retrieve_standards_node, vector_store=vector_store)` returns a new function that calls `retrieve_standards_node` with `vector_store` already filled in — so the caller only needs to pass `state`. LangGraph calls it with `state`, which is all it knows about.

The original node function uses a keyword-only argument (`*`) to make the pattern clear:

```python
def retrieve_standards_node(
    state: AssessmentState, *, vector_store: Chroma
) -> dict:
    ...
```

The `*` before `vector_store` makes it keyword-only. It can never be passed positionally, which prevents accidental misuse and signals to a reader that this argument is meant to be injected, not called directly.

### The Complete Graph Build

The full `build_graph` function in `graph.py` shows how clean the LangGraph assembly is once the nodes are defined:

```python
graph = (
    StateGraph(AssessmentState)
    .add_node("retrieve_standards", retrieve_standards)
    .add_node("criteria_scoring", criteria_scoring_node)
    .add_node("retrieve_samples", retrieve_samples)
    .add_node("comparative_analysis", comparative_analysis_node)
    .add_node("synthesize", synthesize_node)
    .add_edge(START, "retrieve_standards")
    .add_edge("retrieve_standards", "criteria_scoring")
    .add_edge("criteria_scoring", "retrieve_samples")
    .add_edge("retrieve_samples", "comparative_analysis")
    .add_edge("comparative_analysis", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

Notice the builder pattern: each `.add_node()` and `.add_edge()` call returns the `StateGraph` instance, allowing method chaining. The `StateGraph(AssessmentState)` call registers the state schema — LangGraph validates that node return dicts only update fields defined in `AssessmentState`.

---

## 9. Structured Output with RAG

### The Pattern

Each LLM node in this pipeline uses the same pattern from Project 1: `prompt | model.with_structured_output(Schema)`. What's new in Project 3 is that the prompt includes retrieved documents as context.

```python
# From criteria_scoring_node
structured_model = _model.with_structured_output(
    CriteriaScores, method="json_schema"
)
chain = CRITERIA_SCORING_PROMPT | structured_model

standards_text = _format_documents(state["retrieved_standards"])

result = chain.invoke(
    {
        "retrieved_standards": standards_text,  # ← Retrieved documents injected here
        "submission_text": state["submission_text"],
        "submission_context": state["submission_context"],
    }
)
```

The `_format_documents` helper converts the list of `Document` objects into a readable string with metadata headers:

```python
def _format_documents(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
        parts.append(f"--- Document {i} [{meta}] ---\n{doc.page_content}")
    return "\n\n".join(parts)
```

The metadata line (`[type: rubric, dimension: grammar, cefr_level: B1]`) is included in the formatted text. This allows the LLM to reference specific documents in its reasoning — "according to the B1 grammar rubric..." — and makes the assessment traceable.

### Intermediate vs. Final Structured Output

This project uses structured output in two different ways:

**`CriteriaScores` — intermediate structured output**: The LLM produces scores and a preliminary level. This object is stored in state and used both as input to the next LLM call (as `scores_text`) and as a metadata filter trigger (the `preliminary_level` field drives retrieval).

```python
class CriteriaScores(BaseModel):
    scores: list[CriterionScore]
    preliminary_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
    scoring_rationale: str
```

**`Assessment` — final structured output**: The last node synthesizes everything into the complete result. It uses `model_dump_json()` to serialize previous structured outputs into text for the prompt:

```python
# From synthesize_node
scores_text = state["criteria_scores"].model_dump_json(indent=2)
analysis_text = state["comparative_analysis"].model_dump_json(indent=2)

result = chain.invoke(
    {
        "criteria_scores": scores_text,
        "comparative_analysis": analysis_text,
        # ...
    }
)
```

`model_dump_json()` is a Pydantic method that serializes the model to a JSON string. The resulting JSON is well-structured and readable by the LLM — structured data flowing between nodes via JSON serialization is a clean pattern for passing rich intermediate results as prompt context.

### Pydantic Field Constraints in This Project

`models.py` uses a few field constraints worth noting:

```python
score: int = Field(ge=1, le=5, description="Score from 1 (lowest) to 5 (highest)")
```

`ge=1` (greater than or equal to 1) and `le=5` (less than or equal to 5) are validation constraints. Pydantic enforces these at instantiation time — if the LLM returns `score: 7`, Pydantic raises a `ValidationError` before the bad value reaches your code. Combined with `Literal` for enumerated fields like `preliminary_level`, these constraints create a tight contract between the LLM's output and your application's expectations.

---

## 10. LangSmith Tracing for RAG

### What to Look For

RAG pipelines have more moving parts than simple chains, and LangSmith becomes especially valuable for diagnosing retrieval quality. When you run the pipeline with tracing enabled (`LANGSMITH_TRACING=true`), each node's `@traceable` decorator creates a span in the trace.

```python
# From nodes.py
@traceable(name="retrieve_standards", run_type="retriever", tags=_TAGS)
def retrieve_standards_node(state: AssessmentState, *, vector_store: Chroma) -> dict:
    ...
```

Notice `run_type="retriever"` — this tells LangSmith this is a retrieval operation, not a chain or LLM call. LangSmith uses this to categorize spans and display relevant metrics.

### Retrieval Quality

In the LangSmith dashboard, click on a `retrieve_standards` span. You'll see:
- The query text (the student submission)
- The metadata filter applied (`{"$or": [{"type": "rubric"}, {"type": "standard"}]}`)
- The documents returned, including their metadata and content snippets

This is where you verify that retrieval is working as intended. Questions to ask:
- Are the returned documents actually relevant to the submission topic?
- Does the metadata look correct? (Are you getting `type: rubric` documents when you expected them?)
- Is `k=10` returning diverse documents, or are they all very similar chunks from the same source?

### The Phased Retrieval Trace

The trace for a full assessment run shows the two retrieval phases clearly:

```
assessment_run
  ├── retrieve_standards (retriever)
  │     ├── query: "The student writes about their weekend..."
  │     └── results: [rubric/grammar/B1, standard/vocabulary/B2, ...]
  │
  ├── criteria_scoring (chain)
  │     ├── input: {retrieved_standards: "...", submission_text: "..."}
  │     └── output: CriteriaScores(preliminary_level="B1", ...)
  │
  ├── retrieve_samples (retriever)
  │     ├── query: "The student writes about their weekend..."
  │     ├── filter: {type: sample_essay, cefr_level: {$in: [A2, B1, B2]}}
  │     └── results: [sample_essay/B1, sample_essay/A2, sample_essay/B2, ...]
  │
  ├── comparative_analysis (chain)
  │     └── ...
  │
  └── synthesize (chain)
        └── output: Assessment(overall_level="B1", ...)
```

You can directly observe how `preliminary_level="B1"` from `criteria_scoring` became the filter `{$in: [A2, B1, B2]}` in `retrieve_samples`. If the final assessment level doesn't match expectations, you can trace back through each step to find where the reasoning diverged.

### Debugging with Traces

Common issues you can diagnose in LangSmith:

- **Retrieval returns irrelevant documents**: Look at the query text and the returned documents. If the query is too broad, consider adding metadata filters. If the documents are semantically far from the query, consider whether your chunk size is too large (chunks mixing multiple topics).

- **Criteria scoring assigns wrong level**: Click the `criteria_scoring` span and inspect the input — are the retrieved standards actually relevant to the submission? Is the prompt clearly instructing the LLM how to map scores to levels?

- **Final level differs from preliminary level**: This is normal and expected. The comparative analysis phase may reveal that the student's writing is better or worse than the preliminary score suggested. You can trace this in the `synthesize` input to see what evidence tipped the balance.

---

## 11. Key Takeaways

### What You Learned

This project introduced the full RAG stack and showed how retrieval integrates naturally into a LangGraph workflow:

| Concept | What It Does | Where to Look |
|---------|-------------|---------------|
| RAG | Ground LLM responses in your specific documents | `ingestion.py`, `nodes.py` |
| `RecursiveCharacterTextSplitter` | Splits docs into retrievable chunks with overlap | `ingestion.py` — `build_vector_store` |
| HuggingFace embeddings | Converts text to semantic vectors, runs locally | `ingestion.py` — `_get_embeddings` |
| Chroma | Persistent vector store with metadata filtering | `ingestion.py`, `nodes.py` |
| Metadata filtering | Retrieves documents by type, level, or other fields | `nodes.py` — both retrieval nodes |
| Phased retrieval | Intermediate LLM results drive subsequent retrieval | `nodes.py` — `retrieve_samples_node` |
| `functools.partial` | Injects external objects (vector store) into graph nodes | `graph.py` — `build_graph` |
| `model_dump_json()` | Serializes Pydantic objects for use as LLM prompt input | `nodes.py` — `synthesize_node` |
| `run_type="retriever"` | LangSmith categorization for retrieval spans | `nodes.py` — `@traceable` decorators |

### The Pattern You Should Recognize

RAG in a LangGraph pipeline follows a consistent pattern:

```
1. At setup: ingest documents with metadata → build vector store
2. At runtime: use functools.partial to inject the store into retrieval nodes
3. In retrieval nodes: filter by metadata, search by semantic similarity
4. In LLM nodes: format retrieved docs into prompt context, get structured output
5. Use intermediate structured outputs to drive the next retrieval query
```

This pattern scales. You can add more retrieval phases, add branching logic based on retrieval results, or swap Chroma for a production vector database — the fundamental structure stays the same.

### What Comes Next: Project 4

Projects 1-3 have been about LLMs that *read* — analyzing writing, generating plans, assessing submissions. **Project 4** introduces LLMs that *act* — using tools to interact with external systems. You'll see LangGraph's tool-calling support, how to define tools, and how the agent decides when and how to call them. The state management and RAG patterns from Project 3 carry forward; what's new is the agency loop.

---

*This document covers Project 03 of the LinguaFlow learning path. Continue to `docs/04-...` when ready for tool use and agents.*
