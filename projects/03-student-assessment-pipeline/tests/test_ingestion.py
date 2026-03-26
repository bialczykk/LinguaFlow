"""Tests for the document ingestion module.

Verifies that documents are correctly loaded, embedded, and stored
in Chroma with proper metadata for filtered retrieval.
"""

from langchain_core.documents import Document


def test_vector_store_has_documents(vector_store):
    """The vector store should contain all ingested documents."""
    # 12 rubrics + 6 standards + 12 sample essays = 30 base documents
    # After splitting, there may be more chunks, but at least 30
    collection = vector_store._collection
    count = collection.count()
    assert count >= 30, f"Expected at least 30 documents, got {count}"


def test_filter_by_rubric_type(vector_store):
    """Filtering by type='rubric' returns only rubric documents."""
    results = vector_store.similarity_search(
        "grammar accuracy scoring criteria",
        k=20,
        filter={"type": "rubric"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "rubric"


def test_filter_by_standard_type(vector_store):
    """Filtering by type='standard' returns only CEFR level descriptors."""
    results = vector_store.similarity_search(
        "what can a B1 learner do in writing",
        k=10,
        filter={"type": "standard"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "standard"


def test_filter_by_sample_essay_type(vector_store):
    """Filtering by type='sample_essay' returns only sample essays."""
    results = vector_store.similarity_search(
        "student essay about travel and culture",
        k=10,
        filter={"type": "sample_essay"},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "sample_essay"


def test_filter_sample_essays_by_level(vector_store):
    """Can filter sample essays by CEFR level using metadata."""
    results = vector_store.similarity_search(
        "student writing sample",
        k=10,
        filter={"$and": [{"type": "sample_essay"}, {"cefr_level": "B1"}]},
    )
    assert len(results) > 0
    for doc in results:
        assert doc.metadata["type"] == "sample_essay"
        assert doc.metadata["cefr_level"] == "B1"


def test_similarity_search_returns_relevant_docs(vector_store):
    """Similarity search for grammar topics returns grammar-related documents."""
    results = vector_store.similarity_search(
        "grammar errors subject verb agreement tense usage",
        k=5,
    )
    assert len(results) > 0
    # At least one result should be grammar-related
    grammar_related = [
        doc for doc in results
        if "grammar" in doc.page_content.lower()
        or doc.metadata.get("dimension") == "grammar"
    ]
    assert len(grammar_related) > 0


def test_get_vector_store_loads_existing(vector_store, tmp_path):
    """get_vector_store loads an existing persisted Chroma store."""
    from ingestion import build_vector_store, get_vector_store

    # Build a store in a known location
    persist_dir = str(tmp_path / "reload_test")
    build_vector_store(persist_directory=persist_dir)

    # Load it back
    loaded = get_vector_store(persist_directory=persist_dir)
    count = loaded._collection.count()
    assert count >= 30
