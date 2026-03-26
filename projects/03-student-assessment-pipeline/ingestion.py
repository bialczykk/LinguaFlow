"""Document ingestion module for the Student Assessment Pipeline.

Handles loading CEFR rubrics, level descriptors, and sample essays
into a Chroma vector store with metadata for filtered retrieval.

This module is separate from the assessment graph — ingestion is a
one-time setup concern, not part of the assessment workflow.

RAG concepts demonstrated:
- Document loading from Python data structures
- Text splitting with RecursiveCharacterTextSplitter
- Embedding with sentence-transformers (local, no API key)
- Storing in Chroma with metadata for filtered retrieval
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.rubrics import ALL_RUBRICS
from data.standards import ALL_STANDARDS
from data.sample_essays import ALL_SAMPLE_ESSAYS

# Embedding model — runs locally via sentence-transformers
# all-mpnet-base-v2 is a good general-purpose model (768 dimensions)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Collection name used across ingestion and retrieval
COLLECTION_NAME = "linguaflow_assessment"

# Default persist directory (relative to project root)
DEFAULT_PERSIST_DIR = "./chroma_db"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create the HuggingFace embedding model instance.

    Uses all-mpnet-base-v2 which runs locally — no API key needed.
    The same model must be used for both ingestion and retrieval.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Build and persist the Chroma vector store from all document sources.

    Loads rubrics, standards, and sample essays, splits longer documents,
    embeds them, and stores in Chroma with metadata.

    Args:
        persist_directory: Path to persist the Chroma database.

    Returns:
        The populated Chroma vector store instance.
    """
    # Collect all documents from the data modules
    all_documents = ALL_RUBRICS + ALL_STANDARDS + ALL_SAMPLE_ESSAYS

    # Split longer documents into chunks for better retrieval
    # Rubrics and standards are relatively short, but sample essays can be longer
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(all_documents)

    # Create the vector store with embeddings and metadata
    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """Load an existing persisted Chroma vector store.

    Must use the same embedding model that was used during ingestion.

    Args:
        persist_directory: Path where the Chroma database is persisted.

    Returns:
        The loaded Chroma vector store instance.
    """
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
