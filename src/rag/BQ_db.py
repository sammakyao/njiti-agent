"""
To add documents to the BigQuery knowledge base, set the BQ_* and OPENAI_* variables (see .env.example)
and run:
    python -c "from src.rag.BQ_db import create_knowledge_base; create_knowledge_base()"
It will load the PDF documents in src/docs and then add them to the BigQueryVectorStore.
"""

import os
from functools import lru_cache
from typing import Optional

from src.main import API_KEY, BASE_URL, create_embedding_model


def is_configured() -> bool:
    """True when the BigQuery knowledge base settings are present"""
    return bool(API_KEY and os.getenv("BQ_PROJECT_ID") and os.getenv("BQ_DATASET") and os.getenv("BQ_TABLE"))


def _build_store(project_id: str, dataset_name: str, table_name: str, location: str, api_key: str):
    # Imported here so the web app can run without the BigQuery packages installed
    from langchain_google_community import BigQueryVectorStore

    embedding = create_embedding_model(api_key, BASE_URL)
    embedding.show_progress_bar = False

    return BigQueryVectorStore(
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        embedding=embedding,
        location=location
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """The knowledge base store, built once from environment settings"""
    if not is_configured():
        raise RuntimeError("Knowledge base is not configured: set OPENAI_API_KEY, BQ_PROJECT_ID, BQ_DATASET and BQ_TABLE")

    return _build_store(
        os.environ["BQ_PROJECT_ID"],
        os.environ["BQ_DATASET"],
        os.environ["BQ_TABLE"],
        os.getenv("BQ_LOCATION", "us-central1"),
        API_KEY,
    )


def create_knowledge_base(project_id: Optional[str] = None, dataset_name: Optional[str] = None,
                          table_name: Optional[str] = None, location: Optional[str] = None,
                          api_key: Optional[str] = None):
    """Create and populate the knowledge base in BigQuery"""
    from langchain_community.document_loaders import PyPDFLoader

    project_id = project_id or os.getenv("BQ_PROJECT_ID")
    dataset_name = dataset_name or os.getenv("BQ_DATASET")
    table_name = table_name or os.getenv("BQ_TABLE")
    location = location or os.getenv("BQ_LOCATION", "us-central1")
    api_key = api_key or API_KEY

    if not api_key:
        raise ValueError("OpenAI API key is required")
    if not (project_id and dataset_name and table_name):
        raise ValueError("BigQuery project, dataset and table are required")

    store = _build_store(project_id, dataset_name, table_name, location, api_key)

    # Load documents
    documents = []

    # Load PDFs from docs directory
    docs_dir = "src/docs"
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".pdf"):
                pdf_file = os.path.join(docs_dir, filename)
                try:
                    loader = PyPDFLoader(pdf_file)
                    pages = loader.load()
                    documents.extend(pages)
                    print(f"Loaded {len(pages)} pages from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if documents:
        store.add_documents(documents)
        print(f"Added {len(documents)} documents to BigQuery")
    else:
        print("No documents found to add")

    return store


def search_knowledge_base(query: str, k: int = 5):
    """Search the knowledge base for relevant context"""
    return get_vector_store().similarity_search(query, k=k)
