"""
To add documents to the BigQuery knowledge base, we can use this script:
    python -c "from src.rag.BQ_db import create_knowledge_base; create_knowledge_base()"
It will load PDF documents and then add them to the BigQueryVectorStore.
"""

from langchain_google_community import BigQueryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os

def create_knowledge_base(project_id: str, dataset_name: str, table_name: str, 
                         location: str = "us-central1", api_key: str = None):
    """Create and populate the knowledge base in BigQuery"""

    if not api_key:
        raise ValueError("OpenAI API key is required")

    embedding = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002",
        show_progress_bar=True,
        chunk_size=900,
        max_retries=2
    )

    store = BigQueryVectorStore(
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        embedding=embedding,
        location=location
    )

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

def search_knowledge_base(query: str, project_id: str, dataset_name: str, 
                         table_name: str, location: str = "us-central1", 
                         api_key: str = None, k: int = 5):
    """Search the knowledge base for relevant context"""
    
    if not api_key:
        raise ValueError("OpenAI API key is required")

    embedding = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-ada-002",
        show_progress_bar=True,
        chunk_size=900,
        max_retries=2
    )

    # Initialize the BigQueryVectorStore with the embedding model
    store = BigQueryVectorStore(
        project_id=project_id,
        dataset_name=dataset_name,
        table_name=table_name,
        embedding=embedding,
        location=location
    )

    return store.similarity_search(query, k=k)
