from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from tenacity import RetryCallState

import json
import os
from typing import Optional, Any

from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=600)

# Model configurations
SMART = "anthropic.claude-3-7-sonnet-20250219-v1-0"
DUMB = "gemini-2.0-flash-001"

class AuthHeaderHandler(BaseCallbackHandler):
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Simplified retry handler without corporate headers
        pass

def create_model(api_key: str, model_name: str, base_url: Optional[str] = None):
    """Create a chat model instance"""
    model_params = {
        "api_key": api_key,
        "model_name": model_name,
        "temperature": 0.0,
        "max_retries": 2,
    }
    
    if base_url:
        model_params["base_url"] = base_url
        
    return ChatOpenAI(**model_params).with_config(
        config={"tags": ["langsmith:nostream"]}
    )

def create_embedding_model(api_key: str, base_url: Optional[str] = None):
    """Create an embedding model instance"""
    model_params = {
        "openai_api_key": api_key,
        "model": "text-embedding-ada-002",
        "show_progress_bar": True,
        "chunk_size": 900,
        "max_retries": 2,
    }
    
    if base_url:
        model_params["base_url"] = base_url
        
    return OpenAIEmbeddings(**model_params)

def load_medical_documents():
    """Load medical documents for the knowledge base"""
    json_data = {}
    current_dir = os.getcwd()
    docs_dir = os.path.join(current_dir, "src/docs")
    
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, "r") as file:
                    try:
                        json_data[filename.removesuffix(".json")] = json.load(file)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from file {filename}: {e}")
    
    return json_data