import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"  # Change to "gemma:2b" or other supported model as needed

    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Store Settings
    vector_store_path: str = "./vector_store"
    collection_name: str = "healthcare_docs"

    # Document Settings
    data_dir: str = "./data"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # RAG Settings
    top_k_results: int = 4
    similarity_threshold: float = 0.3

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # App
    app_name: str = "Healthcare AI Assistant"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
