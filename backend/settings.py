# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_host: str = "localhost"
    db_name: str = "postgres"
    db_user: str = "postgres"
    db_password: str
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3-32B-AWQ"
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    
    # File upload configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_query_length: int = 1000
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    default_similarity_threshold: float = 0.3
    
    # this tells pydantic to load the variables from .env
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()