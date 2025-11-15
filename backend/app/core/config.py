"""
T.A.R.S. Configuration Management
Loads environment variables and provides configuration objects
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "T.A.R.S. Backend"
    APP_VERSION: str = "v0.1.0-alpha"
    FASTAPI_ENV: str = "development"
    FASTAPI_DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Network
    HOST_IP: str = "192.168.0.11"
    BACKEND_PORT: int = 8000
    OLLAMA_PORT: int = 11434

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://llm.local:3000"

    # JWT Authentication
    JWT_SECRET_KEY: str = "your-secret-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    JWT_REFRESH_EXPIRATION_DAYS: int = 7

    # Ollama Configuration
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral:7b-instruct"
    OLLAMA_GPU_LAYERS: int = -1
    OLLAMA_NUM_THREAD: int = 8
    OLLAMA_CONTEXT_LENGTH: int = 8192

    # Model Parameters
    MODEL_TEMPERATURE: float = 0.7
    MODEL_TOP_P: float = 0.9
    MODEL_TOP_K: int = 40
    MODEL_MAX_TOKENS: int = 2048

    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 10
    WS_MESSAGE_QUEUE_SIZE: int = 100
    WS_TIMEOUT_SECONDS: int = 300

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Security
    HTTPS_ENABLED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60

    # ChromaDB Configuration (Phase 3)
    CHROMA_HOST: str = "http://chromadb:8000"
    CHROMA_COLLECTION_NAME: str = "tars_documents"
    CHROMA_CONVERSATION_COLLECTION: str = "tars_conversations"
    CHROMA_DISTANCE_METRIC: str = "cosine"
    CHROMA_MAX_BATCH_SIZE: int = 100

    # Embedding Configuration (Phase 3)
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIMENSION: int = 384  # MiniLM-L6-v2 output dimension
    EMBED_BATCH_SIZE: int = 32
    EMBED_DEVICE: str = "cuda"  # cuda or cpu
    EMBED_MAX_SEQ_LENGTH: int = 256

    # Document Processing (Phase 3)
    CHUNK_SIZE: int = 512  # tokens per chunk
    CHUNK_OVERLAP: int = 50  # token overlap between chunks
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.md,.csv"
    ENABLE_OCR: bool = False  # OCR for scanned PDFs (requires tesseract)

    # RAG Configuration (Phase 3)
    RAG_TOP_K: int = 5  # Number of chunks to retrieve
    RAG_RELEVANCE_THRESHOLD: float = 0.7  # Minimum similarity score
    RAG_RERANK_ENABLED: bool = True
    RAG_INCLUDE_SOURCES: bool = True
    RAG_MAX_CONTEXT_TOKENS: int = 2048

    # NAS Configuration (Phase 3)
    NAS_MOUNT_POINT: str = "/mnt/nas/LLM_docs"
    NAS_WATCH_ENABLED: bool = False  # File system watcher
    NAS_SCAN_INTERVAL: int = 3600  # Seconds between scans

    # Phase 5: Advanced RAG Configuration
    USE_SEMANTIC_CHUNKING: bool = True
    SEMANTIC_CHUNK_MIN: int = 400
    SEMANTIC_CHUNK_MAX: int = 800

    USE_ADVANCED_RERANKING: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 10
    RERANK_WEIGHT: float = 0.35

    USE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.3  # Weight for BM25 (0-1)

    USE_QUERY_EXPANSION: bool = False  # Disabled by default (adds latency)
    QUERY_EXPANSION_ENABLED: bool = False
    QUERY_EXPANSION_MAX: int = 3

    # Analytics Configuration (Phase 5)
    ANALYTICS_ENABLED: bool = True
    ANALYTICS_LOG_PATH: str = "./logs/analytics.log"

    # Phase 6: Redis Cache Configuration
    REDIS_ENABLED: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_EMBEDDING_TTL: int = 3600  # 60 minutes
    REDIS_RERANKER_TTL: int = 3600  # 60 minutes
    REDIS_DEFAULT_TTL: int = 3600  # 60 minutes

    # Phase 6: PostgreSQL Configuration
    POSTGRES_ENABLED: bool = False
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "tars_analytics"
    POSTGRES_USER: str = "tars"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10

    # Phase 6: Prometheus Metrics
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
