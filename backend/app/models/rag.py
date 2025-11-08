"""
T.A.R.S. RAG Models
Pydantic models for RAG requests, responses, and documents
Phase 3 - Document Indexing & RAG
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ==============================================================================
# DOCUMENT MODELS
# ==============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for indexed documents"""
    file_path: str
    file_name: str
    file_type: str  # pdf, docx, txt, md, csv
    file_size: int  # bytes
    created_at: datetime
    modified_at: datetime
    indexed_at: datetime
    encoding: Optional[str] = "utf-8"
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: int = 0
    hash: Optional[str] = None  # SHA256 hash for deduplication


class DocumentChunk(BaseModel):
    """Individual document chunk with embedding"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int  # Position in document
    token_count: int
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None  # Vector embedding
    similarity_score: Optional[float] = None  # Query similarity (populated during retrieval)


class Document(BaseModel):
    """Complete document representation"""
    document_id: str
    content: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []
    total_chunks: int = 0


# ==============================================================================
# INDEXING REQUEST/RESPONSE MODELS
# ==============================================================================

class DocumentUploadRequest(BaseModel):
    """Request to upload and index a document"""
    file_path: str
    force_reindex: bool = False  # Re-index even if already indexed

    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or v.strip() == "":
            raise ValueError("file_path cannot be empty")
        return v


class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    file_name: str
    status: str  # success, failed, already_indexed
    chunks_created: int
    processing_time_ms: float
    message: Optional[str] = None


class BatchIndexRequest(BaseModel):
    """Request to index multiple documents"""
    file_paths: List[str]
    force_reindex: bool = False
    max_concurrent: int = Field(default=5, ge=1, le=20)


class BatchIndexResponse(BaseModel):
    """Response for batch indexing"""
    total_files: int
    successful: int
    failed: int
    already_indexed: int
    total_chunks: int
    processing_time_ms: float
    results: List[DocumentUploadResponse]


# ==============================================================================
# RAG QUERY MODELS
# ==============================================================================

class RAGQueryRequest(BaseModel):
    """Request for RAG-enhanced query"""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_sources: bool = True
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None  # Metadata filters

    @validator('query')
    def validate_query(cls, v):
        if not v or v.strip() == "":
            raise ValueError("query cannot be empty")
        if len(v) > 10000:
            raise ValueError("query too long (max 10000 characters)")
        return v.strip()


class SourceReference(BaseModel):
    """Reference to source document"""
    document_id: str
    chunk_id: str
    file_name: str
    file_path: str
    chunk_index: int
    similarity_score: float
    excerpt: str  # Relevant text snippet
    page_number: Optional[int] = None


class RAGQueryResponse(BaseModel):
    """Response for RAG query"""
    query: str
    answer: str
    sources: List[SourceReference] = []
    context_used: str  # Combined context sent to LLM
    total_tokens: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model: str
    relevance_scores: List[float] = []


# ==============================================================================
# STREAMING RAG MODELS (for WebSocket)
# ==============================================================================

class RAGStreamToken(BaseModel):
    """Individual token in RAG stream"""
    type: str = "rag_token"
    token: str
    conversation_id: str
    has_sources: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RAGStreamSources(BaseModel):
    """Source citations in stream"""
    type: str = "rag_sources"
    conversation_id: str
    sources: List[SourceReference]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RAGStreamComplete(BaseModel):
    """Stream completion message"""
    type: str = "rag_complete"
    conversation_id: str
    total_tokens: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources_count: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ==============================================================================
# COLLECTION MANAGEMENT MODELS
# ==============================================================================

class CollectionStats(BaseModel):
    """Statistics for a ChromaDB collection"""
    collection_name: str
    total_documents: int
    total_chunks: int
    total_size_mb: float
    last_updated: datetime
    embedding_dimension: int


class DocumentSearchRequest(BaseModel):
    """Request to search indexed documents"""
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class DocumentSearchResponse(BaseModel):
    """Response for document search"""
    query: str
    results: List[DocumentChunk]
    total_results: int
    search_time_ms: float


# ==============================================================================
# HEALTH & STATUS MODELS
# ==============================================================================

class RAGHealthResponse(BaseModel):
    """Health check response for RAG service"""
    status: str  # healthy, degraded, unhealthy
    chromadb_status: str
    embedding_model_status: str
    collections: Dict[str, CollectionStats]
    total_documents: int
    total_chunks: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DocumentDeleteRequest(BaseModel):
    """Request to delete a document"""
    document_id: str


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion"""
    document_id: str
    status: str
    chunks_deleted: int
    message: Optional[str] = None
