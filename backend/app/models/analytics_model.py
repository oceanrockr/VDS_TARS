"""
T.A.R.S. Analytics Database Models
PostgreSQL models for analytics persistence
Phase 6 - Production Scaling & Monitoring
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    Index,
    ForeignKey
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship

from ..core.db import Base


class QueryLog(Base):
    """
    Log of all RAG queries.

    Tracks query performance, features used, and results.
    """
    __tablename__ = 'query_logs'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Query details
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash
    client_id = Column(String(255), nullable=False, index=True)

    # Timing metrics
    retrieval_time_ms = Column(Float, nullable=False)
    generation_time_ms = Column(Float, nullable=False)
    total_time_ms = Column(Float, nullable=False)

    # Results
    sources_count = Column(Integer, nullable=False)
    relevance_scores = Column(JSONB, nullable=True)  # Array of scores
    avg_relevance_score = Column(Float, nullable=True)

    # Model info
    model_used = Column(String(100), nullable=False)
    tokens_generated = Column(Integer, nullable=False, default=0)

    # Advanced RAG features (Phase 5)
    used_reranking = Column(Boolean, nullable=False, default=False)
    used_hybrid_search = Column(Boolean, nullable=False, default=False)
    used_query_expansion = Column(Boolean, nullable=False, default=False)
    expansion_count = Column(Integer, nullable=False, default=0)

    # Status
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)

    # Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    document_accesses = relationship("DocumentAccess", back_populates="query_log")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_query_timestamp', 'timestamp'),
        Index('idx_query_client_timestamp', 'client_id', 'timestamp'),
        Index('idx_query_success', 'success'),
        Index('idx_query_features', 'used_reranking', 'used_hybrid_search', 'used_query_expansion'),
    )

    def __repr__(self):
        return f"<QueryLog(id={self.id}, query='{self.query_text[:50]}...', success={self.success})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'query_text': self.query_text,
            'client_id': self.client_id,
            'retrieval_time_ms': self.retrieval_time_ms,
            'generation_time_ms': self.generation_time_ms,
            'total_time_ms': self.total_time_ms,
            'sources_count': self.sources_count,
            'avg_relevance_score': self.avg_relevance_score,
            'model_used': self.model_used,
            'tokens_generated': self.tokens_generated,
            'used_reranking': self.used_reranking,
            'used_hybrid_search': self.used_hybrid_search,
            'used_query_expansion': self.used_query_expansion,
            'expansion_count': self.expansion_count,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class DocumentAccess(Base):
    """
    Log of document/chunk accesses during RAG queries.

    Tracks which documents are retrieved and their relevance.
    """
    __tablename__ = 'document_accesses'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to query
    query_id = Column(Integer, ForeignKey('query_logs.id', ondelete='CASCADE'), nullable=True, index=True)

    # Document details
    document_id = Column(String(255), nullable=False, index=True)
    file_name = Column(String(512), nullable=False)
    chunk_id = Column(String(255), nullable=True, index=True)

    # Relevance
    relevance_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=True)  # Position in results (1-based)

    # Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    query_log = relationship("QueryLog", back_populates="document_accesses")

    # Indexes
    __table_args__ = (
        Index('idx_doc_access_timestamp', 'timestamp'),
        Index('idx_doc_access_document', 'document_id', 'timestamp'),
        Index('idx_doc_access_query', 'query_id'),
    )

    def __repr__(self):
        return f"<DocumentAccess(id={self.id}, document={self.document_id}, relevance={self.relevance_score})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'query_id': self.query_id,
            'document_id': self.document_id,
            'file_name': self.file_name,
            'chunk_id': self.chunk_id,
            'relevance_score': self.relevance_score,
            'rank': self.rank,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ErrorLog(Base):
    """
    Log of system errors and exceptions.

    Tracks errors for monitoring and debugging.
    """
    __tablename__ = 'error_logs'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Error details
    error_type = Column(String(255), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)

    # Context
    service = Column(String(100), nullable=True, index=True)  # e.g., 'rag_service', 'embedding_service'
    operation = Column(String(100), nullable=True)  # e.g., 'query', 'index_document'
    client_id = Column(String(255), nullable=True, index=True)

    # Severity
    severity = Column(String(20), nullable=False, default='ERROR', index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('idx_error_timestamp', 'timestamp'),
        Index('idx_error_type_timestamp', 'error_type', 'timestamp'),
        Index('idx_error_service', 'service'),
        Index('idx_error_severity', 'severity'),
    )

    def __repr__(self):
        return f"<ErrorLog(id={self.id}, type={self.error_type}, severity={self.severity})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'service': self.service,
            'operation': self.operation,
            'client_id': self.client_id,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class SystemMetrics(Base):
    """
    Periodic snapshots of system metrics.

    Stores resource utilization over time for trend analysis.
    """
    __tablename__ = 'system_metrics'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Resource metrics
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)
    gpu_memory_usage_percent = Column(Float, nullable=True)

    # Application metrics
    active_connections = Column(Integer, nullable=True)
    cache_hit_rate = Column(Float, nullable=True)
    total_documents = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)

    # Performance metrics (aggregates over interval)
    avg_query_time_ms = Column(Float, nullable=True)
    avg_retrieval_time_ms = Column(Float, nullable=True)
    avg_generation_time_ms = Column(Float, nullable=True)
    queries_per_minute = Column(Float, nullable=True)

    # Metadata
    metadata = Column(JSONB, nullable=True)

    # Timestamp (typically collected every 1-5 minutes)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, cpu={self.cpu_usage_percent}%, timestamp={self.timestamp})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'gpu_usage_percent': self.gpu_usage_percent,
            'gpu_memory_usage_percent': self.gpu_memory_usage_percent,
            'active_connections': self.active_connections,
            'cache_hit_rate': self.cache_hit_rate,
            'total_documents': self.total_documents,
            'avg_query_time_ms': self.avg_query_time_ms,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
