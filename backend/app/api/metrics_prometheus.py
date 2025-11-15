"""
T.A.R.S. Prometheus Metrics Endpoint
Exposes application metrics for Prometheus scraping
Phase 6 - Production Scaling & Monitoring
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry
)

from ..core.config import settings
from ..services.redis_cache import redis_cache
from ..services.analytics_service import analytics_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/metrics", tags=["metrics"])

# Create custom registry (optional, for isolation)
# registry = CollectorRegistry()

# ==============================================================================
# Application Info Metrics
# ==============================================================================

app_info = Info(
    'tars_application',
    'T.A.R.S. application information'
)
app_info.info({
    'version': settings.APP_VERSION,
    'environment': settings.FASTAPI_ENV,
    'model': settings.OLLAMA_MODEL
})

# ==============================================================================
# Request Metrics
# ==============================================================================

http_requests_total = Counter(
    'tars_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'tars_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ==============================================================================
# RAG Query Metrics
# ==============================================================================

rag_queries_total = Counter(
    'tars_rag_queries_total',
    'Total RAG queries',
    ['status']  # success, failed
)

rag_query_duration_seconds = Histogram(
    'tars_rag_query_duration_seconds',
    'RAG query total duration',
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0)
)

rag_retrieval_duration_seconds = Histogram(
    'tars_rag_retrieval_duration_seconds',
    'RAG retrieval phase duration',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)

rag_generation_duration_seconds = Histogram(
    'tars_rag_generation_duration_seconds',
    'RAG LLM generation duration',
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
)

rag_sources_retrieved = Histogram(
    'tars_rag_sources_retrieved',
    'Number of sources retrieved per query',
    buckets=(0, 1, 2, 3, 5, 10, 15, 20)
)

rag_relevance_score = Histogram(
    'tars_rag_relevance_score',
    'Average relevance score per query',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# ==============================================================================
# Advanced RAG Feature Metrics
# ==============================================================================

rag_reranking_usage = Counter(
    'tars_rag_reranking_usage_total',
    'Number of queries using advanced reranking'
)

rag_hybrid_search_usage = Counter(
    'tars_rag_hybrid_search_usage_total',
    'Number of queries using hybrid search'
)

rag_query_expansion_usage = Counter(
    'tars_rag_query_expansion_usage_total',
    'Number of queries using query expansion'
)

rag_semantic_chunking_usage = Counter(
    'tars_rag_semantic_chunking_usage_total',
    'Number of documents indexed with semantic chunking'
)

# ==============================================================================
# Document Indexing Metrics
# ==============================================================================

documents_indexed_total = Counter(
    'tars_documents_indexed_total',
    'Total documents indexed',
    ['status']  # success, failed, already_indexed
)

document_chunks_created = Histogram(
    'tars_document_chunks_created',
    'Number of chunks created per document',
    buckets=(1, 5, 10, 25, 50, 100, 200, 500)
)

document_indexing_duration_seconds = Histogram(
    'tars_document_indexing_duration_seconds',
    'Document indexing duration',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# ==============================================================================
# Cache Metrics
# ==============================================================================

cache_operations_total = Counter(
    'tars_cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # operation: get/set, result: hit/miss/success/error
)

cache_hit_rate = Gauge(
    'tars_cache_hit_rate',
    'Current cache hit rate percentage'
)

# ==============================================================================
# WebSocket Metrics
# ==============================================================================

websocket_connections_active = Gauge(
    'tars_websocket_connections_active',
    'Number of active WebSocket connections'
)

websocket_messages_total = Counter(
    'tars_websocket_messages_total',
    'Total WebSocket messages',
    ['direction']  # sent, received
)

# ==============================================================================
# Database Metrics
# ==============================================================================

chromadb_operations_total = Counter(
    'tars_chromadb_operations_total',
    'Total ChromaDB operations',
    ['operation']  # add, query, delete
)

chromadb_query_duration_seconds = Histogram(
    'tars_chromadb_query_duration_seconds',
    'ChromaDB query duration',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
)

# ==============================================================================
# Model Performance Metrics
# ==============================================================================

ollama_generation_tokens_total = Counter(
    'tars_ollama_generation_tokens_total',
    'Total tokens generated by Ollama'
)

ollama_generation_tokens_per_second = Gauge(
    'tars_ollama_generation_tokens_per_second',
    'Ollama token generation rate'
)

embedding_operations_total = Counter(
    'tars_embedding_operations_total',
    'Total embedding operations',
    ['type']  # query, document, batch
)

embedding_duration_seconds = Histogram(
    'tars_embedding_duration_seconds',
    'Embedding generation duration',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
)

# ==============================================================================
# System Resource Metrics (Updated by background task)
# ==============================================================================

system_cpu_usage_percent = Gauge(
    'tars_system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage_percent = Gauge(
    'tars_system_memory_usage_percent',
    'System memory usage percentage'
)

system_gpu_usage_percent = Gauge(
    'tars_system_gpu_usage_percent',
    'GPU usage percentage (if available)'
)

system_gpu_memory_usage_percent = Gauge(
    'tars_system_gpu_memory_usage_percent',
    'GPU memory usage percentage (if available)'
)

# ==============================================================================
# Analytics Metrics
# ==============================================================================

analytics_queries_logged = Counter(
    'tars_analytics_queries_logged_total',
    'Total queries logged to analytics'
)

analytics_documents_accessed = Counter(
    'tars_analytics_documents_accessed_total',
    'Total document accesses logged'
)

# ==============================================================================
# Helper Functions
# ==============================================================================

async def update_cache_metrics():
    """Update cache metrics from Redis stats"""
    try:
        if redis_cache.is_connected:
            stats = redis_cache.get_stats()
            cache_hit_rate.set(stats['hit_rate_percent'])

            # Update operation counters (delta tracking)
            # Note: This is approximate since counters are cumulative
            cache_operations_total.labels(operation='get', result='hit')._value.set(stats['hits'])
            cache_operations_total.labels(operation='get', result='miss')._value.set(stats['misses'])
            cache_operations_total.labels(operation='set', result='success')._value.set(stats['sets'])
            cache_operations_total.labels(operation='all', result='error')._value.set(stats['errors'])

    except Exception as e:
        logger.warning(f"Error updating cache metrics: {e}")


async def update_analytics_metrics():
    """Update metrics from analytics service"""
    try:
        if hasattr(analytics_service, 'get_query_stats'):
            stats = await analytics_service.get_query_stats()

            if stats:
                # Update counters based on total values
                analytics_queries_logged._value.set(stats.get('total_queries', 0))

    except Exception as e:
        logger.warning(f"Error updating analytics metrics: {e}")


async def update_system_metrics():
    """Update system resource metrics"""
    try:
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_cpu_usage_percent.set(cpu_percent)

        # Memory
        memory = psutil.virtual_memory()
        system_memory_usage_percent.set(memory.percent)

        # GPU (if available)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # GPU utilization
                    utilization = torch.cuda.utilization(i)
                    system_gpu_usage_percent.set(utilization)

                    # GPU memory
                    mem_allocated = torch.cuda.memory_allocated(i)
                    mem_total = torch.cuda.get_device_properties(i).total_memory
                    mem_percent = (mem_allocated / mem_total) * 100
                    system_gpu_memory_usage_percent.set(mem_percent)
        except Exception:
            pass  # GPU metrics not available

    except Exception as e:
        logger.warning(f"Error updating system metrics: {e}")


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.get("/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping.
    """
    # Update dynamic metrics before exporting
    await update_cache_metrics()
    await update_analytics_metrics()
    await update_system_metrics()

    # Generate metrics
    metrics_output = generate_latest()

    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/health")
async def metrics_health():
    """
    Health check for metrics endpoint.

    Returns:
        Status of metrics collection
    """
    return {
        "status": "healthy",
        "prometheus_enabled": getattr(settings, 'PROMETHEUS_ENABLED', True),
        "metrics_collected": True
    }


@router.get("/summary")
async def metrics_summary():
    """
    Human-readable metrics summary.

    Returns:
        Dictionary with current metric values
    """
    await update_cache_metrics()
    await update_analytics_metrics()
    await update_system_metrics()

    cache_stats = redis_cache.get_stats() if redis_cache.is_connected else {}

    return {
        "application": {
            "version": settings.APP_VERSION,
            "environment": settings.FASTAPI_ENV,
            "model": settings.OLLAMA_MODEL
        },
        "cache": {
            "enabled": redis_cache.is_connected,
            "hit_rate_percent": cache_stats.get('hit_rate_percent', 0),
            "total_hits": cache_stats.get('hits', 0),
            "total_misses": cache_stats.get('misses', 0),
            "total_sets": cache_stats.get('sets', 0)
        },
        "system": {
            "cpu_percent": system_cpu_usage_percent._value.get(),
            "memory_percent": system_memory_usage_percent._value.get()
        }
    }


# ==============================================================================
# Metric Recording Functions (to be called from other services)
# ==============================================================================

def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics"""
    http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def record_rag_query(
    success: bool,
    total_time: float,
    retrieval_time: float,
    generation_time: float,
    sources_count: int,
    avg_relevance: float,
    used_reranking: bool = False,
    used_hybrid: bool = False,
    used_expansion: bool = False
):
    """Record RAG query metrics"""
    status = 'success' if success else 'failed'
    rag_queries_total.labels(status=status).inc()

    if success:
        rag_query_duration_seconds.observe(total_time)
        rag_retrieval_duration_seconds.observe(retrieval_time)
        rag_generation_duration_seconds.observe(generation_time)
        rag_sources_retrieved.observe(sources_count)
        rag_relevance_score.observe(avg_relevance)

        if used_reranking:
            rag_reranking_usage.inc()
        if used_hybrid:
            rag_hybrid_search_usage.inc()
        if used_expansion:
            rag_query_expansion_usage.inc()


def record_document_indexing(
    status: str,
    chunks_created: int,
    duration: float,
    used_semantic_chunking: bool = False
):
    """Record document indexing metrics"""
    documents_indexed_total.labels(status=status).inc()

    if status == 'success':
        document_chunks_created.observe(chunks_created)
        document_indexing_duration_seconds.observe(duration)

        if used_semantic_chunking:
            rag_semantic_chunking_usage.inc()


def record_cache_operation(operation: str, result: str):
    """Record cache operation"""
    cache_operations_total.labels(operation=operation, result=result).inc()


def record_ollama_generation(tokens: int, duration: float):
    """Record Ollama generation metrics"""
    ollama_generation_tokens_total.inc(tokens)
    if duration > 0:
        tokens_per_second = tokens / duration
        ollama_generation_tokens_per_second.set(tokens_per_second)


def record_embedding_operation(operation_type: str, duration: float):
    """Record embedding operation"""
    embedding_operations_total.labels(type=operation_type).inc()
    embedding_duration_seconds.observe(duration)


def set_websocket_connections(count: int):
    """Update active WebSocket connection count"""
    websocket_connections_active.set(count)


def record_websocket_message(direction: str):
    """Record WebSocket message"""
    websocket_messages_total.labels(direction=direction).inc()
