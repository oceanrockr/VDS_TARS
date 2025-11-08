"""
T.A.R.S. Analytics API
REST endpoints for analytics and usage tracking
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..core.security import verify_token
from ..services.analytics_service import analytics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# Request/Response Models

class QueryStatsResponse(BaseModel):
    """Query statistics response"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate: float
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float
    avg_sources_count: float
    avg_relevance_score: float
    reranking_usage_count: int
    reranking_usage_rate: float
    hybrid_search_usage_count: int
    hybrid_search_usage_rate: float
    query_expansion_usage_count: int
    query_expansion_usage_rate: float
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class DocumentPopularityItem(BaseModel):
    """Document popularity item"""
    document_id: str
    file_name: str
    access_count: int
    last_accessed: str
    avg_relevance_score: float
    total_retrievals: int


class QueryPatternsResponse(BaseModel):
    """Query patterns analysis"""
    total_queries: int
    avg_query_length: float
    min_query_length: int
    max_query_length: int
    top_words: Dict[str, int]
    queries_by_hour: Dict[str, int]


class ExportRequest(BaseModel):
    """CSV export request"""
    output_path: str = Field(
        ...,
        description="Output file path for CSV export",
        example="/tmp/analytics_export.csv"
    )


class HealthResponse(BaseModel):
    """Analytics health response"""
    status: str
    enable_logging: bool
    total_queries: int
    total_errors: int
    tracked_documents: int


# Endpoints

@router.get("/health", response_model=HealthResponse)
async def get_analytics_health():
    """
    Get analytics service health status.

    Returns health information and basic statistics.
    """
    try:
        stats = analytics_service.get_stats()

        return HealthResponse(
            status="healthy",
            enable_logging=stats['enable_logging'],
            total_queries=stats['total_queries'],
            total_errors=stats['total_errors'],
            tracked_documents=stats['tracked_documents']
        )

    except Exception as e:
        logger.error(f"Error getting analytics health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-stats", response_model=QueryStatsResponse)
async def get_query_stats(
    start_time: Optional[str] = Query(
        None,
        description="Start time (ISO format)",
        example="2025-11-07T00:00:00"
    ),
    end_time: Optional[str] = Query(
        None,
        description="End time (ISO format)",
        example="2025-11-07T23:59:59"
    ),
    token: str = Depends(verify_token)
):
    """
    Get aggregated query statistics.

    Returns comprehensive statistics about query performance,
    usage patterns, and success rates.

    Requires authentication.
    """
    try:
        # Parse time range
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        # Get stats
        stats = analytics_service.get_query_stats(start_time=start_dt, end_time=end_dt)

        if not stats:
            raise HTTPException(status_code=404, detail="No statistics available")

        return QueryStatsResponse(**stats)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {e}")
    except Exception as e:
        logger.error(f"Error getting query stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document-popularity", response_model=List[DocumentPopularityItem])
async def get_document_popularity(
    top_n: int = Query(
        10,
        ge=1,
        le=100,
        description="Number of top documents to return"
    ),
    token: str = Depends(verify_token)
):
    """
    Get most popular documents.

    Returns documents ranked by access count and relevance.

    Requires authentication.
    """
    try:
        popularity = analytics_service.get_document_popularity(top_n=top_n)

        return [DocumentPopularityItem(**doc) for doc in popularity]

    except Exception as e:
        logger.error(f"Error getting document popularity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query-patterns", response_model=QueryPatternsResponse)
async def get_query_patterns(
    token: str = Depends(verify_token)
):
    """
    Get query pattern analysis.

    Returns insights about query lengths, common words,
    and temporal patterns.

    Requires authentication.
    """
    try:
        patterns = analytics_service.get_query_patterns()

        if not patterns:
            raise HTTPException(status_code=404, detail="No pattern data available")

        return QueryPatternsResponse(**patterns)

    except Exception as e:
        logger.error(f"Error analyzing query patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=Dict[str, Any])
async def export_analytics(
    request: ExportRequest,
    token: str = Depends(verify_token)
):
    """
    Export analytics data to CSV.

    Exports all query analytics to a CSV file.

    Requires authentication.
    """
    try:
        success = analytics_service.export_to_csv(request.output_path)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to export analytics data"
            )

        return {
            "status": "success",
            "output_path": request.output_path,
            "message": "Analytics exported successfully"
        }

    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=Dict[str, Any])
async def get_analytics_summary(
    token: str = Depends(verify_token)
):
    """
    Get comprehensive analytics summary.

    Returns all analytics data in one response:
    - Query statistics
    - Top documents
    - Query patterns

    Requires authentication.
    """
    try:
        # Get all analytics components
        stats = analytics_service.get_query_stats()
        popularity = analytics_service.get_document_popularity(top_n=10)
        patterns = analytics_service.get_query_patterns()

        return {
            "query_stats": stats,
            "top_documents": popularity,
            "query_patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear", response_model=Dict[str, str])
async def clear_analytics(
    token: str = Depends(verify_token)
):
    """
    Clear all analytics data.

    WARNING: This removes all tracked analytics.
    Use with caution.

    Requires authentication.
    """
    try:
        # Clear in-memory data
        analytics_service.queries.clear()
        analytics_service.document_stats.clear()
        analytics_service.total_queries = 0
        analytics_service.total_errors = 0
        analytics_service.avg_retrieval_time = 0.0
        analytics_service.avg_generation_time = 0.0

        logger.warning("Analytics data cleared")

        return {
            "status": "success",
            "message": "All analytics data has been cleared"
        }

    except Exception as e:
        logger.error(f"Error clearing analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
