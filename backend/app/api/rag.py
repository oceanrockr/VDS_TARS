"""
T.A.R.S. RAG API Router
REST endpoints for document indexing and RAG queries
Phase 3 - Document Indexing & RAG
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from ..models.rag import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    BatchIndexRequest,
    BatchIndexResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    RAGHealthResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    CollectionStats
)
from ..services.rag_service import rag_service
from ..services.chromadb_service import chromadb_service
from ..services.embedding_service import embedding_service
from ..core.middleware import verify_token
import asyncio
import time
import json

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    responses={404: {"description": "Not found"}},
)


# ==============================================================================
# HEALTH & STATUS ENDPOINTS
# ==============================================================================

@router.get(
    "/health",
    response_model=RAGHealthResponse,
    summary="RAG Service Health Check",
    description="Get health status of RAG service components"
)
async def rag_health() -> RAGHealthResponse:
    """
    Check health of all RAG components:
    - ChromaDB connection
    - Embedding model
    - Collection statistics
    """
    try:
        # Check ChromaDB
        chroma_health = chromadb_service.health_check()
        chroma_status = chroma_health.get('status', 'unknown')

        # Check embedding model
        embed_health = embedding_service.health_check()
        embed_status = embed_health.get('status', 'unknown')

        # Get collection stats
        stats = await chromadb_service.get_stats()
        collections = {
            stats.collection_name: stats
        }

        # Overall status
        overall_status = "healthy" if (
            chroma_status == "healthy" and embed_status == "healthy"
        ) else "degraded"

        return RAGHealthResponse(
            status=overall_status,
            chromadb_status=chroma_status,
            embedding_model_status=embed_status,
            collections=collections,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=CollectionStats,
    summary="Collection Statistics",
    description="Get statistics for the document collection"
)
async def get_stats() -> CollectionStats:
    """Get collection statistics"""
    try:
        return await chromadb_service.get_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==============================================================================
# DOCUMENT INDEXING ENDPOINTS
# ==============================================================================

@router.post(
    "/index",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index Document",
    description="Index a single document for RAG retrieval",
    dependencies=[Depends(verify_token)]
)
async def index_document(
    request: DocumentUploadRequest
) -> DocumentUploadResponse:
    """
    Index a document by file path.

    The document will be:
    1. Loaded and validated
    2. Chunked into smaller pieces
    3. Embedded using sentence-transformers
    4. Stored in ChromaDB
    """
    try:
        return await rag_service.index_document(request)
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/index/batch",
    response_model=BatchIndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch Index Documents",
    description="Index multiple documents concurrently",
    dependencies=[Depends(verify_token)]
)
async def batch_index_documents(
    request: BatchIndexRequest
) -> BatchIndexResponse:
    """
    Index multiple documents in parallel.

    Each document is processed independently with configurable concurrency.
    """
    start_time = time.time()

    try:
        total_files = len(request.file_paths)
        successful = 0
        failed = 0
        already_indexed = 0
        total_chunks = 0
        results = []

        # Process in batches with controlled concurrency
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def index_with_semaphore(file_path: str):
            async with semaphore:
                upload_req = DocumentUploadRequest(
                    file_path=file_path,
                    force_reindex=request.force_reindex
                )
                return await rag_service.index_document(upload_req)

        # Execute all indexing tasks
        tasks = [
            index_with_semaphore(file_path)
            for file_path in request.file_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Indexing error: {result}")
            elif result.status == "success":
                successful += 1
                total_chunks += result.chunks_created
            elif result.status == "already_indexed":
                already_indexed += 1
            else:
                failed += 1

        elapsed = (time.time() - start_time) * 1000

        return BatchIndexResponse(
            total_files=total_files,
            successful=successful,
            failed=failed,
            already_indexed=already_indexed,
            total_chunks=total_chunks,
            processing_time_ms=round(elapsed, 2),
            results=[r for r in results if not isinstance(r, Exception)]
        )

    except Exception as e:
        logger.error(f"Error in batch indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/document",
    response_model=DocumentDeleteResponse,
    summary="Delete Document",
    description="Delete a document and all its chunks from the index",
    dependencies=[Depends(verify_token)]
)
async def delete_document(
    request: DocumentDeleteRequest
) -> DocumentDeleteResponse:
    """
    Delete all chunks for a document by document ID.
    """
    try:
        chunks_deleted = await chromadb_service.delete_document(
            request.document_id
        )

        return DocumentDeleteResponse(
            document_id=request.document_id,
            status="success" if chunks_deleted > 0 else "not_found",
            chunks_deleted=chunks_deleted,
            message=f"Deleted {chunks_deleted} chunks"
        )

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==============================================================================
# RAG QUERY ENDPOINTS
# ==============================================================================

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="RAG Query",
    description="Execute a RAG query with retrieval and generation"
)
async def rag_query(
    request: RAGQueryRequest
) -> RAGQueryResponse:
    """
    Execute a RAG query:
    1. Retrieve relevant document chunks
    2. Rerank results (if enabled)
    3. Build context from chunks
    4. Generate answer using Ollama
    5. Return answer with citations
    """
    try:
        return await rag_service.query(request)
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/query/stream",
    summary="RAG Query (Streaming)",
    description="Execute a RAG query with streaming response"
)
async def rag_query_stream(
    request: RAGQueryRequest
):
    """
    Execute a RAG query with streaming response.

    Returns:
    - rag_sources: List of source citations
    - rag_token: Individual response tokens
    - rag_complete: Final statistics
    """
    async def generate():
        try:
            async for message in rag_service.query_stream(request):
                # Send as JSON lines
                yield json.dumps(message) + "\n"
        except Exception as e:
            logger.error(f"Error in RAG stream: {e}")
            error_msg = {
                "type": "error",
                "error": str(e)
            }
            yield json.dumps(error_msg) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@router.post(
    "/search",
    response_model=DocumentSearchResponse,
    summary="Search Documents",
    description="Search indexed documents without generation"
)
async def search_documents(
    request: DocumentSearchRequest
) -> DocumentSearchResponse:
    """
    Search indexed documents for relevant chunks.
    Does not generate an answer, only returns matching chunks.
    """
    try:
        start_time = time.time()

        # Retrieve chunks
        sources = await rag_service.retrieve_context(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        # Convert SourceReferences back to DocumentChunks
        # (This is a simplified conversion - in production might want full chunk data)
        results = []
        for source in sources:
            # Note: This is a minimal reconstruction
            # In production, you'd fetch full chunk data from ChromaDB
            chunk_data = await chromadb_service.get_chunk(source.chunk_id)
            if chunk_data:
                # Build simplified DocumentChunk for response
                results.append({
                    "chunk_id": source.chunk_id,
                    "document_id": source.document_id,
                    "content": source.excerpt,
                    "similarity_score": source.similarity_score,
                    "metadata": chunk_data['metadata']
                })

        elapsed = (time.time() - start_time) * 1000

        return DocumentSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=round(elapsed, 2)
        )

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==============================================================================
# UTILITY ENDPOINTS
# ==============================================================================

@router.post(
    "/initialize",
    summary="Initialize RAG Service",
    description="Initialize all RAG components (ChromaDB, Embeddings, etc.)",
    dependencies=[Depends(verify_token)]
)
async def initialize_rag() -> Dict[str, Any]:
    """
    Initialize RAG service components.
    Useful for manual initialization or recovery.
    """
    try:
        success = await rag_service.initialize()

        if success:
            return {
                "status": "success",
                "message": "RAG service initialized successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service initialization failed"
            )

    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
