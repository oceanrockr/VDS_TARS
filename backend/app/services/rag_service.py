"""
T.A.R.S. RAG Service
Retrieval-Augmented Generation orchestration
Phase 3 - Document Indexing & RAG
"""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import uuid

from ..core.config import settings
from ..models.rag import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SourceReference,
    DocumentChunk,
    RAGStreamToken,
    RAGStreamSources,
    RAGStreamComplete
)
from .document_loader import document_loader
from .embedding_service import embedding_service
from .chromadb_service import chromadb_service
from .ollama_service import ollama_service
from .advanced_reranker import advanced_reranker  # Phase 5
from .semantic_chunker import semantic_chunker  # Phase 5
from .hybrid_search_service import hybrid_search_service  # Phase 5
from .query_expansion import query_expansion_service  # Phase 5
from .analytics_service import analytics_service  # Phase 5
from .redis_cache import redis_cache  # Phase 6

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for Retrieval-Augmented Generation.

    Orchestrates:
    - Document ingestion and chunking
    - Embedding generation
    - Vector storage
    - Similarity search
    - Context reranking
    - LLM generation with citations
    """

    def __init__(self):
        self.top_k = settings.RAG_TOP_K
        self.relevance_threshold = settings.RAG_RELEVANCE_THRESHOLD
        self.rerank_enabled = settings.RAG_RERANK_ENABLED
        self.include_sources = settings.RAG_INCLUDE_SOURCES
        self.max_context_tokens = settings.RAG_MAX_CONTEXT_TOKENS

        # Phase 5: Advanced RAG settings
        self.use_semantic_chunking = getattr(settings, 'USE_SEMANTIC_CHUNKING', True)
        self.use_advanced_reranking = getattr(settings, 'USE_ADVANCED_RERANKING', True)
        self.use_hybrid_search = getattr(settings, 'USE_HYBRID_SEARCH', True)
        self.use_query_expansion = getattr(settings, 'USE_QUERY_EXPANSION', False)

        # Phase 6: Redis caching
        self.use_redis_cache = getattr(settings, 'REDIS_ENABLED', True)

        logger.info("RAGService initialized (Phase 5 enhancements + Phase 6 caching enabled)")

    async def initialize(self) -> bool:
        """
        Initialize all RAG components.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing RAG components...")

            # Connect to ChromaDB
            chroma_ok = await chromadb_service.connect()
            if not chroma_ok:
                logger.error("Failed to connect to ChromaDB")
                return False

            # Load embedding model
            embed_ok = await embedding_service.load_model()
            if not embed_ok:
                logger.error("Failed to load embedding model")
                return False

            # Check Ollama
            ollama_ok = await ollama_service.health_check()
            if not ollama_ok:
                logger.warning("Ollama service not available")

            # Phase 5: Initialize advanced components
            if self.use_advanced_reranking:
                reranker_ok = await advanced_reranker.load_model()
                if not reranker_ok:
                    logger.warning("Advanced reranker not available, using fallback")

            if self.use_semantic_chunking:
                # Semantic chunker is always available, optionally load embedding model
                await semantic_chunker.load_embedding_model()

            # Phase 6: Connect to Redis cache
            if self.use_redis_cache:
                cache_ok = await redis_cache.connect()
                if not cache_ok:
                    logger.warning("Redis cache not available, caching disabled")
                    self.use_redis_cache = False

            logger.info("RAG components initialized successfully (with Phase 5 enhancements + Phase 6 caching)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return False

    async def index_document(
        self,
        request: DocumentUploadRequest
    ) -> DocumentUploadResponse:
        """
        Index a single document.

        Args:
            request: Document upload request

        Returns:
            Upload response with indexing results
        """
        start_time = time.time()

        try:
            logger.info(f"Indexing document: {request.file_path}")

            # Load document
            document = document_loader.load_document(request.file_path)

            # Check if already indexed (unless force_reindex)
            if not request.force_reindex:
                existing = await chromadb_service.get_chunk(
                    f"{document.document_id}_chunk_0"
                )
                if existing:
                    logger.info(f"Document already indexed: {document.document_id}")
                    elapsed = (time.time() - start_time) * 1000
                    return DocumentUploadResponse(
                        document_id=document.document_id,
                        file_name=document.metadata.file_name,
                        status="already_indexed",
                        chunks_created=0,
                        processing_time_ms=elapsed,
                        message="Document already indexed (use force_reindex=true to reindex)"
                    )

            # Create chunks (Phase 5: use semantic chunking if enabled)
            if self.use_semantic_chunking:
                chunks = semantic_chunker.create_chunks(document)
                logger.debug(f"Using semantic chunking: {len(chunks)} chunks created")
            else:
                chunks = document_loader.create_chunks(document)
                logger.debug(f"Using standard chunking: {len(chunks)} chunks created")

            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.embed_batch(chunk_texts)

            # Store in ChromaDB
            chunks_added = await chromadb_service.add_chunks(chunks, embeddings)

            # Phase 5: Add to BM25 index if hybrid search is enabled
            if self.use_hybrid_search:
                hybrid_search_service.add_chunks_to_index(chunks)
                logger.debug(f"Added {len(chunks)} chunks to BM25 index")

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Indexed document: {document.metadata.file_name} "
                f"({chunks_added} chunks, {elapsed:.1f}ms)"
            )

            return DocumentUploadResponse(
                document_id=document.document_id,
                file_name=document.metadata.file_name,
                status="success",
                chunks_created=chunks_added,
                processing_time_ms=elapsed
            )

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            elapsed = (time.time() - start_time) * 1000
            return DocumentUploadResponse(
                document_id="",
                file_name=request.file_path.split("/")[-1],
                status="failed",
                chunks_created=0,
                processing_time_ms=elapsed,
                message=str(e)
            )

    async def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        filters: Optional[Dict[str, Any]] = None,
        use_expansion: bool = None,
        use_hybrid: bool = None,
        use_advanced_rerank: bool = None
    ) -> List[SourceReference]:
        """
        Retrieve relevant document chunks for a query.
        Phase 5: Enhanced with query expansion, hybrid search, and advanced reranking.

        Args:
            query: Search query
            top_k: Number of results to retrieve
            threshold: Minimum relevance score
            filters: Optional metadata filters
            use_expansion: Override query expansion setting
            use_hybrid: Override hybrid search setting
            use_advanced_rerank: Override advanced reranking setting

        Returns:
            List of SourceReference objects
        """
        if top_k is None:
            top_k = self.top_k
        if threshold is None:
            threshold = self.relevance_threshold
        if use_expansion is None:
            use_expansion = self.use_query_expansion
        if use_hybrid is None:
            use_hybrid = self.use_hybrid_search
        if use_advanced_rerank is None:
            use_advanced_rerank = self.use_advanced_reranking

        try:
            start_time = time.time()
            queries_to_search = [query]

            # Phase 5: Query expansion
            if use_expansion:
                expanded_queries = await query_expansion_service.expand_query(
                    query, strategy='general', include_original=True
                )
                queries_to_search = expanded_queries
                logger.debug(f"Expanded query to {len(queries_to_search)} variants")

            # Retrieve from all query variants
            all_sources = []
            for q in queries_to_search:
                # Phase 6: Try to get embedding from cache
                query_embedding = None
                if self.use_redis_cache:
                    query_embedding = await redis_cache.get_embedding(q)
                    if query_embedding:
                        logger.debug(f"Using cached embedding for query: {q[:50]}...")

                # Generate query embedding if not cached
                if query_embedding is None:
                    query_embedding = await embedding_service.embed_query(q)

                    # Cache the embedding
                    if self.use_redis_cache:
                        await redis_cache.set_embedding(q, query_embedding)

                # Search ChromaDB (get more results for reranking)
                search_k = top_k * 3 if use_advanced_rerank else top_k
                results = await chromadb_service.query(
                    query_embedding=query_embedding,
                    top_k=search_k,
                    filters=filters
                )

                # Create SourceReferences
                for result in results:
                    similarity = result['similarity_score']

                    if similarity >= threshold:
                        metadata = result['metadata']
                        content = result['content']

                        # Create excerpt (first 200 chars)
                        excerpt = content[:200] + "..." if len(content) > 200 else content

                        source = SourceReference(
                            document_id=metadata['document_id'],
                            chunk_id=result['chunk_id'],
                            file_name=metadata['file_name'],
                            file_path=metadata['file_path'],
                            chunk_index=metadata['chunk_index'],
                            similarity_score=round(similarity, 4),
                            excerpt=excerpt,
                            page_number=metadata.get('page_count')
                        )
                        all_sources.append(source)

            # Deduplicate by chunk_id (keep highest score)
            sources_map = {}
            for source in all_sources:
                if source.chunk_id not in sources_map or \
                   source.similarity_score > sources_map[source.chunk_id].similarity_score:
                    sources_map[source.chunk_id] = source

            sources = list(sources_map.values())
            sources.sort(key=lambda x: x.similarity_score, reverse=True)

            # Phase 5: Hybrid search
            if use_hybrid:
                sources = await hybrid_search_service.search(
                    query=query,
                    vector_results=sources,
                    top_k=top_k * 2,  # Get more for reranking
                    fusion_method='linear'
                )
                logger.debug(f"Applied hybrid search: {len(sources)} results")

            # Phase 5: Advanced reranking
            if use_advanced_rerank and len(sources) > 1:
                sources = await advanced_reranker.rerank(
                    query=query,
                    sources=sources,
                    top_k=min(10, len(sources))
                )
                logger.debug(f"Applied advanced reranking: {len(sources)} results")

            # Final limit to top_k
            sources = sources[:top_k]

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Retrieved {len(sources)} relevant chunks "
                f"(expansion: {use_expansion}, hybrid: {use_hybrid}, "
                f"rerank: {use_advanced_rerank}, time: {elapsed:.1f}ms)"
            )

            return sources

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def rerank_sources(
        self,
        query: str,
        sources: List[SourceReference]
    ) -> List[SourceReference]:
        """
        Rerank sources using simple scoring.

        Args:
            query: Original query
            sources: List of source references

        Returns:
            Reranked sources
        """
        if not self.rerank_enabled or not sources:
            return sources

        # Simple keyword-based boost
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for source in sources:
            excerpt_lower = source.excerpt.lower()
            excerpt_words = set(excerpt_lower.split())

            # Keyword overlap boost
            overlap = len(query_words.intersection(excerpt_words))
            boost = 1.0 + (overlap * 0.05)  # 5% boost per matching word

            # Apply boost
            source.similarity_score *= boost

        # Re-sort by boosted score
        sources.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.debug(f"Reranked {len(sources)} sources")
        return sources

    def build_context(
        self,
        sources: List[SourceReference],
        max_tokens: int = None
    ) -> str:
        """
        Build context string from sources.

        Args:
            sources: List of source references
            max_tokens: Maximum token count for context

        Returns:
            Combined context string
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens

        context_parts = []
        current_tokens = 0

        for i, source in enumerate(sources, 1):
            # Estimate tokens (words * 1.3)
            source_tokens = int(len(source.excerpt.split()) * 1.3)

            if current_tokens + source_tokens > max_tokens:
                logger.debug(
                    f"Context limit reached at source {i} "
                    f"({current_tokens}/{max_tokens} tokens)"
                )
                break

            context_parts.append(
                f"[Source {i}: {source.file_name}]\n{source.excerpt}\n"
            )
            current_tokens += source_tokens

        context = "\n".join(context_parts)
        logger.debug(f"Built context with {current_tokens} tokens from {len(context_parts)} sources")

        return context

    async def query(
        self,
        request: RAGQueryRequest
    ) -> RAGQueryResponse:
        """
        Execute RAG query with retrieval and generation.

        Args:
            request: RAG query request

        Returns:
            RAG query response with answer and sources
        """
        total_start = time.time()

        try:
            # 1. Retrieve relevant chunks
            retrieval_start = time.time()
            sources = await self.retrieve_context(
                query=request.query,
                top_k=request.top_k,
                threshold=request.relevance_threshold,
                filters=request.filters
            )
            retrieval_time = (time.time() - retrieval_start) * 1000

            # 2. Rerank if enabled
            if request.rerank:
                sources = self.rerank_sources(request.query, sources)

            # 3. Build context
            context = self.build_context(sources)

            # 4. Generate answer with Ollama
            generation_start = time.time()

            # Build prompt with context
            system_prompt = (
                "You are T.A.R.S., a helpful AI assistant. "
                "Use the provided context to answer the question accurately. "
                "If the context doesn't contain relevant information, say so."
            )

            prompt = f"""Context:
{context}

Question: {request.query}

Answer:"""

            # Collect streaming response
            answer_tokens = []
            total_tokens = 0

            async for chunk_data in ollama_service.generate_stream(
                prompt=prompt,
                model=settings.OLLAMA_MODEL,
                temperature=settings.MODEL_TEMPERATURE,
                max_tokens=settings.MODEL_MAX_TOKENS,
                system_prompt=system_prompt
            ):
                if not chunk_data.get('done', False):
                    token = chunk_data.get('token', '')
                    answer_tokens.append(token)
                    total_tokens += 1

            answer = ''.join(answer_tokens)
            generation_time = (time.time() - generation_start) * 1000

            total_time = (time.time() - total_start) * 1000

            # Build response
            response = RAGQueryResponse(
                query=request.query,
                answer=answer,
                sources=sources if request.include_sources else [],
                context_used=context,
                total_tokens=total_tokens,
                retrieval_time_ms=round(retrieval_time, 2),
                generation_time_ms=round(generation_time, 2),
                total_time_ms=round(total_time, 2),
                model=settings.OLLAMA_MODEL,
                relevance_scores=[s.similarity_score for s in sources]
            )

            logger.info(
                f"RAG query completed: {total_tokens} tokens, "
                f"{len(sources)} sources, {total_time:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    async def query_stream(
        self,
        request: RAGQueryRequest,
        conversation_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute RAG query with streaming response.

        Args:
            request: RAG query request
            conversation_id: Optional conversation ID

        Yields:
            Stream messages (tokens, sources, complete)
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        total_start = time.time()

        try:
            # 1. Retrieve relevant chunks
            retrieval_start = time.time()
            sources = await self.retrieve_context(
                query=request.query,
                top_k=request.top_k,
                threshold=request.relevance_threshold,
                filters=request.filters
            )
            retrieval_time = (time.time() - retrieval_start) * 1000

            # 2. Rerank if enabled
            if request.rerank:
                sources = self.rerank_sources(request.query, sources)

            # 3. Send sources first
            if request.include_sources and sources:
                yield RAGStreamSources(
                    conversation_id=conversation_id,
                    sources=sources
                ).dict()

            # 4. Build context and generate
            context = self.build_context(sources)

            system_prompt = (
                "You are T.A.R.S., a helpful AI assistant. "
                "Use the provided context to answer the question accurately. "
                "If the context doesn't contain relevant information, say so."
            )

            prompt = f"""Context:
{context}

Question: {request.query}

Answer:"""

            # 5. Stream generation
            generation_start = time.time()
            total_tokens = 0

            async for chunk_data in ollama_service.generate_stream(
                prompt=prompt,
                model=settings.OLLAMA_MODEL,
                temperature=settings.MODEL_TEMPERATURE,
                max_tokens=settings.MODEL_MAX_TOKENS,
                system_prompt=system_prompt
            ):
                if not chunk_data.get('done', False):
                    token = chunk_data.get('token', '')
                    total_tokens += 1

                    yield RAGStreamToken(
                        token=token,
                        conversation_id=conversation_id,
                        has_sources=len(sources) > 0
                    ).dict()

            generation_time = (time.time() - generation_start) * 1000
            total_time = (time.time() - total_start) * 1000

            # Phase 5: Log analytics
            try:
                await analytics_service.log_query(
                    query_text=request.query,
                    client_id=getattr(request, 'client_id', 'anonymous'),
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=generation_time,
                    total_time_ms=total_time,
                    sources_count=len(sources),
                    relevance_scores=[s.similarity_score for s in sources],
                    model_used=settings.OLLAMA_MODEL,
                    tokens_generated=total_tokens,
                    used_reranking=self.use_advanced_reranking,
                    used_hybrid_search=self.use_hybrid_search,
                    used_query_expansion=self.use_query_expansion,
                    expansion_count=len(queries_to_search) if hasattr(self, 'queries_to_search') else 0,
                    success=True
                )

                # Log document accesses
                for source in sources:
                    await analytics_service.log_document_access(
                        document_id=source.document_id,
                        file_name=source.file_name,
                        relevance_score=source.similarity_score
                    )
            except Exception as analytics_error:
                logger.warning(f"Failed to log analytics: {analytics_error}")

            # 6. Send completion
            yield RAGStreamComplete(
                conversation_id=conversation_id,
                total_tokens=total_tokens,
                retrieval_time_ms=round(retrieval_time, 2),
                generation_time_ms=round(generation_time, 2),
                total_time_ms=round(total_time, 2),
                sources_count=len(sources)
            ).dict()

        except Exception as e:
            logger.error(f"Error in RAG query stream: {e}")
            # Phase 5: Log error in analytics
            try:
                await analytics_service.log_query(
                    query_text=request.query,
                    client_id=getattr(request, 'client_id', 'anonymous'),
                    retrieval_time_ms=0,
                    generation_time_ms=0,
                    total_time_ms=0,
                    sources_count=0,
                    relevance_scores=[],
                    model_used=settings.OLLAMA_MODEL,
                    tokens_generated=0,
                    used_reranking=False,
                    used_hybrid_search=False,
                    used_query_expansion=False,
                    expansion_count=0,
                    success=False,
                    error_message=str(e)
                )
            except:
                pass
            raise


# Global service instance
rag_service = RAGService()
