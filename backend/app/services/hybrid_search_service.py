"""
T.A.R.S. Hybrid Search Service
BM25 keyword search + vector similarity fusion
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import numpy as np

from ..core.config import settings
from ..models.rag import SourceReference, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search with component scores"""
    source: SourceReference
    vector_score: float
    bm25_score: float
    hybrid_score: float


class HybridSearchService:
    """
    Hybrid search combining BM25 keyword retrieval with vector similarity.

    Features:
    - BM25 (Okapi) for keyword matching
    - Vector similarity for semantic search
    - Configurable fusion weights
    - Score normalization
    - Reciprocal Rank Fusion (RRF) support
    """

    def __init__(self):
        self.alpha = getattr(settings, 'HYBRID_ALPHA', 0.3)  # Weight for BM25
        self.bm25_index: Optional[BM25Okapi] = None
        self.indexed_chunks: List[DocumentChunk] = []
        self.chunk_id_map: Dict[str, int] = {}

        logger.info(f"HybridSearchService initialized (alpha: {self.alpha})")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens (lowercase words)
        """
        # Simple whitespace + lowercase tokenization
        # Could be enhanced with stemming/lemmatization
        return text.lower().split()

    def build_bm25_index(self, chunks: List[DocumentChunk]) -> bool:
        """
        Build BM25 index from document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            True if successful
        """
        try:
            logger.info(f"Building BM25 index for {len(chunks)} chunks...")
            start_time = time.time()

            # Tokenize all chunks
            tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)

            # Store chunks and create ID map
            self.indexed_chunks = chunks
            self.chunk_id_map = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"BM25 index built successfully "
                f"({len(chunks)} documents, {elapsed:.1f}ms)"
            )

            return True

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            return False

    def add_chunks_to_index(self, new_chunks: List[DocumentChunk]) -> bool:
        """
        Add new chunks to existing BM25 index.

        Note: This rebuilds the entire index (BM25Okapi doesn't support incremental).

        Args:
            new_chunks: New chunks to add

        Returns:
            True if successful
        """
        try:
            # Combine with existing chunks
            all_chunks = self.indexed_chunks + new_chunks

            # Rebuild index
            return self.build_bm25_index(all_chunks)

        except Exception as e:
            logger.error(f"Error adding chunks to BM25 index: {e}")
            return False

    def _bm25_search(
        self,
        query: str,
        top_k: int = 100
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk, score) tuples
        """
        if not self.bm25_index or not self.indexed_chunks:
            logger.warning("BM25 index not built")
            return []

        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)

            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]

            # Return chunks with scores
            results = [
                (self.indexed_chunks[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0  # Filter zero scores
            ]

            return results

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max normalization.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        if not scores:
            return scores

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores equal
            return [1.0] * len(scores)

        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

        return normalized

    def _fusion_linear(
        self,
        vector_results: List[SourceReference],
        bm25_results: List[Tuple[DocumentChunk, float]],
        alpha: float = None
    ) -> List[HybridSearchResult]:
        """
        Fuse vector and BM25 results using weighted linear combination.

        Hybrid score = (1 - alpha) * vector_score + alpha * bm25_score

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for BM25 (default: self.alpha)

        Returns:
            List of hybrid search results
        """
        if alpha is None:
            alpha = self.alpha

        # Create lookup maps
        vector_map = {src.chunk_id: src for src in vector_results}
        bm25_map = {chunk.chunk_id: score for chunk, score in bm25_results}

        # Normalize vector scores (already 0-1 from cosine similarity)
        vector_scores = [src.similarity_score for src in vector_results]

        # Normalize BM25 scores
        bm25_scores = [score for _, score in bm25_results]
        normalized_bm25 = self._normalize_scores(bm25_scores)
        bm25_normalized_map = {
            chunk.chunk_id: norm_score
            for (chunk, _), norm_score in zip(bm25_results, normalized_bm25)
        }

        # Collect all unique chunk IDs
        all_chunk_ids = set(vector_map.keys()) | set(bm25_map.keys())

        # Compute hybrid scores
        hybrid_results = []

        for chunk_id in all_chunk_ids:
            # Get scores (0.0 if not present)
            vec_score = vector_map[chunk_id].similarity_score if chunk_id in vector_map else 0.0
            bm25_score = bm25_normalized_map.get(chunk_id, 0.0)

            # Compute hybrid score
            hybrid_score = (1 - alpha) * vec_score + alpha * bm25_score

            # Use vector result if available, else create from BM25
            if chunk_id in vector_map:
                source = vector_map[chunk_id]
            else:
                # Find chunk from BM25 results
                chunk = next((c for c, _ in bm25_results if c.chunk_id == chunk_id), None)
                if not chunk:
                    continue

                # Create SourceReference from chunk
                source = SourceReference(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.metadata.get('file_name', 'unknown'),
                    file_path=chunk.metadata.get('file_path', ''),
                    chunk_index=chunk.chunk_index,
                    similarity_score=hybrid_score,
                    excerpt=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    page_number=chunk.metadata.get('page_count')
                )

            # Create hybrid result
            hybrid_result = HybridSearchResult(
                source=source,
                vector_score=vec_score,
                bm25_score=bm25_score,
                hybrid_score=hybrid_score
            )

            hybrid_results.append(hybrid_result)

        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return hybrid_results

    def _fusion_rrf(
        self,
        vector_results: List[SourceReference],
        bm25_results: List[Tuple[DocumentChunk, float]],
        k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each ranking

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default: 60)

        Returns:
            List of hybrid search results
        """
        # Create rank maps
        vector_ranks = {src.chunk_id: i + 1 for i, src in enumerate(vector_results)}
        bm25_ranks = {chunk.chunk_id: i + 1 for i, (chunk, _) in enumerate(bm25_results)}

        # Create lookup maps for scores
        vector_map = {src.chunk_id: src for src in vector_results}
        bm25_map = {chunk.chunk_id: score for chunk, score in bm25_results}

        # Collect all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Compute RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0

            if chunk_id in vector_ranks:
                rrf_score += 1.0 / (k + vector_ranks[chunk_id])

            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk_id])

            rrf_scores[chunk_id] = rrf_score

        # Create hybrid results
        hybrid_results = []

        for chunk_id, rrf_score in rrf_scores.items():
            vec_score = vector_map[chunk_id].similarity_score if chunk_id in vector_map else 0.0
            bm25_score = bm25_map.get(chunk_id, 0.0)

            # Use vector result if available
            if chunk_id in vector_map:
                source = vector_map[chunk_id]
            else:
                chunk = next((c for c, _ in bm25_results if c.chunk_id == chunk_id), None)
                if not chunk:
                    continue

                source = SourceReference(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.metadata.get('file_name', 'unknown'),
                    file_path=chunk.metadata.get('file_path', ''),
                    chunk_index=chunk.chunk_index,
                    similarity_score=rrf_score,
                    excerpt=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    page_number=chunk.metadata.get('page_count')
                )

            hybrid_result = HybridSearchResult(
                source=source,
                vector_score=vec_score,
                bm25_score=bm25_score,
                hybrid_score=rrf_score
            )

            hybrid_results.append(hybrid_result)

        # Sort by RRF score
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return hybrid_results

    async def search(
        self,
        query: str,
        vector_results: List[SourceReference],
        top_k: int = 10,
        fusion_method: str = 'linear',
        alpha: float = None
    ) -> List[SourceReference]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            vector_results: Results from vector search
            top_k: Number of final results
            fusion_method: 'linear' or 'rrf'
            alpha: Weight for BM25 in linear fusion

        Returns:
            List of SourceReference objects with hybrid scores
        """
        start_time = time.time()

        try:
            # Perform BM25 search
            bm25_results = self._bm25_search(query, top_k=100)

            if not bm25_results:
                logger.warning("No BM25 results, returning vector results only")
                return vector_results[:top_k]

            # Fuse results
            if fusion_method == 'rrf':
                hybrid_results = self._fusion_rrf(vector_results, bm25_results)
            else:  # linear (default)
                hybrid_results = self._fusion_linear(vector_results, bm25_results, alpha)

            # Update source similarity scores with hybrid scores
            for result in hybrid_results:
                result.source.similarity_score = round(result.hybrid_score, 4)

                # Add hybrid metadata
                if not hasattr(result.source, 'metadata'):
                    result.source.metadata = {}

                result.source.metadata['hybrid_search'] = True
                result.source.metadata['vector_score'] = round(result.vector_score, 4)
                result.source.metadata['bm25_score'] = round(result.bm25_score, 4)
                result.source.metadata['fusion_method'] = fusion_method

            # Extract sources and limit to top_k
            sources = [r.source for r in hybrid_results[:top_k]]

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Hybrid search completed: {len(sources)} results "
                f"(method: {fusion_method}, alpha: {alpha or self.alpha}, "
                f"time: {elapsed:.1f}ms)"
            )

            return sources

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return vector_results[:top_k]

    def get_stats(self) -> dict:
        """
        Get hybrid search statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'alpha': self.alpha,
            'indexed_chunks': len(self.indexed_chunks),
            'has_bm25_index': self.bm25_index is not None
        }


# Global service instance
hybrid_search_service = HybridSearchService()
