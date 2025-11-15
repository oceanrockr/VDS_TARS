"""
T.A.R.S. Advanced Reranking Service
Cross-Encoder reranking for improved relevance scoring
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
import time
from typing import List, Tuple, Optional
import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder
import torch

from ..core.config import settings
from ..models.rag import SourceReference
from .redis_cache import redis_cache  # Phase 6

logger = logging.getLogger(__name__)


class AdvancedReranker:
    """
    Cross-encoder based reranking service.

    Uses Hugging Face cross-encoder models to compute semantic similarity
    between queries and retrieved documents for improved relevance ranking.

    Features:
    - MS MARCO fine-tuned cross-encoder
    - GPU acceleration support
    - Batch processing for efficiency
    - Configurable fusion weights
    - Model caching
    """

    def __init__(self):
        self.model_name = getattr(settings, 'RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.top_k = getattr(settings, 'RERANK_TOP_K', 10)
        self.rerank_weight = getattr(settings, 'RERANK_WEIGHT', 0.35)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model: Optional[CrossEncoder] = None
        self.is_loaded = False

        # Phase 6: Redis caching
        self.use_redis_cache = getattr(settings, 'REDIS_ENABLED', True)

        logger.info(f"AdvancedReranker initialized (device: {self.device}, caching: {self.use_redis_cache})")

    async def load_model(self) -> bool:
        """
        Load cross-encoder model.

        Returns:
            True if successful, False otherwise
        """
        if self.is_loaded:
            logger.info("Cross-encoder model already loaded")
            return True

        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            start_time = time.time()

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(self.model_name, device=self.device)
            )

            elapsed = (time.time() - start_time) * 1000
            self.is_loaded = True

            logger.info(
                f"Cross-encoder model loaded successfully "
                f"(device: {self.device}, time: {elapsed:.1f}ms)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            return False

    def _create_query_pairs(
        self,
        query: str,
        sources: List[SourceReference]
    ) -> List[Tuple[str, str]]:
        """
        Create query-document pairs for cross-encoder.

        Args:
            query: Search query
            sources: List of source references

        Returns:
            List of (query, document) tuples
        """
        pairs = []
        for source in sources:
            # Use excerpt for reranking (already extracted from chunk)
            pairs.append((query, source.excerpt))
        return pairs

    def _compute_cross_encoder_scores(
        self,
        query: str,
        sources: List[SourceReference]
    ) -> List[float]:
        """
        Compute cross-encoder similarity scores.

        Args:
            query: Search query
            sources: List of source references

        Returns:
            List of relevance scores
        """
        if not self.is_loaded or not self.model:
            logger.warning("Cross-encoder model not loaded, returning zeros")
            return [0.0] * len(sources)

        try:
            # Create query-document pairs
            pairs = self._create_query_pairs(query, sources)

            # Compute scores (returns numpy array)
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Convert to list and normalize to 0-1 range using sigmoid
            # (cross-encoder scores are raw logits)
            scores_list = [float(1 / (1 + torch.exp(-torch.tensor(s)))) for s in scores]

            return scores_list

        except Exception as e:
            logger.error(f"Error computing cross-encoder scores: {e}")
            return [0.0] * len(sources)

    def _fuse_scores(
        self,
        vector_scores: List[float],
        cross_encoder_scores: List[float],
        weight: float = None
    ) -> List[float]:
        """
        Fuse vector similarity and cross-encoder scores.

        Uses weighted linear combination:
        fused_score = (1 - weight) * vector_score + weight * cross_encoder_score

        Args:
            vector_scores: Original vector similarity scores
            cross_encoder_scores: Cross-encoder reranking scores
            weight: Weight for cross-encoder (0-1), defaults to self.rerank_weight

        Returns:
            List of fused scores
        """
        if weight is None:
            weight = self.rerank_weight

        fused = []
        for vec_score, ce_score in zip(vector_scores, cross_encoder_scores):
            fused_score = (1 - weight) * vec_score + weight * ce_score
            fused.append(fused_score)

        return fused

    async def rerank(
        self,
        query: str,
        sources: List[SourceReference],
        top_k: int = None,
        weight: float = None
    ) -> List[SourceReference]:
        """
        Rerank sources using cross-encoder.

        Args:
            query: Original query
            sources: List of source references from vector search
            top_k: Number of top results to rerank (default: RERANK_TOP_K)
            weight: Fusion weight for cross-encoder (default: RERANK_WEIGHT)

        Returns:
            Reranked and potentially filtered sources
        """
        if not sources:
            return sources

        if not self.is_loaded:
            logger.warning("Cross-encoder not loaded, performing simple rerank")
            return self._simple_rerank(query, sources)

        if top_k is None:
            top_k = self.top_k

        start_time = time.time()

        try:
            # Limit to top_k for efficiency (reranking is expensive)
            sources_to_rerank = sources[:top_k]

            # Store original vector scores
            vector_scores = [s.similarity_score for s in sources_to_rerank]

            # Phase 6: Try to get cached reranker scores
            document_ids = [s.chunk_id for s in sources_to_rerank]
            cross_encoder_scores = None

            if self.use_redis_cache:
                cross_encoder_scores = await redis_cache.get_reranker_scores(query, document_ids)
                if cross_encoder_scores:
                    logger.debug(f"Using cached reranker scores for {len(document_ids)} documents")

            # Compute cross-encoder scores if not cached
            if cross_encoder_scores is None:
                # Compute cross-encoder scores (run in thread pool)
                loop = asyncio.get_event_loop()
                cross_encoder_scores = await loop.run_in_executor(
                    None,
                    lambda: self._compute_cross_encoder_scores(query, sources_to_rerank)
                )

                # Cache the scores
                if self.use_redis_cache:
                    await redis_cache.set_reranker_scores(query, document_ids, cross_encoder_scores)

            # Fuse scores
            fused_scores = self._fuse_scores(vector_scores, cross_encoder_scores, weight)

            # Update similarity scores and metadata
            for source, fused_score, ce_score in zip(
                sources_to_rerank, fused_scores, cross_encoder_scores
            ):
                original_score = source.similarity_score
                source.similarity_score = round(fused_score, 4)

                # Store reranking metadata
                if not hasattr(source, 'metadata'):
                    source.metadata = {}
                source.metadata['original_score'] = round(original_score, 4)
                source.metadata['cross_encoder_score'] = round(ce_score, 4)
                source.metadata['reranked'] = True

            # Re-sort by fused score
            sources_to_rerank.sort(key=lambda x: x.similarity_score, reverse=True)

            # Combine reranked sources with remaining
            result = sources_to_rerank + sources[top_k:]

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Reranked {len(sources_to_rerank)} sources "
                f"(weight: {weight or self.rerank_weight}, time: {elapsed:.1f}ms)"
            )

            return result

        except Exception as e:
            logger.error(f"Error in advanced reranking: {e}")
            return sources

    def _simple_rerank(
        self,
        query: str,
        sources: List[SourceReference]
    ) -> List[SourceReference]:
        """
        Fallback simple keyword-based reranking.

        Args:
            query: Original query
            sources: List of source references

        Returns:
            Reranked sources
        """
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

        logger.debug(f"Simple reranked {len(sources)} sources")
        return sources

    async def health_check(self) -> bool:
        """
        Check if reranker is healthy.

        Returns:
            True if model is loaded and functional
        """
        return self.is_loaded and self.model is not None

    def get_stats(self) -> dict:
        """
        Get reranker statistics.

        Returns:
            Dictionary with reranker stats
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'top_k': self.top_k,
            'rerank_weight': self.rerank_weight,
            'gpu_available': torch.cuda.is_available()
        }


# Global service instance
advanced_reranker = AdvancedReranker()
