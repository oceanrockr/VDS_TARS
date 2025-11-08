"""
T.A.R.S. Embedding Service
Generates vector embeddings using sentence-transformers
Phase 3 - Document Indexing & RAG
"""

import logging
import time
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer

from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.

    Features:
    - GPU acceleration (if available)
    - Batch processing
    - Progress tracking
    - Model caching
    """

    def __init__(self):
        self.model_name = settings.EMBED_MODEL
        self.device = self._get_device()
        self.batch_size = settings.EMBED_BATCH_SIZE
        self.max_seq_length = settings.EMBED_MAX_SEQ_LENGTH
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dimension = settings.EMBED_DIMENSION

        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        logger.info(f"Device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device (CUDA, CPU)"""
        if settings.EMBED_DEVICE == "cuda" and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            if settings.EMBED_DEVICE == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
        return device

    async def load_model(self) -> bool:
        """
        Load the embedding model.
        Returns True if successful, False otherwise.
        """
        try:
            start_time = time.time()
            logger.info(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_seq_length

            # Get actual embedding dimension from model
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

            load_time = time.time() - start_time
            logger.info(
                f"Model loaded successfully in {load_time:.2f}s "
                f"(dimension: {self.embedding_dimension})"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the embedding service is healthy.

        Returns:
            Dictionary with health status
        """
        if self.model is None:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "model_name": self.model_name,
                "device": self.device,
            }

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
        }

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            await self.load_model()

        try:
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            await self.load_model()

        if not texts:
            return []

        try:
            start_time = time.time()
            logger.info(f"Generating embeddings for {len(texts)} texts")

            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )

            elapsed = time.time() - start_time
            texts_per_sec = len(texts) / elapsed if elapsed > 0 else 0

            logger.info(
                f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
                f"({texts_per_sec:.1f} texts/sec)"
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses the same method as embed_text but kept separate
        for potential future query-specific optimizations.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return await self.embed_text(query)

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            tensor1 = torch.tensor(embedding1)
            tensor2 = torch.tensor(embedding2)

            similarity = torch.nn.functional.cosine_similarity(
                tensor1.unsqueeze(0),
                tensor2.unsqueeze(0)
            )

            return float(similarity.item())

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    async def close(self):
        """Cleanup resources"""
        if self.model is not None:
            logger.info("Unloading embedding model")
            del self.model
            self.model = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

        logger.info("EmbeddingService closed")


# Global service instance
embedding_service = EmbeddingService()
