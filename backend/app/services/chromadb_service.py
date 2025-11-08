"""
T.A.R.S. ChromaDB Service
Vector database operations for document storage and retrieval
Phase 3 - Document Indexing & RAG
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from ..core.config import settings
from ..models.rag import DocumentChunk, CollectionStats

logger = logging.getLogger(__name__)


class ChromaDBService:
    """
    Service for interacting with ChromaDB vector database.

    Features:
    - Persistent collections
    - Batch operations
    - Similarity search
    - Metadata filtering
    - Collection management
    """

    def __init__(self):
        self.host = settings.CHROMA_HOST
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self.conversation_collection = settings.CHROMA_CONVERSATION_COLLECTION
        self.distance_metric = settings.CHROMA_DISTANCE_METRIC
        self.max_batch_size = settings.CHROMA_MAX_BATCH_SIZE

        self.client: Optional[chromadb.HttpClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.conversation_coll: Optional[chromadb.Collection] = None

        logger.info(f"Initializing ChromaDBService at {self.host}")

    async def connect(self) -> bool:
        """
        Connect to ChromaDB server.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Connecting to ChromaDB at {self.host}")

            # Parse host URL
            if self.host.startswith("http://"):
                host_parts = self.host.replace("http://", "").split(":")
            elif self.host.startswith("https://"):
                host_parts = self.host.replace("https://", "").split(":")
            else:
                host_parts = self.host.split(":")

            chroma_host = host_parts[0]
            chroma_port = int(host_parts[1]) if len(host_parts) > 1 else 8000

            # Create HTTP client
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Test connection
            heartbeat = self.client.heartbeat()
            logger.info(f"ChromaDB heartbeat: {heartbeat}")

            # Get or create collections
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "T.A.R.S. document embeddings"}
            )

            self.conversation_coll = self.client.get_or_create_collection(
                name=self.conversation_collection,
                metadata={"description": "T.A.R.S. conversation history"}
            )

            doc_count = self.collection.count()
            conv_count = self.conversation_coll.count()

            logger.info(
                f"Connected to ChromaDB - "
                f"Documents: {doc_count}, Conversations: {conv_count}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check ChromaDB service health.

        Returns:
            Health status dictionary
        """
        if self.client is None:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Client not initialized"
            }

        try:
            heartbeat = self.client.heartbeat()
            doc_count = self.collection.count() if self.collection else 0
            conv_count = self.conversation_coll.count() if self.conversation_coll else 0

            return {
                "status": "healthy",
                "connected": True,
                "heartbeat": heartbeat,
                "collections": {
                    self.collection_name: doc_count,
                    self.conversation_collection: conv_count
                },
                "host": self.host
            }

        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add document chunks with embeddings to the collection.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors

        Returns:
            Number of chunks added
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: "
                f"{len(chunks)} vs {len(embeddings)}"
            )

        try:
            logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
            start_time = time.time()

            # Prepare data for batch insert
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "file_name": chunk.metadata.file_name,
                    "file_path": chunk.metadata.file_path,
                    "file_type": chunk.metadata.file_type,
                    "file_size": chunk.metadata.file_size,
                    "indexed_at": chunk.metadata.indexed_at.isoformat(),
                    "token_count": chunk.token_count,
                    "page_count": chunk.metadata.page_count or 0,
                }
                for chunk in chunks
            ]

            # Add in batches
            total_added = 0
            for i in range(0, len(chunks), self.max_batch_size):
                batch_end = min(i + self.max_batch_size, len(chunks))

                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

                total_added += (batch_end - i)
                logger.debug(f"Added batch {i//self.max_batch_size + 1}")

            elapsed = time.time() - start_time
            chunks_per_sec = total_added / elapsed if elapsed > 0 else 0

            logger.info(
                f"Added {total_added} chunks in {elapsed:.2f}s "
                f"({chunks_per_sec:.1f} chunks/sec)"
            )

            return total_added

        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the collection for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of result dictionaries with chunks and scores
        """
        try:
            start_time = time.time()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "chunk_id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        # Convert distance to similarity (cosine)
                        "similarity_score": 1 - results['distances'][0][i]
                    })

            logger.info(
                f"Query returned {len(formatted_results)} results in {elapsed:.1f}ms"
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise

    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if result['ids']:
                return {
                    "chunk_id": result['ids'][0],
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0],
                    "embedding": result['embeddings'][0] if result['embeddings'] else None
                }

            return None

        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id}: {e}")
            return None

    async def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of chunks deleted
        """
        try:
            logger.info(f"Deleting document: {document_id}")

            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )

            chunk_count = len(results['ids'])

            if chunk_count > 0:
                self.collection.delete(
                    where={"document_id": document_id}
                )

            logger.info(f"Deleted {chunk_count} chunks for document: {document_id}")
            return chunk_count

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    async def get_stats(self) -> CollectionStats:
        """
        Get collection statistics.

        Returns:
            CollectionStats object
        """
        try:
            total_chunks = self.collection.count()

            # Get unique documents (sample metadata)
            results = self.collection.get(
                limit=1000,
                include=["metadatas"]
            )

            unique_docs = set()
            total_size = 0

            for metadata in results['metadatas']:
                unique_docs.add(metadata['document_id'])
                total_size += metadata.get('file_size', 0)

            total_size_mb = total_size / (1024 * 1024)

            stats = CollectionStats(
                collection_name=self.collection_name,
                total_documents=len(unique_docs),
                total_chunks=total_chunks,
                total_size_mb=round(total_size_mb, 2),
                last_updated=datetime.utcnow(),
                embedding_dimension=settings.EMBED_DIMENSION
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise

    async def reset_collection(self) -> bool:
        """
        Delete all data from the collection (USE WITH CAUTION).

        Returns:
            True if successful
        """
        try:
            logger.warning(f"Resetting collection: {self.collection_name}")

            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "T.A.R.S. document embeddings"}
            )

            logger.info("Collection reset complete")
            return True

        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    async def close(self):
        """Cleanup resources"""
        logger.info("Closing ChromaDB connection")
        self.client = None
        self.collection = None
        self.conversation_coll = None


# Global service instance
chromadb_service = ChromaDBService()
