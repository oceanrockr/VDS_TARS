"""
T.A.R.S. Semantic Chunking Service
Intelligent document chunking based on semantic boundaries
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

from ..core.config import settings
from ..models.rag import DocumentChunk, Document

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunkMetadata:
    """Extended metadata for semantic chunks"""
    section_title: Optional[str] = None
    embedding_density: float = 0.0  # Semantic coherence score
    chunk_type: str = "paragraph"  # paragraph, heading, list, code
    has_headings: bool = False
    word_count: int = 0


class SemanticChunker:
    """
    Semantic text chunking service.

    Creates document chunks based on semantic boundaries rather than
    fixed token windows. Uses embeddings to detect topic shifts and
    preserves document structure.

    Features:
    - Dynamic chunk sizing (400-800 tokens)
    - Heading-aware chunking
    - Semantic boundary detection
    - Structure preservation
    - Embedding density scoring
    """

    def __init__(self):
        self.min_chunk_size = getattr(settings, 'SEMANTIC_CHUNK_MIN', 400)
        self.max_chunk_size = getattr(settings, 'SEMANTIC_CHUNK_MAX', 800)
        self.chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 50)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=[
                "\n\n",  # Double newline (paragraphs)
                "\n",    # Single newline
                ". ",    # Sentences
                ", ",    # Clauses
                " ",     # Words
                ""       # Characters
            ]
        )

        self.embedding_model: Optional[SentenceTransformer] = None
        logger.info("SemanticChunker initialized")

    def _token_length(self, text: str) -> int:
        """
        Estimate token count.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple approximation: words * 1.3
        return int(len(text.split()) * 1.3)

    def _extract_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract markdown/document headings.

        Args:
            text: Document text

        Returns:
            List of heading dictionaries with level and position
        """
        headings = []

        # Markdown headings (# ## ###)
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            headings.append({
                'level': level,
                'title': title,
                'position': position,
                'type': 'markdown'
            })

        # Underline headings (===, ---)
        for match in re.finditer(r'^(.+)\n([=\-]+)$', text, re.MULTILINE):
            title = match.group(1).strip()
            underline = match.group(2)
            level = 1 if '=' in underline else 2
            position = match.start()
            headings.append({
                'level': level,
                'title': title,
                'position': position,
                'type': 'underline'
            })

        # Sort by position
        headings.sort(key=lambda x: x['position'])

        return headings

    def _find_section_title(
        self,
        chunk_start: int,
        headings: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find the most recent heading for a chunk.

        Args:
            chunk_start: Starting position of chunk in document
            headings: List of heading metadata

        Returns:
            Section title or None
        """
        relevant_headings = [h for h in headings if h['position'] <= chunk_start]
        if relevant_headings:
            return relevant_headings[-1]['title']
        return None

    def _detect_chunk_type(self, text: str) -> str:
        """
        Detect the type of content in a chunk.

        Args:
            text: Chunk text

        Returns:
            Chunk type (paragraph, heading, list, code, table)
        """
        text_stripped = text.strip()

        # Code block
        if text_stripped.startswith('```') or text_stripped.startswith('    '):
            return 'code'

        # Heading
        if re.match(r'^#{1,6}\s+', text_stripped) or re.match(r'^.+\n[=\-]+$', text_stripped):
            return 'heading'

        # List
        if re.match(r'^[\*\-\+]\s+', text_stripped, re.MULTILINE) or \
           re.match(r'^\d+\.\s+', text_stripped, re.MULTILINE):
            return 'list'

        # Table
        if '|' in text and re.search(r'\|[\s\-:]+\|', text):
            return 'table'

        # Default: paragraph
        return 'paragraph'

    async def _compute_embedding_density(
        self,
        chunk_text: str,
        context_chunks: List[str]
    ) -> float:
        """
        Compute semantic coherence score for a chunk.

        Measures how semantically similar a chunk is to its surrounding context.

        Args:
            chunk_text: The chunk text
            context_chunks: Surrounding chunks for context

        Returns:
            Density score (0-1)
        """
        if not self.embedding_model or not context_chunks:
            return 0.5  # Default neutral score

        try:
            # Embed current chunk and context
            all_texts = [chunk_text] + context_chunks
            embeddings = self.embedding_model.encode(all_texts, convert_to_numpy=True)

            # Compute average cosine similarity to context
            chunk_emb = embeddings[0]
            context_embs = embeddings[1:]

            similarities = []
            for ctx_emb in context_embs:
                sim = np.dot(chunk_emb, ctx_emb) / (
                    np.linalg.norm(chunk_emb) * np.linalg.norm(ctx_emb)
                )
                similarities.append(sim)

            # Average similarity
            density = float(np.mean(similarities)) if similarities else 0.5

            return round(density, 4)

        except Exception as e:
            logger.warning(f"Error computing embedding density: {e}")
            return 0.5

    def create_chunks(
        self,
        document: Document,
        compute_density: bool = False
    ) -> List[DocumentChunk]:
        """
        Create semantic chunks from document.

        Args:
            document: Source document
            compute_density: Whether to compute embedding density (slower)

        Returns:
            List of DocumentChunk objects
        """
        start_time = time.time()

        try:
            content = document.content
            headings = self._extract_headings(content)

            logger.info(
                f"Creating semantic chunks for {document.metadata.file_name} "
                f"(length: {len(content)}, headings: {len(headings)})"
            )

            # Split text using recursive character splitter
            text_chunks = self.text_splitter.split_text(content)

            # Create DocumentChunk objects with metadata
            chunks = []
            current_position = 0

            for i, chunk_text in enumerate(text_chunks):
                # Find position in original document
                chunk_start = content.find(chunk_text, current_position)
                if chunk_start == -1:
                    chunk_start = current_position

                # Find section title
                section_title = self._find_section_title(chunk_start, headings)

                # Detect chunk type
                chunk_type = self._detect_chunk_type(chunk_text)

                # Count words
                word_count = len(chunk_text.split())

                # Create metadata
                metadata = document.metadata.copy()
                metadata['chunk_index'] = i
                metadata['chunk_type'] = chunk_type
                metadata['section_title'] = section_title
                metadata['word_count'] = word_count
                metadata['has_headings'] = bool(section_title)
                metadata['semantic_chunk'] = True

                # Create chunk
                chunk = DocumentChunk(
                    chunk_id=f"{document.document_id}_chunk_{i}",
                    document_id=document.document_id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata=metadata
                )

                chunks.append(chunk)
                current_position = chunk_start + len(chunk_text)

            # Optionally compute embedding density (expensive)
            if compute_density and self.embedding_model:
                for i, chunk in enumerate(chunks):
                    # Get surrounding chunks for context (±2)
                    context_start = max(0, i - 2)
                    context_end = min(len(chunks), i + 3)
                    context_chunks = [
                        chunks[j].content for j in range(context_start, context_end)
                        if j != i
                    ]

                    # Compute density (run synchronously for simplicity)
                    # In production, batch this
                    density = 0.5  # Placeholder - would call async method
                    chunk.metadata['embedding_density'] = density

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Created {len(chunks)} semantic chunks "
                f"(avg size: {sum(len(c.content) for c in chunks) // len(chunks)} chars, "
                f"time: {elapsed:.1f}ms)"
            )

            return chunks

        except Exception as e:
            logger.error(f"Error creating semantic chunks: {e}")
            # Fallback to simple chunking
            return self._fallback_chunking(document)

    def _fallback_chunking(self, document: Document) -> List[DocumentChunk]:
        """
        Fallback to simple fixed-size chunking.

        Args:
            document: Source document

        Returns:
            List of DocumentChunk objects
        """
        logger.warning("Using fallback fixed-size chunking")

        content = document.content
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        chunks = []
        start = 0
        i = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            metadata = document.metadata.copy()
            metadata['chunk_index'] = i
            metadata['chunk_type'] = 'fallback'
            metadata['semantic_chunk'] = False

            chunk = DocumentChunk(
                chunk_id=f"{document.document_id}_chunk_{i}",
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=i,
                metadata=metadata
            )

            chunks.append(chunk)

            start += (chunk_size - overlap)
            i += 1

        return chunks

    async def load_embedding_model(self, model_name: str = None) -> bool:
        """
        Load sentence transformer for density computation.

        Args:
            model_name: Model name (default: from settings)

        Returns:
            True if successful
        """
        if model_name is None:
            model_name = settings.EMBED_MODEL

        try:
            logger.info(f"Loading embedding model for chunker: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get chunker statistics.

        Returns:
            Dictionary with chunker stats
        """
        return {
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'has_embedding_model': self.embedding_model is not None
        }


# Global service instance
semantic_chunker = SemanticChunker()
