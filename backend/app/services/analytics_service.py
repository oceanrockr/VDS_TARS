"""
T.A.R.S. Analytics Service
Query and usage analytics tracking
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import asyncio

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalytics:
    """Analytics data for a single query"""
    query_id: str
    timestamp: str
    client_id: str
    query_text: str
    query_length: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    sources_count: int
    avg_relevance_score: float
    max_relevance_score: float
    model_used: str
    tokens_generated: int
    used_reranking: bool
    used_hybrid_search: bool
    used_query_expansion: bool
    expansion_count: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class DocumentAnalytics:
    """Analytics for document usage"""
    document_id: str
    file_name: str
    access_count: int
    last_accessed: str
    avg_relevance_score: float
    total_retrievals: int


class AnalyticsService:
    """
    Service for tracking and analyzing query/document usage.

    Features:
    - Query performance tracking
    - Document popularity metrics
    - Usage patterns analysis
    - CSV export support
    - Time-series aggregation
    """

    def __init__(self):
        self.log_path = getattr(settings, 'ANALYTICS_LOG_PATH', '/logs/analytics.log')
        self.enable_logging = getattr(settings, 'ANALYTICS_ENABLED', True)

        # In-memory storage (will persist to file)
        self.queries: List[QueryAnalytics] = []
        self.document_stats: Dict[str, DocumentAnalytics] = {}

        # Aggregated stats
        self.total_queries = 0
        self.total_errors = 0
        self.avg_retrieval_time = 0.0
        self.avg_generation_time = 0.0

        # Ensure log directory exists
        self._ensure_log_directory()

        logger.info(f"AnalyticsService initialized (logging: {self.enable_logging})")

    def _ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created analytics log directory: {log_dir}")
        except Exception as e:
            logger.error(f"Failed to create log directory: {e}")

    def _write_to_log(self, data: dict):
        """
        Write analytics data to log file.

        Args:
            data: Analytics data dictionary
        """
        if not self.enable_logging:
            return

        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write analytics log: {e}")

    async def log_query(
        self,
        query_text: str,
        client_id: str,
        retrieval_time_ms: float,
        generation_time_ms: float,
        total_time_ms: float,
        sources_count: int,
        relevance_scores: List[float],
        model_used: str,
        tokens_generated: int,
        used_reranking: bool = False,
        used_hybrid_search: bool = False,
        used_query_expansion: bool = False,
        expansion_count: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> QueryAnalytics:
        """
        Log query analytics.

        Args:
            query_text: The query text
            client_id: Client identifier
            retrieval_time_ms: Retrieval time in milliseconds
            generation_time_ms: Generation time in milliseconds
            total_time_ms: Total time in milliseconds
            sources_count: Number of sources retrieved
            relevance_scores: List of relevance scores
            model_used: LLM model name
            tokens_generated: Number of tokens generated
            used_reranking: Whether reranking was used
            used_hybrid_search: Whether hybrid search was used
            used_query_expansion: Whether query expansion was used
            expansion_count: Number of query expansions
            success: Whether query succeeded
            error_message: Error message if failed

        Returns:
            QueryAnalytics object
        """
        try:
            # Create analytics record
            query_id = f"q_{int(time.time() * 1000)}"
            timestamp = datetime.now().isoformat()

            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            max_relevance = max(relevance_scores) if relevance_scores else 0.0

            analytics = QueryAnalytics(
                query_id=query_id,
                timestamp=timestamp,
                client_id=client_id,
                query_text=query_text,
                query_length=len(query_text),
                retrieval_time_ms=round(retrieval_time_ms, 2),
                generation_time_ms=round(generation_time_ms, 2),
                total_time_ms=round(total_time_ms, 2),
                sources_count=sources_count,
                avg_relevance_score=round(avg_relevance, 4),
                max_relevance_score=round(max_relevance, 4),
                model_used=model_used,
                tokens_generated=tokens_generated,
                used_reranking=used_reranking,
                used_hybrid_search=used_hybrid_search,
                used_query_expansion=used_query_expansion,
                expansion_count=expansion_count,
                success=success,
                error_message=error_message
            )

            # Store in memory
            self.queries.append(analytics)

            # Update aggregates
            self.total_queries += 1
            if not success:
                self.total_errors += 1

            # Update rolling averages
            n = self.total_queries
            self.avg_retrieval_time = (
                (self.avg_retrieval_time * (n - 1) + retrieval_time_ms) / n
            )
            self.avg_generation_time = (
                (self.avg_generation_time * (n - 1) + generation_time_ms) / n
            )

            # Write to log file
            self._write_to_log(asdict(analytics))

            logger.debug(f"Logged query analytics: {query_id}")

            return analytics

        except Exception as e:
            logger.error(f"Error logging query analytics: {e}")
            return None

    async def log_document_access(
        self,
        document_id: str,
        file_name: str,
        relevance_score: float
    ):
        """
        Log document access for popularity tracking.

        Args:
            document_id: Document identifier
            file_name: Document filename
            relevance_score: Relevance score for this access
        """
        try:
            timestamp = datetime.now().isoformat()

            if document_id in self.document_stats:
                # Update existing stats
                doc_stats = self.document_stats[document_id]
                doc_stats.access_count += 1
                doc_stats.last_accessed = timestamp
                doc_stats.total_retrievals += 1

                # Update rolling average relevance
                n = doc_stats.total_retrievals
                doc_stats.avg_relevance_score = (
                    (doc_stats.avg_relevance_score * (n - 1) + relevance_score) / n
                )

            else:
                # Create new stats
                self.document_stats[document_id] = DocumentAnalytics(
                    document_id=document_id,
                    file_name=file_name,
                    access_count=1,
                    last_accessed=timestamp,
                    avg_relevance_score=relevance_score,
                    total_retrievals=1
                )

        except Exception as e:
            logger.error(f"Error logging document access: {e}")

    def get_query_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated query statistics.

        Args:
            start_time: Start of time range (default: all time)
            end_time: End of time range (default: now)

        Returns:
            Dictionary with query statistics
        """
        try:
            # Filter queries by time range
            filtered_queries = self.queries

            if start_time or end_time:
                filtered_queries = []
                for q in self.queries:
                    q_time = datetime.fromisoformat(q.timestamp)
                    if start_time and q_time < start_time:
                        continue
                    if end_time and q_time > end_time:
                        continue
                    filtered_queries.append(q)

            if not filtered_queries:
                return {
                    'total_queries': 0,
                    'success_rate': 0.0,
                    'avg_retrieval_time_ms': 0.0,
                    'avg_generation_time_ms': 0.0,
                    'avg_total_time_ms': 0.0,
                    'avg_sources_count': 0.0,
                    'avg_relevance_score': 0.0,
                    'reranking_usage_rate': 0.0,
                    'hybrid_search_usage_rate': 0.0,
                    'query_expansion_usage_rate': 0.0
                }

            # Compute stats
            total = len(filtered_queries)
            successful = sum(1 for q in filtered_queries if q.success)

            avg_retrieval = sum(q.retrieval_time_ms for q in filtered_queries) / total
            avg_generation = sum(q.generation_time_ms for q in filtered_queries) / total
            avg_total = sum(q.total_time_ms for q in filtered_queries) / total
            avg_sources = sum(q.sources_count for q in filtered_queries) / total
            avg_relevance = sum(q.avg_relevance_score for q in filtered_queries) / total

            reranking_count = sum(1 for q in filtered_queries if q.used_reranking)
            hybrid_count = sum(1 for q in filtered_queries if q.used_hybrid_search)
            expansion_count = sum(1 for q in filtered_queries if q.used_query_expansion)

            return {
                'total_queries': total,
                'successful_queries': successful,
                'failed_queries': total - successful,
                'success_rate': round(successful / total, 4) if total > 0 else 0.0,
                'avg_retrieval_time_ms': round(avg_retrieval, 2),
                'avg_generation_time_ms': round(avg_generation, 2),
                'avg_total_time_ms': round(avg_total, 2),
                'avg_sources_count': round(avg_sources, 2),
                'avg_relevance_score': round(avg_relevance, 4),
                'reranking_usage_count': reranking_count,
                'reranking_usage_rate': round(reranking_count / total, 4) if total > 0 else 0.0,
                'hybrid_search_usage_count': hybrid_count,
                'hybrid_search_usage_rate': round(hybrid_count / total, 4) if total > 0 else 0.0,
                'query_expansion_usage_count': expansion_count,
                'query_expansion_usage_rate': round(expansion_count / total, 4) if total > 0 else 0.0,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None
            }

        except Exception as e:
            logger.error(f"Error computing query stats: {e}")
            return {}

    def get_document_popularity(
        self,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular documents.

        Args:
            top_n: Number of top documents to return

        Returns:
            List of document statistics
        """
        try:
            # Sort by access count
            sorted_docs = sorted(
                self.document_stats.values(),
                key=lambda x: x.access_count,
                reverse=True
            )

            return [asdict(doc) for doc in sorted_docs[:top_n]]

        except Exception as e:
            logger.error(f"Error getting document popularity: {e}")
            return []

    def get_query_patterns(self) -> Dict[str, Any]:
        """
        Analyze query patterns.

        Returns:
            Dictionary with pattern statistics
        """
        try:
            if not self.queries:
                return {}

            # Query length distribution
            lengths = [q.query_length for q in self.queries]
            avg_length = sum(lengths) / len(lengths)

            # Common words
            word_counter = Counter()
            for q in self.queries:
                words = q.query_text.lower().split()
                word_counter.update(words)

            # Time distribution (hourly)
            hour_counter = Counter()
            for q in self.queries:
                hour = datetime.fromisoformat(q.timestamp).hour
                hour_counter[hour] += 1

            return {
                'total_queries': len(self.queries),
                'avg_query_length': round(avg_length, 2),
                'min_query_length': min(lengths),
                'max_query_length': max(lengths),
                'top_words': dict(word_counter.most_common(20)),
                'queries_by_hour': dict(hour_counter)
            }

        except Exception as e:
            logger.error(f"Error analyzing query patterns: {e}")
            return {}

    def export_to_csv(self, output_path: str) -> bool:
        """
        Export analytics to CSV file.

        Args:
            output_path: Output CSV file path

        Returns:
            True if successful
        """
        try:
            import csv

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not self.queries:
                    logger.warning("No queries to export")
                    return False

                # Write header
                fieldnames = list(asdict(self.queries[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # Write data
                for query in self.queries:
                    writer.writerow(asdict(query))

            logger.info(f"Exported {len(self.queries)} queries to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get analytics service statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'enable_logging': self.enable_logging,
            'log_path': self.log_path,
            'total_queries': self.total_queries,
            'total_errors': self.total_errors,
            'avg_retrieval_time_ms': round(self.avg_retrieval_time, 2),
            'avg_generation_time_ms': round(self.avg_generation_time, 2),
            'tracked_documents': len(self.document_stats)
        }


# Global service instance
analytics_service = AnalyticsService()
