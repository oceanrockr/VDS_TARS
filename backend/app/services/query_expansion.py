"""
T.A.R.S. Query Expansion Service
LLM-based query reformulation and expansion
Phase 5 - Advanced RAG & Semantic Chunking
"""

import logging
import time
from typing import List, Dict, Any, Optional
import asyncio
import re

from ..core.config import settings
from .ollama_service import ollama_service

logger = logging.getLogger(__name__)


class QueryExpansionService:
    """
    Query expansion and reformulation service.

    Uses LLM to generate query variants with:
    - Synonym expansion
    - Phrase variations
    - Domain-specific terminology
    - Question reformulation

    Features:
    - Multiple expansion strategies
    - Configurable expansion count
    - Caching of expansions
    - Async generation
    """

    def __init__(self):
        self.enabled = getattr(settings, 'QUERY_EXPANSION_ENABLED', True)
        self.max_expansions = getattr(settings, 'QUERY_EXPANSION_MAX', 3)
        self.expansion_cache: Dict[str, List[str]] = {}
        self.cache_size = 1000

        logger.info(f"QueryExpansionService initialized (enabled: {self.enabled})")

    def _build_expansion_prompt(self, query: str, strategy: str = 'synonym') -> str:
        """
        Build LLM prompt for query expansion.

        Args:
            query: Original query
            strategy: Expansion strategy (synonym, rephrase, technical)

        Returns:
            Formatted prompt
        """
        if strategy == 'synonym':
            return f"""Given the search query below, generate {self.max_expansions} alternative phrasings using synonyms and related terms.

Original query: "{query}"

Requirements:
- Preserve the original intent and meaning
- Use synonyms and related terminology
- Keep queries concise (under 20 words)
- Output ONLY the alternative queries, one per line
- Do NOT number the queries or add explanations

Alternative queries:"""

        elif strategy == 'rephrase':
            return f"""Given the search query below, rephrase it in {self.max_expansions} different ways while keeping the same meaning.

Original query: "{query}"

Requirements:
- Maintain the original semantic meaning
- Use different sentence structures
- Keep queries natural and concise
- Output ONLY the rephrased queries, one per line
- Do NOT number the queries or add explanations

Rephrased queries:"""

        elif strategy == 'technical':
            return f"""Given the search query below, generate {self.max_expansions} technical variations using domain-specific terminology.

Original query: "{query}"

Requirements:
- Use technical or domain-specific terms
- Expand abbreviations if present
- Include related technical concepts
- Output ONLY the technical variations, one per line
- Do NOT number the queries or add explanations

Technical variations:"""

        else:  # general (default)
            return f"""Given the search query below, generate {self.max_expansions} alternative search queries that would retrieve similar information.

Original query: "{query}"

Requirements:
- Preserve the search intent
- Use different wording and perspectives
- Keep queries clear and focused
- Output ONLY the alternative queries, one per line
- Do NOT number the queries or add explanations

Alternative queries:"""

    def _parse_expansions(self, response: str) -> List[str]:
        """
        Parse LLM response into list of expansions.

        Args:
            response: Raw LLM output

        Returns:
            List of cleaned query expansions
        """
        # Split by newlines
        lines = response.strip().split('\n')

        expansions = []
        for line in lines:
            # Remove numbering (1., 2., -, *, etc.)
            cleaned = re.sub(r'^\s*[\d\-\*\+]+[\.\)]\s*', '', line)

            # Remove quotes
            cleaned = cleaned.strip('"\'')

            # Skip empty lines or very short expansions
            if len(cleaned.strip()) < 5:
                continue

            # Skip if it's exactly the same as original (case-insensitive)
            if cleaned.strip():
                expansions.append(cleaned.strip())

        return expansions[:self.max_expansions]

    async def expand_query(
        self,
        query: str,
        strategy: str = 'general',
        include_original: bool = True
    ) -> List[str]:
        """
        Expand query using LLM.

        Args:
            query: Original query
            strategy: Expansion strategy (synonym, rephrase, technical, general)
            include_original: Whether to include original query in results

        Returns:
            List of query variants (including original if requested)
        """
        if not self.enabled:
            return [query]

        # Check cache
        cache_key = f"{query}:{strategy}"
        if cache_key in self.expansion_cache:
            logger.debug(f"Using cached expansions for: {query}")
            cached = self.expansion_cache[cache_key]
            return [query] + cached if include_original else cached

        start_time = time.time()

        try:
            logger.info(f"Expanding query (strategy: {strategy}): {query}")

            # Build prompt
            prompt = self._build_expansion_prompt(query, strategy)

            # Generate expansions using Ollama
            response_chunks = []
            async for chunk_data in ollama_service.generate_stream(
                prompt=prompt,
                model=settings.OLLAMA_MODEL,
                temperature=0.7,  # Moderate creativity
                max_tokens=200,   # Short responses
                system_prompt="You are a helpful assistant that generates alternative search queries."
            ):
                if not chunk_data.get('done', False):
                    token = chunk_data.get('token', '')
                    response_chunks.append(token)

            response = ''.join(response_chunks)

            # Parse expansions
            expansions = self._parse_expansions(response)

            # Ensure we have at least one expansion (fallback to original)
            if not expansions:
                logger.warning(f"No valid expansions generated for: {query}")
                expansions = [query]

            # Cache result
            if len(self.expansion_cache) >= self.cache_size:
                # Simple LRU: remove first item
                self.expansion_cache.pop(next(iter(self.expansion_cache)))

            self.expansion_cache[cache_key] = expansions

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Generated {len(expansions)} expansions "
                f"(strategy: {strategy}, time: {elapsed:.1f}ms)"
            )

            # Return with original if requested
            if include_original:
                return [query] + expansions
            else:
                return expansions

        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]

    async def multi_strategy_expansion(
        self,
        query: str,
        strategies: List[str] = None
    ) -> List[str]:
        """
        Expand query using multiple strategies.

        Args:
            query: Original query
            strategies: List of strategies to use (default: all)

        Returns:
            Combined list of unique query variants
        """
        if strategies is None:
            strategies = ['synonym', 'rephrase', 'technical']

        start_time = time.time()

        try:
            # Run all strategies in parallel
            tasks = [
                self.expand_query(query, strategy, include_original=False)
                for strategy in strategies
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            all_expansions = [query]  # Start with original

            for result in results:
                if isinstance(result, list):
                    all_expansions.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Strategy failed: {result}")

            # Deduplicate while preserving order
            seen = set()
            unique_expansions = []
            for exp in all_expansions:
                exp_lower = exp.lower().strip()
                if exp_lower not in seen:
                    seen.add(exp_lower)
                    unique_expansions.append(exp)

            elapsed = (time.time() - start_time) * 1000

            logger.info(
                f"Multi-strategy expansion: {len(unique_expansions)} unique queries "
                f"(strategies: {strategies}, time: {elapsed:.1f}ms)"
            )

            return unique_expansions

        except Exception as e:
            logger.error(f"Error in multi-strategy expansion: {e}")
            return [query]

    def clear_cache(self):
        """Clear expansion cache"""
        self.expansion_cache.clear()
        logger.info("Expansion cache cleared")

    def get_stats(self) -> dict:
        """
        Get expansion service statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'enabled': self.enabled,
            'max_expansions': self.max_expansions,
            'cache_size': len(self.expansion_cache),
            'cache_limit': self.cache_size
        }


# Global service instance
query_expansion_service = QueryExpansionService()
