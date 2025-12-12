#!/usr/bin/env python3
"""
Shared Prometheus Client - Async HTTP Client for PromQL Queries

Provides a robust, production-ready Prometheus client with:
- Async HTTP requests (aiohttp)
- Exponential backoff retry logic
- Connection pooling
- Comprehensive error handling
- Normalized response format

Used across Phase 14.5 (GA Day) and Phase 14.6 (7-Day Stability) monitoring.

Author: T.A.R.S. Platform Team
Phase: 14.6 - Shared Infrastructure
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class PrometheusQueryError(Exception):
    """Custom exception for Prometheus query failures."""
    pass


class PrometheusClient:
    """
    Async Prometheus client for metrics collection.

    Features:
    - Context manager support for session lifecycle
    - Automatic retry with exponential backoff
    - Configurable timeouts
    - Normalized response format

    Usage:
        async with PrometheusClient("http://localhost:9090") as prom:
            result = await prom.query('up{job="my-service"}')
            if result:
                print(result["result"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9090",
        max_retries: int = 3,
        timeout_seconds: int = 30
    ):
        """
        Initialize Prometheus client.

        Args:
            base_url: Prometheus server URL (e.g., "http://localhost:9090")
            max_retries: Maximum number of retry attempts for failed queries
            timeout_seconds: Timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Context manager entry - initialize aiohttp session."""
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=30,
            ttl_dns_cache=300  # DNS cache TTL (5 minutes)
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close aiohttp session."""
        if self.session:
            await self.session.close()
            # Allow time for connections to close gracefully
            await asyncio.sleep(0.25)

    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Execute HTTP request with exponential backoff retry logic.

        Args:
            method: HTTP method ("GET" or "POST")
            url: Full request URL
            params: Query parameters
            description: Human-readable description for logging

        Returns:
            Parsed JSON response data, or None on failure

        Raises:
            PrometheusQueryError: If all retries fail
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.get(url, params=params) as resp:
                    # Check HTTP status
                    if resp.status == 503:
                        # Service unavailable - retry
                        logger.warning(
                            f"{description} - Service unavailable (503), "
                            f"attempt {attempt}/{self.max_retries}"
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            raise PrometheusQueryError(
                                f"Prometheus unavailable after {self.max_retries} attempts"
                            )

                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"{description} - HTTP {resp.status}: {error_text}"
                        )
                        raise PrometheusQueryError(
                            f"HTTP {resp.status}: {error_text}"
                        )

                    # Parse JSON response
                    data = await resp.json()

                    # Check Prometheus API status
                    if data.get("status") != "success":
                        error_msg = data.get("error", "Unknown error")
                        error_type = data.get("errorType", "unknown")
                        logger.error(
                            f"{description} - Prometheus error [{error_type}]: {error_msg}"
                        )
                        raise PrometheusQueryError(
                            f"Prometheus query failed [{error_type}]: {error_msg}"
                        )

                    # Return normalized data
                    return data.get("data", {})

            except asyncio.TimeoutError:
                logger.warning(
                    f"{description} - Timeout (>{self.timeout_seconds}s), "
                    f"attempt {attempt}/{self.max_retries}"
                )
                last_error = PrometheusQueryError(
                    f"Query timeout after {self.timeout_seconds}s"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise last_error

            except aiohttp.ClientError as e:
                logger.warning(
                    f"{description} - Network error: {e}, "
                    f"attempt {attempt}/{self.max_retries}"
                )
                last_error = PrometheusQueryError(f"Network error: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise last_error

            except PrometheusQueryError:
                # Don't retry on Prometheus API errors (e.g., bad query syntax)
                raise

            except Exception as e:
                logger.error(
                    f"{description} - Unexpected error: {e}",
                    exc_info=True
                )
                raise PrometheusQueryError(f"Unexpected error: {e}")

        # Should not reach here, but handle gracefully
        if last_error:
            raise last_error
        return None

    async def query(self, promql: str) -> Optional[Dict[str, Any]]:
        """
        Execute an instant PromQL query.

        Args:
            promql: PromQL query string (e.g., 'up{job="my-service"}')

        Returns:
            Query result data with structure:
            {
                "resultType": "vector" | "matrix" | "scalar" | "string",
                "result": [
                    {
                        "metric": {"__name__": "...", "label": "value"},
                        "value": [timestamp, "value"]
                    },
                    ...
                ]
            }

        Raises:
            PrometheusQueryError: On query failure after retries
            RuntimeError: If client not initialized with context manager

        Example:
            result = await prom.query('up{job="my-service"}')
            if result and result.get("result"):
                value = float(result["result"][0]["value"][1])
        """
        url = f"{self.base_url}/api/v1/query"
        params = {"query": promql}
        description = f"PromQL query: {promql[:80]}..."

        return await self._execute_with_retry("GET", url, params, description)

    async def query_range(
        self,
        promql: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a range PromQL query over a time window.

        Args:
            promql: PromQL query string
            start: Start time (datetime object, UTC)
            end: End time (datetime object, UTC)
            step: Query resolution (e.g., "1m", "5m", "1h")

        Returns:
            Range query result data with structure:
            {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"__name__": "...", "label": "value"},
                        "values": [
                            [timestamp1, "value1"],
                            [timestamp2, "value2"],
                            ...
                        ]
                    },
                    ...
                ]
            }

        Raises:
            PrometheusQueryError: On query failure after retries
            RuntimeError: If client not initialized with context manager

        Example:
            result = await prom.query_range(
                'rate(http_requests_total[5m])',
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                step="5m"
            )
            if result and result.get("result"):
                for series in result["result"]:
                    for timestamp, value in series["values"]:
                        print(f"{timestamp}: {value}")
        """
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": promql,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }
        description = f"PromQL range query: {promql[:80]}... (step={step})"

        return await self._execute_with_retry("GET", url, params, description)

    async def health_check(self) -> bool:
        """
        Check if Prometheus server is reachable and healthy.

        Returns:
            True if healthy, False otherwise

        Example:
            if await prom.health_check():
                print("Prometheus is healthy")
            else:
                print("Prometheus is unhealthy")
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        try:
            url = f"{self.base_url}/-/healthy"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Prometheus health check failed: {e}")
            return False

    async def query_safe(self, promql: str, default: Any = None) -> Any:
        """
        Execute a query and return a safe default on failure.

        This is a convenience method for queries where you want to handle
        failures gracefully without raising exceptions.

        Args:
            promql: PromQL query string
            default: Default value to return on failure (default: None)

        Returns:
            Query result or default value on failure

        Example:
            # Returns None if query fails
            result = await prom.query_safe('nonexistent_metric')

            # Returns empty dict if query fails
            result = await prom.query_safe('up', default={})
        """
        try:
            return await self.query(promql)
        except Exception as e:
            logger.debug(f"Query failed (returning default): {e}")
            return default
