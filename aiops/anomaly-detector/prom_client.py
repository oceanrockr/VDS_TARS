"""
Clients for Prometheus, Loki, and Jaeger
"""
import requests
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Client for querying Prometheus metrics"""

    def __init__(self, base_url: str):
        """
        Initialize Prometheus client.

        Args:
            base_url: Base URL of Prometheus server (e.g., http://prometheus:9090)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

    def query(self, query: str) -> List[Dict]:
        """
        Execute instant query.

        Args:
            query: PromQL query string

        Returns:
            List of result metrics
        """
        try:
            url = f"{self.base_url}/api/v1/query"
            response = self.session.get(url, params={'query': query}, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('status') != 'success':
                raise ValueError(f"Prometheus query failed: {data.get('error')}")

            return data.get('data', {}).get('result', [])

        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            raise

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "30s"
    ) -> List[Tuple[datetime, float]]:
        """
        Execute range query.

        Args:
            query: PromQL query string
            start: Start timestamp
            end: End timestamp
            step: Query resolution (e.g., "30s", "1m")

        Returns:
            List of (timestamp, value) tuples
        """
        try:
            url = f"{self.base_url}/api/v1/query_range"
            params = {
                'query': query,
                'start': start.timestamp(),
                'end': end.timestamp(),
                'step': step
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data.get('status') != 'success':
                raise ValueError(f"Prometheus range query failed: {data.get('error')}")

            results = data.get('data', {}).get('result', [])
            if not results:
                return []

            # Extract values from first result series
            values = results[0].get('values', [])

            # Convert to (datetime, float) tuples
            return [
                (datetime.fromtimestamp(float(ts)), float(val))
                for ts, val in values
            ]

        except Exception as e:
            logger.error(f"Prometheus range query failed: {e}")
            raise

    def get_metric_metadata(self, metric_name: str) -> Dict:
        """Get metadata for a metric"""
        try:
            url = f"{self.base_url}/api/v1/metadata"
            response = self.session.get(url, params={'metric': metric_name}, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('data', {}).get(metric_name, [{}])[0]

        except Exception as e:
            logger.warning(f"Failed to get metric metadata: {e}")
            return {}


class LokiClient:
    """Client for querying Loki logs"""

    def __init__(self, base_url: str):
        """
        Initialize Loki client.

        Args:
            base_url: Base URL of Loki server (e.g., http://loki:3100)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query Loki logs over a time range.

        Args:
            query: LogQL query string (e.g., '{namespace="tars",level="ERROR"}')
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of entries

        Returns:
            List of log entries
        """
        try:
            url = f"{self.base_url}/loki/api/v1/query_range"
            params = {
                'query': query,
                'start': int(start.timestamp() * 1e9),  # Nanoseconds
                'end': int(end.timestamp() * 1e9),
                'limit': limit
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data.get('status') != 'success':
                raise ValueError(f"Loki query failed: {data.get('error')}")

            results = data.get('data', {}).get('result', [])
            entries = []

            for result in results:
                stream = result.get('stream', {})
                values = result.get('values', [])
                for timestamp_ns, line in values:
                    entries.append({
                        'timestamp': datetime.fromtimestamp(int(timestamp_ns) / 1e9),
                        'line': line,
                        'labels': stream
                    })

            return entries

        except Exception as e:
            logger.error(f"Loki query failed: {e}")
            raise

    def get_log_rate(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m"
    ) -> List[Tuple[datetime, float]]:
        """
        Get log rate over time.

        Args:
            query: LogQL query
            start: Start timestamp
            end: End timestamp
            step: Time step

        Returns:
            List of (timestamp, rate) tuples
        """
        try:
            # Use rate() function in LogQL
            rate_query = f"rate({query}[1m])"
            url = f"{self.base_url}/loki/api/v1/query_range"
            params = {
                'query': rate_query,
                'start': int(start.timestamp() * 1e9),
                'end': int(end.timestamp() * 1e9),
                'step': step
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data.get('data', {}).get('result', [])

            if not results:
                return []

            values = results[0].get('values', [])
            return [
                (datetime.fromtimestamp(int(ts) / 1e9), float(val))
                for ts, val in values
            ]

        except Exception as e:
            logger.warning(f"Failed to get log rate: {e}")
            return []


class JaegerClient:
    """Client for querying Jaeger traces"""

    def __init__(self, base_url: str):
        """
        Initialize Jaeger client.

        Args:
            base_url: Base URL of Jaeger Query service (e.g., http://jaeger-query:16686)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})

    def get_services(self) -> List[str]:
        """Get list of services"""
        try:
            url = f"{self.base_url}/api/services"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('data', [])

        except Exception as e:
            logger.warning(f"Failed to get Jaeger services: {e}")
            return []

    def find_traces(
        self,
        service: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
        min_duration: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Find traces for a service.

        Args:
            service: Service name
            start: Start timestamp
            end: End timestamp
            limit: Maximum number of traces
            min_duration: Minimum duration filter (e.g., "100ms")
            tags: Additional tag filters

        Returns:
            List of traces
        """
        try:
            url = f"{self.base_url}/api/traces"
            params = {
                'service': service,
                'start': int(start.timestamp() * 1e6),  # Microseconds
                'end': int(end.timestamp() * 1e6),
                'limit': limit
            }

            if min_duration:
                params['minDuration'] = min_duration

            if tags:
                for key, value in tags.items():
                    params[f'tags'] = f'{key}:{value}'

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            return data.get('data', [])

        except Exception as e:
            logger.warning(f"Failed to find traces: {e}")
            return []

    def get_trace_latency_percentiles(
        self,
        service: str,
        start: datetime,
        end: datetime,
        operation: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate latency percentiles from traces.

        Args:
            service: Service name
            start: Start timestamp
            end: End timestamp
            operation: Optional operation name filter

        Returns:
            Dict with p50, p95, p99 latencies in milliseconds
        """
        try:
            traces = self.find_traces(service, start, end, limit=500)

            if not traces:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

            # Extract durations
            durations = []
            for trace in traces:
                spans = trace.get('spans', [])
                if spans:
                    # Root span duration
                    root_span = spans[0]
                    if operation and root_span.get('operationName') != operation:
                        continue
                    duration_us = root_span.get('duration', 0)
                    durations.append(duration_us / 1000.0)  # Convert to ms

            if not durations:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

            durations = sorted(durations)
            n = len(durations)

            return {
                "p50": durations[int(n * 0.50)],
                "p95": durations[int(n * 0.95)],
                "p99": durations[int(n * 0.99)]
            }

        except Exception as e:
            logger.warning(f"Failed to calculate latency percentiles: {e}")
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
