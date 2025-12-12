"""
Production Log Ingestor for T.A.R.S.

Ingests structured JSON logs from production environments (CloudWatch, Stackdriver, etc.)
and extracts telemetry data for analysis.

Features:
- Batch and streaming ingestion modes
- Rate limiting and backpressure handling
- Partial failure safety
- Structured field extraction
- Multi-region support
- CloudWatch and Stackdriver integration

Performance:
- Minimum throughput: 50,000 logs/minute
- Batch size: 100-1000 logs
- Memory efficient streaming
- Async I/O for cloud APIs
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, AsyncIterator, Any, Callable
from collections import deque
import time
import re

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from google.cloud import logging as gcp_logging
    from google.api_core.exceptions import GoogleAPIError
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False


logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(str, Enum):
    """Log source providers."""
    CLOUDWATCH = "cloudwatch"
    STACKDRIVER = "stackdriver"
    FILE = "file"
    STREAM = "stream"


@dataclass
class LogEntry:
    """Structured log entry with extracted fields."""

    # Core fields
    timestamp: datetime
    service: str
    level: LogLevel
    message: str

    # Tracing fields
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Performance fields
    duration_ms: Optional[float] = None

    # Location fields
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    pod_name: Optional[str] = None

    # Request fields
    route: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None

    # Error fields
    error: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None

    # Metadata
    raw_log: Dict[str, Any] = field(default_factory=dict)
    source: LogSource = LogSource.STREAM
    ingestion_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects
        data['timestamp'] = self.timestamp.isoformat()
        data['ingestion_timestamp'] = self.ingestion_timestamp.isoformat()
        # Convert enums
        data['level'] = self.level.value
        data['source'] = self.source.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary."""
        # Parse timestamps
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('ingestion_timestamp'), str):
            data['ingestion_timestamp'] = datetime.fromisoformat(data['ingestion_timestamp'])

        # Parse enums
        if isinstance(data.get('level'), str):
            data['level'] = LogLevel(data['level'])
        if isinstance(data.get('source'), str):
            data['source'] = LogSource(data['source'])

        return cls(**data)


@dataclass
class IngestionStats:
    """Ingestion statistics."""
    total_ingested: int = 0
    total_failed: int = 0
    total_filtered: int = 0
    bytes_ingested: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_ingestion: Optional[datetime] = None

    # Rate metrics
    current_rate: float = 0.0  # logs/sec
    peak_rate: float = 0.0

    # Error tracking
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def record_success(self, log_size: int):
        """Record successful ingestion."""
        self.total_ingested += 1
        self.bytes_ingested += log_size
        self.last_ingestion = datetime.utcnow()

    def record_failure(self, error_type: str):
        """Record failed ingestion."""
        self.total_failed += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def record_filtered(self):
        """Record filtered log."""
        self.total_filtered += 1

    def calculate_rates(self):
        """Calculate current and peak rates."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        if elapsed > 0:
            self.current_rate = self.total_ingested / elapsed
            self.peak_rate = max(self.peak_rate, self.current_rate)

    def summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        self.calculate_rates()
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            'total_ingested': self.total_ingested,
            'total_failed': self.total_failed,
            'total_filtered': self.total_filtered,
            'success_rate': self.total_ingested / max(self.total_ingested + self.total_failed, 1),
            'bytes_ingested': self.bytes_ingested,
            'current_rate_per_sec': round(self.current_rate, 2),
            'current_rate_per_min': round(self.current_rate * 60, 2),
            'peak_rate_per_sec': round(self.peak_rate, 2),
            'peak_rate_per_min': round(self.peak_rate * 60, 2),
            'elapsed_seconds': round(elapsed, 2),
            'errors_by_type': dict(self.errors_by_type),
            'last_ingestion': self.last_ingestion.isoformat() if self.last_ingestion else None,
        }


class LogParser:
    """Parses structured logs into LogEntry objects."""

    # Common field mappings for different log formats
    FIELD_MAPPINGS = {
        'cloudwatch': {
            'timestamp': ['@timestamp', 'timestamp', 'time'],
            'message': ['message', 'msg'],
            'level': ['level', 'severity', 'logLevel'],
            'service': ['service', 'service_name', 'application'],
            'trace_id': ['trace_id', 'traceId', 'request_id'],
            'span_id': ['span_id', 'spanId'],
            'duration_ms': ['duration', 'duration_ms', 'latency_ms'],
            'region': ['region', 'aws_region'],
            'error': ['error', 'error_message', 'exception'],
        },
        'stackdriver': {
            'timestamp': ['timestamp', 'receiveTimestamp'],
            'message': ['textPayload', 'message'],
            'level': ['severity'],
            'service': ['resource.labels.service', 'labels.service_name'],
            'trace_id': ['trace', 'labels.trace_id'],
            'span_id': ['spanId', 'labels.span_id'],
            'region': ['resource.labels.zone', 'resource.labels.region'],
        },
    }

    def __init__(self, source: LogSource = LogSource.STREAM):
        """Initialize parser.

        Args:
            source: Log source type for field mapping
        """
        self.source = source
        self.field_map = self.FIELD_MAPPINGS.get(source.value, {})

    def parse(self, raw_log: Dict[str, Any]) -> Optional[LogEntry]:
        """Parse raw log into structured LogEntry.

        Args:
            raw_log: Raw log dictionary

        Returns:
            Parsed LogEntry or None if parsing fails
        """
        try:
            # Extract timestamp
            timestamp = self._extract_field(raw_log, 'timestamp')
            if isinstance(timestamp, str):
                timestamp = self._parse_timestamp(timestamp)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.utcnow()

            # Extract required fields
            service = self._extract_field(raw_log, 'service') or 'unknown'
            message = self._extract_field(raw_log, 'message') or ''

            # Extract level
            level_str = self._extract_field(raw_log, 'level') or 'INFO'
            level = self._parse_level(level_str)

            # Extract optional fields
            trace_id = self._extract_field(raw_log, 'trace_id')
            span_id = self._extract_field(raw_log, 'span_id')
            duration_ms = self._extract_field(raw_log, 'duration_ms')
            region = self._extract_field(raw_log, 'region')
            error = self._extract_field(raw_log, 'error')

            # Extract from message if fields not found
            if not trace_id:
                trace_id = self._extract_trace_from_message(message)

            # Build LogEntry
            return LogEntry(
                timestamp=timestamp,
                service=service,
                level=level,
                message=message,
                trace_id=trace_id,
                span_id=span_id,
                duration_ms=float(duration_ms) if duration_ms else None,
                region=region,
                error=error,
                raw_log=raw_log,
                source=self.source,
            )

        except Exception as e:
            logger.error(f"Failed to parse log: {e}", exc_info=True)
            return None

    def _extract_field(self, data: Dict[str, Any], field_name: str) -> Optional[Any]:
        """Extract field using field mappings.

        Args:
            data: Log data
            field_name: Field to extract

        Returns:
            Field value or None
        """
        # Try mapped field names
        field_paths = self.field_map.get(field_name, [field_name])

        for path in field_paths:
            value = self._get_nested_field(data, path)
            if value is not None:
                return value

        # Try direct field name
        return data.get(field_name)

    def _get_nested_field(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        """Get nested field using dot notation.

        Args:
            data: Data dictionary
            path: Dot-separated path (e.g., 'resource.labels.service')

        Returns:
            Field value or None
        """
        parts = path.split('.')
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            else:
                return None

        return current

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string.

        Args:
            ts_str: Timestamp string

        Returns:
            Parsed datetime
        """
        # Try ISO format
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except ValueError:
            pass

        # Try other common formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue

        # Fallback
        logger.warning(f"Failed to parse timestamp: {ts_str}")
        return datetime.utcnow()

    def _parse_level(self, level_str: str) -> LogLevel:
        """Parse log level string.

        Args:
            level_str: Level string

        Returns:
            LogLevel enum
        """
        level_upper = str(level_str).upper()

        # Map common level names
        level_map = {
            'DEBUG': LogLevel.DEBUG,
            'INFO': LogLevel.INFO,
            'INFORMATION': LogLevel.INFO,
            'WARN': LogLevel.WARNING,
            'WARNING': LogLevel.WARNING,
            'ERROR': LogLevel.ERROR,
            'ERR': LogLevel.ERROR,
            'CRITICAL': LogLevel.CRITICAL,
            'FATAL': LogLevel.CRITICAL,
        }

        return level_map.get(level_upper, LogLevel.INFO)

    def _extract_trace_from_message(self, message: str) -> Optional[str]:
        """Extract trace ID from message text.

        Args:
            message: Log message

        Returns:
            Trace ID or None
        """
        # Look for trace ID patterns (UUID-like)
        pattern = r'trace[_-]?id[:\s=]+([a-f0-9\-]{32,36})'
        match = re.search(pattern, message, re.IGNORECASE)

        if match:
            return match.group(1)

        return None


class RateLimiter:
    """Token bucket rate limiter for ingestion."""

    def __init__(self, rate: float, burst: int):
        """Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait(self, tokens: int = 1):
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire
        """
        while not await self.acquire(tokens):
            await asyncio.sleep(0.01)


class ProductionLogIngestor:
    """Production log ingestor with batch and streaming support."""

    def __init__(
        self,
        source: LogSource = LogSource.STREAM,
        batch_size: int = 100,
        rate_limit: float = 1000.0,  # logs/sec
        max_retries: int = 3,
        filter_func: Optional[Callable[[LogEntry], bool]] = None,
    ):
        """Initialize ingestor.

        Args:
            source: Log source type
            batch_size: Batch size for processing
            rate_limit: Maximum logs per second
            max_retries: Maximum retry attempts
            filter_func: Optional filter function
        """
        self.source = source
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.filter_func = filter_func

        self.parser = LogParser(source)
        self.rate_limiter = RateLimiter(rate=rate_limit, burst=int(rate_limit * 2))
        self.stats = IngestionStats()

        self._buffer: deque = deque(maxlen=batch_size * 10)
        self._running = False

    async def ingest_batch(
        self,
        raw_logs: List[Dict[str, Any]],
    ) -> List[LogEntry]:
        """Ingest a batch of raw logs.

        Args:
            raw_logs: List of raw log dictionaries

        Returns:
            List of parsed LogEntry objects
        """
        entries = []

        for raw_log in raw_logs:
            try:
                # Rate limiting
                await self.rate_limiter.wait()

                # Parse log
                entry = self.parser.parse(raw_log)

                if entry is None:
                    self.stats.record_failure('parse_error')
                    continue

                # Apply filter
                if self.filter_func and not self.filter_func(entry):
                    self.stats.record_filtered()
                    continue

                # Record success
                log_size = len(json.dumps(raw_log))
                self.stats.record_success(log_size)

                entries.append(entry)

            except Exception as e:
                logger.error(f"Failed to ingest log: {e}")
                self.stats.record_failure(type(e).__name__)

        return entries

    async def ingest_stream(
        self,
        log_stream: AsyncIterator[Dict[str, Any]],
        callback: Optional[Callable[[List[LogEntry]], None]] = None,
    ):
        """Ingest logs from async stream.

        Args:
            log_stream: Async iterator of raw logs
            callback: Optional callback for processed batches
        """
        self._running = True
        batch = []

        try:
            async for raw_log in log_stream:
                if not self._running:
                    break

                # Rate limiting
                await self.rate_limiter.wait()

                # Parse log
                entry = self.parser.parse(raw_log)

                if entry is None:
                    self.stats.record_failure('parse_error')
                    continue

                # Apply filter
                if self.filter_func and not self.filter_func(entry):
                    self.stats.record_filtered()
                    continue

                # Record success
                log_size = len(json.dumps(raw_log))
                self.stats.record_success(log_size)

                batch.append(entry)

                # Process batch
                if len(batch) >= self.batch_size:
                    if callback:
                        callback(batch)
                    batch = []

            # Process remaining
            if batch and callback:
                callback(batch)

        finally:
            self._running = False

    def stop(self):
        """Stop streaming ingestion."""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics.

        Returns:
            Statistics dictionary
        """
        return self.stats.summary()


class CloudWatchIngestor(ProductionLogIngestor):
    """CloudWatch Logs ingestor."""

    def __init__(
        self,
        log_group: str,
        region: str = 'us-east-1',
        **kwargs,
    ):
        """Initialize CloudWatch ingestor.

        Args:
            log_group: CloudWatch log group name
            region: AWS region
            **kwargs: Additional arguments for ProductionLogIngestor
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 required for CloudWatch ingestion")

        super().__init__(source=LogSource.CLOUDWATCH, **kwargs)

        self.log_group = log_group
        self.region = region
        self.client = boto3.client('logs', region_name=region)

    async def fetch_logs(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        filter_pattern: str = '',
    ) -> List[LogEntry]:
        """Fetch logs from CloudWatch.

        Args:
            start_time: Start time for log query
            end_time: End time for log query (default: now)
            filter_pattern: CloudWatch filter pattern

        Returns:
            List of parsed log entries
        """
        if end_time is None:
            end_time = datetime.utcnow()

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        raw_logs = []
        next_token = None

        try:
            while True:
                kwargs = {
                    'logGroupName': self.log_group,
                    'startTime': start_ms,
                    'endTime': end_ms,
                }

                if filter_pattern:
                    kwargs['filterPattern'] = filter_pattern

                if next_token:
                    kwargs['nextToken'] = next_token

                response = self.client.filter_log_events(**kwargs)

                for event in response.get('events', []):
                    # Parse JSON message if possible
                    try:
                        message = json.loads(event['message'])
                    except json.JSONDecodeError:
                        message = {'message': event['message']}

                    message['timestamp'] = event['timestamp'] / 1000  # Convert to seconds
                    raw_logs.append(message)

                next_token = response.get('nextToken')
                if not next_token:
                    break

        except (ClientError, BotoCoreError) as e:
            logger.error(f"CloudWatch API error: {e}")
            self.stats.record_failure('cloudwatch_api_error')

        return await self.ingest_batch(raw_logs)


class StackdriverIngestor(ProductionLogIngestor):
    """Google Cloud Stackdriver Logging ingestor."""

    def __init__(
        self,
        project_id: str,
        **kwargs,
    ):
        """Initialize Stackdriver ingestor.

        Args:
            project_id: GCP project ID
            **kwargs: Additional arguments for ProductionLogIngestor
        """
        if not GCP_AVAILABLE:
            raise ImportError("google-cloud-logging required for Stackdriver ingestion")

        super().__init__(source=LogSource.STACKDRIVER, **kwargs)

        self.project_id = project_id
        self.client = gcp_logging.Client(project=project_id)

    async def fetch_logs(
        self,
        filter_str: str = '',
        max_results: int = 1000,
    ) -> List[LogEntry]:
        """Fetch logs from Stackdriver.

        Args:
            filter_str: Stackdriver filter string
            max_results: Maximum number of results

        Returns:
            List of parsed log entries
        """
        raw_logs = []

        try:
            entries = self.client.list_entries(
                filter_=filter_str,
                max_results=max_results,
            )

            for entry in entries:
                log_data = {
                    'timestamp': entry.timestamp.isoformat(),
                    'severity': entry.severity,
                    'message': entry.payload,
                    'resource': entry.resource._asdict() if entry.resource else {},
                    'labels': dict(entry.labels) if entry.labels else {},
                }

                raw_logs.append(log_data)

        except GoogleAPIError as e:
            logger.error(f"Stackdriver API error: {e}")
            self.stats.record_failure('stackdriver_api_error')

        return await self.ingest_batch(raw_logs)


# Example usage
if __name__ == '__main__':
    # Example: Batch ingestion
    async def example_batch():
        ingestor = ProductionLogIngestor(
            source=LogSource.STREAM,
            batch_size=100,
            rate_limit=1000.0,
        )

        # Sample logs
        sample_logs = [
            {
                'timestamp': datetime.utcnow().isoformat(),
                'service': 'eval-engine',
                'level': 'INFO',
                'message': 'Evaluation completed',
                'trace_id': 'abc123',
                'duration_ms': 150.5,
                'region': 'us-east-1',
            }
            for _ in range(1000)
        ]

        entries = await ingestor.ingest_batch(sample_logs)

        print(f"Ingested {len(entries)} logs")
        print(json.dumps(ingestor.get_stats(), indent=2))

    # Run example
    asyncio.run(example_batch())
