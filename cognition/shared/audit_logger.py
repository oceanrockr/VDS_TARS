"""
T.A.R.S. Audit Logging Module
Comprehensive audit logging for all security and administrative events
Phase 12
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel
from collections import deque
import asyncio

# Try to import Redis for persistent storage
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_TOKEN_EXPIRED = "auth.token.expired"
    AUTH_TOKEN_INVALID = "auth.token.invalid"

    # API Key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_ROTATED = "api_key.rotated"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED = "api_key.used"
    API_KEY_INVALID = "api_key.invalid"

    # Authorization events
    AUTHZ_ACCESS_GRANTED = "authz.access.granted"
    AUTHZ_ACCESS_DENIED = "authz.access.denied"
    AUTHZ_INSUFFICIENT_PERMISSIONS = "authz.insufficient_permissions"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    RATE_LIMIT_WARNING = "rate_limit.warning"

    # Admin actions
    ADMIN_AGENT_RELOAD = "admin.agent.reload"
    ADMIN_MODEL_PROMOTE = "admin.model.promote"
    ADMIN_HYPERSYNC_APPROVE = "admin.hypersync.approve"
    ADMIN_HYPERSYNC_DENY = "admin.hypersync.deny"
    ADMIN_CONFIG_CHANGE = "admin.config.change"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


class AuditEventSeverity(str, Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Audit event model"""
    event_id: str
    event_type: AuditEventType
    severity: AuditEventSeverity
    timestamp: datetime
    service: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "user_id": self.user_id,
            "username": self.username,
            "ip_address": self.ip_address,
            "endpoint": self.endpoint,
            "method": self.method,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "error": self.error
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Audit logger for security and administrative events

    Features:
    - Async logging with minimal performance impact
    - Redis backend for persistence (with in-memory fallback)
    - Prometheus metrics integration
    - Structured logging with JSON format
    - Configurable retention policies
    """

    def __init__(
        self,
        service_name: str,
        redis_url: Optional[str] = None,
        max_memory_events: int = 10000,
        retention_days: int = 90
    ):
        self.service_name = service_name
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.max_memory_events = max_memory_events
        self.retention_days = retention_days

        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = REDIS_AVAILABLE and os.getenv("USE_REDIS", "true").lower() == "true"

        # In-memory fallback storage
        self.memory_events: deque = deque(maxlen=max_memory_events)

        # Event counter for generating IDs
        self._event_counter = 0
        self._lock = asyncio.Lock()

        # Prometheus metrics (if available)
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            from prometheus_client import Counter, Gauge

            self.audit_events_total = Counter(
                'tars_audit_events_total',
                'Total audit events',
                ['service', 'event_type', 'severity']
            )

            self.audit_events_by_user = Counter(
                'tars_audit_events_by_user_total',
                'Audit events by user',
                ['service', 'username', 'event_type']
            )

            self.auth_failures_total = Counter(
                'tars_auth_failures_total',
                'Total authentication failures',
                ['service', 'reason']
            )

            self.authz_denials_total = Counter(
                'tars_authz_denials_total',
                'Total authorization denials',
                ['service', 'endpoint', 'username']
            )

            self.rate_limit_events_total = Counter(
                'tars_rate_limit_events_total',
                'Total rate limit events',
                ['service', 'endpoint']
            )

            self.metrics_available = True
            logger.info("Audit logger Prometheus metrics initialized")

        except ImportError:
            self.metrics_available = False
            logger.warning("Prometheus client not available. Metrics disabled.")

    async def connect(self):
        """Connect to Redis backend"""
        if not self.use_redis:
            logger.info(f"Audit logger for {self.service_name} using in-memory storage")
            return

        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Audit logger for {self.service_name} connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for audit logging: {e}")
            logger.warning("Falling back to in-memory audit logging")
            self.redis_client = None

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Audit logger disconnected from Redis")
            except Exception as e:
                logger.error(f"Error disconnecting from Redis: {e}")

    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Log an audit event

        Returns:
            event_id: Unique identifier for the event
        """
        async with self._lock:
            self._event_counter += 1
            event_id = f"{self.service_name}-{datetime.utcnow().strftime('%Y%m%d')}-{self._event_counter:08d}"

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            service=self.service_name,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            resource=resource,
            action=action,
            result=result,
            details=details,
            error=error
        )

        # Update Prometheus metrics
        if self.metrics_available:
            self._update_metrics(event)

        # Log to Python logger
        self._log_to_python_logger(event)

        # Store event
        await self._store_event(event)

        return event_id

    def _update_metrics(self, event: AuditEvent):
        """Update Prometheus metrics"""
        try:
            self.audit_events_total.labels(
                service=self.service_name,
                event_type=event.event_type.value,
                severity=event.severity.value
            ).inc()

            if event.username:
                self.audit_events_by_user.labels(
                    service=self.service_name,
                    username=event.username,
                    event_type=event.event_type.value
                ).inc()

            # Update specific metrics
            if event.event_type in [
                AuditEventType.AUTH_LOGIN_FAILURE,
                AuditEventType.AUTH_TOKEN_INVALID,
                AuditEventType.API_KEY_INVALID
            ]:
                reason = event.event_type.value.split('.')[-1]
                self.auth_failures_total.labels(
                    service=self.service_name,
                    reason=reason
                ).inc()

            if event.event_type in [
                AuditEventType.AUTHZ_ACCESS_DENIED,
                AuditEventType.AUTHZ_INSUFFICIENT_PERMISSIONS
            ]:
                self.authz_denials_total.labels(
                    service=self.service_name,
                    endpoint=event.endpoint or "unknown",
                    username=event.username or "anonymous"
                ).inc()

            if event.event_type in [
                AuditEventType.RATE_LIMIT_EXCEEDED,
                AuditEventType.RATE_LIMIT_WARNING
            ]:
                self.rate_limit_events_total.labels(
                    service=self.service_name,
                    endpoint=event.endpoint or "unknown"
                ).inc()

        except Exception as e:
            logger.error(f"Failed to update audit metrics: {e}")

    def _log_to_python_logger(self, event: AuditEvent):
        """Log to Python logger"""
        log_level = {
            AuditEventSeverity.DEBUG: logging.DEBUG,
            AuditEventSeverity.INFO: logging.INFO,
            AuditEventSeverity.WARNING: logging.WARNING,
            AuditEventSeverity.ERROR: logging.ERROR,
            AuditEventSeverity.CRITICAL: logging.CRITICAL
        }.get(event.severity, logging.INFO)

        log_message = f"[AUDIT] {event.event_type.value}"
        if event.username:
            log_message += f" | User: {event.username}"
        if event.endpoint:
            log_message += f" | Endpoint: {event.method} {event.endpoint}"
        if event.resource:
            log_message += f" | Resource: {event.resource}"
        if event.action:
            log_message += f" | Action: {event.action}"
        if event.result:
            log_message += f" | Result: {event.result}"
        if event.error:
            log_message += f" | Error: {event.error}"

        logger.log(log_level, log_message)

    async def _store_event(self, event: AuditEvent):
        """Store event to Redis or memory"""
        try:
            if self.redis_client:
                await self._store_to_redis(event)
            else:
                await self._store_to_memory(event)
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            # Fall back to memory if Redis fails
            await self._store_to_memory(event)

    async def _store_to_redis(self, event: AuditEvent):
        """Store event to Redis"""
        try:
            # Store in sorted set by timestamp
            key = f"audit:events:{self.service_name}"
            score = event.timestamp.timestamp()
            value = event.to_json()

            await self.redis_client.zadd(key, {value: score})

            # Also store by event type for faster queries
            type_key = f"audit:events:{self.service_name}:{event.event_type.value}"
            await self.redis_client.zadd(type_key, {value: score})

            # Set TTL based on retention policy
            ttl_seconds = self.retention_days * 24 * 60 * 60
            await self.redis_client.expire(key, ttl_seconds)
            await self.redis_client.expire(type_key, ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to store event to Redis: {e}")
            raise

    async def _store_to_memory(self, event: AuditEvent):
        """Store event to in-memory deque"""
        self.memory_events.append(event.to_dict())

    async def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query audit events

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            username: Filter by username
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of audit events
        """
        if self.redis_client:
            return await self._get_events_from_redis(
                event_type, user_id, username, start_time, end_time, limit, offset
            )
        else:
            return await self._get_events_from_memory(
                event_type, user_id, username, start_time, end_time, limit, offset
            )

    async def _get_events_from_redis(
        self,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        username: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """Query events from Redis"""
        try:
            # Choose key based on event_type filter
            if event_type:
                key = f"audit:events:{self.service_name}:{event_type.value}"
            else:
                key = f"audit:events:{self.service_name}"

            # Convert timestamps to scores
            min_score = start_time.timestamp() if start_time else "-inf"
            max_score = end_time.timestamp() if end_time else "+inf"

            # Query Redis sorted set
            raw_events = await self.redis_client.zrangebyscore(
                key,
                min_score,
                max_score,
                start=offset,
                num=limit,
                withscores=False
            )

            # Parse JSON events
            events = []
            for raw_event in raw_events:
                try:
                    event = json.loads(raw_event)

                    # Apply additional filters
                    if user_id and event.get("user_id") != user_id:
                        continue
                    if username and event.get("username") != username:
                        continue

                    events.append(event)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse audit event: {raw_event}")

            return events

        except Exception as e:
            logger.error(f"Failed to query events from Redis: {e}")
            return []

    async def _get_events_from_memory(
        self,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        username: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """Query events from in-memory storage"""
        events = list(self.memory_events)

        # Apply filters
        filtered_events = []
        for event in events:
            # Convert timestamp string to datetime
            event_time = datetime.fromisoformat(event["timestamp"])

            if event_type and event["event_type"] != event_type.value:
                continue
            if user_id and event.get("user_id") != user_id:
                continue
            if username and event.get("username") != username:
                continue
            if start_time and event_time < start_time:
                continue
            if end_time and event_time > end_time:
                continue

            filtered_events.append(event)

        # Sort by timestamp (descending)
        filtered_events.sort(key=lambda e: e["timestamp"], reverse=True)

        # Apply pagination
        return filtered_events[offset:offset+limit]

    async def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get audit event statistics

        Returns:
            Dictionary with event counts by type, severity, user, etc.
        """
        events = await self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Get more events for stats
        )

        stats = {
            "total_events": len(events),
            "by_type": {},
            "by_severity": {},
            "by_user": {},
            "auth_failures": 0,
            "authz_denials": 0,
            "rate_limits": 0
        }

        for event in events:
            # Count by type
            event_type = event["event_type"]
            stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1

            # Count by severity
            severity = event["severity"]
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

            # Count by user
            username = event.get("username")
            if username:
                stats["by_user"][username] = stats["by_user"].get(username, 0) + 1

            # Count specific event categories
            if "login.failure" in event_type or "token.invalid" in event_type or "api_key.invalid" in event_type:
                stats["auth_failures"] += 1

            if "authz.access.denied" in event_type or "authz.insufficient_permissions" in event_type:
                stats["authz_denials"] += 1

            if "rate_limit" in event_type:
                stats["rate_limits"] += 1

        return stats


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(service_name: str = "tars") -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger

    if _audit_logger is None:
        _audit_logger = AuditLogger(service_name=service_name)

    return _audit_logger


async def init_audit_logger(service_name: str = "tars"):
    """Initialize global audit logger"""
    global _audit_logger

    _audit_logger = AuditLogger(service_name=service_name)
    await _audit_logger.connect()

    # Log system startup
    await _audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_STARTUP,
        severity=AuditEventSeverity.INFO,
        action="service_start",
        result="success"
    )

    return _audit_logger


async def shutdown_audit_logger():
    """Shutdown global audit logger"""
    global _audit_logger

    if _audit_logger:
        # Log system shutdown
        await _audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            severity=AuditEventSeverity.INFO,
            action="service_stop",
            result="success"
        )

        await _audit_logger.disconnect()
        _audit_logger = None
