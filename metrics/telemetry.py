"""
Telemetry Collector for T.A.R.S. Internal Metrics

Tracks:
- CLI command execution (duration, exit codes)
- API request latency
- Error events
- Report generation metrics
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import time
import functools
import json

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway, start_http_server


class TelemetryCollector:
    """
    Collects internal telemetry for T.A.R.S. operations.

    Metrics:
    - CLI command duration
    - CLI exit codes
    - API request count and latency
    - Error events
    - Report sizes
    """

    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: int = 9101,
        log_file: Optional[Path] = None,
    ):
        """
        Initialize telemetry collector.

        Args:
            enable_prometheus: Enable Prometheus metrics export
            prometheus_port: Port for Prometheus metrics server
            log_file: Path to telemetry log file (JSONL format)
        """
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.log_file = log_file

        # Prometheus registry
        self.registry = CollectorRegistry()

        # CLI metrics
        self.cli_command_duration = Histogram(
            'tars_cli_command_duration_seconds',
            'CLI command execution duration',
            ['command'],
            registry=self.registry,
        )

        self.cli_command_total = Counter(
            'tars_cli_command_total',
            'Total CLI command executions',
            ['command', 'status'],
            registry=self.registry,
        )

        # API metrics
        self.api_request_duration = Histogram(
            'tars_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint', 'status'],
            registry=self.registry,
        )

        self.api_request_total = Counter(
            'tars_api_request_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry,
        )

        # Error metrics
        self.error_total = Counter(
            'tars_error_total',
            'Total errors',
            ['component', 'error_type'],
            registry=self.registry,
        )

        # Report metrics
        self.report_generation_duration = Histogram(
            'tars_report_generation_duration_seconds',
            'Report generation duration',
            ['report_type'],
            registry=self.registry,
        )

        self.report_size_bytes = Gauge(
            'tars_report_size_bytes',
            'Report file size',
            ['report_type'],
            registry=self.registry,
        )

        # Start Prometheus server if enabled
        if self.enable_prometheus:
            self._start_prometheus_server()

        # Ensure log file parent directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def track_cli_command(
        self,
        command: str,
        duration_seconds: float,
        exit_code: int,
    ):
        """
        Track CLI command execution.

        Args:
            command: Command name
            duration_seconds: Execution duration
            exit_code: Command exit code
        """
        status = "success" if exit_code == 0 else "error"

        # Prometheus metrics
        self.cli_command_duration.labels(command=command).observe(duration_seconds)
        self.cli_command_total.labels(command=command, status=status).inc()

        # Log to file
        self._log_event({
            "event_type": "cli_command",
            "command": command,
            "duration_seconds": duration_seconds,
            "exit_code": exit_code,
            "status": status,
        })

    def track_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ):
        """
        Track API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration_seconds: Request duration
        """
        # Prometheus metrics
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint,
            status=status_code
        ).observe(duration_seconds)

        self.api_request_total.labels(
            method=method,
            endpoint=endpoint,
            status=status_code
        ).inc()

        # Log to file
        self._log_event({
            "event_type": "api_request",
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_seconds": duration_seconds,
        })

    def track_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Track error event.

        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        # Prometheus metrics
        self.error_total.labels(component=component, error_type=error_type).inc()

        # Log to file
        self._log_event({
            "event_type": "error",
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        })

    def track_report_generation(
        self,
        report_type: str,
        duration_seconds: float,
        file_size_bytes: int,
    ):
        """
        Track report generation.

        Args:
            report_type: Type of report (ga_kpi, retrospective, etc.)
            duration_seconds: Generation duration
            file_size_bytes: Report file size
        """
        # Prometheus metrics
        self.report_generation_duration.labels(report_type=report_type).observe(duration_seconds)
        self.report_size_bytes.labels(report_type=report_type).set(file_size_bytes)

        # Log to file
        self._log_event({
            "event_type": "report_generation",
            "report_type": report_type,
            "duration_seconds": duration_seconds,
            "file_size_bytes": file_size_bytes,
        })

    def _start_prometheus_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(self.prometheus_port, registry=self.registry)
            print(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            print(f"Failed to start Prometheus server: {e}")

    def _log_event(self, event: Dict[str, Any]):
        """Log event to JSONL file."""
        if not self.log_file:
            return

        event["timestamp"] = datetime.utcnow().isoformat() + "Z"

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")


# Global telemetry instance
_telemetry: Optional[TelemetryCollector] = None


def get_telemetry() -> TelemetryCollector:
    """Get global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryCollector(
            enable_prometheus=False,  # Disabled by default
            log_file=Path("logs/tars_telemetry.log"),
        )
    return _telemetry


def track_command(command_name: str):
    """
    Decorator to track CLI command execution.

    Usage:
        @track_command("ga_kpi_collector")
        def main():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            exit_code = 0

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                exit_code = 1
                track_error(
                    component=command_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise
            finally:
                duration = time.time() - start_time
                get_telemetry().track_cli_command(
                    command=command_name,
                    duration_seconds=duration,
                    exit_code=exit_code,
                )

        return wrapper
    return decorator


def track_error(
    component: str,
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Track error event.

    Args:
        component: Component where error occurred
        error_type: Type of error
        error_message: Error message
        context: Additional context
    """
    get_telemetry().track_error(
        component=component,
        error_type=error_type,
        error_message=error_message,
        context=context,
    )
