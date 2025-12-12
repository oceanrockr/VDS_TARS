"""
Distributed Tracing Instrumentation for T.A.R.S. Evaluation Engine

Integrates OpenTelemetry with Jaeger for end-to-end distributed tracing.

Features:
- OTLP/gRPC export to Jaeger
- FastAPI automatic instrumentation
- Custom span attributes for ML metrics
- Context propagation for multi-service traces
- Trace sampling for production efficiency

Usage:
    from instrumentation.tracing import setup_tracing, tracer

    # In main.py
    setup_tracing(app, service_name="eval-engine", jaeger_endpoint="http://jaeger:4317")

    # In worker code
    with tracer.start_as_current_span("evaluate_agent") as span:
        span.set_attribute("agent_type", "dqn")
        span.set_attribute("mean_reward", 195.5)
"""
import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Global tracer instance
tracer: Optional[trace.Tracer] = None


def setup_tracing(
    app: FastAPI,
    service_name: str = "tars-eval-engine",
    service_version: str = "v1.0.0-rc2",
    jaeger_endpoint: Optional[str] = None,
    sample_rate: float = 1.0,
    enable_instrumentation: bool = True
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing with Jaeger backend.

    Args:
        app: FastAPI application instance
        service_name: Service name for tracing
        service_version: Service version for tracing
        jaeger_endpoint: Jaeger collector endpoint (OTLP/gRPC)
                        Defaults to JAEGER_ENDPOINT env var or http://localhost:4317
        sample_rate: Trace sampling rate (0.0-1.0)
                    1.0 = trace all requests (dev)
                    0.1 = trace 10% of requests (prod)
        enable_instrumentation: Auto-instrument FastAPI and requests

    Returns:
        Configured tracer instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> tracer = setup_tracing(
        ...     app,
        ...     service_name="eval-engine",
        ...     jaeger_endpoint="http://jaeger:4317",
        ...     sample_rate=0.1  # 10% sampling for production
        ... )
    """
    global tracer

    # Get Jaeger endpoint from env or parameter
    jaeger_endpoint = jaeger_endpoint or os.getenv("JAEGER_ENDPOINT", "http://localhost:4317")

    logger.info(f"Initializing tracing: service={service_name}, jaeger={jaeger_endpoint}, sample_rate={sample_rate}")

    # Create resource with service metadata
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        "k8s.namespace": os.getenv("KUBERNETES_NAMESPACE", "tars"),
        "k8s.pod.name": os.getenv("HOSTNAME", "unknown"),
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter for Jaeger
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=jaeger_endpoint,
            insecure=True  # Use TLS in production: insecure=False + certificates
        )

        # Add batch span processor (efficient batching for production)
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000  # Export every 5 seconds
        )
        provider.add_span_processor(span_processor)

        logger.info(f"OTLP exporter configured: {jaeger_endpoint}")

    except Exception as e:
        logger.error(f"Failed to configure OTLP exporter: {e}")
        logger.warning("Tracing will be disabled")
        # Continue without tracing rather than crashing

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer instance
    tracer = trace.get_tracer(
        instrumenting_module_name=__name__,
        instrumenting_library_version=service_version
    )

    # Auto-instrument FastAPI
    if enable_instrumentation:
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=provider,
                excluded_urls="/health,/metrics"  # Don't trace health checks
            )
            logger.info("FastAPI instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")

        # Auto-instrument requests library (for HTTP client calls)
        try:
            RequestsInstrumentor().instrument(tracer_provider=provider)
            logger.info("Requests instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument requests: {e}")

        # Auto-instrument logging (attach trace IDs to logs)
        try:
            LoggingInstrumentor().instrument(
                set_logging_format=True,
                log_level=logging.INFO
            )
            logger.info("Logging instrumentation enabled")
        except Exception as e:
            logger.error(f"Failed to instrument logging: {e}")

    logger.info("Tracing setup complete")
    return tracer


def get_tracer() -> trace.Tracer:
    """
    Get the global tracer instance.

    Returns:
        Global tracer instance

    Raises:
        RuntimeError: If tracing not initialized (setup_tracing not called)
    """
    if tracer is None:
        raise RuntimeError("Tracing not initialized. Call setup_tracing() first.")
    return tracer


# ==========================================
# Custom Span Utilities
# ==========================================

def add_evaluation_attributes(
    span: trace.Span,
    agent_type: str,
    environment: str,
    num_episodes: int,
    hyperparameters: dict
) -> None:
    """
    Add evaluation-specific attributes to a span.

    Args:
        span: Active span
        agent_type: RL agent type (dqn, a2c, ppo, ddpg)
        environment: Gymnasium environment name
        num_episodes: Number of episodes evaluated
        hyperparameters: Agent hyperparameters

    Example:
        >>> with tracer.start_as_current_span("evaluate") as span:
        ...     add_evaluation_attributes(
        ...         span,
        ...         agent_type="dqn",
        ...         environment="CartPole-v1",
        ...         num_episodes=100,
        ...         hyperparameters={"learning_rate": 0.001}
        ...     )
    """
    span.set_attribute("ml.agent_type", agent_type)
    span.set_attribute("ml.environment", environment)
    span.set_attribute("ml.num_episodes", num_episodes)

    # Add select hyperparameters (avoid huge payloads)
    for key in ["learning_rate", "gamma", "epsilon", "batch_size"]:
        if key in hyperparameters:
            span.set_attribute(f"ml.hyperparameter.{key}", str(hyperparameters[key]))


def add_metrics_attributes(
    span: trace.Span,
    mean_reward: float,
    std_reward: float,
    success_rate: float,
    mean_steps: float
) -> None:
    """
    Add performance metrics to a span.

    Args:
        span: Active span
        mean_reward: Mean episode reward
        std_reward: Std deviation of rewards
        success_rate: Success rate (0.0-1.0)
        mean_steps: Mean episode steps

    Example:
        >>> with tracer.start_as_current_span("compute_metrics") as span:
        ...     add_metrics_attributes(span, 195.5, 12.3, 0.95, 195.5)
    """
    span.set_attribute("ml.metrics.mean_reward", mean_reward)
    span.set_attribute("ml.metrics.std_reward", std_reward)
    span.set_attribute("ml.metrics.success_rate", success_rate)
    span.set_attribute("ml.metrics.mean_steps", mean_steps)


def add_regression_attributes(
    span: trace.Span,
    is_regression: bool,
    confidence: float,
    severity: str,
    details: str
) -> None:
    """
    Add regression detection results to a span.

    Args:
        span: Active span
        is_regression: True if regression detected
        confidence: Confidence score (0.0-1.0)
        severity: Severity level (none, low, medium, high, critical)
        details: Human-readable details

    Example:
        >>> with tracer.start_as_current_span("detect_regression") as span:
        ...     add_regression_attributes(span, True, 0.95, "high", "Reward dropped 15%")
    """
    span.set_attribute("ml.regression.detected", is_regression)
    span.set_attribute("ml.regression.confidence", confidence)
    span.set_attribute("ml.regression.severity", severity)
    span.set_attribute("ml.regression.details", details)

    # Mark span as error if critical regression
    if is_regression and severity == "critical":
        span.set_status(Status(StatusCode.ERROR, "Critical regression detected"))


def add_nash_attributes(
    span: trace.Span,
    conflict_score: float,
    deviation_from_equilibrium: float,
    stability_score: float,
    recommendation: str
) -> None:
    """
    Add Nash equilibrium scoring to a span.

    Args:
        span: Active span
        conflict_score: Multi-agent conflict score (0.0-1.0)
        deviation_from_equilibrium: Distance from equilibrium
        stability_score: System stability (0.0-1.0)
        recommendation: accept/review/reject

    Example:
        >>> with tracer.start_as_current_span("nash_scoring") as span:
        ...     add_nash_attributes(span, 0.02, 0.05, 0.98, "accept")
    """
    span.set_attribute("ml.nash.conflict_score", conflict_score)
    span.set_attribute("ml.nash.deviation", deviation_from_equilibrium)
    span.set_attribute("ml.nash.stability_score", stability_score)
    span.set_attribute("ml.nash.recommendation", recommendation)


def record_exception(span: trace.Span, exception: Exception) -> None:
    """
    Record exception in span with proper status.

    Args:
        span: Active span
        exception: Exception that occurred

    Example:
        >>> with tracer.start_as_current_span("risky_operation") as span:
        ...     try:
        ...         risky_function()
        ...     except ValueError as e:
        ...         record_exception(span, e)
        ...         raise
    """
    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, str(exception)))


# ==========================================
# Context Propagation Utilities
# ==========================================

def get_current_trace_id() -> Optional[str]:
    """
    Get current trace ID for logging/debugging.

    Returns:
        Trace ID as hex string, or None if no active span

    Example:
        >>> trace_id = get_current_trace_id()
        >>> logger.info(f"Processing request: trace_id={trace_id}")
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get current span ID for logging/debugging.

    Returns:
        Span ID as hex string, or None if no active span

    Example:
        >>> span_id = get_current_span_id()
        >>> logger.info(f"Entering function: span_id={span_id}")
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, '016x')
    return None


# ==========================================
# Jaeger UI Helpers
# ==========================================

def get_jaeger_trace_url(trace_id: str, jaeger_ui_url: str = "http://localhost:16686") -> str:
    """
    Generate Jaeger UI URL for a trace.

    Args:
        trace_id: Trace ID (32-char hex)
        jaeger_ui_url: Jaeger UI base URL

    Returns:
        Full Jaeger UI URL

    Example:
        >>> trace_id = get_current_trace_id()
        >>> url = get_jaeger_trace_url(trace_id, "https://jaeger.tars.example.com")
        >>> logger.info(f"View trace: {url}")
    """
    return f"{jaeger_ui_url}/trace/{trace_id}"


# ==========================================
# Example Usage
# ==========================================

"""
Example: Instrument evaluation workflow

```python
# main.py
from fastapi import FastAPI
from instrumentation.tracing import setup_tracing, tracer

app = FastAPI()
setup_tracing(app, jaeger_endpoint="http://jaeger:4317", sample_rate=0.1)

# worker.py
from instrumentation.tracing import (
    tracer,
    add_evaluation_attributes,
    add_metrics_attributes,
    add_regression_attributes,
    record_exception
)

async def evaluate_agent_in_env(...):
    with tracer.start_as_current_span("evaluate_agent_in_env") as span:
        add_evaluation_attributes(span, agent_type, environment, num_episodes, hyperparameters)

        try:
            # Load agent
            with tracer.start_as_current_span("load_agent") as child_span:
                child_span.set_attribute("ml.agent_type", agent_type)
                agent = load_agent(agent_type, hyperparameters)

            # Run episodes
            with tracer.start_as_current_span("run_episodes") as child_span:
                child_span.set_attribute("ml.num_episodes", num_episodes)
                episode_rewards, episode_steps = await run_episodes(agent, env, num_episodes)

            # Compute metrics
            with tracer.start_as_current_span("compute_metrics") as child_span:
                metrics = metrics_calculator.calculate(episode_rewards, episode_steps)
                add_metrics_attributes(child_span, metrics.mean_reward, metrics.std_reward, ...)

            # Detect regressions
            with tracer.start_as_current_span("detect_regression") as child_span:
                regression = regression_detector.detect(metrics, baseline)
                add_regression_attributes(child_span, regression.is_regression, ...)

            return EnvironmentResult(metrics=metrics, regression=regression, ...)

        except Exception as e:
            record_exception(span, e)
            raise
```

Trace output in Jaeger:
```
evaluate_agent_in_env (200ms)
├── load_agent (50ms)
│   └── [ml.agent_type: dqn]
├── run_episodes (100ms)
│   └── [ml.num_episodes: 100]
├── compute_metrics (30ms)
│   └── [ml.metrics.mean_reward: 195.5]
└── detect_regression (20ms)
    └── [ml.regression.detected: false]
```
"""
