"""
Instrumentation Package for T.A.R.S. Evaluation Engine

Provides distributed tracing, metrics, and observability utilities.
"""
from .tracing import (
    setup_tracing,
    get_tracer,
    tracer,
    add_evaluation_attributes,
    add_metrics_attributes,
    add_regression_attributes,
    add_nash_attributes,
    record_exception,
    get_current_trace_id,
    get_current_span_id,
    get_jaeger_trace_url
)

__all__ = [
    "setup_tracing",
    "get_tracer",
    "tracer",
    "add_evaluation_attributes",
    "add_metrics_attributes",
    "add_regression_attributes",
    "add_nash_attributes",
    "record_exception",
    "get_current_trace_id",
    "get_current_span_id",
    "get_jaeger_trace_url"
]
