"""
Jaeger Trace Context Propagation Tests

TARS-1003: Jaeger Trace Context Tests
--------------------------------------
Tests for validating trace context propagation through Redis Streams.

Success Criteria:
- 100% parent-child span linking
- <1ms trace context overhead per message
- Multi-region trace continuity
- No trace breaks in Redis Streams

Test Coverage:
1. W3C Trace Context parsing and formatting
2. Trace context injection into Redis messages
3. Trace context extraction from Redis messages
4. Parent-child span linking
5. Multi-region trace propagation
6. Performance overhead measurement

Author: T.A.R.S. Engineering Team
Version: 1.0.1
Date: 2025-11-20
"""

import pytest
import json
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import hashlib

# Import the trace context propagation fix
from trace_context_patch import (
    TraceContext,
    RedisStreamsTraceContextPropagator,
    MultiRegionTracePropagator
)


# =================================================================
# TEST CONFIGURATION
# =================================================================

# Performance thresholds
TRACE_INJECTION_OVERHEAD_MS = 1  # <1ms overhead
TRACE_EXTRACTION_OVERHEAD_MS = 1  # <1ms overhead

# Sample trace IDs (W3C format)
SAMPLE_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SAMPLE_SPAN_ID = "00f067aa0ba902b7"
SAMPLE_TRACE_FLAGS = "01"

# =================================================================
# MOCK REDIS CLIENT
# =================================================================

class MockRedisClient:
    """Mock Redis client for testing"""

    def __init__(self):
        self.streams: Dict[str, List[tuple]] = {}
        self.message_counter = 0

    def xadd(self, stream_key: str, message: Dict[str, Any]) -> str:
        """Add message to stream"""
        self.message_counter += 1
        message_id = f"{int(time.time() * 1000)}-{self.message_counter}"

        if stream_key not in self.streams:
            self.streams[stream_key] = []

        self.streams[stream_key].append((message_id, message))

        return message_id

    def xread(
        self,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None
    ) -> List[tuple]:
        """Read messages from streams"""
        results = []

        for stream_key, last_id in streams.items():
            if stream_key in self.streams:
                messages = self.streams[stream_key]
                if count:
                    messages = messages[:count]
                results.append((stream_key, messages))

        return results

    def reset(self):
        """Reset mock state"""
        self.streams = {}
        self.message_counter = 0


# =================================================================
# MOCK JAEGER SPAN
# =================================================================

class MockSpanContext:
    """Mock Jaeger span context"""

    def __init__(self, trace_id: int, span_id: int, parent_id: Optional[int] = None, flags: int = 1):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.flags = flags


class MockSpan:
    """Mock Jaeger span"""

    def __init__(self, operation_name: str, context: MockSpanContext):
        self.operation_name = operation_name
        self.context = context
        self.tags: Dict[str, Any] = {}
        self.is_finished = False

    def set_tag(self, key: str, value: Any):
        """Set span tag"""
        self.tags[key] = value
        return self

    def finish(self):
        """Finish span"""
        self.is_finished = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


# =================================================================
# W3C TRACE CONTEXT TESTS
# =================================================================

class TestTraceContext:
    """Test W3C Trace Context parsing and formatting"""

    def test_traceparent_format(self):
        """Test traceparent header formatting"""
        trace_context = TraceContext(
            trace_id=SAMPLE_TRACE_ID,
            span_id=SAMPLE_SPAN_ID,
            trace_flags=SAMPLE_TRACE_FLAGS
        )

        traceparent = trace_context.traceparent

        assert traceparent == f"00-{SAMPLE_TRACE_ID}-{SAMPLE_SPAN_ID}-{SAMPLE_TRACE_FLAGS}"

    def test_traceparent_parsing(self):
        """Test traceparent header parsing"""
        traceparent = f"00-{SAMPLE_TRACE_ID}-{SAMPLE_SPAN_ID}-{SAMPLE_TRACE_FLAGS}"

        trace_context = TraceContext.from_traceparent(traceparent)

        assert trace_context is not None
        assert trace_context.trace_id == SAMPLE_TRACE_ID
        assert trace_context.span_id == SAMPLE_SPAN_ID
        assert trace_context.trace_flags == SAMPLE_TRACE_FLAGS

    def test_traceparent_invalid_format(self):
        """Test invalid traceparent handling"""
        invalid_traceparents = [
            "invalid",
            "00-12345",  # Too few parts
            "01-{SAMPLE_TRACE_ID}-{SAMPLE_SPAN_ID}-01",  # Wrong version
            "",
            None
        ]

        for invalid in invalid_traceparents:
            trace_context = TraceContext.from_traceparent(invalid) if invalid else None
            assert trace_context is None or trace_context.trace_id != SAMPLE_TRACE_ID

    def test_trace_context_to_dict(self):
        """Test trace context serialization to dict"""
        trace_context = TraceContext(
            trace_id=SAMPLE_TRACE_ID,
            span_id=SAMPLE_SPAN_ID,
            trace_flags=SAMPLE_TRACE_FLAGS,
            trace_state="rojo=00f067aa0ba902b7"
        )

        data = trace_context.to_dict()

        assert "traceparent" in data
        assert data["traceparent"] == trace_context.traceparent
        assert "tracestate" in data
        assert data["tracestate"] == "rojo=00f067aa0ba902b7"

    def test_trace_context_from_dict(self):
        """Test trace context deserialization from dict"""
        data = {
            "traceparent": f"00-{SAMPLE_TRACE_ID}-{SAMPLE_SPAN_ID}-{SAMPLE_TRACE_FLAGS}",
            "tracestate": "rojo=00f067aa0ba902b7"
        }

        trace_context = TraceContext.from_dict(data)

        assert trace_context is not None
        assert trace_context.trace_id == SAMPLE_TRACE_ID
        assert trace_context.span_id == SAMPLE_SPAN_ID
        assert trace_context.trace_state == "rojo=00f067aa0ba902b7"

    def test_trace_context_roundtrip(self):
        """Test trace context serialization roundtrip"""
        original = TraceContext(
            trace_id=SAMPLE_TRACE_ID,
            span_id=SAMPLE_SPAN_ID,
            trace_flags=SAMPLE_TRACE_FLAGS
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        reconstructed = TraceContext.from_dict(data)

        assert reconstructed.trace_id == original.trace_id
        assert reconstructed.span_id == original.span_id
        assert reconstructed.trace_flags == original.trace_flags


# =================================================================
# TRACE INJECTION TESTS
# =================================================================

class TestTraceInjection:
    """Test trace context injection into Redis messages"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis client"""
        return MockRedisClient()

    @pytest.fixture
    def propagator(self, redis_client):
        """Trace context propagator"""
        return RedisStreamsTraceContextPropagator(redis_client)

    @pytest.fixture
    def mock_span(self):
        """Mock Jaeger span"""
        context = MockSpanContext(
            trace_id=int(SAMPLE_TRACE_ID, 16),
            span_id=int(SAMPLE_SPAN_ID, 16),
            flags=int(SAMPLE_TRACE_FLAGS, 16)
        )
        return MockSpan("test_operation", context)

    def test_inject_trace_context(self, propagator, mock_span):
        """Test trace context injection into message"""
        message = {"data": "test_value"}

        message_with_trace = propagator.inject_trace_context(message, mock_span)

        assert "_trace_context" in message_with_trace
        assert message_with_trace["data"] == "test_value"

        # Parse trace context
        trace_json = json.loads(message_with_trace["_trace_context"])
        assert "traceparent" in trace_json

    def test_inject_preserves_original_data(self, propagator, mock_span):
        """Test injection preserves original message data"""
        message = {
            "agent_id": "dqn_agent_1",
            "action": "update",
            "data": {"weights": [0.1, 0.2, 0.3]}
        }

        message_with_trace = propagator.inject_trace_context(message, mock_span)

        assert message_with_trace["agent_id"] == "dqn_agent_1"
        assert message_with_trace["action"] == "update"
        assert message_with_trace["data"] == {"weights": [0.1, 0.2, 0.3]}

    def test_inject_without_span(self, propagator):
        """Test injection without active span (should not crash)"""
        message = {"data": "test"}

        message_with_trace = propagator.inject_trace_context(message, None)

        # Should return message unchanged (or with empty trace context)
        assert "data" in message_with_trace

    def test_inject_performance_overhead(self, propagator, mock_span):
        """Test trace injection performance overhead"""
        message = {"data": "test"}

        start_time = time.time()

        for _ in range(1000):
            propagator.inject_trace_context(message.copy(), mock_span)

        elapsed_ms = (time.time() - start_time) * 1000
        avg_overhead_ms = elapsed_ms / 1000

        assert avg_overhead_ms < TRACE_INJECTION_OVERHEAD_MS, \
            f"Trace injection overhead {avg_overhead_ms:.2f}ms exceeds threshold {TRACE_INJECTION_OVERHEAD_MS}ms"


# =================================================================
# TRACE EXTRACTION TESTS
# =================================================================

class TestTraceExtraction:
    """Test trace context extraction from Redis messages"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis client"""
        return MockRedisClient()

    @pytest.fixture
    def propagator(self, redis_client):
        """Trace context propagator"""
        return RedisStreamsTraceContextPropagator(redis_client)

    def test_extract_trace_context(self, propagator):
        """Test trace context extraction from message"""
        # Create message with trace context
        trace_context = TraceContext(
            trace_id=SAMPLE_TRACE_ID,
            span_id=SAMPLE_SPAN_ID,
            trace_flags=SAMPLE_TRACE_FLAGS
        )

        message = {
            "data": "test",
            "_trace_context": json.dumps(trace_context.to_dict())
        }

        extracted = propagator.extract_trace_context(message)

        assert extracted is not None
        assert extracted.trace_id == SAMPLE_TRACE_ID
        assert extracted.span_id == SAMPLE_SPAN_ID

    def test_extract_without_trace_context(self, propagator):
        """Test extraction from message without trace context"""
        message = {"data": "test"}

        extracted = propagator.extract_trace_context(message)

        assert extracted is None

    def test_extract_with_invalid_trace_context(self, propagator):
        """Test extraction with malformed trace context"""
        message = {
            "data": "test",
            "_trace_context": "invalid_json"
        }

        extracted = propagator.extract_trace_context(message)

        assert extracted is None

    def test_extract_performance_overhead(self, propagator):
        """Test trace extraction performance overhead"""
        trace_context = TraceContext(
            trace_id=SAMPLE_TRACE_ID,
            span_id=SAMPLE_SPAN_ID,
            trace_flags=SAMPLE_TRACE_FLAGS
        )

        message = {
            "data": "test",
            "_trace_context": json.dumps(trace_context.to_dict())
        }

        start_time = time.time()

        for _ in range(1000):
            propagator.extract_trace_context(message)

        elapsed_ms = (time.time() - start_time) * 1000
        avg_overhead_ms = elapsed_ms / 1000

        assert avg_overhead_ms < TRACE_EXTRACTION_OVERHEAD_MS, \
            f"Trace extraction overhead {avg_overhead_ms:.2f}ms exceeds threshold {TRACE_EXTRACTION_OVERHEAD_MS}ms"


# =================================================================
# END-TO-END TRACE PROPAGATION TESTS
# =================================================================

class TestEndToEndTracePropagation:
    """Test complete trace propagation through Redis Streams"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis client"""
        return MockRedisClient()

    @pytest.fixture
    def propagator(self, redis_client):
        """Trace context propagator"""
        return RedisStreamsTraceContextPropagator(redis_client)

    @pytest.fixture
    def mock_span(self):
        """Mock Jaeger span"""
        context = MockSpanContext(
            trace_id=int(SAMPLE_TRACE_ID, 16),
            span_id=int(SAMPLE_SPAN_ID, 16),
            flags=int(SAMPLE_TRACE_FLAGS, 16)
        )
        return MockSpan("producer_operation", context)

    def test_producer_consumer_trace_continuity(
        self,
        redis_client,
        propagator,
        mock_span
    ):
        """Test trace continuity from producer to consumer"""
        # Producer: Inject trace context
        message = {"agent_id": "dqn_agent_1", "action": "update"}
        message_with_trace = propagator.inject_trace_context(message, mock_span)

        # Send to Redis Stream
        message_id = redis_client.xadd("test_stream", message_with_trace)

        # Consumer: Read message
        messages = redis_client.xread({"test_stream": ">"})

        assert len(messages) > 0

        stream, msg_list = messages[0]
        msg_id, message_data = msg_list[0]

        # Extract trace context
        extracted_trace = propagator.extract_trace_context(message_data)

        assert extracted_trace is not None
        assert extracted_trace.trace_id == SAMPLE_TRACE_ID
        assert extracted_trace.span_id == SAMPLE_SPAN_ID

        # Verify trace continuity (same trace_id)
        assert extracted_trace.trace_id == SAMPLE_TRACE_ID

    def test_multiple_messages_preserve_trace_context(
        self,
        redis_client,
        propagator,
        mock_span
    ):
        """Test multiple messages maintain separate trace contexts"""
        # Send 3 messages with different trace contexts
        for i in range(3):
            # Create unique span for each message
            context = MockSpanContext(
                trace_id=int(SAMPLE_TRACE_ID, 16) + i,
                span_id=int(SAMPLE_SPAN_ID, 16) + i,
                flags=1
            )
            span = MockSpan(f"operation_{i}", context)

            message = {"message_num": i}
            message_with_trace = propagator.inject_trace_context(message, span)
            redis_client.xadd("test_stream", message_with_trace)

        # Read all messages
        messages = redis_client.xread({"test_stream": ">"})
        stream, msg_list = messages[0]

        # Verify each message has correct trace context
        for idx, (msg_id, message_data) in enumerate(msg_list):
            extracted = propagator.extract_trace_context(message_data)

            expected_trace_id = format(int(SAMPLE_TRACE_ID, 16) + idx, '032x')
            assert extracted.trace_id == expected_trace_id


# =================================================================
# MULTI-REGION TRACE PROPAGATION TESTS
# =================================================================

class TestMultiRegionTracePropagation:
    """Test trace propagation across regions"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis client"""
        return MockRedisClient()

    @pytest.fixture
    def us_west_propagator(self, redis_client):
        """US-West-2 propagator"""
        return MultiRegionTracePropagator(redis_client, "us-west-2")

    @pytest.fixture
    def eu_central_propagator(self, redis_client):
        """EU-Central-1 propagator"""
        return MultiRegionTracePropagator(redis_client, "eu-central-1")

    @pytest.fixture
    def mock_span(self):
        """Mock Jaeger span"""
        context = MockSpanContext(
            trace_id=int(SAMPLE_TRACE_ID, 16),
            span_id=int(SAMPLE_SPAN_ID, 16),
            flags=1
        )
        return MockSpan("cross_region_operation", context)

    def test_cross_region_message_send(
        self,
        redis_client,
        us_west_propagator,
        mock_span
    ):
        """Test sending cross-region message with trace context"""
        message = {"data": "cross_region_update"}

        message_id = us_west_propagator.send_cross_region_message(
            target_region="eu-central-1",
            stream_key="cross_region_stream",
            message=message,
            current_span=mock_span
        )

        assert message_id is not None

        # Verify message in stream
        messages = redis_client.xread({"cross_region_stream": ">"})
        assert len(messages) > 0

        stream, msg_list = messages[0]
        msg_id, message_data = msg_list[0]

        # Verify region metadata
        assert message_data["_source_region"] == "us-west-2"
        assert message_data["_target_region"] == "eu-central-1"

        # Verify trace context
        assert "_trace_context" in message_data

    def test_cross_region_trace_continuity(
        self,
        redis_client,
        us_west_propagator,
        eu_central_propagator,
        mock_span
    ):
        """Test trace continuity across regions"""
        # US-West-2: Send message
        message = {"data": "test"}
        message_id = us_west_propagator.send_cross_region_message(
            target_region="eu-central-1",
            stream_key="cross_region_stream",
            message=message,
            current_span=mock_span
        )

        # EU-Central-1: Receive message
        messages = redis_client.xread({"cross_region_stream": ">"})
        stream, msg_list = messages[0]
        msg_id, message_data = msg_list[0]

        # Extract trace context
        trace_context = eu_central_propagator.propagator.extract_trace_context(message_data)

        # Verify trace continuity (same trace_id)
        assert trace_context is not None
        assert trace_context.trace_id == SAMPLE_TRACE_ID

    def test_multi_hop_trace_propagation(
        self,
        redis_client,
        us_west_propagator,
        eu_central_propagator,
        mock_span
    ):
        """Test trace propagation across multiple hops"""
        # Hop 1: US-West-2 → EU-Central-1
        message = {"data": "hop1"}
        us_west_propagator.send_cross_region_message(
            target_region="eu-central-1",
            stream_key="hop1_stream",
            message=message,
            current_span=mock_span
        )

        # EU-Central-1 receives and forwards to another region
        messages = redis_client.xread({"hop1_stream": ">"})
        stream, msg_list = messages[0]
        msg_id, message_data = msg_list[0]

        # Extract trace context for forwarding
        trace_context = eu_central_propagator.propagator.extract_trace_context(message_data)

        # Create child span with extracted context
        child_context = MockSpanContext(
            trace_id=int(trace_context.trace_id, 16),
            span_id=int(trace_context.span_id, 16) + 1,
            parent_id=int(trace_context.span_id, 16),
            flags=1
        )
        child_span = MockSpan("forward_operation", child_context)

        # Hop 2: EU-Central-1 → another region
        eu_central_propagator.send_cross_region_message(
            target_region="ap-south-1",
            stream_key="hop2_stream",
            message={"data": "hop2"},
            current_span=child_span
        )

        # Verify trace_id continuity across both hops
        messages_hop2 = redis_client.xread({"hop2_stream": ">"})
        stream2, msg_list2 = messages_hop2[0]
        msg_id2, message_data2 = msg_list2[0]

        trace_context2 = eu_central_propagator.propagator.extract_trace_context(message_data2)

        # Same trace_id across all hops
        assert trace_context2.trace_id == SAMPLE_TRACE_ID


# =================================================================
# PERFORMANCE TESTS
# =================================================================

class TestPerformance:
    """Test performance overhead of trace propagation"""

    @pytest.fixture
    def redis_client(self):
        """Mock Redis client"""
        return MockRedisClient()

    @pytest.fixture
    def propagator(self, redis_client):
        """Trace context propagator"""
        return RedisStreamsTraceContextPropagator(redis_client)

    def test_end_to_end_overhead(self, redis_client, propagator):
        """Test total end-to-end overhead"""
        context = MockSpanContext(
            trace_id=int(SAMPLE_TRACE_ID, 16),
            span_id=int(SAMPLE_SPAN_ID, 16),
            flags=1
        )
        span = MockSpan("test", context)

        iterations = 1000
        message = {"data": "test"}

        start_time = time.time()

        for _ in range(iterations):
            # Inject
            msg_with_trace = propagator.inject_trace_context(message.copy(), span)
            # Send
            redis_client.xadd("test_stream", msg_with_trace)
            # Read
            messages = redis_client.xread({"test_stream": ">"})
            # Extract
            propagator.extract_trace_context(messages[0][1][-1][1])

        elapsed_ms = (time.time() - start_time) * 1000
        avg_overhead_ms = elapsed_ms / iterations

        print(f"\nEnd-to-end trace overhead: {avg_overhead_ms:.2f}ms per message")

        # Total overhead should be <2ms per message
        assert avg_overhead_ms < 2.0, \
            f"End-to-end overhead {avg_overhead_ms:.2f}ms exceeds 2ms threshold"


# =================================================================
# MAIN TEST EXECUTION
# =================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
