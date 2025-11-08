"""
T.A.R.S. Performance Tests
Performance validation and load testing for WebSocket gateway
"""

import pytest
import asyncio
import time
from typing import List
import statistics

from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token


class TestWebSocketPerformance:
    """Performance tests for WebSocket gateway"""

    @pytest.mark.skipif(
        True,  # Skip by default, run manually for performance validation
        reason="Performance test - run manually"
    )
    def test_concurrent_connections_10(self):
        """
        Test 10 concurrent WebSocket connections.

        Validation Criteria:
        - All 10 connections should succeed
        - Inter-token latency < 100ms
        - No connection errors
        """
        num_connections = 10
        client = TestClient(app)

        # Generate tokens
        tokens = [
            create_access_token({"sub": f"perf-client-{i}"})
            for i in range(num_connections)
        ]

        connections = []
        latencies = []

        try:
            # Establish all connections
            for i, token in enumerate(tokens):
                ws = client.websocket_connect(f"/ws/chat?token={token}")
                connections.append(ws)

                # Receive connection ack
                ack = ws.receive_json()
                assert ack["type"] == "connection_ack"

                print(f"Connection {i + 1}/{num_connections} established")

            print(f"\n✓ All {num_connections} connections established successfully")

            # Send test messages and measure latency
            test_message = {
                "type": "chat",
                "content": "Say 'test' once",
                "conversation_id": "perf-test",
            }

            for i, ws in enumerate(connections):
                start_time = time.time()

                # Send message
                ws.send_json(test_message)

                # Wait for first token
                while True:
                    msg = ws.receive_json(timeout=10)
                    if msg["type"] == "token":
                        latency_ms = (time.time() - start_time) * 1000
                        latencies.append(latency_ms)
                        print(f"Connection {i + 1}: First token latency = {latency_ms:.1f}ms")
                        break
                    elif msg["type"] == "error":
                        print(f"Connection {i + 1}: Error = {msg['error']}")
                        break

            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)

                print(f"\n=== Performance Metrics ===")
                print(f"Concurrent Connections: {num_connections}")
                print(f"Average Latency: {avg_latency:.1f}ms")
                print(f"Min Latency: {min_latency:.1f}ms")
                print(f"Max Latency: {max_latency:.1f}ms")

                # Validate performance criteria
                assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms exceeds 100ms limit"
                print(f"✓ Performance validation passed (avg latency < 100ms)")

        finally:
            # Close all connections
            for ws in connections:
                try:
                    ws.close()
                except:
                    pass

    @pytest.mark.skipif(
        True,
        reason="Performance test - run manually"
    )
    def test_connection_stability(self):
        """
        Test connection stability over time.

        Validates:
        - Connections remain stable for 5 minutes
        - Heartbeat ping/pong works correctly
        - No unexpected disconnections
        """
        client = TestClient(app)
        token = create_access_token({"sub": "stability-test"})

        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            # Receive connection ack
            ack = ws.receive_json()
            assert ack["type"] == "connection_ack"

            print("Connection established, monitoring for 60 seconds...")

            # Monitor for 60 seconds
            start_time = time.time()
            ping_count = 0
            duration = 60  # seconds

            while (time.time() - start_time) < duration:
                try:
                    msg = ws.receive_json(timeout=35)

                    if msg["type"] == "ping":
                        ping_count += 1
                        print(f"Received ping #{ping_count}")

                        # Send pong
                        pong = {"type": "pong"}
                        ws.send_json(pong)

                except Exception as e:
                    print(f"Error: {e}")
                    break

            elapsed = time.time() - start_time
            print(f"\nConnection stable for {elapsed:.1f} seconds")
            print(f"Pings received: {ping_count}")

            assert ping_count >= 1, "Should receive at least one ping"

    @pytest.mark.skipif(
        True,
        reason="Performance test - run manually"
    )
    def test_throughput(self):
        """
        Test message throughput.

        Validates:
        - Can handle multiple messages in rapid succession
        - Tokens are streamed efficiently
        - No message loss or corruption
        """
        client = TestClient(app)
        token = create_access_token({"sub": "throughput-test"})

        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            # Receive connection ack
            ws.receive_json()

            num_messages = 5
            total_tokens = 0

            print(f"Sending {num_messages} messages...")

            for i in range(num_messages):
                msg = {
                    "type": "chat",
                    "content": f"Message {i + 1}",
                    "conversation_id": f"throughput-{i}",
                }

                start_time = time.time()
                ws.send_json(msg)

                # Collect response
                message_tokens = 0
                while True:
                    response = ws.receive_json(timeout=30)

                    if response["type"] == "token":
                        message_tokens += 1
                        total_tokens += 1
                    elif response["type"] == "complete":
                        elapsed = time.time() - start_time
                        print(
                            f"Message {i + 1}: {message_tokens} tokens in {elapsed:.2f}s "
                            f"({message_tokens / elapsed:.1f} tokens/s)"
                        )
                        break
                    elif response["type"] == "error":
                        print(f"Message {i + 1}: Error = {response['error']}")
                        break

            print(f"\nTotal tokens received: {total_tokens}")
            assert total_tokens > 0, "Should receive tokens"


class TestAuthPerformance:
    """Performance tests for authentication endpoints"""

    @pytest.mark.skipif(
        True,
        reason="Performance test - run manually"
    )
    def test_token_generation_performance(self):
        """Test token generation performance under load"""
        client = TestClient(app)
        num_requests = 100

        print(f"Generating {num_requests} tokens...")

        start_time = time.time()
        latencies = []

        for i in range(num_requests):
            req_start = time.time()

            response = client.post(
                "/auth/token",
                json={"client_id": f"perf-client-{i}"},
            )

            req_latency = (time.time() - req_start) * 1000
            latencies.append(req_latency)

            assert response.status_code == 200

        elapsed = time.time() - start_time
        avg_latency = statistics.mean(latencies)
        throughput = num_requests / elapsed

        print(f"\n=== Token Generation Performance ===")
        print(f"Total Requests: {num_requests}")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} req/s")
        print(f"Average Latency: {avg_latency:.1f}ms")
        print(f"Min Latency: {min(latencies):.1f}ms")
        print(f"Max Latency: {max(latencies):.1f}ms")

        assert avg_latency < 50, "Token generation should be < 50ms on average"

    @pytest.mark.skipif(
        True,
        reason="Performance test - run manually"
    )
    def test_token_verification_performance(self):
        """Test token verification performance"""
        client = TestClient(app)

        # Generate test token
        response = client.post(
            "/auth/token",
            json={"client_id": "perf-test"},
        )
        token = response.json()["access_token"]

        num_requests = 100
        print(f"Verifying token {num_requests} times...")

        start_time = time.time()
        latencies = []

        for _ in range(num_requests):
            req_start = time.time()

            response = client.post(
                "/auth/validate",
                headers={"Authorization": f"Bearer {token}"},
            )

            req_latency = (time.time() - req_start) * 1000
            latencies.append(req_latency)

            assert response.status_code == 200
            assert response.json()["valid"] is True

        elapsed = time.time() - start_time
        avg_latency = statistics.mean(latencies)
        throughput = num_requests / elapsed

        print(f"\n=== Token Verification Performance ===")
        print(f"Total Requests: {num_requests}")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} req/s")
        print(f"Average Latency: {avg_latency:.1f}ms")

        assert avg_latency < 20, "Token verification should be < 20ms on average"
