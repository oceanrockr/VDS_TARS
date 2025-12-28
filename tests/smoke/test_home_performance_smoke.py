"""
T.A.R.S. Home Network Performance & Stability Smoke Tests

Version: v1.0.10 (GA)
Phase: 22 - Deployment Validation
Target: Ubuntu 22.04 LTS, Home Network Deployment

These tests are designed to be:
- Non-destructive
- Fast (< 5 minutes total)
- Runnable against a live deployment
- Informational (failures are warnings, not blockers)

Usage:
    pytest tests/smoke/test_home_performance_smoke.py -v

    Or with custom endpoint:
    TARS_API_URL=http://192.168.1.100:8000 pytest tests/smoke/ -v
"""

import os
import time
import statistics
import pytest
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuration
API_BASE_URL = os.getenv("TARS_API_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("TARS_OLLAMA_URL", "http://localhost:11434")
CHROMA_URL = os.getenv("TARS_CHROMA_URL", "http://localhost:8001")

# Thresholds (relaxed for home deployment)
HEALTH_CHECK_MAX_MS = 500
RAG_QUERY_MAX_MS = 30000  # 30 seconds for LLM inference
INFERENCE_MAX_MS = 60000  # 60 seconds for cold start
CHROMA_QUERY_MAX_MS = 2000  # 2 seconds for vector search


@dataclass
class TimingResult:
    """Result of a timed operation."""
    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None


class TestHealthPerformance:
    """Performance tests for health endpoints."""

    def test_health_endpoint_latency(self):
        """Test /health endpoint responds within threshold."""
        latencies = []

        for _ in range(5):
            start = time.perf_counter()
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            duration_ms = (time.perf_counter() - start) * 1000
            latencies.append(duration_ms)

            assert response.status_code == 200, f"Health check failed: {response.status_code}"

        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n  Health endpoint latency:")
        print(f"    Average: {avg_latency:.1f}ms")
        print(f"    P95: {p95_latency:.1f}ms")
        print(f"    Threshold: {HEALTH_CHECK_MAX_MS}ms")

        assert avg_latency < HEALTH_CHECK_MAX_MS, (
            f"Health endpoint too slow: {avg_latency:.1f}ms > {HEALTH_CHECK_MAX_MS}ms"
        )

    def test_ready_endpoint_latency(self):
        """Test /ready endpoint responds within threshold."""
        start = time.perf_counter()
        response = requests.get(f"{API_BASE_URL}/ready", timeout=15)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, f"Ready check failed: {response.status_code}"

        data = response.json()
        print(f"\n  Ready endpoint latency: {duration_ms:.1f}ms")
        print(f"    Status: {data.get('status')}")
        print(f"    Checks: {data.get('checks', {})}")

        # Ready can be slower due to service checks
        assert duration_ms < HEALTH_CHECK_MAX_MS * 5, (
            f"Ready endpoint too slow: {duration_ms:.1f}ms"
        )


class TestOllamaPerformance:
    """Performance tests for Ollama LLM inference."""

    def test_ollama_health(self):
        """Test Ollama API is responsive."""
        start = time.perf_counter()
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, "Ollama not responding"

        models = response.json().get("models", [])
        print(f"\n  Ollama latency: {duration_ms:.1f}ms")
        print(f"    Models available: {len(models)}")
        for model in models[:3]:
            print(f"      - {model.get('name', 'unknown')}")

    def test_inference_latency_cold(self):
        """Test LLM inference latency (may include cold start)."""
        prompt = "Say 'hello' in exactly one word."

        start = time.perf_counter()
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral:7b-instruct",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        duration_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            pytest.skip(f"Ollama inference failed: {response.status_code}")

        data = response.json()
        tokens = data.get("eval_count", 0)
        eval_time_ms = data.get("eval_duration", 0) / 1e6  # ns to ms

        print(f"\n  Cold inference:")
        print(f"    Total time: {duration_ms:.1f}ms")
        print(f"    Eval time: {eval_time_ms:.1f}ms")
        print(f"    Tokens: {tokens}")
        if tokens > 0:
            print(f"    Tokens/sec: {tokens / (eval_time_ms / 1000):.1f}")

        assert duration_ms < INFERENCE_MAX_MS, (
            f"Inference too slow: {duration_ms:.1f}ms > {INFERENCE_MAX_MS}ms"
        )

    def test_inference_latency_warm(self):
        """Test LLM inference latency (warm, model loaded)."""
        # First request to warm up
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral:7b-instruct", "prompt": "Hi", "stream": False},
            timeout=60
        )

        # Timed request
        prompt = "What is 2+2? Answer with just the number."

        latencies = []
        for _ in range(3):
            start = time.perf_counter()
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "mistral:7b-instruct", "prompt": prompt, "stream": False},
                timeout=60
            )
            duration_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                latencies.append(duration_ms)

        if not latencies:
            pytest.skip("No successful warm inference requests")

        avg_latency = statistics.mean(latencies)
        print(f"\n  Warm inference (3 samples):")
        print(f"    Average: {avg_latency:.1f}ms")
        print(f"    Min: {min(latencies):.1f}ms")
        print(f"    Max: {max(latencies):.1f}ms")


class TestChromaPerformance:
    """Performance tests for ChromaDB vector search."""

    def test_chroma_heartbeat(self):
        """Test ChromaDB heartbeat latency."""
        start = time.perf_counter()
        response = requests.get(f"{CHROMA_URL}/api/v1/heartbeat", timeout=10)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, "ChromaDB not responding"
        print(f"\n  ChromaDB heartbeat: {duration_ms:.1f}ms")

    def test_chroma_collection_query(self):
        """Test ChromaDB collection listing latency."""
        start = time.perf_counter()
        response = requests.get(f"{CHROMA_URL}/api/v1/collections", timeout=10)
        duration_ms = (time.perf_counter() - start) * 1000

        assert response.status_code == 200, "ChromaDB collections query failed"

        collections = response.json()
        print(f"\n  ChromaDB collection list: {duration_ms:.1f}ms")
        print(f"    Collections: {len(collections)}")


class TestRAGPerformance:
    """Performance tests for RAG pipeline."""

    def test_rag_search_latency(self):
        """Test RAG search (retrieval only) latency."""
        start = time.perf_counter()
        response = requests.post(
            f"{API_BASE_URL}/rag/search",
            json={"query": "test query", "top_k": 3},
            timeout=30
        )
        duration_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            pytest.skip(f"RAG search unavailable: {response.status_code}")

        data = response.json()
        print(f"\n  RAG search latency: {duration_ms:.1f}ms")
        print(f"    Results: {data.get('total_results', 0)}")
        print(f"    Search time (server): {data.get('search_time_ms', 'N/A')}ms")

        assert duration_ms < CHROMA_QUERY_MAX_MS, (
            f"RAG search too slow: {duration_ms:.1f}ms > {CHROMA_QUERY_MAX_MS}ms"
        )

    def test_rag_query_full_latency(self):
        """Test full RAG query (retrieval + generation) latency."""
        start = time.perf_counter()
        response = requests.post(
            f"{API_BASE_URL}/rag/query",
            json={"query": "What is this system?", "top_k": 2},
            timeout=120
        )
        duration_ms = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            pytest.skip(f"RAG query unavailable: {response.status_code}")

        data = response.json()
        print(f"\n  Full RAG query latency: {duration_ms:.1f}ms")
        print(f"    Retrieval time: {data.get('retrieval_time_ms', 'N/A')}ms")
        print(f"    Generation time: {data.get('generation_time_ms', 'N/A')}ms")
        print(f"    Total tokens: {data.get('total_tokens', 0)}")
        print(f"    Sources: {len(data.get('sources', []))}")

        assert duration_ms < RAG_QUERY_MAX_MS, (
            f"Full RAG query too slow: {duration_ms:.1f}ms > {RAG_QUERY_MAX_MS}ms"
        )


class TestMemoryPressure:
    """Tests for memory and resource usage."""

    def test_concurrent_health_checks(self):
        """Test system handles concurrent requests."""
        num_requests = 20

        def make_request(i):
            start = time.perf_counter()
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=10)
                duration_ms = (time.perf_counter() - start) * 1000
                return TimingResult(
                    operation=f"health_{i}",
                    duration_ms=duration_ms,
                    success=response.status_code == 200
                )
            except Exception as e:
                return TimingResult(
                    operation=f"health_{i}",
                    duration_ms=0,
                    success=False,
                    error=str(e)
                )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n  Concurrent health checks ({num_requests} requests):")
        print(f"    Successful: {len(successful)}")
        print(f"    Failed: {len(failed)}")

        if successful:
            latencies = [r.duration_ms for r in successful]
            print(f"    Avg latency: {statistics.mean(latencies):.1f}ms")
            print(f"    Max latency: {max(latencies):.1f}ms")

        assert len(successful) >= num_requests * 0.9, (
            f"Too many failures: {len(failed)}/{num_requests}"
        )


class TestRestartResilience:
    """Tests for restart and recovery behavior."""

    def test_service_recovery_check(self):
        """Verify services are stable (no recent restarts)."""
        response = requests.get(f"{API_BASE_URL}/ready", timeout=15)

        if response.status_code != 200:
            pytest.skip("Ready endpoint unavailable")

        data = response.json()
        uptime_seconds = data.get("uptime_seconds", 0)

        print(f"\n  Service uptime: {uptime_seconds}s ({uptime_seconds/60:.1f} minutes)")

        # Warn if uptime is very short (might indicate recent restart)
        if uptime_seconds < 60:
            print("    WARNING: Service recently started (< 1 minute)")

    def test_repeated_queries_stability(self):
        """Test system remains stable after repeated queries."""
        num_queries = 10
        results = []

        for i in range(num_queries):
            start = time.perf_counter()
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=10)
                duration_ms = (time.perf_counter() - start) * 1000
                results.append(TimingResult(
                    operation=f"stability_{i}",
                    duration_ms=duration_ms,
                    success=response.status_code == 200
                ))
            except Exception as e:
                results.append(TimingResult(
                    operation=f"stability_{i}",
                    duration_ms=0,
                    success=False,
                    error=str(e)
                ))

            time.sleep(0.1)  # Small delay between requests

        successful = [r for r in results if r.success]

        print(f"\n  Stability test ({num_queries} sequential requests):")
        print(f"    Successful: {len(successful)}/{num_queries}")

        if successful:
            latencies = [r.duration_ms for r in successful]
            variance = statistics.variance(latencies) if len(latencies) > 1 else 0
            print(f"    Latency variance: {variance:.2f}")

        assert len(successful) == num_queries, "Some stability requests failed"


class TestTokenThroughput:
    """Tests for LLM token generation throughput."""

    def test_token_throughput(self):
        """Measure approximate tokens per second."""
        # Use a prompt that generates a predictable response length
        prompt = "Write the numbers 1 through 20, one per line."

        start = time.perf_counter()
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral:7b-instruct",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        total_time = time.perf_counter() - start

        if response.status_code != 200:
            pytest.skip(f"Ollama not available: {response.status_code}")

        data = response.json()
        tokens = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 0)
        eval_time = eval_ns / 1e9  # Convert to seconds

        if tokens > 0 and eval_time > 0:
            tokens_per_sec = tokens / eval_time
            print(f"\n  Token throughput:")
            print(f"    Tokens generated: {tokens}")
            print(f"    Eval time: {eval_time:.2f}s")
            print(f"    Throughput: {tokens_per_sec:.1f} tokens/sec")
            print(f"    Total request time: {total_time:.2f}s")

            # Rough expectation for 7B model on RTX GPU
            assert tokens_per_sec > 5, (
                f"Token throughput too low: {tokens_per_sec:.1f} tokens/sec"
            )


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Mark certain tests as slow."""
    slow_tests = [
        "test_inference_latency_cold",
        "test_inference_latency_warm",
        "test_rag_query_full_latency",
        "test_token_throughput"
    ]

    for item in items:
        if item.name in slow_tests:
            item.add_marker(pytest.mark.slow)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
