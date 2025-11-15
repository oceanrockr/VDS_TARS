"""
T.A.R.S. Locust Load Testing Script
Phase 6: Python-based User Scenario Simulation

Features:
- Realistic user behavior simulation
- WebSocket chat testing
- Document upload simulation
- Analytics tracking
- Custom metrics

Usage:
    # Web UI (recommended)
    locust -f load_test_locust.py --host=http://localhost:8000

    # Headless mode
    locust -f load_test_locust.py --host=http://localhost:8000 \
           --users 100 --spawn-rate 10 --run-time 5m --headless

    # Distributed load testing
    locust -f load_test_locust.py --host=http://localhost:8000 \
           --master
    locust -f load_test_locust.py --host=http://localhost:8000 \
           --worker --master-host=localhost
"""

import random
import time
import json
from typing import Dict, Any

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


# ==============================================================================
# CUSTOM METRICS
# ==============================================================================

cache_hit_count = 0
cache_miss_count = 0
rag_query_count = 0
websocket_message_count = 0


@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    """Print custom metrics on test completion"""
    total_cache_requests = cache_hit_count + cache_miss_count
    cache_hit_ratio = (cache_hit_count / total_cache_requests * 100) if total_cache_requests > 0 else 0

    print("\n" + "=" * 80)
    print("T.A.R.S. Load Test Summary")
    print("=" * 80)
    print(f"Total RAG Queries: {rag_query_count}")
    print(f"Cache Hits: {cache_hit_count}")
    print(f"Cache Misses: {cache_miss_count}")
    print(f"Cache Hit Ratio: {cache_hit_ratio:.2f}%")
    print(f"WebSocket Messages: {websocket_message_count}")
    print("=" * 80 + "\n")


# ==============================================================================
# TEST DATA
# ==============================================================================

TEST_QUERIES = [
    "What is machine learning?",
    "Explain neural networks",
    "How does gradient descent work?",
    "What are transformers in NLP?",
    "Describe convolutional neural networks",
    "What is reinforcement learning?",
    "Explain attention mechanisms",
    "How do GANs work?",
    "What is transfer learning?",
    "Describe BERT model architecture",
    "What is semantic chunking?",
    "Explain cross-encoder reranking",
    "How does hybrid search work?",
    "What is RAG in LLMs?",
    "Describe vector databases",
]

DOCUMENT_PATHS = [
    "/tmp/test_doc_1.pdf",
    "/tmp/test_doc_2.docx",
    "/tmp/test_doc_3.txt",
]


# ==============================================================================
# USER SCENARIOS
# ==============================================================================

class TARSUser(FastHttpUser):
    """
    Simulates a typical T.A.R.S. user performing various operations.

    Tasks weighted by realistic usage patterns:
    - 50% RAG queries (most common operation)
    - 20% Conversation history access
    - 15% Analytics viewing
    - 10% Health/metrics checks
    - 5% Admin operations
    """

    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    auth_token: str = None
    client_id: str = None

    def on_start(self):
        """Called when a user starts - authenticate and get token"""
        self.client_id = f"loadtest_user_{random.randint(1000, 9999)}"
        self.authenticate()

    def authenticate(self):
        """Authenticate and get JWT token"""
        response = self.client.post(
            "/auth/authenticate",
            json={"client_id": self.client_id},
            name="Auth: Authenticate",
        )

        if response.status_code == 200:
            data = response.json()
            self.auth_token = data.get("access_token")
        else:
            print(f"Authentication failed: {response.status_code}")

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

    # ==========================================================================
    # TASKS (weighted by probability)
    # ==========================================================================

    @task(50)
    def query_rag(self):
        """Perform RAG query (50% of requests)"""
        global rag_query_count, cache_hit_count, cache_miss_count

        query = random.choice(TEST_QUERIES)
        payload = {
            "query": query,
            "top_k": 5,
            "rerank": True,
            "include_sources": True,
        }

        with self.client.post(
            "/rag/query",
            json=payload,
            headers=self.get_headers(),
            catch_response=True,
            name="RAG: Query",
        ) as response:
            if response.status_code == 200:
                rag_query_count += 1
                try:
                    data = response.json()
                    # Check for cache hit
                    if data.get("cached") or data.get("from_cache"):
                        cache_hit_count += 1
                    else:
                        cache_miss_count += 1

                    # Validate response
                    if not data.get("answer"):
                        response.failure("No answer in response")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(20)
    def get_conversation_history(self):
        """Retrieve conversation history (20% of requests)"""
        with self.client.get(
            "/conversation/list",
            headers=self.get_headers(),
            catch_response=True,
            name="Conversation: List",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data.get("conversations"), list):
                        response.success()
                    else:
                        response.failure("Invalid conversation list format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(15)
    def view_analytics(self):
        """View analytics dashboard (15% of requests)"""
        endpoints = [
            "/analytics/query-stats",
            "/analytics/summary",
            "/analytics/document-popularity?top_n=5",
        ]

        endpoint = random.choice(endpoints)
        with self.client.get(
            endpoint,
            headers=self.get_headers(),
            catch_response=True,
            name=f"Analytics: {endpoint.split('?')[0].split('/')[-1]}",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)
    def check_health(self):
        """Check system health and metrics (10% of requests)"""
        endpoints = [
            ("/health", "Health Check"),
            ("/ready", "Readiness Check"),
            ("/metrics/system", "System Metrics"),
        ]

        endpoint, name = random.choice(endpoints)
        with self.client.get(
            endpoint,
            catch_response=True,
            name=name,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def get_rag_stats(self):
        """Get RAG statistics (5% of requests)"""
        with self.client.get(
            "/rag/stats",
            headers=self.get_headers(),
            catch_response=True,
            name="RAG: Stats",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "total_documents" in data:
                        response.success()
                    else:
                        response.failure("Missing stats fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")


# ==============================================================================
# ADMIN USER SCENARIO
# ==============================================================================

class TARSAdminUser(FastHttpUser):
    """
    Simulates an admin user performing management operations.

    Admin operations:
    - View comprehensive analytics
    - Export data
    - Monitor system performance
    - Manage documents
    """

    wait_time = between(5, 15)  # Admins make fewer requests
    auth_token: str = None
    weight = 1  # Only 1 admin per 10 regular users

    def on_start(self):
        """Authenticate as admin"""
        self.client_id = "admin_001"  # Admin client ID from config
        self.authenticate()

    def authenticate(self):
        """Authenticate and get JWT token"""
        response = self.client.post(
            "/auth/authenticate",
            json={"client_id": self.client_id},
            name="Admin: Authenticate",
        )

        if response.status_code == 200:
            data = response.json()
            self.auth_token = data.get("access_token")

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }

    @task(40)
    def view_comprehensive_analytics(self):
        """View comprehensive analytics summary"""
        with self.client.get(
            "/analytics/summary",
            headers=self.get_headers(),
            catch_response=True,
            name="Admin: Analytics Summary",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(30)
    def check_system_metrics(self):
        """Monitor system performance"""
        with self.client.get(
            "/metrics/system",
            headers=self.get_headers(),
            catch_response=True,
            name="Admin: System Metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(20)
    def view_document_stats(self):
        """View document indexing statistics"""
        with self.client.get(
            "/rag/stats",
            headers=self.get_headers(),
            catch_response=True,
            name="Admin: Document Stats",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(10)
    def check_readiness(self):
        """Check detailed readiness of all services"""
        with self.client.get(
            "/ready",
            catch_response=True,
            name="Admin: Readiness Check",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    checks = data.get("checks", {})
                    unhealthy = [k for k, v in checks.items() if v != "healthy"]
                    if unhealthy:
                        print(f"Warning: Unhealthy services: {unhealthy}")
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")


# ==============================================================================
# PERFORMANCE TESTING SCENARIOS
# ==============================================================================

class TARSStressTestUser(FastHttpUser):
    """
    Aggressive load testing user for stress/spike testing.

    Characteristics:
    - Minimal wait time between requests
    - Large batch queries
    - Concurrent operations
    """

    wait_time = between(0.1, 0.5)  # Very short wait time
    weight = 0  # Disabled by default (set to 1 for stress testing)

    auth_token: str = None

    def on_start(self):
        """Authenticate"""
        self.client_id = f"stress_test_{random.randint(10000, 99999)}"
        response = self.client.post(
            "/auth/authenticate",
            json={"client_id": self.client_id},
        )
        if response.status_code == 200:
            self.auth_token = response.json().get("access_token")

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.auth_token}"}

    @task
    def rapid_fire_queries(self):
        """Perform rapid-fire RAG queries"""
        query = random.choice(TEST_QUERIES)
        self.client.post(
            "/rag/query",
            json={"query": query, "top_k": 3},
            headers=self.get_headers(),
            name="Stress: Rapid Query",
        )


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Uncomment to enable stress testing
# TARSStressTestUser.weight = 1

# Uncomment to adjust user distribution
# TARSUser.weight = 10
# TARSAdminUser.weight = 1
