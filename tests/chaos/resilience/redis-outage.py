#!/usr/bin/env python3
"""
T.A.R.S. Chaos Testing - Redis Outage Simulation

Tests system resilience when Redis becomes unavailable.
Expected behavior:
- JWT fallback to legacy single-key mode
- API key fallback to in-memory store
- Graceful degradation with warnings
"""

import asyncio
import requests
import time
import subprocess
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:3001"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
REDIS_POD_LABEL = "app=redis"
NAMESPACE = "tars"
TEST_DURATION = 300  # 5 minutes


@dataclass
class TestMetrics:
    """Metrics collected during the test"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    degraded_requests: int = 0
    avg_latency_ms: float = 0.0
    error_types: Dict[str, int] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = {}


class RedisOutageTest:
    """Redis outage chaos test"""

    def __init__(self):
        self.base_url = BASE_URL
        self.token: Optional[str] = None
        self.metrics = TestMetrics()
        self.latencies: List[float] = []

    def login(self) -> bool:
        """Login to get auth token"""
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                return True
        except Exception as e:
            print(f"Login failed: {e}")
        return False

    def kill_redis(self) -> bool:
        """Kill Redis pod in Kubernetes"""
        try:
            print("\n[CHAOS] Killing Redis pod...")
            cmd = [
                "kubectl",
                "delete",
                "pod",
                "-l",
                REDIS_POD_LABEL,
                "-n",
                NAMESPACE,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("[CHAOS] Redis pod deleted successfully")
                return True
            else:
                print(f"[ERROR] Failed to kill Redis: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception killing Redis: {e}")
            return False

    def make_request(self, endpoint: str, method: str = "GET") -> Optional[requests.Response]:
        """Make authenticated request and track metrics"""
        if not self.token:
            return None

        headers = {"Authorization": f"Bearer {self.token}"}
        start_time = time.time()

        try:
            if method == "GET":
                response = requests.get(
                    f"{self.base_url}{endpoint}", headers=headers, timeout=5
                )
            else:
                response = requests.post(
                    f"{self.base_url}{endpoint}", headers=headers, timeout=5
                )

            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)

            self.metrics.total_requests += 1

            if response.status_code == 200:
                self.metrics.successful_requests += 1
            elif response.status_code in [503, 502]:
                self.metrics.degraded_requests += 1
            else:
                self.metrics.failed_requests += 1
                error_key = f"{response.status_code}"
                self.metrics.error_types[error_key] = (
                    self.metrics.error_types.get(error_key, 0) + 1
                )

            return response

        except requests.exceptions.Timeout:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.error_types["timeout"] = (
                self.metrics.error_types.get("timeout", 0) + 1
            )
        except Exception as e:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.error_types["exception"] = (
                self.metrics.error_types.get("exception", 0) + 1
            )

        return None

    def run_load(self, duration_sec: int):
        """Run continuous load during the test"""
        print(f"\n[LOAD] Running load for {duration_sec}s...")
        start_time = time.time()
        request_count = 0

        while time.time() - start_time < duration_sec:
            # Mix of different endpoints
            endpoints = [
                "/admin/jwt/status",
                "/admin/agents",
                "/admin/api-keys",
                "/admin/health",
            ]

            for endpoint in endpoints:
                self.make_request(endpoint)
                request_count += 1

                if request_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rps = request_count / elapsed
                    success_rate = (
                        self.metrics.successful_requests / self.metrics.total_requests
                        if self.metrics.total_requests > 0
                        else 0
                    )
                    print(
                        f"[LOAD] {request_count} reqs, {rps:.1f} RPS, "
                        f"{success_rate * 100:.1f}% success"
                    )

            time.sleep(0.1)

    def calculate_metrics(self):
        """Calculate final metrics"""
        if self.latencies:
            self.metrics.avg_latency_ms = sum(self.latencies) / len(self.latencies)

    def print_summary(self):
        """Print test summary"""
        self.calculate_metrics()

        print("\n" + "=" * 60)
        print("Redis Outage Test Summary")
        print("=" * 60)
        print(f"Total Requests: {self.metrics.total_requests}")
        print(f"Successful: {self.metrics.successful_requests}")
        print(f"Degraded (503/502): {self.metrics.degraded_requests}")
        print(f"Failed: {self.metrics.failed_requests}")
        print(f"Average Latency: {self.metrics.avg_latency_ms:.2f}ms")

        if self.metrics.total_requests > 0:
            success_rate = (
                self.metrics.successful_requests / self.metrics.total_requests
            ) * 100
            print(f"Success Rate: {success_rate:.2f}%")

        if self.metrics.error_types:
            print("\nError Breakdown:")
            for error_type, count in self.metrics.error_types.items():
                print(f"  {error_type}: {count}")

        print("=" * 60)

    def run(self):
        """Run the full chaos test"""
        print("=" * 60)
        print("T.A.R.S. Redis Outage Chaos Test")
        print("=" * 60)

        # Step 1: Baseline - verify system is healthy
        print("\n[STEP 1] Verifying system health...")
        if not self.login():
            print("[ERROR] Failed to login. Is the system running?")
            return False

        response = self.make_request("/admin/health")
        if not response or response.status_code != 200:
            print("[ERROR] System health check failed")
            return False

        print("[OK] System is healthy")

        # Step 2: Run baseline load (30s)
        print("\n[STEP 2] Running baseline load (30s)...")
        self.run_load(30)
        baseline_success_rate = (
            self.metrics.successful_requests / self.metrics.total_requests
            if self.metrics.total_requests > 0
            else 0
        )
        print(f"[OK] Baseline success rate: {baseline_success_rate * 100:.2f}%")

        # Step 3: Kill Redis
        print("\n[STEP 3] Inducing Redis outage...")
        if not self.kill_redis():
            print("[ERROR] Failed to kill Redis. Continuing anyway...")

        # Wait for failure to propagate
        print("[WAIT] Waiting 10s for failure to propagate...")
        time.sleep(10)

        # Step 4: Run load during outage
        print("\n[STEP 4] Running load during Redis outage (60s)...")
        outage_start_metrics = self.metrics.total_requests
        self.run_load(60)
        outage_requests = self.metrics.total_requests - outage_start_metrics

        # Step 5: Wait for Redis to recover (K8s should restart it)
        print("\n[STEP 5] Waiting for Redis recovery (60s)...")
        time.sleep(60)

        # Step 6: Run load after recovery
        print("\n[STEP 6] Running load after recovery (30s)...")
        self.run_load(30)

        # Print final summary
        self.print_summary()

        # Evaluate results
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)

        if baseline_success_rate > 0.95:
            print("✅ Baseline: PASS (>95% success)")
        else:
            print("❌ Baseline: FAIL (<95% success)")

        # During outage, we expect graceful degradation
        # Success rate may drop, but system should not crash
        final_success_rate = (
            self.metrics.successful_requests / self.metrics.total_requests
            if self.metrics.total_requests > 0
            else 0
        )

        if final_success_rate > 0.5:
            print("✅ Resilience: PASS (graceful degradation)")
        else:
            print("⚠️  Resilience: DEGRADED (high failure rate)")

        return True


def main():
    """Main entry point"""
    test = RedisOutageTest()
    try:
        test.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test stopped by user")
        test.print_summary()
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
