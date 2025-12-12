#!/usr/bin/env python3
"""
T.A.R.S. Chaos Testing - Pod Kill Test

Tests graceful recovery when service pods are killed.
Targets: Orchestration, AutoML, HyperSync services.
"""

import subprocess
import time
import requests
import sys
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ServiceType(Enum):
    """Target services for pod kill"""

    ORCHESTRATION = "orchestration"
    AUTOML = "automl"
    HYPERSYNC = "hypersync"
    INSIGHT = "insight"


@dataclass
class PodKillResult:
    """Result of a pod kill test"""

    service: ServiceType
    kill_success: bool
    recovery_time_sec: float
    requests_during_kill: int
    successful_requests: int
    failed_requests: int
    max_downtime_sec: float


class PodKillTest:
    """Pod kill chaos test"""

    def __init__(self, namespace: str = "tars", base_url: str = "http://localhost:3001"):
        self.namespace = namespace
        self.base_url = base_url
        self.token: Optional[str] = None

    def login(self) -> bool:
        """Get auth token"""
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"username": "admin", "password": "admin123"},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                return True
        except Exception as e:
            print(f"Login failed: {e}")
        return False

    def get_pod_name(self, service: ServiceType) -> Optional[str]:
        """Get pod name for service"""
        label_map = {
            ServiceType.ORCHESTRATION: "app.kubernetes.io/component=orchestration",
            ServiceType.AUTOML: "app.kubernetes.io/component=automl",
            ServiceType.HYPERSYNC: "app.kubernetes.io/component=hypersync",
            ServiceType.INSIGHT: "app.kubernetes.io/component=insight",
        }

        label = label_map.get(service)
        if not label:
            return None

        try:
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-l",
                label,
                "-n",
                self.namespace,
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
        except Exception as e:
            print(f"Error getting pod name: {e}")

        return None

    def kill_pod(self, pod_name: str) -> bool:
        """Kill a specific pod"""
        try:
            print(f"\n[CHAOS] Killing pod: {pod_name}")
            cmd = ["kubectl", "delete", "pod", pod_name, "-n", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"[OK] Pod {pod_name} deleted")
                return True
            else:
                print(f"[ERROR] Failed to delete pod: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
            return False

    def check_service_health(self, service: ServiceType) -> bool:
        """Check if service is healthy"""
        if not self.token:
            return False

        endpoint_map = {
            ServiceType.ORCHESTRATION: "/admin/agents",
            ServiceType.AUTOML: "/admin/automl/trials",
            ServiceType.HYPERSYNC: "/admin/hypersync/proposals",
            ServiceType.INSIGHT: "/admin/health",
        }

        endpoint = endpoint_map.get(service)
        if not endpoint:
            return False

        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                f"{self.base_url}{endpoint}", headers=headers, timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_recovery(
        self, service: ServiceType, max_wait_sec: int = 120
    ) -> tuple[bool, float]:
        """Wait for service to recover after pod kill"""
        print(f"\n[WAIT] Waiting for {service.value} recovery...")
        start_time = time.time()

        while time.time() - start_time < max_wait_sec:
            if self.check_service_health(service):
                recovery_time = time.time() - start_time
                print(f"[OK] {service.value} recovered in {recovery_time:.2f}s")
                return True, recovery_time

            time.sleep(2)

        print(f"[TIMEOUT] {service.value} did not recover within {max_wait_sec}s")
        return False, max_wait_sec

    def measure_downtime(self, service: ServiceType, duration_sec: int = 60) -> tuple[int, int, float]:
        """Measure downtime by continuously probing service"""
        print(f"\n[MEASURE] Measuring {service.value} availability for {duration_sec}s...")

        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        max_consecutive_failures = 0
        current_consecutive_failures = 0
        last_failure_time = None
        max_downtime = 0.0

        start_time = time.time()

        while time.time() - start_time < duration_sec:
            healthy = self.check_service_health(service)
            total_requests += 1

            if healthy:
                successful_requests += 1
                # End of failure window
                if current_consecutive_failures > 0:
                    downtime = time.time() - last_failure_time
                    max_downtime = max(max_downtime, downtime)
                current_consecutive_failures = 0
            else:
                failed_requests += 1
                current_consecutive_failures += 1
                if current_consecutive_failures == 1:
                    last_failure_time = time.time()
                max_consecutive_failures = max(
                    max_consecutive_failures, current_consecutive_failures
                )

            time.sleep(1)

        # Handle case where failures extend to end of test
        if current_consecutive_failures > 0:
            downtime = time.time() - last_failure_time
            max_downtime = max(max_downtime, downtime)

        return total_requests, successful_requests, max_downtime

    def test_service(self, service: ServiceType) -> PodKillResult:
        """Run pod kill test for a specific service"""
        print("\n" + "=" * 60)
        print(f"Testing {service.value.upper()} Pod Kill")
        print("=" * 60)

        # Get pod name
        pod_name = self.get_pod_name(service)
        if not pod_name:
            print(f"[ERROR] Could not find pod for {service.value}")
            return PodKillResult(
                service=service,
                kill_success=False,
                recovery_time_sec=0,
                requests_during_kill=0,
                successful_requests=0,
                failed_requests=0,
                max_downtime_sec=0,
            )

        print(f"[INFO] Target pod: {pod_name}")

        # Verify service is healthy before test
        if not self.check_service_health(service):
            print(f"[ERROR] {service.value} is not healthy before test")
            return PodKillResult(
                service=service,
                kill_success=False,
                recovery_time_sec=0,
                requests_during_kill=0,
                successful_requests=0,
                failed_requests=0,
                max_downtime_sec=0,
            )

        print(f"[OK] {service.value} is healthy")

        # Kill the pod
        kill_success = self.kill_pod(pod_name)
        if not kill_success:
            return PodKillResult(
                service=service,
                kill_success=False,
                recovery_time_sec=0,
                requests_during_kill=0,
                successful_requests=0,
                failed_requests=0,
                max_downtime_sec=0,
            )

        # Measure downtime while waiting for recovery
        total_reqs, successful_reqs, max_downtime = self.measure_downtime(service, 120)
        failed_reqs = total_reqs - successful_reqs

        # Final health check
        recovered, recovery_time = self.wait_for_recovery(service, max_wait_sec=30)

        return PodKillResult(
            service=service,
            kill_success=True,
            recovery_time_sec=recovery_time if recovered else 120,
            requests_during_kill=total_reqs,
            successful_requests=successful_reqs,
            failed_requests=failed_reqs,
            max_downtime_sec=max_downtime,
        )

    def run(self, services: list[ServiceType] = None):
        """Run pod kill tests for all services"""
        if services is None:
            services = [
                ServiceType.ORCHESTRATION,
                ServiceType.AUTOML,
                ServiceType.HYPERSYNC,
            ]

        print("=" * 60)
        print("T.A.R.S. Pod Kill Chaos Test")
        print("=" * 60)

        # Login
        if not self.login():
            print("[ERROR] Failed to login")
            return

        results: list[PodKillResult] = []

        # Test each service
        for service in services:
            result = self.test_service(service)
            results.append(result)

            # Wait between tests
            if service != services[-1]:
                print("\n[WAIT] Cooling down for 30s before next test...")
                time.sleep(30)

        # Print summary
        self.print_summary(results)

    def print_summary(self, results: list[PodKillResult]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("Pod Kill Test Summary")
        print("=" * 60)

        for result in results:
            print(f"\n{result.service.value.upper()}:")
            print(f"  Kill Success: {'✅' if result.kill_success else '❌'}")
            print(f"  Recovery Time: {result.recovery_time_sec:.2f}s")
            print(f"  Max Downtime: {result.max_downtime_sec:.2f}s")
            print(f"  Requests: {result.requests_during_kill}")
            print(f"  Success: {result.successful_requests}")
            print(f"  Failed: {result.failed_requests}")

            if result.requests_during_kill > 0:
                availability = (
                    result.successful_requests / result.requests_during_kill
                ) * 100
                print(f"  Availability: {availability:.2f}%")

            # Evaluation
            if result.recovery_time_sec < 30:
                print("  ✅ PASS: Fast recovery (<30s)")
            elif result.recovery_time_sec < 60:
                print("  ⚠️  WARN: Moderate recovery (30-60s)")
            else:
                print("  ❌ FAIL: Slow recovery (>60s)")

        print("=" * 60)


def main():
    """Main entry point"""
    test = PodKillTest()
    try:
        test.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
