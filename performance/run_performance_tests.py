#!/usr/bin/env python3
"""
T.A.R.S. Performance Testing Suite

Comprehensive performance testing tool for T.A.R.S. v1.0.2+

Features:
- Latency testing (p50, p95, p99)
- Throughput measurement
- CPU/memory profiling
- Stress testing
- Regression detection
- JSON and Markdown reporting

Usage:
    python performance/run_performance_tests.py \\
        --url https://tars.company.com \\
        --duration 300 \\
        --concurrency 50 \\
        --output-json results.json \\
        --output-md report.md \\
        --baseline baseline.json \\
        --verbose

Author: T.A.R.S. Team
Version: 1.0.2
"""

import argparse
import json
import time
import sys
import statistics
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess
import platform

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("Error: 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Warning: 'psutil' not found. CPU/memory profiling disabled.")
    psutil = None


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TestConfig:
    """Test configuration"""
    url: str
    duration: int
    concurrency: int
    profile: str
    baseline_path: Optional[str]
    output_json: Optional[str]
    output_md: Optional[str]
    verbose: bool
    config_path: Optional[str]

    # Enterprise integration
    encryption_enabled: bool = False
    signing_enabled: bool = False
    auth_token: Optional[str] = None


@dataclass
class RequestResult:
    """Single request result"""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    response_size: int
    timestamp: float
    success: bool
    error: Optional[str] = None


@dataclass
class EndpointStats:
    """Statistics for a single endpoint"""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float

    # Latency statistics
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float

    # Throughput
    requests_per_second: float

    # Response size
    avg_response_size: int


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    timestamp: float


@dataclass
class PerformanceReport:
    """Complete performance test report"""
    test_id: str
    timestamp: str
    config: Dict
    duration_seconds: float

    # Overall statistics
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float

    # Latency
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float

    # Throughput
    requests_per_second: float

    # Per-endpoint stats
    endpoint_stats: List[Dict]

    # System metrics
    peak_cpu: float
    peak_memory_mb: float
    avg_cpu: float
    avg_memory_mb: float

    # Regression analysis
    regression_detected: bool
    regression_details: Optional[Dict] = None


# ============================================================================
# HTTP Client
# ============================================================================

class PerformanceHTTPClient:
    """HTTP client with connection pooling and retry logic"""

    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token

        # Session with connection pooling
        self.session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=0.3
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Default headers
        self.session.headers.update({
            'User-Agent': 'T.A.R.S-PerformanceTest/1.0.2'
        })

        if self.auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.auth_token}'
            })

    def request(self, method: str, endpoint: str, **kwargs) -> RequestResult:
        """Execute a single HTTP request and measure performance"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        start_time = time.time()
        timestamp = start_time

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=30,
                **kwargs
            )

            latency_ms = (time.time() - start_time) * 1000

            return RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                latency_ms=latency_ms,
                response_size=len(response.content),
                timestamp=timestamp,
                success=response.status_code < 400,
                error=None
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            return RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                latency_ms=latency_ms,
                response_size=0,
                timestamp=timestamp,
                success=False,
                error=str(e)
            )


# ============================================================================
# System Monitoring
# ============================================================================

class SystemMonitor:
    """Monitor system resources during tests"""

    def __init__(self):
        self.metrics: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None

        if psutil is None:
            print("Warning: psutil not available, system monitoring disabled")

    def start(self):
        """Start monitoring system resources"""
        if psutil is None:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """Monitoring loop (runs in background thread)"""
        process = psutil.Process()

        while self.monitoring:
            try:
                metric = SystemMetrics(
                    cpu_percent=process.cpu_percent(interval=0.1),
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    memory_percent=process.memory_percent(),
                    timestamp=time.time()
                )
                self.metrics.append(metric)
            except Exception:
                pass

            time.sleep(1)

    def get_stats(self) -> Dict:
        """Get aggregated statistics"""
        if not self.metrics:
            return {
                'peak_cpu': 0.0,
                'avg_cpu': 0.0,
                'peak_memory_mb': 0.0,
                'avg_memory_mb': 0.0
            }

        cpu_values = [m.cpu_percent for m in self.metrics]
        mem_values = [m.memory_mb for m in self.metrics]

        return {
            'peak_cpu': max(cpu_values),
            'avg_cpu': statistics.mean(cpu_values),
            'peak_memory_mb': max(mem_values),
            'avg_memory_mb': statistics.mean(mem_values)
        }


# ============================================================================
# Test Scenarios
# ============================================================================

class TestScenario:
    """Test scenario definition"""

    @staticmethod
    def get_test_endpoints() -> List[Tuple[str, str]]:
        """Get list of endpoints to test (method, endpoint)"""
        return [
            ('GET', '/health'),
            ('GET', '/api/agents'),
            ('GET', '/api/agents/dqn'),
            ('GET', '/api/agents/a2c'),
            ('GET', '/api/hyperparameters'),
            ('GET', '/api/training/status'),
            ('GET', '/api/metrics'),
            ('POST', '/api/agents/dqn/train'),  # Quick training
        ]

    @staticmethod
    def get_request_kwargs(method: str, endpoint: str) -> Dict:
        """Get request kwargs for endpoint"""
        if method == 'POST' and '/train' in endpoint:
            return {
                'json': {
                    'episodes': 10,
                    'profile': 'quick'
                }
            }
        return {}


# ============================================================================
# Performance Tester
# ============================================================================

class PerformanceTester:
    """Main performance testing engine"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.client = PerformanceHTTPClient(
            base_url=config.url,
            auth_token=config.auth_token
        )
        self.monitor = SystemMonitor()
        self.results: List[RequestResult] = []
        self.results_lock = threading.Lock()

    def run_tests(self) -> PerformanceReport:
        """Run complete performance test suite"""
        test_id = str(uuid.uuid4())

        if self.config.verbose:
            print(f"Starting performance test: {test_id}")
            print(f"Target: {self.config.url}")
            print(f"Duration: {self.config.duration}s")
            print(f"Concurrency: {self.config.concurrency}")
            print()

        # Start system monitoring
        self.monitor.start()

        # Run tests
        start_time = time.time()
        self._run_load_test()
        end_time = time.time()

        # Stop monitoring
        self.monitor.stop()

        duration = end_time - start_time

        if self.config.verbose:
            print(f"\nTest completed in {duration:.2f}s")
            print(f"Total requests: {len(self.results)}")

        # Generate report
        report = self._generate_report(test_id, duration)

        # Regression detection
        if self.config.baseline_path:
            report = self._detect_regressions(report)

        return report

    def _run_load_test(self):
        """Run load test with specified concurrency and duration"""
        end_time = time.time() + self.config.duration
        endpoints = TestScenario.get_test_endpoints()

        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = []

            # Submit initial batch
            for _ in range(self.config.concurrency):
                future = executor.submit(
                    self._worker_loop,
                    endpoints,
                    end_time
                )
                futures.append(future)

            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    if self.config.verbose:
                        print(f"Worker error: {e}")

    def _worker_loop(self, endpoints: List[Tuple[str, str]], end_time: float):
        """Worker loop - make requests until time expires"""
        endpoint_idx = 0

        while time.time() < end_time:
            method, endpoint = endpoints[endpoint_idx % len(endpoints)]
            kwargs = TestScenario.get_request_kwargs(method, endpoint)

            result = self.client.request(method, endpoint, **kwargs)

            with self.results_lock:
                self.results.append(result)

            if self.config.verbose and len(self.results) % 100 == 0:
                print(f"  Completed {len(self.results)} requests...")

            endpoint_idx += 1

            # Small delay to avoid overwhelming server
            time.sleep(0.01)

    def _generate_report(self, test_id: str, duration: float) -> PerformanceReport:
        """Generate performance report from results"""

        # Overall statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        error_rate = (failed / total * 100) if total > 0 else 0

        # Latency statistics
        latencies = [r.latency_ms for r in self.results]
        latencies.sort()

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            return data[f] + (k - f) * (data[c] - data[f])

        # Per-endpoint statistics
        endpoint_stats = self._calculate_endpoint_stats(duration)

        # System metrics
        system_stats = self.monitor.get_stats()

        # Throughput
        rps = total / duration if duration > 0 else 0

        report = PerformanceReport(
            test_id=test_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            config={
                'url': self.config.url,
                'duration': self.config.duration,
                'concurrency': self.config.concurrency,
                'profile': self.config.profile
            },
            duration_seconds=duration,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            error_rate=error_rate,
            mean_latency=statistics.mean(latencies) if latencies else 0,
            median_latency=statistics.median(latencies) if latencies else 0,
            p95_latency=percentile(latencies, 0.95),
            p99_latency=percentile(latencies, 0.99),
            min_latency=min(latencies) if latencies else 0,
            max_latency=max(latencies) if latencies else 0,
            requests_per_second=rps,
            endpoint_stats=endpoint_stats,
            peak_cpu=system_stats['peak_cpu'],
            peak_memory_mb=system_stats['peak_memory_mb'],
            avg_cpu=system_stats['avg_cpu'],
            avg_memory_mb=system_stats['avg_memory_mb'],
            regression_detected=False
        )

        return report

    def _calculate_endpoint_stats(self, duration: float) -> List[Dict]:
        """Calculate per-endpoint statistics"""
        from collections import defaultdict

        endpoint_results = defaultdict(list)
        for result in self.results:
            key = f"{result.method} {result.endpoint}"
            endpoint_results[key].append(result)

        stats = []

        for key, results in endpoint_results.items():
            method, endpoint = key.split(' ', 1)

            total = len(results)
            successful = sum(1 for r in results if r.success)
            failed = total - successful

            latencies = [r.latency_ms for r in results]
            latencies.sort()

            sizes = [r.response_size for r in results if r.success]

            def percentile(data: List[float], p: float) -> float:
                if not data:
                    return 0.0
                k = (len(data) - 1) * p
                f = int(k)
                c = f + 1
                if c >= len(data):
                    return data[-1]
                return data[f] + (k - f) * (data[c] - data[f])

            stat = EndpointStats(
                endpoint=endpoint,
                method=method,
                total_requests=total,
                successful_requests=successful,
                failed_requests=failed,
                error_rate=(failed / total * 100) if total > 0 else 0,
                min_latency=min(latencies) if latencies else 0,
                max_latency=max(latencies) if latencies else 0,
                mean_latency=statistics.mean(latencies) if latencies else 0,
                median_latency=statistics.median(latencies) if latencies else 0,
                p95_latency=percentile(latencies, 0.95),
                p99_latency=percentile(latencies, 0.99),
                requests_per_second=total / duration if duration > 0 else 0,
                avg_response_size=int(statistics.mean(sizes)) if sizes else 0
            )

            stats.append(asdict(stat))

        return stats

    def _detect_regressions(self, report: PerformanceReport) -> PerformanceReport:
        """Detect performance regressions compared to baseline"""

        try:
            with open(self.config.baseline_path, 'r') as f:
                baseline = json.load(f)

            regressions = []

            # Check p95 latency
            baseline_p95 = baseline.get('p95_latency', 0)
            if baseline_p95 > 0:
                change_pct = ((report.p95_latency - baseline_p95) / baseline_p95) * 100
                if change_pct > 10:  # 10% regression threshold
                    regressions.append({
                        'metric': 'p95_latency',
                        'baseline': baseline_p95,
                        'current': report.p95_latency,
                        'change_percent': change_pct
                    })

            # Check p99 latency
            baseline_p99 = baseline.get('p99_latency', 0)
            if baseline_p99 > 0:
                change_pct = ((report.p99_latency - baseline_p99) / baseline_p99) * 100
                if change_pct > 15:  # 15% regression threshold
                    regressions.append({
                        'metric': 'p99_latency',
                        'baseline': baseline_p99,
                        'current': report.p99_latency,
                        'change_percent': change_pct
                    })

            # Check throughput
            baseline_rps = baseline.get('requests_per_second', 0)
            if baseline_rps > 0:
                change_pct = ((report.requests_per_second - baseline_rps) / baseline_rps) * 100
                if change_pct < -10:  # 10% throughput degradation
                    regressions.append({
                        'metric': 'requests_per_second',
                        'baseline': baseline_rps,
                        'current': report.requests_per_second,
                        'change_percent': change_pct
                    })

            # Check error rate
            baseline_error = baseline.get('error_rate', 0)
            if report.error_rate - baseline_error > 1.0:  # 1% increase in errors
                regressions.append({
                    'metric': 'error_rate',
                    'baseline': baseline_error,
                    'current': report.error_rate,
                    'change_percent': None
                })

            if regressions:
                report.regression_detected = True
                report.regression_details = {
                    'baseline_file': self.config.baseline_path,
                    'regressions': regressions
                }

        except FileNotFoundError:
            if self.config.verbose:
                print(f"Warning: Baseline file not found: {self.config.baseline_path}")
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Error loading baseline: {e}")

        return report


# ============================================================================
# Report Generation
# ============================================================================

class ReportGenerator:
    """Generate JSON and Markdown reports"""

    @staticmethod
    def generate_json(report: PerformanceReport, output_path: str):
        """Generate JSON report"""
        report_dict = asdict(report)

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

    @staticmethod
    def generate_markdown(report: PerformanceReport, output_path: str):
        """Generate Markdown report"""

        md = [
            "# T.A.R.S. Performance Test Report",
            "",
            f"**Test ID:** `{report.test_id}`",
            f"**Timestamp:** {report.timestamp}",
            f"**Duration:** {report.duration_seconds:.2f}s",
            "",
            "---",
            "",
            "## Configuration",
            "",
            f"- **Target URL:** {report.config['url']}",
            f"- **Duration:** {report.config['duration']}s",
            f"- **Concurrency:** {report.config['concurrency']} workers",
            f"- **Profile:** {report.config['profile']}",
            "",
            "---",
            "",
            "## Overall Results",
            "",
            "### Request Statistics",
            "",
            f"- **Total Requests:** {report.total_requests:,}",
            f"- **Successful:** {report.successful_requests:,} ({(report.successful_requests/report.total_requests*100):.2f}%)" if report.total_requests > 0 else "- **Successful:** 0",
            f"- **Failed:** {report.failed_requests:,} ({report.error_rate:.2f}%)",
            f"- **Error Rate:** {report.error_rate:.2f}%",
            "",
            "### Latency",
            "",
            f"- **Mean:** {report.mean_latency:.2f}ms",
            f"- **Median (p50):** {report.median_latency:.2f}ms",
            f"- **p95:** {report.p95_latency:.2f}ms",
            f"- **p99:** {report.p99_latency:.2f}ms",
            f"- **Min:** {report.min_latency:.2f}ms",
            f"- **Max:** {report.max_latency:.2f}ms",
            "",
            "### Throughput",
            "",
            f"- **Requests/sec:** {report.requests_per_second:.2f}",
            "",
            "### System Resources",
            "",
            f"- **Peak CPU:** {report.peak_cpu:.2f}%",
            f"- **Average CPU:** {report.avg_cpu:.2f}%",
            f"- **Peak Memory:** {report.peak_memory_mb:.2f} MB",
            f"- **Average Memory:** {report.avg_memory_mb:.2f} MB",
            "",
        ]

        # Regression analysis
        if report.regression_detected and report.regression_details:
            md.extend([
                "---",
                "",
                "## Regression Analysis",
                "",
                f"**Status:** ❌ REGRESSIONS DETECTED",
                "",
                f"**Baseline:** `{report.regression_details['baseline_file']}`",
                "",
                "### Detected Regressions",
                "",
                "| Metric | Baseline | Current | Change |",
                "|--------|----------|---------|--------|"
            ])

            for reg in report.regression_details['regressions']:
                metric = reg['metric']
                baseline = f"{reg['baseline']:.2f}"
                current = f"{reg['current']:.2f}"

                if reg['change_percent'] is not None:
                    change = f"{reg['change_percent']:+.2f}%"
                else:
                    change = "N/A"

                md.append(f"| {metric} | {baseline} | {current} | {change} |")

            md.append("")

        # Per-endpoint statistics
        if report.endpoint_stats:
            md.extend([
                "---",
                "",
                "## Per-Endpoint Statistics",
                "",
                "| Endpoint | Method | Requests | Success Rate | p95 Latency | RPS |",
                "|----------|--------|----------|--------------|-------------|-----|"
            ])

            for stat in report.endpoint_stats:
                success_rate = (stat['successful_requests'] / stat['total_requests'] * 100) if stat['total_requests'] > 0 else 0
                md.append(
                    f"| `{stat['endpoint']}` | {stat['method']} | "
                    f"{stat['total_requests']:,} | {success_rate:.2f}% | "
                    f"{stat['p95_latency']:.2f}ms | {stat['requests_per_second']:.2f} |"
                )

            md.append("")

        # Footer
        md.extend([
            "---",
            "",
            f"*Generated by T.A.R.S. Performance Testing Suite v1.0.2*",
            f"*Platform: {platform.system()} {platform.release()}*",
            f"*Python: {platform.python_version()}*",
            ""
        ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(md))


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='T.A.R.S. Performance Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python performance/run_performance_tests.py --url http://localhost:3001

  # Full test with regression detection
  python performance/run_performance_tests.py \\
      --url https://tars.company.com \\
      --duration 600 \\
      --concurrency 100 \\
      --baseline baseline.json \\
      --output-json results.json \\
      --output-md report.md \\
      --verbose

  # Stress test
  python performance/run_performance_tests.py \\
      --url https://tars-staging.company.com \\
      --duration 1800 \\
      --concurrency 200 \\
      --profile stress
        """
    )

    parser.add_argument(
        '--url',
        required=True,
        help='Target URL (e.g., https://tars.company.com)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=10,
        help='Number of concurrent workers (default: 10)'
    )

    parser.add_argument(
        '--profile',
        default='standard',
        choices=['quick', 'standard', 'stress'],
        help='Test profile (default: standard)'
    )

    parser.add_argument(
        '--baseline',
        dest='baseline_path',
        help='Path to baseline JSON for regression detection'
    )

    parser.add_argument(
        '--output-json',
        help='Output JSON report path'
    )

    parser.add_argument(
        '--output-md',
        help='Output Markdown report path'
    )

    parser.add_argument(
        '--config',
        dest='config_path',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--auth-token',
        help='JWT authentication token'
    )

    parser.add_argument(
        '--encryption-enabled',
        action='store_true',
        help='Enable encryption (requires enterprise_config)'
    )

    parser.add_argument(
        '--signing-enabled',
        action='store_true',
        help='Enable signing (requires enterprise_config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create test config
    config = TestConfig(
        url=args.url,
        duration=args.duration,
        concurrency=args.concurrency,
        profile=args.profile,
        baseline_path=args.baseline_path,
        output_json=args.output_json,
        output_md=args.output_md,
        verbose=args.verbose,
        config_path=args.config_path,
        encryption_enabled=args.encryption_enabled,
        signing_enabled=args.signing_enabled,
        auth_token=args.auth_token
    )

    # Run tests
    tester = PerformanceTester(config)

    try:
        report = tester.run_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST SUMMARY")
    print("="*80)
    print(f"Total Requests:     {report.total_requests:,}")
    print(f"Successful:         {report.successful_requests:,} ({(report.successful_requests/report.total_requests*100):.2f}%)" if report.total_requests > 0 else "Successful: 0")
    print(f"Failed:             {report.failed_requests:,} ({report.error_rate:.2f}%)")
    print()
    print(f"Mean Latency:       {report.mean_latency:.2f}ms")
    print(f"Median Latency:     {report.median_latency:.2f}ms")
    print(f"p95 Latency:        {report.p95_latency:.2f}ms")
    print(f"p99 Latency:        {report.p99_latency:.2f}ms")
    print()
    print(f"Throughput:         {report.requests_per_second:.2f} req/s")
    print()
    print(f"Peak CPU:           {report.peak_cpu:.2f}%")
    print(f"Peak Memory:        {report.peak_memory_mb:.2f} MB")
    print("="*80)

    # Regression detection
    if report.regression_detected:
        print()
        print("⚠️  REGRESSIONS DETECTED")
        print()
        for reg in report.regression_details['regressions']:
            metric = reg['metric']
            if reg['change_percent'] is not None:
                print(f"  - {metric}: {reg['change_percent']:+.2f}% (baseline: {reg['baseline']:.2f}, current: {reg['current']:.2f})")
            else:
                print(f"  - {metric}: baseline={reg['baseline']:.2f}, current={reg['current']:.2f}")
        print()

    # Generate reports
    if config.output_json:
        ReportGenerator.generate_json(report, config.output_json)
        print(f"\n✓ JSON report saved: {config.output_json}")

    if config.output_md:
        ReportGenerator.generate_markdown(report, config.output_md)
        print(f"✓ Markdown report saved: {config.output_md}")

    # Exit code
    if report.regression_detected:
        sys.exit(2)  # Regressions detected
    elif report.error_rate > 1.0:
        sys.exit(3)  # High error rate
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
