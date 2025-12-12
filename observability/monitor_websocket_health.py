#!/usr/bin/env python3
"""
WebSocket Health Monitor - TARS-1001 Validation

Validates the WebSocket auto-reconnection fix (TARS-1001) in the GA environment.
Tests reconnection behavior, tracks success rates, and reports health metrics.

Usage:
    python monitor_websocket_health.py --endpoint ws://localhost:8080/ws --duration 3600
    python monitor_websocket_health.py --endpoint wss://tars.prod/ws --iterations 100 --slack-webhook <url>

Author: T.A.R.S. Platform Team
Phase: 14.4 - GA Day Monitoring
JIRA: TARS-1001 (WebSocket Auto-Reconnect)
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ReconnectionAttempt:
    """Records a single reconnection attempt."""
    timestamp: str
    attempt_number: int
    success: bool
    latency_ms: float
    error_message: Optional[str] = None
    jitter_ms: float = 0.0  # Time variation from expected reconnect interval


@dataclass
class WebSocketHealthMetrics:
    """Aggregated WebSocket health metrics."""
    test_start: str
    test_end: str
    duration_seconds: float

    # Connection metrics
    total_connections: int
    total_disconnections: int
    forced_disconnections: int

    # Reconnection metrics
    total_reconnection_attempts: int
    successful_reconnections: int
    failed_reconnections: int
    reconnection_success_rate: float  # percentage

    # Latency metrics
    avg_reconnection_latency_ms: float
    min_reconnection_latency_ms: float
    max_reconnection_latency_ms: float
    p50_reconnection_latency_ms: float
    p95_reconnection_latency_ms: float
    p99_reconnection_latency_ms: float

    # Jitter metrics
    avg_jitter_ms: float
    max_jitter_ms: float

    # Downtime tracking
    total_downtime_seconds: float
    max_downtime_seconds: float

    # TARS-1001 specific validation
    tars_1001_compliant: bool
    tars_1001_notes: List[str] = field(default_factory=list)

    # Detailed attempts
    attempts: List[ReconnectionAttempt] = field(default_factory=list)


class WebSocketHealthMonitor:
    """Monitors WebSocket health and validates auto-reconnection."""

    def __init__(
        self,
        endpoint: str,
        reconnect_interval_ms: int = 5000,
        max_reconnect_attempts: int = 5,
        slack_webhook: Optional[str] = None
    ):
        self.endpoint = endpoint
        self.reconnect_interval_ms = reconnect_interval_ms
        self.max_reconnect_attempts = max_reconnect_attempts
        self.slack_webhook = slack_webhook

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.reconnection_attempts: List[ReconnectionAttempt] = []
        self.downtime_periods: List[tuple[float, float]] = []  # (start, end) timestamps

        self.test_start_time: Optional[datetime] = None
        self.test_end_time: Optional[datetime] = None

        self.connection_count = 0
        self.disconnection_count = 0
        self.forced_disconnection_count = 0

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            logger.info(f"Connecting to {self.endpoint}...")
            self.ws = await websockets.connect(
                self.endpoint,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.connection_count += 1
            logger.info(f"Connected successfully (connection #{self.connection_count})")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self, forced: bool = False):
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
                self.disconnection_count += 1
                if forced:
                    self.forced_disconnection_count += 1
                logger.info(f"Disconnected ({'forced' if forced else 'graceful'})")
            except Exception as e:
                logger.error(f"Disconnect error: {e}")

    async def test_reconnection(self, iteration: int) -> ReconnectionAttempt:
        """Test a single reconnection cycle."""
        logger.info(f"\n--- Reconnection Test #{iteration} ---")

        # 1. Force disconnect
        disconnect_time = time.time()
        await self.disconnect(forced=True)

        # 2. Wait for auto-reconnect (with timeout)
        reconnect_start = time.time()
        expected_reconnect_time = self.reconnect_interval_ms / 1000.0

        success = False
        latency_ms = 0.0
        error_message = None
        attempt_count = 0

        # Wait for reconnection with retries
        while attempt_count < self.max_reconnect_attempts:
            attempt_count += 1
            attempt_start = time.time()

            try:
                # Try to reconnect
                logger.info(f"Reconnection attempt {attempt_count}/{self.max_reconnect_attempts}...")
                connected = await self.connect()

                if connected:
                    # Verify connection is working
                    await self.ws.send(json.dumps({"type": "ping"}))
                    response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)

                    reconnect_end = time.time()
                    latency_ms = (reconnect_end - reconnect_start) * 1000.0
                    success = True

                    logger.info(f"✅ Reconnection successful in {latency_ms:.2f}ms")
                    break

            except asyncio.TimeoutError:
                error_message = f"Attempt {attempt_count}: Timeout waiting for response"
                logger.warning(error_message)
            except ConnectionClosed as e:
                error_message = f"Attempt {attempt_count}: Connection closed: {e}"
                logger.warning(error_message)
            except Exception as e:
                error_message = f"Attempt {attempt_count}: {type(e).__name__}: {e}"
                logger.error(error_message)

            # Wait before next retry (exponential backoff)
            if attempt_count < self.max_reconnect_attempts:
                backoff_seconds = min(2 ** attempt_count, 30)
                await asyncio.sleep(backoff_seconds)

        if not success:
            reconnect_end = time.time()
            latency_ms = (reconnect_end - reconnect_start) * 1000.0
            logger.error(f"❌ Reconnection failed after {attempt_count} attempts")

            # Track downtime
            downtime_seconds = reconnect_end - disconnect_time
            self.downtime_periods.append((disconnect_time, reconnect_end))

        # Calculate jitter (deviation from expected reconnect interval)
        jitter_ms = abs(latency_ms - expected_reconnect_time * 1000.0)

        attempt = ReconnectionAttempt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            attempt_number=iteration,
            success=success,
            latency_ms=round(latency_ms, 2),
            error_message=error_message,
            jitter_ms=round(jitter_ms, 2)
        )

        self.reconnection_attempts.append(attempt)
        return attempt

    async def run_health_checks(
        self,
        duration_seconds: Optional[int] = None,
        iterations: Optional[int] = None,
        interval_seconds: int = 60
    ) -> WebSocketHealthMetrics:
        """Run WebSocket health checks."""
        self.test_start_time = datetime.now(timezone.utc)
        logger.info(f"Starting WebSocket health monitoring at {self.test_start_time}")

        if duration_seconds:
            logger.info(f"Running for {duration_seconds} seconds with {interval_seconds}s intervals")
        elif iterations:
            logger.info(f"Running {iterations} reconnection tests")
        else:
            raise ValueError("Must specify either duration_seconds or iterations")

        # Initial connection
        await self.connect()

        iteration = 1

        if duration_seconds:
            end_time = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)

            while datetime.now(timezone.utc) < end_time:
                try:
                    await self.test_reconnection(iteration)
                    iteration += 1

                    # Wait before next test
                    await asyncio.sleep(interval_seconds)

                except KeyboardInterrupt:
                    logger.warning("Health check interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error during health check: {e}", exc_info=True)
                    await asyncio.sleep(10)

        elif iterations:
            for i in range(iterations):
                try:
                    await self.test_reconnection(iteration)
                    iteration += 1

                    # Wait before next test (except last)
                    if i < iterations - 1:
                        await asyncio.sleep(interval_seconds)

                except KeyboardInterrupt:
                    logger.warning("Health check interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error during health check: {e}", exc_info=True)
                    await asyncio.sleep(10)

        # Clean up connection
        if self.ws:
            await self.disconnect()

        self.test_end_time = datetime.now(timezone.utc)
        logger.info(f"Health monitoring complete at {self.test_end_time}")

        # Calculate metrics
        return self.calculate_metrics()

    def calculate_metrics(self) -> WebSocketHealthMetrics:
        """Calculate aggregated health metrics."""
        if not self.test_start_time or not self.test_end_time:
            raise ValueError("Test times not set")

        duration = (self.test_end_time - self.test_start_time).total_seconds()

        # Reconnection metrics
        total_attempts = len(self.reconnection_attempts)
        successful = sum(1 for a in self.reconnection_attempts if a.success)
        failed = total_attempts - successful
        success_rate = (successful / total_attempts * 100.0) if total_attempts > 0 else 0.0

        # Latency metrics
        successful_latencies = [a.latency_ms for a in self.reconnection_attempts if a.success]

        avg_latency = statistics.mean(successful_latencies) if successful_latencies else 0.0
        min_latency = min(successful_latencies) if successful_latencies else 0.0
        max_latency = max(successful_latencies) if successful_latencies else 0.0

        p50_latency = statistics.median(successful_latencies) if successful_latencies else 0.0
        p95_latency = statistics.quantiles(successful_latencies, n=20)[18] if len(successful_latencies) >= 20 else max_latency
        p99_latency = statistics.quantiles(successful_latencies, n=100)[98] if len(successful_latencies) >= 100 else max_latency

        # Jitter metrics
        all_jitter = [a.jitter_ms for a in self.reconnection_attempts]
        avg_jitter = statistics.mean(all_jitter) if all_jitter else 0.0
        max_jitter = max(all_jitter) if all_jitter else 0.0

        # Downtime metrics
        total_downtime = sum(end - start for start, end in self.downtime_periods)
        max_downtime = max((end - start for start, end in self.downtime_periods), default=0.0)

        # TARS-1001 compliance validation
        tars_1001_notes = []
        tars_1001_compliant = True

        # Requirement 1: >= 95% reconnection success rate
        if success_rate < 95.0:
            tars_1001_compliant = False
            tars_1001_notes.append(f"❌ Success rate {success_rate:.2f}% < 95% threshold")
        else:
            tars_1001_notes.append(f"✅ Success rate {success_rate:.2f}% >= 95%")

        # Requirement 2: Average reconnection latency < 10s
        if avg_latency > 10000.0:
            tars_1001_compliant = False
            tars_1001_notes.append(f"❌ Avg latency {avg_latency:.2f}ms > 10s threshold")
        else:
            tars_1001_notes.append(f"✅ Avg latency {avg_latency:.2f}ms < 10s")

        # Requirement 3: P99 latency < 30s
        if p99_latency > 30000.0:
            tars_1001_compliant = False
            tars_1001_notes.append(f"❌ P99 latency {p99_latency:.2f}ms > 30s threshold")
        else:
            tars_1001_notes.append(f"✅ P99 latency {p99_latency:.2f}ms < 30s")

        # Requirement 4: Max downtime < 60s
        if max_downtime > 60.0:
            tars_1001_compliant = False
            tars_1001_notes.append(f"❌ Max downtime {max_downtime:.2f}s > 60s threshold")
        else:
            tars_1001_notes.append(f"✅ Max downtime {max_downtime:.2f}s < 60s")

        # Requirement 5: Jitter < 2s
        if avg_jitter > 2000.0:
            tars_1001_notes.append(f"⚠️ Avg jitter {avg_jitter:.2f}ms > 2s (warning only)")
        else:
            tars_1001_notes.append(f"✅ Avg jitter {avg_jitter:.2f}ms < 2s")

        return WebSocketHealthMetrics(
            test_start=self.test_start_time.isoformat(),
            test_end=self.test_end_time.isoformat(),
            duration_seconds=round(duration, 2),
            total_connections=self.connection_count,
            total_disconnections=self.disconnection_count,
            forced_disconnections=self.forced_disconnection_count,
            total_reconnection_attempts=total_attempts,
            successful_reconnections=successful,
            failed_reconnections=failed,
            reconnection_success_rate=round(success_rate, 2),
            avg_reconnection_latency_ms=round(avg_latency, 2),
            min_reconnection_latency_ms=round(min_latency, 2),
            max_reconnection_latency_ms=round(max_latency, 2),
            p50_reconnection_latency_ms=round(p50_latency, 2),
            p95_reconnection_latency_ms=round(p95_latency, 2),
            p99_reconnection_latency_ms=round(p99_latency, 2),
            avg_jitter_ms=round(avg_jitter, 2),
            max_jitter_ms=round(max_jitter, 2),
            total_downtime_seconds=round(total_downtime, 2),
            max_downtime_seconds=round(max_downtime, 2),
            tars_1001_compliant=tars_1001_compliant,
            tars_1001_notes=tars_1001_notes,
            attempts=self.reconnection_attempts
        )

    async def send_slack_notification(
        self,
        metrics: WebSocketHealthMetrics,
        output_file: Optional[Path] = None
    ):
        """Send health metrics to Slack."""
        if not self.slack_webhook:
            logger.info("No Slack webhook configured, skipping notification")
            return

        # Build Slack message
        status_emoji = "✅" if metrics.tars_1001_compliant else "❌"
        status_text = "PASS" if metrics.tars_1001_compliant else "FAIL"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} WebSocket Health Check - {status_text}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Endpoint:*\n{self.endpoint}"},
                    {"type": "mrkdwn", "text": f"*Duration:*\n{metrics.duration_seconds}s"},
                    {"type": "mrkdwn", "text": f"*Success Rate:*\n{metrics.reconnection_success_rate}%"},
                    {"type": "mrkdwn", "text": f"*Avg Latency:*\n{metrics.avg_reconnection_latency_ms}ms"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*TARS-1001 Validation:*\n" + "\n".join(metrics.tars_1001_notes)
                }
            }
        ]

        if output_file:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Report:* `{output_file}`"
                }
            })

        payload = {
            "blocks": blocks
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        logger.error(f"Slack notification failed: {resp.status}")
                    else:
                        logger.info("Slack notification sent successfully")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    def save_metrics(self, metrics: WebSocketHealthMetrics, output_file: Path):
        """Save health metrics to JSON file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        logger.info(f"Metrics saved: {output_file}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="T.A.R.S. WebSocket Health Monitor")
    parser.add_argument("--endpoint", type=str, required=True, help="WebSocket endpoint URL")
    parser.add_argument("--duration", type=int, help="Test duration in seconds")
    parser.add_argument("--iterations", type=int, help="Number of reconnection tests")
    parser.add_argument("--interval", type=int, default=60, help="Interval between tests in seconds (default: 60)")
    parser.add_argument("--reconnect-interval", type=int, default=5000, help="Expected reconnect interval in ms (default: 5000)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max reconnection attempts (default: 5)")
    parser.add_argument("--output", type=str, default="ws_health_metrics.json", help="Output file (default: ws_health_metrics.json)")
    parser.add_argument("--slack-webhook", type=str, help="Slack webhook URL for notifications")

    args = parser.parse_args()

    if not args.duration and not args.iterations:
        parser.error("Must specify either --duration or --iterations")

    # Create monitor
    monitor = WebSocketHealthMonitor(
        endpoint=args.endpoint,
        reconnect_interval_ms=args.reconnect_interval,
        max_reconnect_attempts=args.max_retries,
        slack_webhook=args.slack_webhook
    )

    # Run health checks
    metrics = await monitor.run_health_checks(
        duration_seconds=args.duration,
        iterations=args.iterations,
        interval_seconds=args.interval
    )

    # Save metrics
    output_file = Path(args.output)
    monitor.save_metrics(metrics, output_file)

    # Send Slack notification
    if args.slack_webhook:
        await monitor.send_slack_notification(metrics, output_file)

    # Print summary
    print("\n" + "="*80)
    print("WebSocket Health Check Summary")
    print("="*80)
    print(f"Endpoint: {args.endpoint}")
    print(f"Duration: {metrics.duration_seconds}s")
    print(f"Total Tests: {metrics.total_reconnection_attempts}")
    print(f"Success Rate: {metrics.reconnection_success_rate}%")
    print(f"Avg Latency: {metrics.avg_reconnection_latency_ms}ms")
    print(f"P99 Latency: {metrics.p99_reconnection_latency_ms}ms")
    print(f"Max Downtime: {metrics.max_downtime_seconds}s")
    print(f"\nTARS-1001 Compliance: {'✅ PASS' if metrics.tars_1001_compliant else '❌ FAIL'}")
    for note in metrics.tars_1001_notes:
        print(f"  {note}")
    print("="*80)

    # Exit with appropriate code
    sys.exit(0 if metrics.tars_1001_compliant else 1)


if __name__ == "__main__":
    asyncio.run(main())
