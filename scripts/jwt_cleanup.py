"""
T.A.R.S. JWT Key Cleanup Script
Periodic cleanup job for expired JWT keys
Phase 12 Part 2
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add cognition shared to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cognition', 'shared'))

from jwt_key_store import jwt_key_store

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, start_http_server
    METRICS_AVAILABLE = True

    jwt_cleanup_total = Counter(
        'jwt_cleanup_total',
        'Total JWT cleanup runs'
    )

    jwt_keys_cleaned_total = Counter(
        'jwt_keys_cleaned_total',
        'Total JWT keys cleaned up'
    )

    jwt_cleanup_duration_seconds = Histogram(
        'jwt_cleanup_duration_seconds',
        'JWT cleanup duration in seconds'
    )

    jwt_cleanup_errors_total = Counter(
        'jwt_cleanup_errors_total',
        'Total JWT cleanup errors'
    )

except ImportError:
    METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/jwt_cleanup.log')
    ]
)

logger = logging.getLogger(__name__)


def cleanup_expired_keys():
    """
    Clean up expired JWT keys

    Returns:
        int: Number of keys cleaned up
    """
    logger.info("Starting JWT key cleanup")

    start_time = time.time()

    try:
        # Run cleanup
        deleted_count = jwt_key_store.cleanup_expired()

        duration = time.time() - start_time

        # Metrics
        if METRICS_AVAILABLE:
            jwt_cleanup_total.inc()
            jwt_keys_cleaned_total.inc(deleted_count)
            jwt_cleanup_duration_seconds.observe(duration)

        logger.info(f"JWT key cleanup completed: {deleted_count} keys cleaned up in {duration:.2f}s")

        return deleted_count

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"JWT key cleanup failed after {duration:.2f}s: {e}", exc_info=True)

        if METRICS_AVAILABLE:
            jwt_cleanup_errors_total.inc()

        raise


def health_check():
    """
    Health check for JWT key store

    Returns:
        dict: Health status
    """
    try:
        health = jwt_key_store.health_check()
        logger.info(f"JWT key store health: {health}")
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def run_once():
    """Run cleanup once and exit"""
    logger.info("Running JWT cleanup (one-time)")

    # Health check
    health = health_check()
    if health.get("status") == "unhealthy":
        logger.error("JWT key store is unhealthy, aborting cleanup")
        sys.exit(1)

    # Run cleanup
    try:
        deleted = cleanup_expired_keys()
        logger.info(f"Cleanup complete: {deleted} keys deleted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


def run_daemon(interval_hours: int = 24):
    """
    Run cleanup as a daemon (continuous loop)

    Args:
        interval_hours: Hours between cleanup runs (default: 24)
    """
    logger.info(f"Starting JWT cleanup daemon (interval: {interval_hours}h)")

    # Start Prometheus metrics server if available
    if METRICS_AVAILABLE:
        metrics_port = int(os.getenv("METRICS_PORT", "9999"))
        try:
            start_http_server(metrics_port)
            logger.info(f"Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start metrics server: {e}")

    interval_seconds = interval_hours * 3600

    while True:
        try:
            # Health check
            health = health_check()
            if health.get("status") == "unhealthy":
                logger.warning("JWT key store is unhealthy, skipping cleanup")
            else:
                # Run cleanup
                cleanup_expired_keys()

        except Exception as e:
            logger.error(f"Cleanup iteration failed: {e}", exc_info=True)

        # Sleep until next iteration
        logger.info(f"Next cleanup in {interval_hours}h")
        time.sleep(interval_seconds)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="T.A.R.S. JWT Key Cleanup")
    parser.add_argument(
        "--mode",
        choices=["once", "daemon"],
        default="once",
        help="Run mode: once (single run) or daemon (continuous)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Cleanup interval in hours (daemon mode only)"
    )

    args = parser.parse_args()

    logger.info(f"T.A.R.S. JWT Cleanup starting in {args.mode} mode")

    if args.mode == "once":
        run_once()
    elif args.mode == "daemon":
        run_daemon(interval_hours=args.interval)


if __name__ == "__main__":
    main()
