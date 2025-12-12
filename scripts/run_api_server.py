#!/usr/bin/env python3
"""
T.A.R.S. Enterprise API Server Launcher

Production-ready API server with:
- Enterprise configuration loading
- Security manager initialization
- CORS and rate limiting
- TLS support
- Health checks
- Graceful shutdown
- Prometheus metrics

Usage:
    python scripts/run_api_server.py
    python scripts/run_api_server.py --profile prod
    python scripts/run_api_server.py --profile prod --port 8443
"""

import argparse
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from enterprise_config import load_enterprise_config
from security import SecurityManager
from compliance import ComplianceEnforcer
from metrics import get_logger, setup_telemetry

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="T.A.R.S. Enterprise API Server"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="local",
        choices=["local", "dev", "staging", "prod"],
        help="Configuration profile to use"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override API port (default from config)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override API host (default from config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    parser.add_argument(
        "--no-tls",
        action="store_true",
        help="Disable TLS even if configured"
    )
    return parser.parse_args()


def setup_security(config):
    """Initialize security manager."""
    logger.info("Initializing security manager")

    security_config = {
        "encryption_enabled": config.security.encryption_enabled,
        "signing_enabled": config.security.signing_enabled,
    }

    if config.security.encryption_key_path:
        security_config["encryption_key_path"] = config.security.encryption_key_path

    if config.security.signing_key_path:
        security_config["signing_key_path"] = config.security.signing_key_path

    if config.security.public_key_path:
        security_config["public_key_path"] = config.security.public_key_path

    try:
        security_manager = SecurityManager(**security_config)
        logger.info(
            "Security manager initialized",
            extra={
                "encryption_enabled": security_config.get("encryption_enabled", False),
                "signing_enabled": security_config.get("signing_enabled", False)
            }
        )
        return security_manager
    except Exception as e:
        logger.warning(f"Failed to initialize security manager: {e}")
        logger.warning("Running without security features")
        return None


def setup_compliance(config):
    """Initialize compliance enforcer."""
    logger.info("Initializing compliance enforcer")

    try:
        enforcer = ComplianceEnforcer(
            enabled_standards=config.compliance.enabled_standards,
            enforcement_mode=config.compliance.enforcement_mode
        )

        status = enforcer.get_compliance_status()
        logger.info(
            "Compliance enforcer initialized",
            extra={
                "enabled_standards": config.compliance.enabled_standards,
                "enforcement_mode": config.compliance.enforcement_mode,
                "compliance_percentage": status.get("compliance_percentage", 0)
            }
        )
        return enforcer
    except Exception as e:
        logger.warning(f"Failed to initialize compliance enforcer: {e}")
        logger.warning("Running without compliance enforcement")
        return None


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, gracefully shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_tls_config(config):
    """Validate TLS configuration."""
    if not config.api.tls_enabled:
        return True

    cert_path = Path(config.api.tls_cert_path) if config.api.tls_cert_path else None
    key_path = Path(config.api.tls_key_path) if config.api.tls_key_path else None

    if not cert_path or not cert_path.exists():
        logger.error(f"TLS certificate not found: {config.api.tls_cert_path}")
        return False

    if not key_path or not key_path.exists():
        logger.error(f"TLS key not found: {config.api.tls_key_path}")
        return False

    logger.info(
        "TLS configuration validated",
        extra={
            "cert_path": str(cert_path),
            "key_path": str(key_path)
        }
    )
    return True


def main():
    """Main entry point."""
    args = parse_args()

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Load enterprise configuration
    logger.info(f"Loading enterprise configuration (profile: {args.profile})")

    overrides = {}
    if args.port:
        overrides["api.port"] = args.port
    if args.host:
        overrides["api.host"] = args.host
    if args.workers:
        overrides["api.workers"] = args.workers

    try:
        config = load_enterprise_config(profile=args.profile, overrides=overrides)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Setup telemetry
    setup_telemetry(
        prometheus_enabled=config.observability.prometheus_enabled,
        log_level=config.observability.log_level,
        log_format=config.observability.log_format
    )

    # Initialize security and compliance
    security_manager = setup_security(config)
    compliance_enforcer = setup_compliance(config)

    # Store in app state (will be available to enterprise_api)
    import enterprise_api.main as api_module
    if hasattr(api_module, "app"):
        api_module.app.state.security_manager = security_manager
        api_module.app.state.compliance_enforcer = compliance_enforcer
        api_module.app.state.config = config

    # Validate TLS configuration if enabled
    if config.api.tls_enabled and not args.no_tls:
        if not validate_tls_config(config):
            logger.error("TLS validation failed")
            sys.exit(1)

    # Prepare uvicorn configuration
    uvicorn_config = {
        "app": "enterprise_api.main:app",
        "host": config.api.host,
        "port": config.api.port,
        "workers": config.api.workers if not args.reload else 1,
        "log_level": config.observability.log_level.lower(),
        "access_log": True,
        "reload": args.reload,
    }

    # Add TLS if enabled
    if config.api.tls_enabled and not args.no_tls:
        uvicorn_config["ssl_certfile"] = config.api.tls_cert_path
        uvicorn_config["ssl_keyfile"] = config.api.tls_key_path
        protocol = "https"
    else:
        protocol = "http"

    # Log startup information
    logger.info(
        "Starting T.A.R.S. Enterprise API Server",
        extra={
            "profile": args.profile,
            "url": f"{protocol}://{config.api.host}:{config.api.port}",
            "workers": uvicorn_config["workers"],
            "tls_enabled": config.api.tls_enabled and not args.no_tls,
            "cors_enabled": config.api.cors_enabled,
            "rate_limit_enabled": config.api.rate_limit_enabled,
            "compliance_standards": config.compliance.enabled_standards,
            "reload": args.reload
        }
    )

    # Production warnings
    if args.profile == "prod":
        if args.reload:
            logger.warning("Auto-reload is enabled in production mode!")
        if args.no_tls and config.api.tls_enabled:
            logger.warning("TLS is disabled via --no-tls in production!")
        if not config.security.encryption_enabled:
            logger.warning("Encryption is disabled in production!")
        if not config.security.signing_enabled:
            logger.warning("Signing is disabled in production!")
        if config.compliance.enforcement_mode != "block":
            logger.warning(
                f"Compliance enforcement mode is '{config.compliance.enforcement_mode}' "
                f"(recommended: 'block' for production)"
            )

    # Start server
    try:
        uvicorn.run(**uvicorn_config)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
