# TLS Certificate Expiration Monitoring

Production-grade TLS certificate monitoring system for T.A.R.S. security infrastructure.

## Overview

The Certificate Monitor module provides comprehensive certificate expiration tracking with:

- **Remote Certificate Checking**: Validate TLS certificates from network endpoints
- **Local File Monitoring**: Inspect PEM-encoded certificate files
- **Multi-level Alerting**: Configurable severity thresholds (CRITICAL, HIGH, WARNING, INFO)
- **Prometheus Integration**: Export metrics for monitoring dashboards
- **Production-Ready**: Full error handling, logging, and caching

## Quick Start

### Check a Single Domain

```python
from security import check_domain_certificate

# Quick certificate check
cert_info = check_domain_certificate("example.com")

if cert_info:
    print(f"Expires in {cert_info.days_until_expiry} days")
    print(f"Issuer: {cert_info.issuer}")
```

### Monitor Multiple Domains

```python
from security import CertificateMonitor

# Create monitor with domains to watch
monitor = CertificateMonitor(
    monitored_domains=[
        "api.example.com:443",
        "www.example.com:443"
    ]
)

# Check all and generate alerts
alerts = monitor.check_all()

for alert in alerts:
    if alert.severity == "CRITICAL":
        print(f"URGENT: {alert.message}")
```

### Check Certificate Files

```python
from pathlib import Path
from security import CertificateMonitor

# Monitor local certificate files
monitor = CertificateMonitor(
    monitored_files=[
        Path("/etc/ssl/certs/server.crt"),
        Path("/etc/ssl/certs/client.crt")
    ]
)

alerts = monitor.check_all()
```

## Architecture

### Data Models

#### CertificateInfo

Container for certificate details:

```python
@dataclass
class CertificateInfo:
    domain: str                  # Domain or file path
    issuer: str                  # Certificate issuer (CA)
    subject: str                 # Certificate subject
    not_before: datetime         # Valid from date
    not_after: datetime          # Expiration date
    serial_number: str           # Certificate serial number
    days_until_expiry: int       # Days remaining (negative if expired)
    is_expired: bool             # Expiration status
```

#### CertificateAlert

Alert generated for expiring certificates:

```python
@dataclass
class CertificateAlert:
    domain: str                  # Certificate identifier
    severity: str                # CRITICAL, HIGH, WARNING, INFO
    message: str                 # Human-readable message
    days_remaining: int          # Days until expiration
    expires_at: datetime         # Expiration timestamp
```

### Severity Thresholds

| Severity | Threshold | Use Case |
|----------|-----------|----------|
| CRITICAL | ≤ 7 days | Immediate action required |
| HIGH | ≤ 14 days | Plan renewal soon |
| WARNING | ≤ 30 days | Schedule renewal |
| INFO | > 30 days | Normal monitoring |

Thresholds are configurable via `CertificateMonitor.THRESHOLDS`:

```python
monitor = CertificateMonitor()
monitor.THRESHOLDS["CRITICAL"] = 3  # Change to 3 days
```

## API Reference

### CertificateMonitor

Main monitoring class.

#### Constructor

```python
CertificateMonitor(
    monitored_domains: Optional[List[str]] = None,
    monitored_files: Optional[List[Path]] = None
)
```

**Parameters:**
- `monitored_domains`: List of domains in "domain:port" format (default port: 443)
- `monitored_files`: List of Path objects to certificate files

#### Methods

##### check_certificate(domain, port=443)

Check certificate from network endpoint.

```python
cert_info = monitor.check_certificate("example.com", 443)
```

**Returns:** `CertificateInfo` or `None` if check fails

##### check_certificate_file(cert_path)

Check certificate from local file.

```python
cert_info = monitor.check_certificate_file(Path("/etc/ssl/cert.pem"))
```

**Returns:** `CertificateInfo` or `None` if parsing fails

##### check_all()

Check all monitored certificates.

```python
alerts = monitor.check_all()
```

**Returns:** `List[CertificateAlert]`

##### get_alert_severity(days_remaining)

Calculate severity level.

```python
severity = monitor.get_alert_severity(5)  # Returns "CRITICAL"
```

**Returns:** Severity string

##### get_prometheus_metrics()

Generate Prometheus metrics.

```python
metrics = monitor.get_prometheus_metrics()
print(metrics)
```

**Returns:** Prometheus text format metrics

##### get_certificate_info(identifier)

Retrieve cached certificate information.

```python
cached = monitor.get_certificate_info("example.com:443")
```

**Returns:** `CertificateInfo` or `None`

##### clear_cache()

Clear certificate cache.

```python
monitor.clear_cache()
```

### Convenience Functions

#### check_domain_certificate(domain, port=443)

Quick one-off domain check without creating a monitor instance.

```python
from security import check_domain_certificate

cert = check_domain_certificate("google.com")
```

#### check_certificate_file(cert_path)

Quick one-off file check.

```python
from security import check_certificate_file
from pathlib import Path

cert = check_certificate_file(Path("/etc/ssl/server.crt"))
```

## Prometheus Metrics

The module exports four metric families:

### certificate_expiry_days

Days until certificate expires.

```
certificate_expiry_days{domain="example.com:443"} 45
```

### certificate_expiry_timestamp

Unix timestamp of expiration.

```
certificate_expiry_timestamp{domain="example.com:443"} 1735689600
```

### certificate_expired

Boolean indicator (1 = expired, 0 = valid).

```
certificate_expired{domain="example.com:443"} 0
```

### certificate_severity

Severity level (0=INFO, 1=WARNING, 2=HIGH, 3=CRITICAL).

```
certificate_severity{domain="example.com:443",severity="WARNING"} 1
```

## Integration Examples

### Prometheus Alerting

```yaml
# prometheus/alerts.yml
groups:
  - name: certificate_alerts
    rules:
      - alert: CertificateExpiringSoon
        expr: certificate_expiry_days < 14
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Certificate expiring soon"
          description: "Certificate for {{ $labels.domain }} expires in {{ $value }} days"

      - alert: CertificateCritical
        expr: certificate_expiry_days < 7
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Certificate expiring very soon"
          description: "URGENT: Certificate for {{ $labels.domain }} expires in {{ $value }} days"
```

### Scheduled Monitoring

```python
import schedule
import time
from security import CertificateMonitor

def check_certificates():
    monitor = CertificateMonitor(
        monitored_domains=[
            "api.example.com:443",
            "www.example.com:443"
        ]
    )

    alerts = monitor.check_all()

    # Send critical alerts to Slack/PagerDuty
    critical = [a for a in alerts if a.severity == "CRITICAL"]
    for alert in critical:
        send_alert(alert.message)

# Run every 6 hours
schedule.every(6).hours.do(check_certificates)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Flask/FastAPI Integration

```python
from flask import Flask, Response
from security import CertificateMonitor

app = Flask(__name__)
monitor = CertificateMonitor(monitored_domains=["api.example.com:443"])

@app.route("/metrics")
def metrics():
    monitor.check_all()
    return Response(
        monitor.get_prometheus_metrics(),
        mimetype="text/plain"
    )

if __name__ == "__main__":
    app.run(port=9090)
```

## Error Handling

The module handles errors gracefully:

- **Connection Timeouts**: Returns `None`, logs error
- **DNS Resolution Failures**: Returns `None`, logs error
- **SSL Errors**: Returns `None`, logs error
- **Invalid Certificate Files**: Returns `None`, logs error
- **Missing Files**: Returns `None`, logs error

All errors are logged at ERROR level with descriptive messages.

## Logging

Configure logging to capture certificate checks:

```python
import logging

# Enable certificate monitor logging
logging.basicConfig(level=logging.INFO)

# Or configure specific logger
logger = logging.getLogger('security.certificate_monitor')
logger.setLevel(logging.DEBUG)
```

Log levels:
- **DEBUG**: Individual certificate checks
- **INFO**: Check results, cache operations
- **ERROR**: Failures and errors

## Testing

Run comprehensive test suite:

```bash
# All tests
pytest tests/test_certificate_monitor.py -v

# Unit tests only (no network)
pytest tests/test_certificate_monitor.py -v -m "not integration"

# Integration tests (requires network)
pytest tests/test_certificate_monitor.py -v -m integration

# Coverage report
pytest tests/test_certificate_monitor.py --cov=security.certificate_monitor
```

## Performance

- **Connection Timeout**: 10 seconds per domain
- **Concurrent Checks**: Sequential (to avoid overwhelming servers)
- **Memory**: Minimal (certificates cached as dataclasses)
- **Recommended Check Frequency**: Every 6-12 hours

## Security Considerations

1. **Certificate Validation**: Uses system's default SSL context
2. **No Private Keys**: Only reads public certificates
3. **Read-Only**: No certificate modification capabilities
4. **Safe Defaults**: Validates certificates using system trust store

## Troubleshooting

### "No certificate received"

- Check network connectivity
- Verify domain and port are correct
- Ensure TLS is enabled on the endpoint

### "Invalid certificate format"

- Ensure file is PEM-encoded
- Check file permissions
- Verify certificate is not corrupted

### "Connection timeout"

- Check firewall rules
- Verify endpoint is accessible
- Increase timeout if needed (modify source)

## Examples

See `examples/certificate_monitoring_demo.py` for comprehensive usage examples.

## Dependencies

- `cryptography`: X.509 certificate parsing
- `ssl`: TLS connection handling
- Python 3.8+

## License

Proprietary - Veleron Dev Studios

## Support

For issues or questions:
- GitHub Issues: [VDS_TARS Issues](https://github.com/veleron-dev/tars/issues)
- Documentation: `docs/`
- Tests: `tests/test_certificate_monitor.py`
