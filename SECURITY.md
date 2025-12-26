# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Features

T.A.R.S. is designed with security and privacy as core principles:

### Data Protection
- **Encryption at Rest:** AES-256-GCM for all sensitive data
- **Encryption in Transit:** TLS 1.3 for all network communications
- **Data Sovereignty:** 100% on-premises, no cloud dependencies

### Authentication & Authorization
- **JWT Authentication:** HS256 with 60-minute access tokens
- **Refresh Tokens:** 7-day refresh tokens with rotation
- **RBAC:** Role-based access control (admin, sre, readonly)
- **Rate Limiting:** Redis-backed sliding window rate limiting

### Compliance
- **SOC 2 Type II:** 18 security controls
- **ISO 27001:** 20 information security controls
- **GDPR:** PII redaction and data minimization

### Supply Chain Security
- **SBOM:** Software Bill of Materials (CycloneDX, SPDX)
- **SLSA:** Level 3 provenance for releases
- **Dependency Scanning:** Regular security audits

## Reporting a Vulnerability

If you discover a security vulnerability in T.A.R.S., please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Send an email to the project maintainers with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Include

- Type of vulnerability (e.g., injection, authentication bypass, XSS)
- Affected component(s) and version(s)
- Proof of concept or reproduction steps
- Your assessment of severity

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial Assessment:** Within 5 business days
- **Fix Development:** Depends on severity
- **Disclosure:** Coordinated with reporter

## Security Best Practices

### For Operators

1. **Change Default Credentials**
   - Replace default JWT secret
   - Replace default API keys
   - Use strong, unique passwords

2. **Key Management**
   - Generate unique AES-256 keys for each environment
   - Generate unique RSA-4096 signing keys
   - Rotate keys periodically (recommended: 90 days)

3. **Network Security**
   - Enable TLS for all endpoints
   - Use network policies in Kubernetes
   - Restrict ingress to trusted sources

4. **Secrets Management**
   - Use Vault, AWS Secrets Manager, or similar
   - Never commit secrets to version control
   - Use environment-specific configurations

### For Developers

1. **Never commit:**
   - `.env` files with real credentials
   - Private keys or certificates
   - API keys or tokens
   - Database connection strings with passwords

2. **Always verify:**
   - Input validation on all endpoints
   - Authentication on sensitive operations
   - Authorization checks for role-based access

3. **Use secure defaults:**
   - HTTPS only in production
   - Strong password policies
   - Short token lifetimes

## Security Configuration

### Example Secure Configuration

```yaml
# enterprise_config/prod.yaml
security:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_path: "/etc/tars/secrets/aes.key"  # Generated, not default
  signing:
    enabled: true
    algorithm: "RSA-PSS"
    key_path: "/etc/tars/secrets/rsa.key"  # Generated, not default

compliance:
  enabled: true
  standards: ["soc2", "iso27001"]
  mode: "block"  # Enforce compliance

api:
  rate_limit:
    enabled: true
    requests_per_minute: 60
  auth:
    jwt_secret: "${JWT_SECRET}"  # From environment/vault
    token_expiry: 3600  # 1 hour
```

### Key Generation

```bash
# Generate AES-256 encryption key
openssl rand -hex 32 > /etc/tars/secrets/aes.key
chmod 600 /etc/tars/secrets/aes.key

# Generate RSA-4096 signing key
openssl genrsa -out /etc/tars/secrets/rsa.key 4096
chmod 600 /etc/tars/secrets/rsa.key

# Generate JWT secret
openssl rand -base64 32 > /etc/tars/secrets/jwt.key
chmod 600 /etc/tars/secrets/jwt.key
```

## Known Security Considerations

1. **Local Deployment Only:** T.A.R.S. is designed for on-premises deployment. External exposure requires additional hardening.

2. **Rate Limiting Requires Redis:** Rate limiting functionality depends on Redis. Without Redis, rate limiting is disabled.

3. **Default Credentials:** The default API key and JWT secret in example configs are for development only. Always change for production.

4. **Audit Logging:** Enable structured logging in production for security audit trails.

---

**Security Contact:** [To be configured by operator]

**Last Updated:** December 26, 2025
