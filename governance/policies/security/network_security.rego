# Security Policy: Network Security
package tars.security.network_security

default allow = false

# Allow network communication if security checks pass
allow {
    input.action == "network_connect"
    is_allowed_protocol
    is_allowed_destination
    is_encrypted
}

# Check if protocol is allowed
is_allowed_protocol {
    input.protocol in data.allowed_protocols
}

# Check if destination is allowed
is_allowed_destination {
    input.destination_type == "internal"
}

is_allowed_destination {
    input.destination_type == "external"
    some allowed_domain in data.allowed_external_domains
    glob.match(allowed_domain, ["."], input.destination)
}

# Check if connection is encrypted
is_encrypted {
    input.protocol in ["https", "grpcs", "wss"]
}

is_encrypted {
    input.tls_enabled == true
}

# Violations for disallowed protocols
violations[msg] {
    input.action == "network_connect"
    not is_allowed_protocol
    msg := sprintf("Protocol '%s' not allowed - use one of: %s", [input.protocol, concat(", ", data.allowed_protocols)])
}

# Violations for disallowed destinations
violations[msg] {
    input.action == "network_connect"
    not is_allowed_destination
    msg := sprintf("Destination '%s' not in allowlist", [input.destination])
}

# Violations for unencrypted connections
violations[msg] {
    input.action == "network_connect"
    not is_encrypted
    msg := "Unencrypted connections not allowed - TLS required"
}

# Deny connections to known malicious IPs
violations[msg] {
    input.action == "network_connect"
    input.destination in data.blocked_ips
    msg := sprintf("Destination IP '%s' is blocked (known malicious)", [input.destination])
}

# Deny egress to high-risk ports
violations[msg] {
    input.action == "network_connect"
    input.destination_type == "external"
    input.destination_port in data.blocked_ports
    msg := sprintf("Egress to port %d is blocked (high-risk)", [input.destination_port])
}

# Warn on large data transfers
warn[msg] {
    input.action == "data_transfer"
    input.size_bytes > data.limits.warn_transfer_size_bytes
    msg := sprintf("Large data transfer detected: %.2f MB", [input.size_bytes / 1048576])
}

# Network security configuration
data := {
    "allowed_protocols": [
        "https",
        "grpcs",
        "wss",
        "tcp",
        "udp"
    ],
    "allowed_external_domains": [
        "*.amazonaws.com",
        "*.googleapis.com",
        "*.azurewebsites.net",
        "*.github.com",
        "*.dockerhub.com"
    ],
    "blocked_ips": [
        # Example malicious IPs (placeholder)
        "192.0.2.1",
        "198.51.100.1"
    ],
    "blocked_ports": [
        # High-risk ports
        23,    # Telnet
        445,   # SMB
        3389,  # RDP
        5900   # VNC
    ],
    "limits": {
        "warn_transfer_size_bytes": 104857600  # 100 MB
    }
}
