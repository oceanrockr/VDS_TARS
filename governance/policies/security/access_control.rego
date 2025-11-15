# Security Policy: Access Control and Authentication
package tars.security.access_control

default allow = false

# Allow authenticated and authorized requests
allow {
    is_authenticated
    is_authorized
    not is_rate_limited
}

# Check if request is authenticated
is_authenticated {
    input.principal != ""
    input.principal != "anonymous"
    input.auth_token != ""
}

# Check if principal is authorized for action on resource
is_authorized {
    some role in input.principal_roles
    role_can_perform_action(role, input.action, input.resource)
}

# Role-based access control
role_can_perform_action(role, action, resource) {
    some permission in data.rbac[role]
    permission.action == action
    glob.match(permission.resource_pattern, ["/"], resource)
}

# Check if request is rate limited
is_rate_limited {
    input.request_count > data.limits.max_requests_per_minute
}

# Violations for unauthenticated access
violations[msg] {
    not is_authenticated
    msg := "Authentication required"
}

# Violations for unauthorized access
violations[msg] {
    is_authenticated
    not is_authorized
    msg := sprintf("Principal '%s' not authorized for action '%s' on resource '%s'", [input.principal, input.action, input.resource])
}

# Violations for rate limiting
violations[msg] {
    is_rate_limited
    msg := sprintf("Rate limit exceeded: %d requests (max: %d/min)", [input.request_count, data.limits.max_requests_per_minute])
}

# Deny privilege escalation
violations[msg] {
    input.action == "modify_permissions"
    input.target_principal == input.principal
    msg := "Self-privilege escalation not allowed"
}

# Deny access to sensitive resources without MFA
violations[msg] {
    input.resource_sensitivity == "high"
    not input.mfa_verified
    msg := "Multi-factor authentication required for high-sensitivity resources"
}

# Warn on admin actions
warn[msg] {
    some role in input.principal_roles
    role == "admin"
    input.action in ["delete", "modify_permissions", "deploy"]
    msg := sprintf("Admin action logged: %s on %s", [input.action, input.resource])
}

# RBAC configuration
data := {
    "rbac": {
        "admin": [
            {"action": "read", "resource_pattern": "*"},
            {"action": "write", "resource_pattern": "*"},
            {"action": "delete", "resource_pattern": "*"},
            {"action": "deploy", "resource_pattern": "*"},
            {"action": "modify_permissions", "resource_pattern": "*"}
        ],
        "operator": [
            {"action": "read", "resource_pattern": "*"},
            {"action": "write", "resource_pattern": "deployments/*"},
            {"action": "scale", "resource_pattern": "deployments/*"},
            {"action": "restart", "resource_pattern": "pods/*"}
        ],
        "developer": [
            {"action": "read", "resource_pattern": "*"},
            {"action": "write", "resource_pattern": "applications/*"},
            {"action": "deploy", "resource_pattern": "applications/*"}
        ],
        "viewer": [
            {"action": "read", "resource_pattern": "*"}
        ]
    },
    "limits": {
        "max_requests_per_minute": 100
    }
}
