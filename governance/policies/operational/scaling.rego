# Operational Policy: Auto-Scaling Limits
package tars.operational.scaling

default allow = false

# Allow scaling if within limits
allow {
    input.action == "scale_out"
    input.resource_type == "deployment"
    input.target_replicas <= data.limits.max_replicas
    input.target_replicas >= data.limits.min_replicas
}

# Deny scaling beyond max replicas
violations[msg] {
    input.action == "scale_out"
    input.target_replicas > data.limits.max_replicas
    msg := sprintf("Target replicas %d exceeds max limit %d", [input.target_replicas, data.limits.max_replicas])
}

# Deny scaling below min replicas
violations[msg] {
    input.action == "scale_in"
    input.target_replicas < data.limits.min_replicas
    msg := sprintf("Target replicas %d below min limit %d", [input.target_replicas, data.limits.min_replicas])
}

# Deny rapid scaling (within cooldown period)
violations[msg] {
    input.action == "scale_out"
    input.last_scale_timestamp != ""
    time_since_last_scale := time.now_ns() - time.parse_rfc3339_ns(input.last_scale_timestamp)
    cooldown_ns := data.limits.scale_cooldown_seconds * 1000000000
    time_since_last_scale < cooldown_ns
    msg := sprintf("Scaling too soon - cooldown period is %d seconds", [data.limits.scale_cooldown_seconds])
}

# Warn on expensive scaling
warn[msg] {
    input.action == "scale_out"
    input.target_replicas > 10
    input.estimated_cost_usd > 100
    msg := sprintf("High cost scaling: %d replicas (~$%.2f/month)", [input.target_replicas, input.estimated_cost_usd])
}

# Default limits (can be overridden by data document)
data := {
    "limits": {
        "max_replicas": 20,
        "min_replicas": 1,
        "scale_cooldown_seconds": 300
    }
}
