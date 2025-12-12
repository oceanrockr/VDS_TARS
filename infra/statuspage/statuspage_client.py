"""
Phase 13.9 - Statuspage.io Integration Client
=============================================

Provides integration with Statuspage.io (or custom status page) for:
- Component status updates
- Incident management
- Scheduled maintenance
- Subscriber notifications

Supported Status Levels:
-----------------------
- operational: Component is working normally
- degraded_performance: Component is experiencing issues
- partial_outage: Some functionality unavailable
- major_outage: Component is unavailable

Author: T.A.R.S. SRE Team
Date: 2025-11-19
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import httpx


# ============================================================================
# CONFIGURATION
# ============================================================================

STATUSPAGE_API_KEY = os.getenv("STATUSPAGE_API_KEY", "test-api-key")
STATUSPAGE_PAGE_ID = os.getenv("STATUSPAGE_PAGE_ID", "test-page-id")
STATUSPAGE_BASE_URL = os.getenv(
    "STATUSPAGE_BASE_URL", "https://api.statuspage.io/v1"
)

# Component IDs (would be created in Statuspage.io dashboard)
COMPONENT_IDS = {
    "eval_api": "comp-eval-api-001",
    "hypersync": "comp-hypersync-001",
    "rl_agents": "comp-rl-agents-001",
    "dashboard_api": "comp-dashboard-api-001",
    "dashboard_frontend": "comp-dashboard-frontend-001",
    "multi_region_replication": "comp-replication-001",
}

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# ENUMS
# ============================================================================


class ComponentStatus(str, Enum):
    """Component status levels."""

    OPERATIONAL = "operational"
    DEGRADED_PERFORMANCE = "degraded_performance"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    UNDER_MAINTENANCE = "under_maintenance"


class IncidentStatus(str, Enum):
    """Incident status levels."""

    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"


class IncidentImpact(str, Enum):
    """Incident impact levels."""

    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


# ============================================================================
# STATUSPAGE CLIENT
# ============================================================================


class StatuspageClient:
    """
    Client for interacting with Statuspage.io API.

    Supports:
    - Component status updates
    - Incident creation and management
    - Scheduled maintenance
    """

    def __init__(
        self,
        api_key: str = STATUSPAGE_API_KEY,
        page_id: str = STATUSPAGE_PAGE_ID,
        base_url: str = STATUSPAGE_BASE_URL,
    ):
        """Initialize Statuspage client."""
        self.api_key = api_key
        self.page_id = page_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"OAuth {api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = httpx.Timeout(30.0)

    # ========================================================================
    # COMPONENT STATUS METHODS
    # ========================================================================

    async def get_component(self, component_id: str) -> Dict:
        """
        Get component details.

        Args:
            component_id: Component ID

        Returns:
            Component details
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/pages/{self.page_id}/components/{component_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    async def update_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        description: Optional[str] = None,
    ) -> Dict:
        """
        Update component status.

        Args:
            component_id: Component ID
            status: New status
            description: Optional status description

        Returns:
            Updated component details
        """
        payload = {"component": {"status": status.value}}

        if description:
            payload["component"]["description"] = description

        logger.info(
            f"Updating component {component_id} to status: {status.value}"
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                f"{self.base_url}/pages/{self.page_id}/components/{component_id}",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def bulk_update_components(
        self, updates: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Update multiple components at once.

        Args:
            updates: List of {component_id, status, description}

        Returns:
            List of updated components
        """
        results = []
        for update in updates:
            component_id = update["component_id"]
            status = ComponentStatus(update["status"])
            description = update.get("description")

            result = await self.update_component_status(
                component_id, status, description
            )
            results.append(result)

        return results

    # ========================================================================
    # INCIDENT MANAGEMENT METHODS
    # ========================================================================

    async def create_incident(
        self,
        name: str,
        status: IncidentStatus,
        impact: IncidentImpact,
        body: str,
        component_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Create a new incident.

        Args:
            name: Incident title
            status: Incident status
            impact: Impact level
            body: Incident description
            component_ids: Affected component IDs

        Returns:
            Created incident details
        """
        payload = {
            "incident": {
                "name": name,
                "status": status.value,
                "impact_override": impact.value,
                "body": body,
            }
        }

        if component_ids:
            payload["incident"]["component_ids"] = component_ids

        logger.info(f"Creating incident: {name} (impact: {impact.value})")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/pages/{self.page_id}/incidents",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def update_incident(
        self,
        incident_id: str,
        status: IncidentStatus,
        body: str,
    ) -> Dict:
        """
        Update an existing incident.

        Args:
            incident_id: Incident ID
            status: New status
            body: Update message

        Returns:
            Updated incident details
        """
        payload = {
            "incident": {
                "status": status.value,
                "body": body,
            }
        }

        logger.info(f"Updating incident {incident_id} to status: {status.value}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(
                f"{self.base_url}/pages/{self.page_id}/incidents/{incident_id}",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def resolve_incident(self, incident_id: str, resolution: str) -> Dict:
        """
        Resolve an incident.

        Args:
            incident_id: Incident ID
            resolution: Resolution message

        Returns:
            Resolved incident details
        """
        return await self.update_incident(
            incident_id, IncidentStatus.RESOLVED, resolution
        )

    async def list_active_incidents(self) -> List[Dict]:
        """
        List all active (unresolved) incidents.

        Returns:
            List of active incidents
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/pages/{self.page_id}/incidents/unresolved",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    # ========================================================================
    # SCHEDULED MAINTENANCE METHODS
    # ========================================================================

    async def create_maintenance(
        self,
        name: str,
        scheduled_for: datetime,
        scheduled_until: datetime,
        component_ids: List[str],
        body: str,
    ) -> Dict:
        """
        Schedule maintenance window.

        Args:
            name: Maintenance title
            scheduled_for: Start time (UTC)
            scheduled_until: End time (UTC)
            component_ids: Affected component IDs
            body: Maintenance description

        Returns:
            Created maintenance details
        """
        payload = {
            "scheduled_maintenance": {
                "name": name,
                "status": "scheduled",
                "scheduled_for": scheduled_for.isoformat(),
                "scheduled_until": scheduled_until.isoformat(),
                "component_ids": component_ids,
                "body": body,
            }
        }

        logger.info(f"Scheduling maintenance: {name} ({scheduled_for} - {scheduled_until})")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/pages/{self.page_id}/incidents",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def get_page_summary(self) -> Dict:
        """
        Get status page summary.

        Returns:
            Page summary with all components and incidents
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/pages/{self.page_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """
        Check if Statuspage API is accessible.

        Returns:
            True if API is accessible
        """
        try:
            await self.get_page_summary()
            return True
        except Exception as e:
            logger.error(f"Statuspage health check failed: {e}")
            return False


# ============================================================================
# HIGH-LEVEL HELPER FUNCTIONS
# ============================================================================


async def set_component_operational(component_name: str) -> Dict:
    """Set component to operational status."""
    client = StatuspageClient()
    component_id = COMPONENT_IDS.get(component_name)

    if not component_id:
        raise ValueError(f"Unknown component: {component_name}")

    return await client.update_component_status(
        component_id, ComponentStatus.OPERATIONAL, "All systems operational"
    )


async def set_component_degraded(
    component_name: str, reason: str
) -> Dict:
    """Set component to degraded status."""
    client = StatuspageClient()
    component_id = COMPONENT_IDS.get(component_name)

    if not component_id:
        raise ValueError(f"Unknown component: {component_name}")

    return await client.update_component_status(
        component_id, ComponentStatus.DEGRADED_PERFORMANCE, reason
    )


async def set_component_outage(
    component_name: str, is_partial: bool, reason: str
) -> Dict:
    """Set component to outage status."""
    client = StatuspageClient()
    component_id = COMPONENT_IDs.get(component_name)

    if not component_id:
        raise ValueError(f"Unknown component: {component_name}")

    status = (
        ComponentStatus.PARTIAL_OUTAGE
        if is_partial
        else ComponentStatus.MAJOR_OUTAGE
    )

    return await client.update_component_status(component_id, status, reason)


async def create_incident_from_alert(
    alert_name: str,
    alert_severity: str,
    alert_description: str,
    affected_components: List[str],
) -> Dict:
    """
    Create incident from monitoring alert.

    Args:
        alert_name: Alert name
        alert_severity: Alert severity (critical, high, medium, low)
        alert_description: Alert description
        affected_components: List of affected component names

    Returns:
        Created incident
    """
    client = StatuspageClient()

    # Map alert severity to incident impact
    severity_to_impact = {
        "critical": IncidentImpact.CRITICAL,
        "high": IncidentImpact.MAJOR,
        "medium": IncidentImpact.MINOR,
        "low": IncidentImpact.NONE,
    }

    impact = severity_to_impact.get(
        alert_severity.lower(), IncidentImpact.MINOR
    )

    # Get component IDs
    component_ids = [
        COMPONENT_IDS[comp]
        for comp in affected_components
        if comp in COMPONENT_IDS
    ]

    # Create incident
    incident = await client.create_incident(
        name=alert_name,
        status=IncidentStatus.INVESTIGATING,
        impact=impact,
        body=alert_description,
        component_ids=component_ids,
    )

    # Update affected components to degraded/outage
    if impact in [IncidentImpact.MAJOR, IncidentImpact.CRITICAL]:
        status = ComponentStatus.MAJOR_OUTAGE
    else:
        status = ComponentStatus.DEGRADED_PERFORMANCE

    for component_id in component_ids:
        await client.update_component_status(
            component_id, status, f"Affected by incident: {alert_name}"
        )

    return incident


async def auto_resolve_incident_if_healthy(
    incident_id: str, component_ids: List[str]
) -> Optional[Dict]:
    """
    Auto-resolve incident if all affected components are healthy.

    Args:
        incident_id: Incident ID
        component_ids: Affected component IDs

    Returns:
        Resolved incident if conditions met, None otherwise
    """
    client = StatuspageClient()

    # Check if all components are operational
    all_operational = True
    for component_id in component_ids:
        component = await client.get_component(component_id)
        if component.get("status") != ComponentStatus.OPERATIONAL.value:
            all_operational = False
            break

    if all_operational:
        # Auto-resolve incident
        return await client.resolve_incident(
            incident_id, "All affected systems have recovered. Monitoring for stability."
        )

    return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


async def example_usage():
    """Example usage of Statuspage client."""
    client = StatuspageClient()

    # Health check
    is_healthy = await client.health_check()
    print(f"Statuspage API health: {is_healthy}")

    # Update component status
    await client.update_component_status(
        COMPONENT_IDS["eval_api"],
        ComponentStatus.OPERATIONAL,
        "Evaluation API is running normally",
    )

    # Create incident
    incident = await client.create_incident(
        name="High Latency on Evaluation API",
        status=IncidentStatus.INVESTIGATING,
        impact=IncidentImpact.MINOR,
        body="We are investigating reports of increased latency on the Evaluation API.",
        component_ids=[COMPONENT_IDS["eval_api"]],
    )

    incident_id = incident.get("id")

    # Update incident
    await client.update_incident(
        incident_id,
        IncidentStatus.IDENTIFIED,
        "Issue identified: Database connection pool exhaustion. Scaling up.",
    )

    # Resolve incident
    await client.resolve_incident(
        incident_id,
        "Database connection pool increased. Latency has returned to normal.",
    )

    # Schedule maintenance
    start_time = datetime.utcnow() + timedelta(days=7)
    end_time = start_time + timedelta(hours=2)

    await client.create_maintenance(
        name="Database Maintenance",
        scheduled_for=start_time,
        scheduled_until=end_time,
        component_ids=[COMPONENT_IDS["eval_api"]],
        body="We will be performing scheduled database maintenance. Minimal impact expected.",
    )


if __name__ == "__main__":
    asyncio.run(example_usage())
