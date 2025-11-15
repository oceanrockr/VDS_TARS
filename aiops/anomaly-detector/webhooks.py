"""
Alertmanager Webhook Handler
Processes Alertmanager webhooks, correlates with anomaly detection, and generates incident IDs.
"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import uuid

from prom_client import PrometheusClient

logger = logging.getLogger(__name__)


class AlertmanagerWebhook:
    """Handles Alertmanager webhook payloads and correlates with anomaly detection"""

    def __init__(self, anomaly_detector, prom_client: PrometheusClient):
        """
        Initialize webhook handler.

        Args:
            anomaly_detector: AnomalyDetector instance
            prom_client: PrometheusClient instance
        """
        self.detector = anomaly_detector
        self.prom_client = prom_client

        # Incident correlation
        self.active_incidents = {}  # incident_id -> incident data
        self.alert_to_incident = {}  # alert fingerprint -> incident_id

        # Deduplication window
        self.dedup_window = timedelta(minutes=5)

    async def process(self, payload: Dict) -> Dict:
        """
        Process Alertmanager webhook payload.

        Args:
            payload: Alertmanager webhook JSON payload

        Returns:
            Response dict with incident IDs and correlations
        """
        try:
            alerts = payload.get('alerts', [])
            if not alerts:
                return {"status": "no_alerts", "incidents": []}

            logger.info(f"Processing {len(alerts)} alerts from Alertmanager")

            incidents = []
            for alert in alerts:
                incident = await self._process_alert(alert)
                if incident:
                    incidents.append(incident)

            # Deduplicate and correlate
            correlated_incidents = self._correlate_incidents(incidents)

            return {
                "status": "processed",
                "alerts_received": len(alerts),
                "incidents_created": len(correlated_incidents),
                "incidents": correlated_incidents
            }

        except Exception as e:
            logger.error(f"Webhook processing failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _process_alert(self, alert: Dict) -> Optional[Dict]:
        """Process a single alert"""
        try:
            # Extract alert metadata
            labels = alert.get('labels', {})
            annotations = alert.get('annotations', {})
            status = alert.get('status', 'unknown')
            fingerprint = alert.get('fingerprint', '')

            alert_name = labels.get('alertname', 'unknown')
            service = labels.get('service', labels.get('job', 'unknown'))
            severity = labels.get('severity', 'medium')

            # Skip resolved alerts for now
            if status == 'resolved':
                logger.info(f"Alert {alert_name} resolved, skipping")
                return None

            # Check if this alert is already part of an incident
            existing_incident_id = self.alert_to_incident.get(fingerprint)
            if existing_incident_id and existing_incident_id in self.active_incidents:
                # Update existing incident
                incident = self.active_incidents[existing_incident_id]
                incident['alert_count'] += 1
                incident['last_seen'] = datetime.now()
                logger.info(f"Alert {alert_name} added to existing incident {existing_incident_id}")
                return incident

            # Run anomaly detection on related metrics
            anomaly_results = await self._detect_anomalies_for_alert(alert_name, service, labels)

            # Create incident
            incident_id = self._generate_incident_id(alert_name, service, fingerprint)
            incident = {
                "incident_id": incident_id,
                "alert_name": alert_name,
                "service": service,
                "severity": self._escalate_severity(severity, anomaly_results),
                "status": "active",
                "created_at": datetime.now(),
                "last_seen": datetime.now(),
                "alert_count": 1,
                "fingerprints": [fingerprint],
                "labels": labels,
                "annotations": annotations,
                "anomaly_results": anomaly_results,
                "grafana_link": self._generate_grafana_link(service),
                "trace_link": self._generate_trace_link(service),
                "log_snippet": await self._fetch_recent_logs(service, labels),
                "recommendations": self._generate_recommendations(
                    alert_name, service, anomaly_results
                )
            }

            # Store incident
            self.active_incidents[incident_id] = incident
            self.alert_to_incident[fingerprint] = incident_id

            logger.info(f"Created incident {incident_id} for alert {alert_name}")

            return incident

        except Exception as e:
            logger.error(f"Alert processing failed: {e}", exc_info=True)
            return None

    async def _detect_anomalies_for_alert(
        self,
        alert_name: str,
        service: str,
        labels: Dict
    ) -> List[Dict]:
        """Run anomaly detection on metrics related to the alert"""
        anomaly_results = []

        try:
            # Map alert names to relevant metric queries
            metric_queries = self._get_metric_queries_for_alert(alert_name, service, labels)

            for signal_name, query in metric_queries.items():
                try:
                    # Query last 1 hour of data
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=1)

                    data = self.prom_client.query_range(
                        query,
                        start=start_time,
                        end=end_time,
                        step="30s"
                    )

                    if data:
                        # Run anomaly detection
                        result = self.detector.detect(
                            signal_name=signal_name,
                            service_name=service,
                            data=data
                        )
                        anomaly_results.append({
                            "signal": signal_name,
                            **result
                        })

                except Exception as e:
                    logger.warning(f"Anomaly detection failed for {signal_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to detect anomalies for alert: {e}")

        return anomaly_results

    def _get_metric_queries_for_alert(
        self,
        alert_name: str,
        service: str,
        labels: Dict
    ) -> Dict[str, str]:
        """Map alert to relevant Prometheus queries"""
        queries = {}

        # Common patterns
        if "latency" in alert_name.lower() or "slow" in alert_name.lower():
            queries["latency_p95"] = (
                f'histogram_quantile(0.95, sum(rate('
                f'http_server_duration_seconds_bucket{{service="{service}"}}[5m])) by (le))'
            )

        if "error" in alert_name.lower() or "5xx" in alert_name.lower():
            queries["error_rate"] = (
                f'sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[5m])) '
                f'/ sum(rate(http_requests_total{{service="{service}"}}[5m]))'
            )

        if "cpu" in alert_name.lower():
            queries["cpu_usage"] = (
                f'sum(rate(container_cpu_usage_seconds_total{{'
                f'pod=~"{service}.*"}}[5m])) by (pod)'
            )

        if "memory" in alert_name.lower():
            queries["memory_usage"] = (
                f'sum(container_memory_working_set_bytes{{'
                f'pod=~"{service}.*"}}) by (pod)'
            )

        # Fallback: use alert expression if available
        if not queries and "expr" in labels:
            queries["alert_metric"] = labels["expr"]

        return queries

    def _escalate_severity(self, base_severity: str, anomaly_results: List[Dict]) -> str:
        """Escalate severity based on anomaly detection results"""
        # Check if any anomalies are critical
        for result in anomaly_results:
            if result.get("severity") == "critical":
                return "critical"
            elif result.get("severity") == "high" and base_severity != "critical":
                return "high"

        return base_severity

    def _generate_incident_id(self, alert_name: str, service: str, fingerprint: str) -> str:
        """Generate unique incident ID"""
        # Use deterministic hash for deduplication
        data = f"{alert_name}:{service}:{fingerprint}"
        hash_value = hashlib.sha256(data.encode()).hexdigest()[:12]
        return f"INC-{hash_value}"

    def _generate_grafana_link(self, service: str) -> str:
        """Generate Grafana dashboard link"""
        # Assumes Grafana is accessible
        base_url = "http://grafana:3000"
        dashboard = "tars-advanced-rag-observability"
        return f"{base_url}/d/{dashboard}?var-service={service}"

    def _generate_trace_link(self, service: str) -> str:
        """Generate Jaeger trace link"""
        base_url = "http://jaeger-query:16686"
        lookback = "1h"
        return f"{base_url}/search?service={service}&lookback={lookback}"

    async def _fetch_recent_logs(self, service: str, labels: Dict) -> List[str]:
        """Fetch recent error logs for context"""
        try:
            # This would use LokiClient in practice
            # For now, return placeholder
            return [
                f"[Example] Recent logs for {service} would appear here",
                "Implementation requires LokiClient integration"
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch logs: {e}")
            return []

    def _generate_recommendations(
        self,
        alert_name: str,
        service: str,
        anomaly_results: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Add anomaly-based recommendations
        for result in anomaly_results:
            if result.get("recommendations"):
                recommendations.extend(result["recommendations"])

        # Add alert-specific recommendations
        if "latency" in alert_name.lower():
            recommendations.append("Consider scaling out service replicas")
            recommendations.append("Check database query performance")

        if "error" in alert_name.lower():
            recommendations.append("Review recent deployments for issues")
            recommendations.append("Check application logs for error details")

        if "memory" in alert_name.lower():
            recommendations.append("Investigate for memory leaks")
            recommendations.append("Consider increasing memory limits")

        # Generic recommendations
        recommendations.append(f"View metrics in Grafana dashboard")
        recommendations.append(f"Check distributed traces in Jaeger")

        # Deduplicate and limit
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        return unique_recs[:8]

    def _correlate_incidents(self, incidents: List[Dict]) -> List[Dict]:
        """Correlate related incidents"""
        # Simple time-based correlation for now
        # In production, would use ML-based correlation

        if not incidents:
            return []

        # Group by service and time window
        grouped = defaultdict(list)
        for incident in incidents:
            service = incident["service"]
            grouped[service].append(incident)

        # Return deduplicated incidents
        correlated = []
        for service, service_incidents in grouped.items():
            # If multiple incidents for same service within dedup window, merge
            if len(service_incidents) > 1:
                primary = service_incidents[0]
                primary["alert_count"] = sum(i["alert_count"] for i in service_incidents)
                primary["fingerprints"] = [
                    fp for i in service_incidents for fp in i["fingerprints"]
                ]
                correlated.append(primary)
            else:
                correlated.extend(service_incidents)

        return correlated

    def cleanup_resolved_incidents(self, max_age_hours: int = 24):
        """Clean up old resolved incidents"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        to_remove = []
        for incident_id, incident in self.active_incidents.items():
            if incident["last_seen"] < cutoff:
                to_remove.append(incident_id)

        for incident_id in to_remove:
            incident = self.active_incidents.pop(incident_id)
            # Remove alert mappings
            for fp in incident.get("fingerprints", []):
                self.alert_to_incident.pop(fp, None)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} resolved incidents")
