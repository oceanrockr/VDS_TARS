"""
Federation Agent - Sidecar for nodes to communicate with coordination hub
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import httpx
from prometheus_client import Gauge, Counter, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
node_status_gauge = Gauge('tars_federation_node_status', 'Node status', ['node_id'])
heartbeat_counter = Counter('tars_federation_heartbeats_total', 'Total heartbeats sent', ['node_id'])
gossip_received_counter = Counter('tars_federation_gossip_received_total', 'Total gossip messages received', ['message_type'])
gossip_sent_counter = Counter('tars_federation_gossip_sent_total', 'Total gossip messages sent', ['message_type'])


class FederationAgent:
    """Federation agent for node-to-hub communication"""

    def __init__(
        self,
        node_id: str,
        cluster_name: str,
        region: str,
        hub_url: str,
        capabilities: list,
        version: str = "0.7.0-alpha"
    ):
        self.node_id = node_id
        self.cluster_name = cluster_name
        self.region = region
        self.hub_url = hub_url.rstrip('/')
        self.capabilities = capabilities
        self.version = version
        self.registered = False
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info(f"Initialized Federation Agent: {node_id} in {region}")

    async def register(self) -> bool:
        """Register node with coordination hub"""
        try:
            endpoint = f"grpcs://{self.node_id}.tars.local:50051"

            registration_data = {
                "node_id": self.node_id,
                "cluster_name": self.cluster_name,
                "region": self.region,
                "endpoint": endpoint,
                "capabilities": self.capabilities,
                "version": self.version,
                "status": "healthy",
                "metadata": {
                    "hostname": os.getenv("HOSTNAME", "unknown"),
                    "pod_ip": os.getenv("POD_IP", "unknown")
                }
            }

            response = await self.client.post(
                f"{self.hub_url}/api/v1/nodes/register",
                json=registration_data
            )

            if response.status_code == 200:
                self.registered = True
                logger.info(f"Successfully registered node {self.node_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    async def send_heartbeat(self, metrics: Dict[str, float], anomaly_score: Optional[float] = None) -> bool:
        """Send heartbeat to coordination hub"""
        try:
            heartbeat_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "metrics": metrics,
                "anomaly_score": anomaly_score,
                "active_remediations": 0
            }

            response = await self.client.post(
                f"{self.hub_url}/api/v1/nodes/{self.node_id}/heartbeat",
                json=heartbeat_data
            )

            if response.status_code == 200:
                heartbeat_counter.labels(node_id=self.node_id).inc()
                logger.debug(f"Heartbeat sent for {self.node_id}")
                return True
            else:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return False

    async def get_cluster_registry(self) -> Optional[Dict[str, Any]]:
        """Get cluster registry from hub"""
        try:
            response = await self.client.get(
                f"{self.hub_url}/api/v1/cluster/registry"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get cluster registry: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Registry fetch error: {e}")
            return None

    async def cast_vote(self, vote_id: str, approve: bool) -> bool:
        """Cast a vote on consensus decision"""
        try:
            response = await self.client.post(
                f"{self.hub_url}/api/v1/votes/{vote_id}/cast",
                params={"node_id": self.node_id, "approve": approve}
            )

            if response.status_code == 200:
                logger.info(f"Vote cast: {vote_id} = {'APPROVE' if approve else 'REJECT'}")
                return True
            else:
                logger.warning(f"Vote casting failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Vote casting error: {e}")
            return False

    async def fetch_metrics_from_prometheus(self) -> Dict[str, float]:
        """Fetch current metrics from local Prometheus"""
        # Placeholder: query local Prometheus for key metrics
        # In production, use promql queries
        return {
            "cpu_usage": 0.45,
            "memory_usage": 0.62,
            "request_rate": 125.3,
            "error_rate": 0.02
        }

    async def fetch_anomaly_score(self) -> Optional[float]:
        """Fetch current anomaly score from local anomaly detector"""
        # Placeholder: query local anomaly detector
        # In production, query via HTTP or Prometheus
        return None

    async def heartbeat_loop(self, interval: int = 10) -> None:
        """Background loop to send periodic heartbeats"""
        while True:
            try:
                if not self.registered:
                    logger.info("Not registered, attempting registration...")
                    await self.register()
                    await asyncio.sleep(5)
                    continue

                # Fetch metrics
                metrics = await self.fetch_metrics_from_prometheus()
                anomaly_score = await self.fetch_anomaly_score()

                # Send heartbeat
                success = await self.send_heartbeat(metrics, anomaly_score)

                if success:
                    node_status_gauge.labels(node_id=self.node_id).set(1)
                else:
                    node_status_gauge.labels(node_id=self.node_id).set(0)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(interval)

    async def governance_listener(self) -> None:
        """Listen for governance directives from hub"""
        # Placeholder: implement WebSocket or gRPC streaming
        # to receive real-time governance updates
        logger.info("Governance listener started (stub)")
        while True:
            await asyncio.sleep(60)

    async def start(self, heartbeat_interval: int = 10) -> None:
        """Start agent background tasks"""
        logger.info(f"Starting Federation Agent for {self.node_id}")

        # Start Prometheus metrics server
        start_http_server(9091)
        logger.info("Prometheus metrics server started on port 9091")

        # Start background tasks
        await asyncio.gather(
            self.heartbeat_loop(heartbeat_interval),
            self.governance_listener()
        )


async def main():
    """Main entry point"""
    # Configuration from environment
    node_id = os.getenv("NODE_ID", "node-001")
    cluster_name = os.getenv("CLUSTER_NAME", "tars-cluster")
    region = os.getenv("REGION", "us-east-1")
    hub_url = os.getenv("FEDERATION_HUB_URL", "http://federation-hub:8080")
    capabilities = os.getenv("CAPABILITIES", "compute,inference,anomaly_detection").split(",")
    heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "10"))

    # Create agent
    agent = FederationAgent(
        node_id=node_id,
        cluster_name=cluster_name,
        region=region,
        hub_url=hub_url,
        capabilities=capabilities
    )

    # Start agent
    await agent.start(heartbeat_interval)


if __name__ == "__main__":
    asyncio.run(main())
