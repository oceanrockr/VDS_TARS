#!/usr/bin/env python3
"""
Federation Simulator - Test federation features with mock nodes
"""
import asyncio
import argparse
import random
import logging
from datetime import datetime
from typing import List, Dict
import httpx
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockNode:
    """Mock federation node"""

    def __init__(self, node_id: str, region: str, hub_url: str):
        self.node_id = node_id
        self.region = region
        self.hub_url = hub_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        self.registered = False

    async def register(self) -> bool:
        """Register node with coordination hub"""
        try:
            registration_data = {
                "node_id": self.node_id,
                "cluster_name": f"cluster-{self.region}",
                "region": self.region,
                "endpoint": f"grpcs://{self.node_id}.sim.local:50051",
                "capabilities": ["compute", "inference", "anomaly_detection"],
                "version": "0.7.0-alpha",
                "status": "healthy",
                "metadata": {}
            }

            response = await self.client.post(
                f"{self.hub_url}/api/v1/nodes/register",
                json=registration_data
            )

            if response.status_code == 200:
                self.registered = True
                logger.info(f"[{self.node_id}] Registered successfully")
                return True
            else:
                logger.error(f"[{self.node_id}] Registration failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[{self.node_id}] Registration error: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """Send heartbeat to hub"""
        if not self.registered:
            return False

        try:
            # Generate mock metrics
            heartbeat_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "healthy",
                "metrics": {
                    "cpu_usage": random.uniform(0.2, 0.8),
                    "memory_usage": random.uniform(0.3, 0.7),
                    "request_rate": random.uniform(50, 200),
                    "error_rate": random.uniform(0, 0.05)
                },
                "anomaly_score": random.uniform(0.1, 0.4),
                "active_remediations": 0
            }

            response = await self.client.post(
                f"{self.hub_url}/api/v1/nodes/{self.node_id}/heartbeat",
                json=heartbeat_data
            )

            if response.status_code == 200:
                logger.debug(f"[{self.node_id}] Heartbeat sent")
                return True
            else:
                logger.warning(f"[{self.node_id}] Heartbeat failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[{self.node_id}] Heartbeat error: {e}")
            return False

    async def cast_vote(self, vote_id: str, approve: bool) -> bool:
        """Cast a vote on consensus decision"""
        try:
            response = await self.client.post(
                f"{self.hub_url}/api/v1/votes/{vote_id}/cast",
                params={"node_id": self.node_id, "approve": approve}
            )

            if response.status_code == 200:
                logger.info(f"[{self.node_id}] Voted {'APPROVE' if approve else 'REJECT'} on {vote_id}")
                return True
            else:
                logger.warning(f"[{self.node_id}] Vote failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[{self.node_id}] Vote error: {e}")
            return False


class FederationSimulator:
    """Federation simulator orchestrator"""

    def __init__(self, hub_url: str, num_nodes: int = 3):
        self.hub_url = hub_url
        self.nodes: List[MockNode] = []
        self.client = httpx.AsyncClient(timeout=30.0)

        # Create mock nodes
        regions = ["us-east-1", "us-west-2", "eu-west-1"]
        for i in range(num_nodes):
            region = regions[i % len(regions)]
            node = MockNode(f"sim-node-{i+1}", region, hub_url)
            self.nodes.append(node)

        logger.info(f"Created {num_nodes} mock nodes")

    async def register_all_nodes(self) -> None:
        """Register all nodes with hub"""
        logger.info("Registering all nodes...")
        tasks = [node.register() for node in self.nodes]
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r)
        logger.info(f"Registered {success_count}/{len(self.nodes)} nodes")

    async def heartbeat_loop(self, duration: int) -> None:
        """Run heartbeat loop for specified duration"""
        logger.info(f"Starting heartbeat loop for {duration} seconds...")

        start_time = asyncio.get_event_loop().time()
        interval = 5  # 5 second heartbeat interval

        while (asyncio.get_event_loop().time() - start_time) < duration:
            # Send heartbeats from all nodes
            tasks = [node.send_heartbeat() for node in self.nodes]
            await asyncio.gather(*tasks)

            await asyncio.sleep(interval)

        logger.info("Heartbeat loop completed")

    async def simulate_policy_vote(self) -> None:
        """Simulate a policy vote"""
        logger.info("Simulating policy vote...")

        try:
            # Submit a test policy bundle
            policy_bundle = {
                "bundle_id": "test-policy-001",
                "name": "Test Scaling Policy",
                "version": "1.0.0",
                "policy_type": "operational",
                "rules": ["package test\ndefault allow = true"],
                "checksum": "abc123"
            }

            response = await self.client.post(
                f"{self.hub_url}/api/v1/policies/submit",
                json=policy_bundle
            )

            if response.status_code == 200:
                logger.info("Policy submitted successfully")

                # Wait a bit for vote to be created
                await asyncio.sleep(1)

                # Get cluster registry to find vote
                registry_response = await self.client.get(
                    f"{self.hub_url}/api/v1/cluster/registry"
                )

                if registry_response.status_code == 200:
                    # Simulate nodes voting (majority approve)
                    for i, node in enumerate(self.nodes):
                        approve = i < (len(self.nodes) * 2 // 3)  # 2/3 approve
                        # In real scenario, would get vote_id from webhook
                        # For simulation, we assume vote_id generation
                        vote_id = f"vote-{policy_bundle['bundle_id']}"
                        await node.cast_vote(vote_id, approve)
                        await asyncio.sleep(0.5)

            else:
                logger.error(f"Policy submission failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Policy vote simulation error: {e}")

    async def measure_consensus_latency(self, num_votes: int = 5) -> List[float]:
        """Measure consensus latency"""
        logger.info(f"Measuring consensus latency ({num_votes} votes)...")

        latencies = []

        for i in range(num_votes):
            start_time = asyncio.get_event_loop().time()

            # Submit a config change that requires consensus
            config_sync = {
                "config_key": f"test_config_{i}",
                "value": {"test": True},
                "node_id": "simulator",
                "consensus_required": True
            }

            try:
                response = await self.client.post(
                    f"{self.hub_url}/api/v1/config/sync",
                    json=config_sync
                )

                if response.status_code == 200:
                    # Simulate nodes voting
                    vote_tasks = []
                    for node in self.nodes:
                        # Assume vote_id is returned in response or generated
                        vote_id = f"config-vote-{i}"
                        vote_tasks.append(node.cast_vote(vote_id, True))

                    await asyncio.gather(*vote_tasks)

                    end_time = asyncio.get_event_loop().time()
                    latency = (end_time - start_time) * 1000  # Convert to ms

                    latencies.append(latency)
                    logger.info(f"Vote {i+1}: {latency:.2f}ms")

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Consensus latency test error: {e}")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            logger.info(f"Average consensus latency: {avg_latency:.2f}ms")

        return latencies

    async def get_cluster_status(self) -> Dict:
        """Get cluster registry status"""
        try:
            response = await self.client.get(
                f"{self.hub_url}/api/v1/cluster/registry"
            )

            if response.status_code == 200:
                registry = response.json()
                logger.info(f"Cluster Status: {registry['total_nodes']} total, {registry['healthy_nodes']} healthy")
                return registry
            else:
                logger.error(f"Failed to get cluster status: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Cluster status error: {e}")
            return {}

    async def run(self, duration: int, num_policies: int = 1) -> None:
        """Run full simulation"""
        logger.info("=" * 60)
        logger.info("T.A.R.S. Federation Simulator")
        logger.info("=" * 60)

        # Step 1: Register all nodes
        await self.register_all_nodes()
        await asyncio.sleep(2)

        # Step 2: Get initial cluster status
        await self.get_cluster_status()

        # Step 3: Run heartbeats in background
        heartbeat_task = asyncio.create_task(self.heartbeat_loop(duration))

        # Step 4: Simulate policy votes
        for i in range(num_policies):
            logger.info(f"\nSimulating policy vote {i+1}/{num_policies}...")
            await self.simulate_policy_vote()
            await asyncio.sleep(5)

        # Step 5: Measure consensus latency
        await asyncio.sleep(2)
        latencies = await self.measure_consensus_latency(num_votes=3)

        # Wait for heartbeat loop to complete
        await heartbeat_task

        # Final cluster status
        await self.get_cluster_status()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Simulation Summary")
        logger.info("=" * 60)
        logger.info(f"Nodes: {len(self.nodes)}")
        logger.info(f"Duration: {duration}s")
        logger.info(f"Policies: {num_policies}")
        if latencies:
            logger.info(f"Avg Consensus Latency: {sum(latencies)/len(latencies):.2f}ms")
            logger.info(f"Max Consensus Latency: {max(latencies):.2f}ms")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="T.A.R.S. Federation Simulator")
    parser.add_argument(
        "--hub-url",
        default="http://localhost:8080",
        help="Federation hub URL"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=3,
        help="Number of mock nodes to create"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--policies",
        type=int,
        default=3,
        help="Number of policy votes to simulate"
    )

    args = parser.parse_args()

    # Create simulator
    simulator = FederationSimulator(
        hub_url=args.hub_url,
        num_nodes=args.nodes
    )

    # Run simulation
    await simulator.run(
        duration=args.duration,
        num_policies=args.policies
    )


if __name__ == "__main__":
    asyncio.run(main())
