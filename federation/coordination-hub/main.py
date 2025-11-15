"""
Federation Coordination Hub - Main Service
Implements secure gossip, consensus, and cluster registry
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import uvicorn

from models import (
    FederationNode, Heartbeat, PolicyBundle, ConsensusVote,
    GossipMessage, ClusterRegistry, ConfigSync,
    NodeStatus, VoteStatus, ConsensusAlgorithm
)
from consensus import RaftConsensus, PBFTConsensus
from storage import StorageBackend, PostgresStorage, EtcdStorage
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoordinationHub:
    """Federation Coordination Hub"""

    def __init__(self, consensus_algorithm: ConsensusAlgorithm, storage: StorageBackend):
        self.nodes: Dict[str, FederationNode] = {}
        self.votes: Dict[str, ConsensusVote] = {}
        self.policy_bundles: Dict[str, PolicyBundle] = {}
        self.config_state: Dict[str, ConfigSync] = {}
        self.storage = storage

        # Initialize consensus mechanism
        if consensus_algorithm == ConsensusAlgorithm.RAFT:
            self.consensus = RaftConsensus(storage=storage)
        elif consensus_algorithm == ConsensusAlgorithm.PBFT:
            self.consensus = PBFTConsensus(storage=storage)
        else:
            raise ValueError(f"Unsupported consensus algorithm: {consensus_algorithm}")

        logger.info(f"Initialized CoordinationHub with {consensus_algorithm.value} consensus")

    async def register_node(self, node: FederationNode) -> FederationNode:
        """Register a new federation node"""
        logger.info(f"Registering node {node.node_id} from {node.region}")

        # Validate node
        if not node.endpoint.startswith(("grpc://", "grpcs://")):
            raise ValueError("Invalid gRPC endpoint format")

        # Store in memory and persistent storage
        self.nodes[node.node_id] = node
        await self.storage.set(f"node:{node.node_id}", node.dict())

        # Notify other nodes via gossip
        await self._gossip_message(GossipMessage(
            message_id=self._generate_message_id(),
            sender_node_id=node.node_id,
            message_type="node_join",
            payload={"node": node.dict()}
        ))

        return node

    async def update_heartbeat(self, heartbeat: Heartbeat) -> None:
        """Process node heartbeat"""
        node_id = heartbeat.node_id

        if node_id not in self.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")

        # Update node status
        node = self.nodes[node_id]
        node.last_heartbeat = heartbeat.timestamp
        node.status = heartbeat.status
        node.metadata.update({
            "anomaly_score": heartbeat.anomaly_score,
            "active_remediations": heartbeat.active_remediations,
            "metrics": heartbeat.metrics
        })

        # Persist
        await self.storage.set(f"node:{node_id}", node.dict())

        logger.debug(f"Heartbeat from {node_id}: status={heartbeat.status.value}")

    async def submit_policy_bundle(self, bundle: PolicyBundle) -> PolicyBundle:
        """Submit a policy bundle for federation-wide deployment"""
        logger.info(f"Submitting policy bundle {bundle.bundle_id} ({bundle.policy_type.value})")

        # Calculate checksum
        rules_concat = "".join(bundle.rules)
        bundle.checksum = hashlib.sha256(rules_concat.encode()).hexdigest()

        # Store bundle
        self.policy_bundles[bundle.bundle_id] = bundle
        await self.storage.set(f"policy:{bundle.bundle_id}", bundle.dict())

        # Initiate consensus vote
        vote = await self._initiate_vote(
            subject=f"policy:{bundle.bundle_id}",
            subject_type="policy",
            initiated_by="coordination-hub"
        )

        return bundle

    async def cast_vote(self, vote_id: str, node_id: str, approve: bool) -> ConsensusVote:
        """Cast a vote on a consensus decision"""
        if vote_id not in self.votes:
            raise HTTPException(status_code=404, detail=f"Vote {vote_id} not found")

        vote = self.votes[vote_id]

        # Check if vote is still active
        if vote.status != VoteStatus.PENDING:
            raise HTTPException(status_code=400, detail=f"Vote {vote_id} is not active")

        if datetime.utcnow() > vote.expires_at:
            vote.status = VoteStatus.TIMEOUT
            raise HTTPException(status_code=400, detail=f"Vote {vote_id} has expired")

        # Record vote
        vote.votes[node_id] = approve
        logger.info(f"Node {node_id} voted {'APPROVE' if approve else 'REJECT'} on {vote_id}")

        # Check if quorum reached
        await self._check_vote_result(vote)

        # Persist
        await self.storage.set(f"vote:{vote_id}", vote.dict())

        return vote

    async def get_cluster_registry(self) -> ClusterRegistry:
        """Get current cluster registry"""
        healthy_count = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)

        return ClusterRegistry(
            nodes=self.nodes,
            total_nodes=len(self.nodes),
            healthy_nodes=healthy_count,
            last_updated=datetime.utcnow()
        )

    async def sync_config(self, config: ConfigSync) -> ConfigSync:
        """Synchronize configuration across federation"""
        logger.info(f"Syncing config {config.config_key} from {config.node_id}")

        if config.consensus_required:
            # Initiate vote
            vote = await self._initiate_vote(
                subject=f"config:{config.config_key}",
                subject_type="config",
                initiated_by=config.node_id
            )
            config.approved_by = []  # Will be populated when vote passes
        else:
            # Direct sync
            config.approved_by = [config.node_id]

        self.config_state[config.config_key] = config
        await self.storage.set(f"config:{config.config_key}", config.dict())

        return config

    async def _initiate_vote(self, subject: str, subject_type: str, initiated_by: str) -> ConsensusVote:
        """Initiate a consensus vote"""
        vote_id = self._generate_vote_id(subject)

        # Calculate quorum (simple majority)
        quorum = (len(self.nodes) // 2) + 1

        vote = ConsensusVote(
            vote_id=vote_id,
            subject=subject,
            subject_type=subject_type,
            initiated_by=initiated_by,
            required_votes=quorum,
            expires_at=datetime.utcnow() + timedelta(seconds=config.VOTE_TIMEOUT_SECONDS)
        )

        self.votes[vote_id] = vote
        await self.storage.set(f"vote:{vote_id}", vote.dict())

        # Gossip vote to all nodes
        await self._gossip_message(GossipMessage(
            message_id=self._generate_message_id(),
            sender_node_id=initiated_by,
            message_type="vote_initiated",
            payload={"vote": vote.dict()}
        ))

        logger.info(f"Initiated vote {vote_id} for {subject} (quorum: {quorum})")
        return vote

    async def _check_vote_result(self, vote: ConsensusVote) -> None:
        """Check if vote has reached consensus"""
        approve_count = sum(1 for v in vote.votes.values() if v)
        reject_count = len(vote.votes) - approve_count

        if approve_count >= vote.required_votes:
            vote.status = VoteStatus.APPROVED
            vote.result = True
            logger.info(f"Vote {vote.vote_id} APPROVED ({approve_count}/{vote.required_votes})")

            # Execute approved action
            await self._execute_vote_result(vote)

        elif reject_count > (len(self.nodes) - vote.required_votes):
            vote.status = VoteStatus.REJECTED
            vote.result = False
            logger.info(f"Vote {vote.vote_id} REJECTED ({reject_count} rejections)")

    async def _execute_vote_result(self, vote: ConsensusVote) -> None:
        """Execute the result of an approved vote"""
        if vote.subject_type == "policy":
            bundle_id = vote.subject.split(":", 1)[1]
            logger.info(f"Deploying policy bundle {bundle_id} to federation")

            # Gossip deployment command
            await self._gossip_message(GossipMessage(
                message_id=self._generate_message_id(),
                sender_node_id="coordination-hub",
                message_type="policy_deploy",
                payload={"bundle_id": bundle_id}
            ))

        elif vote.subject_type == "config":
            config_key = vote.subject.split(":", 1)[1]
            logger.info(f"Applying config {config_key} to federation")

            # Update config approval list
            if config_key in self.config_state:
                self.config_state[config_key].approved_by = list(vote.votes.keys())

    async def _gossip_message(self, message: GossipMessage) -> None:
        """Broadcast message via gossip protocol"""
        # TODO: Implement actual gRPC gossip
        # For now, log the message
        logger.debug(f"Gossip: {message.message_type} from {message.sender_node_id}")

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return hashlib.sha256(f"{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]

    def _generate_vote_id(self, subject: str) -> str:
        """Generate unique vote ID"""
        return hashlib.sha256(f"{subject}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]

    async def cleanup_stale_nodes(self) -> None:
        """Background task to cleanup stale nodes"""
        while True:
            try:
                now = datetime.utcnow()
                timeout = timedelta(seconds=config.NODE_HEARTBEAT_TIMEOUT)

                for node_id, node in list(self.nodes.items()):
                    if (now - node.last_heartbeat) > timeout:
                        if node.status != NodeStatus.UNREACHABLE:
                            logger.warning(f"Node {node_id} is unreachable")
                            node.status = NodeStatus.UNREACHABLE
                            await self.storage.set(f"node:{node_id}", node.dict())

                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(30)


# Global hub instance
hub: Optional[CoordinationHub] = None


async def get_hub() -> CoordinationHub:
    """Dependency injection for hub"""
    if hub is None:
        raise HTTPException(status_code=503, detail="Hub not initialized")
    return hub


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    global hub

    # Initialize storage
    if config.STORAGE_BACKEND == "postgres":
        storage = PostgresStorage(config.POSTGRES_URL)
    elif config.STORAGE_BACKEND == "etcd":
        storage = EtcdStorage(config.ETCD_ENDPOINTS)
    else:
        raise ValueError(f"Unsupported storage backend: {config.STORAGE_BACKEND}")

    await storage.connect()

    # Initialize hub
    hub = CoordinationHub(
        consensus_algorithm=ConsensusAlgorithm(config.CONSENSUS_ALGORITHM),
        storage=storage
    )

    # Start background tasks
    asyncio.create_task(hub.cleanup_stale_nodes())

    logger.info("Coordination Hub started")
    yield

    # Cleanup
    await storage.disconnect()
    logger.info("Coordination Hub stopped")


# FastAPI app
app = FastAPI(
    title="T.A.R.S. Federation Coordination Hub",
    version="0.7.0-alpha",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "coordination-hub"}


@app.post("/api/v1/nodes/register", response_model=FederationNode)
async def register_node(node: FederationNode, hub: CoordinationHub = Depends(get_hub)):
    """Register a federation node"""
    return await hub.register_node(node)


@app.post("/api/v1/nodes/{node_id}/heartbeat")
async def update_heartbeat(node_id: str, heartbeat: Heartbeat, hub: CoordinationHub = Depends(get_hub)):
    """Update node heartbeat"""
    heartbeat.node_id = node_id
    await hub.update_heartbeat(heartbeat)
    return {"status": "ok"}


@app.get("/api/v1/cluster/registry", response_model=ClusterRegistry)
async def get_cluster_registry(hub: CoordinationHub = Depends(get_hub)):
    """Get cluster registry"""
    return await hub.get_cluster_registry()


@app.post("/api/v1/policies/submit", response_model=PolicyBundle)
async def submit_policy_bundle(bundle: PolicyBundle, hub: CoordinationHub = Depends(get_hub)):
    """Submit policy bundle for federation deployment"""
    return await hub.submit_policy_bundle(bundle)


@app.post("/api/v1/votes/{vote_id}/cast", response_model=ConsensusVote)
async def cast_vote(vote_id: str, node_id: str, approve: bool, hub: CoordinationHub = Depends(get_hub)):
    """Cast a vote on consensus decision"""
    return await hub.cast_vote(vote_id, node_id, approve)


@app.get("/api/v1/votes/{vote_id}", response_model=ConsensusVote)
async def get_vote(vote_id: str, hub: CoordinationHub = Depends(get_hub)):
    """Get vote status"""
    if vote_id not in hub.votes:
        raise HTTPException(status_code=404, detail="Vote not found")
    return hub.votes[vote_id]


@app.post("/api/v1/config/sync", response_model=ConfigSync)
async def sync_config(config_sync: ConfigSync, hub: CoordinationHub = Depends(get_hub)):
    """Synchronize configuration"""
    return await hub.sync_config(config_sync)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
