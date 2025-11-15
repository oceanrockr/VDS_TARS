"""
Data models for Federation Coordination Hub
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    MAINTENANCE = "maintenance"


class NodeCapability(str, Enum):
    """Node capabilities"""
    COMPUTE = "compute"
    STORAGE = "storage"
    INFERENCE = "inference"
    REMEDIATION = "remediation"
    ANOMALY_DETECTION = "anomaly_detection"


class ConsensusAlgorithm(str, Enum):
    """Supported consensus algorithms"""
    RAFT = "raft"
    PBFT = "pbft"


class PolicyType(str, Enum):
    """Policy types"""
    OPERATIONAL = "operational"
    ETHICAL = "ethical"
    SECURITY = "security"


class VoteStatus(str, Enum):
    """Consensus vote status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class FederationNode(BaseModel):
    """Federation node registration and metadata"""
    node_id: str = Field(..., description="Unique node identifier")
    cluster_name: str = Field(..., description="Kubernetes cluster name")
    region: str = Field(..., description="Geographic region (e.g., us-east-1)")
    endpoint: str = Field(..., description="gRPC endpoint URL")
    capabilities: List[NodeCapability] = Field(default_factory=list)
    status: NodeStatus = Field(default=NodeStatus.HEALTHY)
    version: str = Field(..., description="T.A.R.S. version")
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Heartbeat(BaseModel):
    """Node heartbeat message"""
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: NodeStatus
    metrics: Dict[str, float] = Field(default_factory=dict)
    anomaly_score: Optional[float] = None
    active_remediations: int = 0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PolicyBundle(BaseModel):
    """Policy bundle for governance"""
    bundle_id: str = Field(..., description="Unique bundle identifier")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Semantic version")
    policy_type: PolicyType
    rules: List[str] = Field(..., description="OPA Rego rule files")
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str = Field(..., description="SHA256 checksum of rules")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConsensusVote(BaseModel):
    """Consensus vote for policy or action"""
    vote_id: str = Field(..., description="Unique vote identifier")
    subject: str = Field(..., description="What is being voted on")
    subject_type: str = Field(..., description="Type: policy, remediation, config")
    initiated_by: str = Field(..., description="Node ID that initiated vote")
    required_votes: int = Field(..., description="Minimum votes needed (quorum)")
    votes: Dict[str, bool] = Field(default_factory=dict, description="node_id -> approve/reject")
    status: VoteStatus = Field(default=VoteStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Vote expiration time")
    result: Optional[bool] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GossipMessage(BaseModel):
    """Gossip protocol message"""
    message_id: str
    sender_node_id: str
    message_type: str = Field(..., description="heartbeat, alert, metric, vote")
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl: int = Field(default=3, description="Time-to-live (hops)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClusterRegistry(BaseModel):
    """Registry of all federated clusters"""
    nodes: Dict[str, FederationNode] = Field(default_factory=dict)
    total_nodes: int = 0
    healthy_nodes: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConfigSync(BaseModel):
    """Configuration synchronization state"""
    config_key: str
    value: Any
    version: int = 1
    node_id: str = Field(..., description="Node that proposed this config")
    consensus_required: bool = True
    approved_by: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
