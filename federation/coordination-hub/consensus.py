"""
Consensus algorithms: Raft and PBFT implementations
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class ConsensusEngine(ABC):
    """Abstract base class for consensus algorithms"""

    def __init__(self, storage):
        self.storage = storage
        self.state: Dict[str, Any] = {}

    @abstractmethod
    async def propose(self, key: str, value: Any) -> bool:
        """Propose a value for consensus"""
        pass

    @abstractmethod
    async def get_value(self, key: str) -> Optional[Any]:
        """Get consensus value"""
        pass


class RaftConsensus(ConsensusEngine):
    """
    Simplified Raft consensus implementation
    Supports leader election, log replication, and state machine
    """

    def __init__(self, storage):
        super().__init__(storage)
        self.role = "follower"  # follower, candidate, leader
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: list = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: Optional[str] = None
        self.election_timeout = random.uniform(5, 10)  # seconds
        self.heartbeat_interval = 2  # seconds
        self.last_heartbeat = datetime.utcnow()

        logger.info("Initialized Raft consensus engine")

    async def propose(self, key: str, value: Any) -> bool:
        """
        Propose a value for consensus
        Only leader can propose; followers must forward to leader
        """
        if self.role != "leader":
            logger.warning(f"Cannot propose: not leader (current role: {self.role})")
            return False

        # Append to log
        log_entry = {
            "term": self.current_term,
            "key": key,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log.append(log_entry)

        logger.info(f"Proposed {key}={value} in term {self.current_term}")

        # In production, replicate to followers here
        # For now, immediately commit (single-node simulation)
        self.commit_index = len(self.log) - 1
        await self._apply_log_entry(log_entry)

        return True

    async def get_value(self, key: str) -> Optional[Any]:
        """Get committed consensus value"""
        return self.state.get(key)

    async def _apply_log_entry(self, entry: Dict[str, Any]) -> None:
        """Apply committed log entry to state machine"""
        self.state[entry["key"]] = entry["value"]
        self.last_applied += 1

        # Persist to storage
        await self.storage.set(f"raft:state:{entry['key']}", entry["value"])

        logger.debug(f"Applied log entry: {entry['key']}={entry['value']}")

    async def start_election(self) -> None:
        """Start leader election"""
        self.role = "candidate"
        self.current_term += 1
        self.voted_for = "self"  # Vote for self

        logger.info(f"Starting election for term {self.current_term}")

        # In production, request votes from other nodes
        # For now, immediately become leader (single-node simulation)
        await self._become_leader()

    async def _become_leader(self) -> None:
        """Transition to leader role"""
        self.role = "leader"
        self.leader_id = "self"

        logger.info(f"Became leader for term {self.current_term}")

    async def receive_heartbeat(self, leader_id: str, term: int) -> None:
        """Receive heartbeat from leader"""
        if term >= self.current_term:
            self.current_term = term
            self.role = "follower"
            self.leader_id = leader_id
            self.last_heartbeat = datetime.utcnow()

            logger.debug(f"Received heartbeat from {leader_id} (term {term})")

    async def send_heartbeat(self) -> None:
        """Send heartbeat to followers (leader only)"""
        if self.role != "leader":
            return

        # In production, send to all followers
        # For now, just log
        logger.debug(f"Sending heartbeat (term {self.current_term})")


class PBFTConsensus(ConsensusEngine):
    """
    Simplified PBFT (Practical Byzantine Fault Tolerance) implementation
    Supports 3-phase commit: pre-prepare, prepare, commit
    """

    def __init__(self, storage):
        super().__init__(storage)
        self.sequence_number = 0
        self.view = 0
        self.primary_id: Optional[str] = None
        self.prepared_certificates: Dict[int, Dict[str, bool]] = {}
        self.committed_certificates: Dict[int, Dict[str, bool]] = {}

        logger.info("Initialized PBFT consensus engine")

    async def propose(self, key: str, value: Any) -> bool:
        """
        Propose a value for consensus via 3-phase PBFT
        """
        self.sequence_number += 1
        seq = self.sequence_number

        # Pre-prepare phase (primary only)
        request = {
            "view": self.view,
            "sequence": seq,
            "key": key,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"PBFT Pre-prepare: seq={seq}, {key}={value}")

        # In production, broadcast to replicas
        # For now, simulate immediate consensus (single-node)
        await self._commit_request(request)

        return True

    async def get_value(self, key: str) -> Optional[Any]:
        """Get committed consensus value"""
        return self.state.get(key)

    async def _prepare_phase(self, seq: int, request: Dict[str, Any]) -> bool:
        """
        Prepare phase: collect 2f+1 prepare messages
        """
        # Initialize prepare certificate
        if seq not in self.prepared_certificates:
            self.prepared_certificates[seq] = {}

        # In production, count votes from replicas
        # For now, simulate immediate quorum
        self.prepared_certificates[seq]["self"] = True

        # Check if we have 2f+1 prepares (quorum)
        # For 3f+1 total nodes, need 2f+1 = (2*(3f+1-1)/3)+1
        # Simplified: assume single node = immediate quorum
        return True

    async def _commit_phase(self, seq: int, request: Dict[str, Any]) -> bool:
        """
        Commit phase: collect 2f+1 commit messages
        """
        # Initialize commit certificate
        if seq not in self.committed_certificates:
            self.committed_certificates[seq] = {}

        # In production, count votes from replicas
        self.committed_certificates[seq]["self"] = True

        # Check if we have 2f+1 commits (quorum)
        return True

    async def _commit_request(self, request: Dict[str, Any]) -> None:
        """
        Execute committed request
        """
        seq = request["sequence"]

        # Apply to state machine
        self.state[request["key"]] = request["value"]

        # Persist to storage
        await self.storage.set(f"pbft:state:{request['key']}", request["value"])

        logger.info(f"PBFT Committed: seq={seq}, {request['key']}={request['value']}")

    async def view_change(self, new_view: int) -> None:
        """
        Initiate view change (change primary)
        """
        self.view = new_view
        logger.info(f"View changed to {new_view}")
