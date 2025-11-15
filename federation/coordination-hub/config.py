"""
Configuration for Federation Coordination Hub
"""
import os
from typing import List

# Storage
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "postgres")  # postgres or etcd
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://tars:tars@postgres:5432/federation")
ETCD_ENDPOINTS = os.getenv("ETCD_ENDPOINTS", "http://etcd:2379").split(",")

# Consensus
CONSENSUS_ALGORITHM = os.getenv("CONSENSUS_ALGORITHM", "raft")  # raft or pbft
VOTE_TIMEOUT_SECONDS = int(os.getenv("VOTE_TIMEOUT_SECONDS", "300"))  # 5 minutes

# Node management
NODE_HEARTBEAT_TIMEOUT = int(os.getenv("NODE_HEARTBEAT_TIMEOUT", "30"))  # seconds
NODE_CLEANUP_INTERVAL = int(os.getenv("NODE_CLEANUP_INTERVAL", "30"))  # seconds

# gRPC
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
GRPC_MAX_WORKERS = int(os.getenv("GRPC_MAX_WORKERS", "10"))

# TLS/mTLS
TLS_ENABLED = os.getenv("TLS_ENABLED", "true").lower() == "true"
TLS_CERT_PATH = os.getenv("TLS_CERT_PATH", "/certs/tls.crt")
TLS_KEY_PATH = os.getenv("TLS_KEY_PATH", "/certs/tls.key")
TLS_CA_PATH = os.getenv("TLS_CA_PATH", "/certs/ca.crt")
MTLS_ENABLED = os.getenv("MTLS_ENABLED", "true").lower() == "true"

# Gossip
GOSSIP_INTERVAL = int(os.getenv("GOSSIP_INTERVAL", "10"))  # seconds
GOSSIP_FANOUT = int(os.getenv("GOSSIP_FANOUT", "3"))  # number of nodes to gossip to
