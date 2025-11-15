"""
ML Model Parameter Bridge - Federated Learning Coordination
Implements FedAvg/FedProx for model aggregation across federation nodes
"""
import asyncio
import logging
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelUpdate(BaseModel):
    """Model parameter update from a node"""
    node_id: str
    model_name: str
    model_version: str
    parameters: Dict[str, List[float]]  # layer_name -> weights
    sample_count: int  # Number of samples used for training
    timestamp: datetime = datetime.utcnow()
    checksum: str


class AggregatedModel(BaseModel):
    """Aggregated model from federation"""
    model_name: str
    model_version: str
    parameters: Dict[str, List[float]]
    contributing_nodes: List[str]
    total_samples: int
    aggregation_method: str  # fedavg, fedprox
    timestamp: datetime = datetime.utcnow()
    checksum: str
    signed: bool = False
    signature: Optional[str] = None


class FederatedAggregator:
    """Federated learning model aggregator"""

    def __init__(self, storage_dir: str = "/app/models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.pending_updates: Dict[str, List[ModelUpdate]] = {}
        logger.info(f"Initialized FederatedAggregator (storage: {storage_dir})")

    def register_update(self, update: ModelUpdate) -> None:
        """Register a model update from a node"""
        model_key = f"{update.model_name}:{update.model_version}"

        if model_key not in self.pending_updates:
            self.pending_updates[model_key] = []

        self.pending_updates[model_key].append(update)

        logger.info(f"Registered update from {update.node_id} for {model_key} ({update.sample_count} samples)")

    def aggregate_fedavg(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate model parameters using FedAvg (Federated Averaging)
        Weighted average by number of samples
        """
        if not updates:
            raise ValueError("No updates to aggregate")

        # Calculate total samples
        total_samples = sum(u.sample_count for u in updates)

        # Get layer names from first update
        layer_names = list(updates[0].parameters.keys())

        aggregated = {}

        for layer_name in layer_names:
            # Convert to numpy arrays
            layer_params = [
                np.array(update.parameters[layer_name]) * (update.sample_count / total_samples)
                for update in updates
            ]

            # Weighted sum
            aggregated[layer_name] = np.sum(layer_params, axis=0)

        logger.info(f"FedAvg aggregation: {len(updates)} nodes, {total_samples} samples")

        return aggregated

    def aggregate_fedprox(
        self,
        updates: List[ModelUpdate],
        global_model: Optional[Dict[str, np.ndarray]] = None,
        mu: float = 0.01
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model parameters using FedProx
        FedProx adds a proximal term to handle heterogeneous data
        """
        # FedProx aggregation (simplified)
        # In production, implement full FedProx with proximal term

        # Start with FedAvg
        aggregated = self.aggregate_fedavg(updates)

        # Apply proximal regularization if global model exists
        if global_model:
            for layer_name in aggregated.keys():
                if layer_name in global_model:
                    # Mix aggregated with global model
                    aggregated[layer_name] = (
                        (1 - mu) * aggregated[layer_name] +
                        mu * global_model[layer_name]
                    )

        logger.info(f"FedProx aggregation: {len(updates)} nodes, mu={mu}")

        return aggregated

    def aggregate_models(
        self,
        model_key: str,
        method: str = "fedavg",
        min_nodes: int = 2
    ) -> Optional[AggregatedModel]:
        """
        Aggregate pending updates for a model
        """
        if model_key not in self.pending_updates:
            return None

        updates = self.pending_updates[model_key]

        if len(updates) < min_nodes:
            logger.info(f"Not enough updates for {model_key}: {len(updates)}/{min_nodes}")
            return None

        # Load global model if exists (for FedProx)
        global_model = self._load_global_model(model_key)

        # Aggregate based on method
        if method == "fedavg":
            aggregated_params = self.aggregate_fedavg(updates)
        elif method == "fedprox":
            aggregated_params = self.aggregate_fedprox(updates, global_model)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Convert numpy arrays back to lists for JSON serialization
        parameters_dict = {
            layer_name: params.tolist()
            for layer_name, params in aggregated_params.items()
        }

        # Calculate checksum
        params_json = json.dumps(parameters_dict, sort_keys=True)
        checksum = hashlib.sha256(params_json.encode()).hexdigest()

        # Create aggregated model
        aggregated = AggregatedModel(
            model_name=updates[0].model_name,
            model_version=updates[0].model_version,
            parameters=parameters_dict,
            contributing_nodes=[u.node_id for u in updates],
            total_samples=sum(u.sample_count for u in updates),
            aggregation_method=method,
            checksum=checksum
        )

        # Save aggregated model
        self._save_global_model(model_key, aggregated)

        # Clear pending updates
        self.pending_updates[model_key] = []

        logger.info(f"Aggregated model {model_key}: {len(updates)} nodes, {aggregated.total_samples} samples")

        return aggregated

    def _save_global_model(self, model_key: str, model: AggregatedModel) -> None:
        """Save aggregated model to disk"""
        model_file = self.storage_dir / f"{model_key.replace(':', '_')}.json"

        with open(model_file, 'w') as f:
            json.dump(model.dict(), f)

        logger.info(f"Saved global model to {model_file}")

    def _load_global_model(self, model_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Load global model from disk"""
        model_file = self.storage_dir / f"{model_key.replace(':', '_')}.json"

        if not model_file.exists():
            return None

        try:
            with open(model_file, 'r') as f:
                model_data = json.load(f)

            # Convert to numpy arrays
            parameters = {
                layer_name: np.array(params)
                for layer_name, params in model_data['parameters'].items()
            }

            return parameters

        except Exception as e:
            logger.error(f"Failed to load global model: {e}")
            return None

    async def sign_model(self, model: AggregatedModel) -> AggregatedModel:
        """Sign model using cosign"""
        # In production, call cosign to sign model
        # For now, create a simple hash-based signature
        signature_input = f"{model.checksum}:{model.timestamp.isoformat()}"
        model.signature = hashlib.sha256(signature_input.encode()).hexdigest()
        model.signed = True

        logger.info(f"Signed model {model.model_name}:{model.model_version}")

        return model


# FastAPI app
app = FastAPI(
    title="T.A.R.S. ML Parameter Bridge",
    version="0.7.0-alpha"
)

# Global aggregator
aggregator: Optional[FederatedAggregator] = None


@app.on_event("startup")
async def startup():
    """Initialize aggregator on startup"""
    global aggregator

    storage_dir = os.getenv("MODEL_STORAGE_DIR", "/app/models")
    aggregator = FederatedAggregator(storage_dir)

    logger.info("ML Parameter Bridge started")


@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "mlsync-parameter-bridge"}


@app.post("/api/v1/updates/submit", response_model=Dict[str, str])
async def submit_update(update: ModelUpdate):
    """Submit a model update from a node"""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Aggregator not initialized")

    aggregator.register_update(update)

    model_key = f"{update.model_name}:{update.model_version}"
    pending_count = len(aggregator.pending_updates.get(model_key, []))

    return {
        "status": "accepted",
        "model_key": model_key,
        "pending_updates": str(pending_count)
    }


@app.post("/api/v1/aggregate/{model_name}/{model_version}", response_model=AggregatedModel)
async def aggregate_model(
    model_name: str,
    model_version: str,
    method: str = "fedavg",
    min_nodes: int = 2
):
    """Trigger model aggregation"""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Aggregator not initialized")

    model_key = f"{model_name}:{model_version}"
    aggregated = aggregator.aggregate_models(model_key, method, min_nodes)

    if not aggregated:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot aggregate: insufficient updates (need {min_nodes} nodes)"
        )

    # Sign the model
    aggregated = await aggregator.sign_model(aggregated)

    return aggregated


@app.get("/api/v1/models/{model_name}/{model_version}", response_model=AggregatedModel)
async def get_model(model_name: str, model_version: str):
    """Get aggregated model"""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Aggregator not initialized")

    model_key = f"{model_name}:{model_version}"
    model_file = aggregator.storage_dir / f"{model_key.replace(':', '_')}.json"

    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    with open(model_file, 'r') as f:
        model_data = json.load(f)

    return AggregatedModel(**model_data)


@app.get("/api/v1/updates/pending/{model_name}/{model_version}")
async def get_pending_updates(model_name: str, model_version: str):
    """Get pending updates for a model"""
    if not aggregator:
        raise HTTPException(status_code=503, detail="Aggregator not initialized")

    model_key = f"{model_name}:{model_version}"
    updates = aggregator.pending_updates.get(model_key, [])

    return {
        "model_key": model_key,
        "pending_count": len(updates),
        "nodes": [u.node_id for u in updates],
        "total_samples": sum(u.sample_count for u in updates)
    }


if __name__ == "__main__":
    uvicorn.run(
        "parameter-bridge:app",
        host="0.0.0.0",
        port=8083,
        log_level="info"
    )
