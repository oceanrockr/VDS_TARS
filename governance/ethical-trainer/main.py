"""
Ethical Fairness Trainer Service
FastAPI service for training and serving fairness models
"""
import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from train import EthicalFairnessTrainer, FairnessFeatureExtractor
from explainability import SimpleLIMEExplainer, generate_explanation_summary

# Configuration
DB_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://tars:tars@postgres:5432/tars"
)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/fairness_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/scaler.pkl")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
FAIRNESS_MODEL_ACCURACY = Gauge(
    'tars_fairness_model_accuracy',
    'Current fairness model accuracy'
)

FAIRNESS_PREDICTIONS = Counter(
    'tars_fairness_predictions_total',
    'Total fairness predictions made',
    ['prediction']
)

FAIRNESS_TRAINING_RUNS = Counter(
    'tars_fairness_training_runs_total',
    'Total model training runs',
    ['status']
)

# Global trainer
trainer: EthicalFairnessTrainer = None
explainer: SimpleLIMEExplainer = None

app = FastAPI(
    title="T.A.R.S. Ethical Fairness Trainer",
    description="Federated ethical learning with explainability",
    version="0.8.0-alpha"
)


@app.on_event("startup")
async def startup():
    """Initialize trainer on startup"""
    global trainer, explainer

    logger.info("Starting Ethical Fairness Trainer")

    trainer = EthicalFairnessTrainer(
        db_url=DB_URL,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )

    feature_extractor = FairnessFeatureExtractor()
    explainer = SimpleLIMEExplainer(feature_extractor.get_feature_names())

    logger.info("Ethical Fairness Trainer ready")


class TrainRequest(BaseModel):
    """Request to train model"""
    lookback_days: int = Field(default=30, ge=1, le=365)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)


class PredictRequest(BaseModel):
    """Request for fairness prediction"""
    fairness_score: float
    training_data_distribution: Dict[str, float]
    outcome_by_group: Dict[str, float]
    sample_size: int


class ThresholdRequest(BaseModel):
    """Request for threshold suggestion"""
    current_threshold: float = Field(..., ge=0.0, le=1.0)
    target_fairness_rate: float = Field(default=0.85, ge=0.5, le=1.0)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "ethical-fairness-trainer",
        "version": "0.8.0-alpha"
    }


@app.post("/api/v1/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train fairness model"""

    async def train_task():
        try:
            result = await trainer.train(
                lookback_days=request.lookback_days,
                test_size=request.test_size
            )

            if result["success"]:
                FAIRNESS_MODEL_ACCURACY.set(result["accuracy"])
                FAIRNESS_TRAINING_RUNS.labels(status="success").inc()
                logger.info(f"Training successful: accuracy={result['accuracy']:.3f}")
            else:
                FAIRNESS_TRAINING_RUNS.labels(status="failed").inc()
                logger.warning(f"Training failed: {result.get('reason')}")

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            FAIRNESS_TRAINING_RUNS.labels(status="error").inc()

    background_tasks.add_task(train_task)

    return {
        "status": "training_started",
        "lookback_days": request.lookback_days,
        "test_size": request.test_size
    }


@app.post("/api/v1/predict")
async def predict_fairness(request: PredictRequest):
    """Predict if decision will be fair"""

    try:
        features = {
            "fairness_score": request.fairness_score,
            "training_data_distribution": request.training_data_distribution,
            "outcome_by_group": request.outcome_by_group,
            "sample_size": request.sample_size
        }

        prediction, probability = trainer.predict_fairness(features)

        FAIRNESS_PREDICTIONS.labels(
            prediction="fair" if prediction == 1 else "unfair"
        ).inc()

        # Generate explanation
        if trainer.model:
            feature_extractor = FairnessFeatureExtractor()
            feature_vector = feature_extractor.extract_features({"metadata": features})

            if feature_vector is not None:
                feature_importance = dict(zip(
                    feature_extractor.get_feature_names(),
                    trainer.model.feature_importances_
                ))

                explanation = explainer.explain_decision(
                    feature_vector,
                    prediction,
                    probability,
                    feature_importance
                )

                explanation_summary = generate_explanation_summary(explanation)
            else:
                explanation = None
                explanation_summary = "Unable to generate explanation"
        else:
            explanation = None
            explanation_summary = "Model not trained"

        return {
            "prediction": "fair" if prediction == 1 else "unfair",
            "probability": probability,
            "confidence": probability if prediction == 1 else (1 - probability),
            "explanation": explanation,
            "explanation_summary": explanation_summary
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/suggest-threshold")
async def suggest_threshold(request: ThresholdRequest):
    """Suggest new fairness threshold"""

    try:
        suggestion = trainer.suggest_fairness_threshold(
            current_threshold=request.current_threshold,
            target_fairness_rate=request.target_fairness_rate
        )

        return suggestion

    except Exception as e:
        logger.error(f"Threshold suggestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/statistics")
async def get_statistics():
    """Get model statistics"""

    try:
        stats = trainer.get_model_statistics()
        return stats

    except Exception as e:
        logger.error(f"Statistics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8093, log_level="info")
