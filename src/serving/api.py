"""
API de predicao de fraudes com FastAPI.
Endpoints para inferencia em tempo real.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict


# ============================================
# SCHEMAS
# ============================================


class TransactionInput(BaseModel):
    """Schema de entrada para uma transacao."""

    amount: float = Field(..., ge=0, description="Valor da transacao")
    v1: float = Field(default=0.0)
    v2: float = Field(default=0.0)
    v3: float = Field(default=0.0)
    v4: float = Field(default=0.0)
    v5: float = Field(default=0.0)
    v6: float = Field(default=0.0)
    v7: float = Field(default=0.0)
    v8: float = Field(default=0.0)
    v9: float = Field(default=0.0)
    v10: float = Field(default=0.0)
    v11: float = Field(default=0.0)
    v12: float = Field(default=0.0)
    v13: float = Field(default=0.0)
    v14: float = Field(default=0.0)
    v15: float = Field(default=0.0)
    v16: float = Field(default=0.0)
    v17: float = Field(default=0.0)
    v18: float = Field(default=0.0)
    v19: float = Field(default=0.0)
    v20: float = Field(default=0.0)
    v21: float = Field(default=0.0)
    v22: float = Field(default=0.0)
    v23: float = Field(default=0.0)
    v24: float = Field(default=0.0)
    v25: float = Field(default=0.0)
    v26: float = Field(default=0.0)
    v27: float = Field(default=0.0)
    v28: float = Field(default=0.0)

    # Features derivadas (opcionais - serao calculadas se nao fornecidas)
    avg_amount_last_100: Optional[float] = None
    std_amount_last_100: Optional[float] = None
    max_amount_last_100: Optional[float] = None
    min_amount_last_100: Optional[float] = None
    tx_count_last_50: Optional[float] = None
    amount_deviation: Optional[float] = None
    amount_percentile: Optional[float] = None
    amount_change: Optional[float] = None
    amount_ratio: Optional[float] = None
    amount_zscore: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 149.62,
                "v1": -1.359807,
                "v2": -0.072781,
                "v3": 2.536346,
                "v4": 1.378155,
                "v5": -0.338321,
                "v14": -0.5278,
            }
        }
    )


class PredictionOutput(BaseModel):
    """Schema de saida da predicao."""

    transaction_id: str = Field(..., description="ID unico da transacao")
    is_fraud: bool = Field(..., description="Predicao de fraude")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probabilidade de fraude")
    risk_level: str = Field(..., description="Nivel de risco: low, medium, high")
    timestamp: str = Field(..., description="Timestamp da predicao")


class BatchInput(BaseModel):
    """Schema para predicao em batch."""

    transactions: List[TransactionInput]


class BatchOutput(BaseModel):
    """Schema de saida para batch."""

    predictions: List[PredictionOutput]
    total: int
    fraud_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Schema de health check."""

    status: str
    model_loaded: bool
    model_path: str
    timestamp: str


# ============================================
# MODEL LOADING
# ============================================

# Variaveis globais para o modelo
model = None
feature_columns = None


def load_model():
    """Carrega o modelo e feature columns."""
    global model, feature_columns

    model_path = os.environ.get("MODEL_PATH", "models/fraud_model.joblib")
    feature_cols_path = Path(model_path).parent / "feature_columns.joblib"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    model = joblib.load(model_path)

    if feature_cols_path.exists():
        feature_columns = joblib.load(feature_cols_path)
    else:
        # Colunas padrao
        feature_columns = (
            ["amount"]
            + [f"v{i}" for i in range(1, 29)]
            + [
                "avg_amount_last_100",
                "std_amount_last_100",
                "max_amount_last_100",
                "min_amount_last_100",
                "tx_count_last_50",
                "amount_deviation",
                "amount_percentile",
                "amount_change",
                "amount_ratio",
                "amount_zscore",
            ]
        )

    print(f"[OK] Modelo carregado: {model_path}")
    print(f"[OK] Features: {len(feature_columns)} colunas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager para carregar modelo no startup."""
    load_model()
    yield


# ============================================
# APP
# ============================================

app = FastAPI(
    title="Fraud Detection API",
    description="API para deteccao de fraudes em transacoes financeiras",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================
# HELPERS
# ============================================


def get_risk_level(probability: float) -> str:
    """Retorna nivel de risco baseado na probabilidade."""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    return "high"


def prepare_features(transaction: TransactionInput) -> np.ndarray:
    """Prepara features para predicao."""
    features = {}

    # Features basicas
    features["amount"] = transaction.amount
    for i in range(1, 29):
        features[f"v{i}"] = getattr(transaction, f"v{i}", 0.0)

    # Features derivadas (usar valores fornecidos ou defaults)
    features["avg_amount_last_100"] = transaction.avg_amount_last_100 or transaction.amount
    features["std_amount_last_100"] = transaction.std_amount_last_100 or 0.0
    features["max_amount_last_100"] = transaction.max_amount_last_100 or transaction.amount
    features["min_amount_last_100"] = transaction.min_amount_last_100 or transaction.amount
    features["tx_count_last_50"] = transaction.tx_count_last_50 or 1.0
    features["amount_deviation"] = transaction.amount_deviation or 0.0
    features["amount_percentile"] = transaction.amount_percentile or 0.5
    features["amount_change"] = transaction.amount_change or 0.0
    features["amount_ratio"] = transaction.amount_ratio or 1.0
    features["amount_zscore"] = transaction.amount_zscore or 0.0

    # Ordenar de acordo com feature_columns
    feature_array = [features.get(col, 0.0) for col in feature_columns]

    return np.array([feature_array])


# ============================================
# ENDPOINTS
# ============================================


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz."""
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica saude da API."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=os.environ.get("MODEL_PATH", "models/fraud_model.joblib"),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(transaction: TransactionInput):
    """
    Realiza predicao de fraude para uma transacao.

    - **amount**: Valor da transacao (obrigatorio)
    - **v1-v28**: Features PCA do dataset original
    - **features derivadas**: Calculadas automaticamente se nao fornecidas
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo nao carregado",
        )

    try:
        # Preparar features
        X = prepare_features(transaction)

        # Predicao
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        # Gerar ID unico
        transaction_id = f"tx_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        return PredictionOutput(
            transaction_id=transaction_id,
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=get_risk_level(probability),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na predicao: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    """
    Realiza predicao de fraude para multiplas transacoes.

    Maximo de 1000 transacoes por request.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo nao carregado",
        )

    if len(batch.transactions) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximo de 1000 transacoes por request",
        )

    start_time = datetime.now()

    try:
        predictions = []
        fraud_count = 0

        for i, transaction in enumerate(batch.transactions):
            X = prepare_features(transaction)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]

            transaction_id = f"tx_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i:04d}"

            pred_output = PredictionOutput(
                transaction_id=transaction_id,
                is_fraud=bool(prediction),
                fraud_probability=float(probability),
                risk_level=get_risk_level(probability),
                timestamp=datetime.now().isoformat(),
            )
            predictions.append(pred_output)

            if prediction:
                fraud_count += 1

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return BatchOutput(
            predictions=predictions,
            total=len(predictions),
            fraud_count=fraud_count,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no batch: {str(e)}",
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Retorna informacoes do modelo carregado."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo nao carregado",
        )

    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "model_path": os.environ.get("MODEL_PATH", "models/fraud_model.joblib"),
    }
