"""Testes para a API de deteccao de fraudes."""

import sys
from pathlib import Path
import pytest

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.serving.api import app, get_risk_level, TransactionInput


# Verificar se modelo existe para pular testes que dependem dele
MODEL_EXISTS = Path("models/fraud_model.joblib").exists()


class TestRiskLevel:
    """Testes para funcao de nivel de risco."""

    def test_low_risk(self):
        """Testa nivel de risco baixo."""
        assert get_risk_level(0.1) == "low"
        assert get_risk_level(0.29) == "low"

    def test_medium_risk(self):
        """Testa nivel de risco medio."""
        assert get_risk_level(0.3) == "medium"
        assert get_risk_level(0.5) == "medium"
        assert get_risk_level(0.69) == "medium"

    def test_high_risk(self):
        """Testa nivel de risco alto."""
        assert get_risk_level(0.7) == "high"
        assert get_risk_level(0.9) == "high"
        assert get_risk_level(1.0) == "high"


class TestTransactionInput:
    """Testes para schema de entrada."""

    def test_minimal_input(self):
        """Testa input minimo (apenas amount)."""
        transaction = TransactionInput(amount=100.0)
        assert transaction.amount == 100.0
        assert transaction.v1 == 0.0

    def test_full_input(self):
        """Testa input completo."""
        transaction = TransactionInput(
            amount=150.0,
            v1=-1.5,
            v14=-0.5,
            avg_amount_last_100=88.0,
        )
        assert transaction.amount == 150.0
        assert transaction.v1 == -1.5
        assert transaction.v14 == -0.5
        assert transaction.avg_amount_last_100 == 88.0


class TestAPIEndpoints:
    """Testes para endpoints da API."""

    def setup_method(self):
        """Setup do cliente de teste."""
        self.client = TestClient(app)

    def test_root(self):
        """Testa endpoint raiz."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Fraud Detection API"

    def test_health(self):
        """Testa health check."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_predict_valid(self):
        """Testa predicao valida."""
        response = self.client.post(
            "/predict",
            json={"amount": 100.0, "v1": -1.0, "v14": -0.5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert data["fraud_probability"] >= 0
        assert data["fraud_probability"] <= 1
        assert data["risk_level"] in ["low", "medium", "high"]

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_predict_minimal(self):
        """Testa predicao com input minimo."""
        response = self.client.post(
            "/predict",
            json={"amount": 50.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_predict_batch(self):
        """Testa predicao em batch."""
        response = self.client.post(
            "/predict/batch",
            json={
                "transactions": [
                    {"amount": 100.0},
                    {"amount": 5000.0, "v14": -5.0},
                    {"amount": 10.0},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3
        assert "fraud_count" in data
        assert "processing_time_ms" in data

    def test_predict_negative_amount(self):
        """Testa que amount negativo e rejeitado."""
        response = self.client.post(
            "/predict",
            json={"amount": -100.0},
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_model_info(self):
        """Testa informacoes do modelo."""
        response = self.client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "n_features" in data
        assert "feature_columns" in data


class TestBatchLimits:
    """Testes para limites de batch."""

    def setup_method(self):
        """Setup do cliente de teste."""
        self.client = TestClient(app)

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_batch_limit_exceeded(self):
        """Testa limite de batch excedido."""
        # Criar mais de 1000 transacoes
        transactions = [{"amount": 100.0} for _ in range(1001)]

        response = self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
        )
        assert response.status_code == 400
        assert "1000" in response.json()["detail"]

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not available in CI")
    def test_batch_at_limit(self):
        """Testa batch no limite."""
        transactions = [{"amount": 100.0} for _ in range(100)]

        response = self.client.post(
            "/predict/batch",
            json={"transactions": transactions},
        )
        assert response.status_code == 200
        assert response.json()["total"] == 100
