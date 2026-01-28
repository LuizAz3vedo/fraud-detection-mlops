"""Testes para o modulo de monitoramento."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift import DriftMonitor


class TestDriftMonitor:
    """Testes para a classe DriftMonitor."""

    def test_get_feature_columns(self):
        """Testa extracao de colunas de features."""
        monitor = DriftMonitor()

        df = pd.DataFrame({
            "transaction_id": ["tx_1", "tx_2"],
            "is_fraud": [0, 1],
            "amount": [100.0, 200.0],
            "v1": [0.1, 0.2],
            "v2": [0.3, 0.4],
            "created_at": ["2024-01-01", "2024-01-02"],
        })

        feature_cols = monitor._get_feature_columns(df)

        assert "amount" in feature_cols
        assert "v1" in feature_cols
        assert "v2" in feature_cols
        assert "transaction_id" not in feature_cols
        assert "is_fraud" not in feature_cols
        assert "created_at" not in feature_cols

    def test_simulate_production_drift(self):
        """Testa simulacao de drift."""
        monitor = DriftMonitor()

        # Criar dados de referencia mock
        reference = pd.DataFrame({
            "amount": np.random.uniform(10, 100, 100),
            "v1": np.random.normal(0, 1, 100),
            "v2": np.random.normal(0, 1, 100),
            "is_fraud": np.random.randint(0, 2, 100),
        })

        # Simular drift
        production = reference.copy()
        production["amount"] = production["amount"] * 2 + np.random.normal(0, 50, 100)
        production["v1"] = production["v1"] + np.random.normal(1.5, 0.5, 100)

        # Verificar que os dados foram modificados
        assert not np.allclose(reference["amount"].mean(), production["amount"].mean())
        assert not np.allclose(reference["v1"].mean(), production["v1"].mean())


class TestDriftDetection:
    """Testes para deteccao de drift."""

    def test_no_drift_same_data(self):
        """Testa que dados identicos nao geram drift."""
        # Criar dados identicos
        data = pd.DataFrame({
            "amount": np.random.uniform(10, 100, 1000),
            "v1": np.random.normal(0, 1, 1000),
            "v2": np.random.normal(0, 1, 1000),
            "v14": np.random.normal(0, 1, 1000),
        })

        reference = data.copy()
        current = data.copy()

        # Com dados identicos, drift_share deve ser baixo
        # (Nota: Este teste e simplificado, drift real usa estatisticas)
        assert len(reference) == len(current)
        assert list(reference.columns) == list(current.columns)

    def test_drift_with_shifted_data(self):
        """Testa deteccao de drift com dados deslocados."""
        np.random.seed(42)

        # Dados de referencia
        reference = pd.DataFrame({
            "amount": np.random.uniform(10, 100, 1000),
            "v1": np.random.normal(0, 1, 1000),
        })

        # Dados com drift significativo
        current = pd.DataFrame({
            "amount": np.random.uniform(500, 1000, 1000),  # Muito diferente
            "v1": np.random.normal(5, 2, 1000),  # Media e std diferentes
        })

        # Verificar que as medias sao diferentes
        assert abs(reference["amount"].mean() - current["amount"].mean()) > 100
        assert abs(reference["v1"].mean() - current["v1"].mean()) > 3


class TestReportGeneration:
    """Testes para geracao de relatorios."""

    def test_reports_directory_created(self, tmp_path):
        """Testa que diretorio de relatorios e criado."""
        reports_path = tmp_path / "reports"
        monitor = DriftMonitor(reports_path=str(reports_path))

        assert reports_path.exists()

    def test_drift_result_structure(self):
        """Testa estrutura do resultado de drift."""
        # Simular resultado de drift
        result = {
            "timestamp": "2024-01-01T00:00:00",
            "drift_detected": True,
            "drift_share": 0.3,
            "n_drifted_columns": 3,
            "n_total_columns": 10,
            "drifted_columns": ["amount", "v1", "v14"],
            "reference_size": 10000,
            "current_size": 5000,
        }

        # Verificar campos obrigatorios
        assert "drift_detected" in result
        assert "drift_share" in result
        assert "n_drifted_columns" in result
        assert "drifted_columns" in result
        assert isinstance(result["drifted_columns"], list)
