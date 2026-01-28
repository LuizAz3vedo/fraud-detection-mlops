"""Testes para o modulo de treinamento."""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import FraudModelTrainer


class TestFraudModelTrainer:
    """Testes para a classe FraudModelTrainer."""

    def test_get_feature_columns(self):
        """Testa extracao de colunas de features."""
        trainer = FraudModelTrainer()

        df = pd.DataFrame({
            "transaction_id": ["tx_1", "tx_2"],
            "is_fraud": [0, 1],
            "amount": [100.0, 200.0],
            "v1": [0.1, 0.2],
            "v2": [0.3, 0.4],
        })

        feature_cols = trainer._get_feature_columns(df)

        assert "amount" in feature_cols
        assert "v1" in feature_cols
        assert "v2" in feature_cols
        assert "transaction_id" not in feature_cols
        assert "is_fraud" not in feature_cols

    def test_calculate_metrics(self):
        """Testa calculo de metricas."""
        trainer = FraudModelTrainer()

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.4])

        metrics = trainer._calculate_metrics(y_true, y_pred, y_prob)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert "true_positives" in metrics
        assert "false_positives" in metrics

        # Verificar valores
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_negatives"] == 1

    def test_calculate_metrics_perfect(self):
        """Testa metricas com predicao perfeita."""
        trainer = FraudModelTrainer()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = trainer._calculate_metrics(y_true, y_pred, y_prob)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_calculate_metrics_all_negative(self):
        """Testa metricas quando prediz tudo como negativo."""
        trainer = FraudModelTrainer()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = trainer._calculate_metrics(y_true, y_pred, y_prob)

        # Precision = 0 (zero division handled)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestMetricsCalculation:
    """Testes detalhados de calculo de metricas."""

    def test_roc_auc_range(self):
        """Testa que ROC-AUC esta entre 0 e 1."""
        trainer = FraudModelTrainer()

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.4])

        metrics = trainer._calculate_metrics(y_true, y_pred, y_prob)

        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1

    def test_confusion_matrix_sum(self):
        """Testa que matriz de confusao soma ao total."""
        trainer = FraudModelTrainer()

        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_prob = np.random.random(n_samples)

        metrics = trainer._calculate_metrics(y_true, y_pred, y_prob)

        total = (
            metrics["true_positives"]
            + metrics["true_negatives"]
            + metrics["false_positives"]
            + metrics["false_negatives"]
        )

        assert total == n_samples


class TestFeatureColumns:
    """Testes para selecao de features."""

    def test_exclude_metadata_columns(self):
        """Testa exclusao de colunas de metadados."""
        trainer = FraudModelTrainer()

        df = pd.DataFrame({
            "transaction_id": ["tx_1"],
            "is_fraud": [0],
            "created_at": ["2024-01-01"],
            "ingested_at": ["2024-01-01"],
            "time_elapsed": [100],
            "amount": [50.0],
            "feature_1": [0.5],
        })

        feature_cols = trainer._get_feature_columns(df)

        assert "transaction_id" not in feature_cols
        assert "is_fraud" not in feature_cols
        assert "created_at" not in feature_cols
        assert "ingested_at" not in feature_cols
        assert "time_elapsed" not in feature_cols
        assert "amount" in feature_cols
        assert "feature_1" in feature_cols

    def test_all_v_columns_included(self):
        """Testa que todas as colunas V sao incluidas."""
        trainer = FraudModelTrainer()

        # Simular dataset original com V1-V28
        columns = {"transaction_id": ["tx_1"], "is_fraud": [0], "amount": [100.0]}
        for i in range(1, 29):
            columns[f"v{i}"] = [0.1]

        df = pd.DataFrame(columns)
        feature_cols = trainer._get_feature_columns(df)

        # Verificar que todas V columns estao presentes
        for i in range(1, 29):
            assert f"v{i}" in feature_cols
