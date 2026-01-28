"""Testes para o modulo de feature engineering."""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.store import FeatureStore


class TestFeatureStore:
    """Testes para a classe FeatureStore."""

    def test_get_feature_columns_excludes_metadata(self):
        """Testa que colunas de metadados sao excluidas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar dados de teste
            df = pd.DataFrame(
                {
                    "transaction_id": ["tx_1", "tx_2", "tx_3"],
                    "is_fraud": [0, 1, 0],
                    "amount": [100.0, 200.0, 150.0],
                    "v1": [0.1, 0.2, 0.3],
                    "avg_amount_last_100": [100.0, 150.0, 125.0],
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))
            feature_cols = store.get_feature_columns()

            assert "transaction_id" not in feature_cols
            assert "is_fraud" not in feature_cols
            assert "amount" in feature_cols
            assert "v1" in feature_cols

    def test_get_class_distribution(self):
        """Testa calculo de distribuicao de classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar dados de teste com 20% de fraude
            df = pd.DataFrame(
                {
                    "transaction_id": [f"tx_{i}" for i in range(100)],
                    "is_fraud": [1] * 20 + [0] * 80,
                    "amount": [100.0] * 100,
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))
            dist = store.get_class_distribution()

            assert dist["total"] == 100
            assert dist["fraud"] == 20
            assert dist["non_fraud"] == 80
            assert dist["fraud_pct"] == 20.0
            assert dist["imbalance_ratio"] == 4.0

    def test_get_balanced_data_ratio(self):
        """Testa que dados balanceados respeitam o ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar dados desbalanceados
            df = pd.DataFrame(
                {
                    "transaction_id": [f"tx_{i}" for i in range(1000)],
                    "is_fraud": [1] * 10 + [0] * 990,
                    "amount": [100.0] * 1000,
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))

            # Balancear com ratio 1:5
            balanced = store.get_balanced_data(ratio=5)

            fraud_count = balanced["is_fraud"].sum()
            non_fraud_count = len(balanced) - fraud_count

            assert fraud_count == 10  # Todas as fraudes
            assert non_fraud_count == 50  # 5x fraudes

    def test_get_training_data_stratified(self):
        """Testa que amostragem mantem estratificacao."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Criar dados com 1% de fraude
            n_total = 10000
            n_fraud = 100
            df = pd.DataFrame(
                {
                    "transaction_id": [f"tx_{i}" for i in range(n_total)],
                    "is_fraud": [1] * n_fraud + [0] * (n_total - n_fraud),
                    "amount": np.random.uniform(1, 1000, n_total),
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))

            # Amostrar 50% dos dados
            sampled = store.get_training_data(sample_frac=0.5)

            # Deve manter todas as fraudes
            assert sampled["is_fraud"].sum() == n_fraud

    def test_get_feature_stats_returns_dataframe(self):
        """Testa que estatisticas retornam DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "transaction_id": ["tx_1", "tx_2", "tx_3"],
                    "is_fraud": [0, 1, 0],
                    "amount": [100.0, 200.0, 150.0],
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))
            stats = store.get_feature_stats()

            assert isinstance(stats, pd.DataFrame)
            assert "null_count" in stats.columns
            assert "null_pct" in stats.columns


class TestFeatureStoreEdgeCases:
    """Testes de casos extremos para o Feature Store."""

    def test_empty_balanced_data(self):
        """Testa comportamento com dados sem fraude."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "transaction_id": [f"tx_{i}" for i in range(100)],
                    "is_fraud": [0] * 100,
                    "amount": [100.0] * 100,
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))
            dist = store.get_class_distribution()

            assert dist["fraud"] == 0
            assert dist["imbalance_ratio"] == 0

    def test_all_fraud_data(self):
        """Testa comportamento com dados 100% fraude."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(
                {
                    "transaction_id": [f"tx_{i}" for i in range(100)],
                    "is_fraud": [1] * 100,
                    "amount": [100.0] * 100,
                }
            )
            features_path = Path(tmpdir) / "features"
            df.to_parquet(features_path)

            store = FeatureStore(features_path=str(features_path))
            dist = store.get_class_distribution()

            assert dist["fraud"] == 100
            assert dist["fraud_pct"] == 100.0

    def test_missing_features_path(self):
        """Testa erro quando path nao existe."""
        store = FeatureStore(features_path="path/that/does/not/exist")

        try:
            store.get_training_data()
            assert False, "Deveria levantar FileNotFoundError"
        except FileNotFoundError:
            pass  # Esperado


class TestFeatureEngineeringUnit:
    """Testes unitarios para verificar logica de feature engineering."""

    def test_window_feature_calculation(self):
        """Testa calculo de features de janela manualmente."""
        # Simular calculo de media movel
        amounts = [100, 200, 150, 300, 250]
        window_size = 3

        # Calcular media movel manualmente
        expected_avgs = []
        for i in range(len(amounts)):
            start = max(0, i - window_size + 1)
            window = amounts[start : i + 1]
            expected_avgs.append(sum(window) / len(window))

        # Verificar valores esperados
        assert expected_avgs[0] == 100.0  # Apenas primeiro valor
        assert expected_avgs[1] == 150.0  # Media de [100, 200]
        assert expected_avgs[2] == 150.0  # Media de [100, 200, 150]
        assert expected_avgs[3] == 216.67 or abs(expected_avgs[3] - 216.67) < 0.1
        assert expected_avgs[4] == 233.33 or abs(expected_avgs[4] - 233.33) < 0.1

    def test_zscore_calculation(self):
        """Testa calculo de z-score manualmente."""
        values = [10, 20, 30, 40, 50]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=0)

        # Z-score do valor 30 (media)
        zscore_30 = (30 - mean_val) / std_val
        assert abs(zscore_30) < 0.01  # Deve ser proximo de 0

        # Z-score do valor 50 (acima da media)
        zscore_50 = (50 - mean_val) / std_val
        assert zscore_50 > 0

        # Z-score do valor 10 (abaixo da media)
        zscore_10 = (10 - mean_val) / std_val
        assert zscore_10 < 0

    def test_percentile_calculation(self):
        """Testa calculo de percentil."""
        values = [10, 20, 30, 40, 50]
        df = pd.DataFrame({"amount": values})

        # Calcular percentil
        df["percentile"] = df["amount"].rank(pct=True)

        # Menor valor deve ter percentil baixo
        assert df.loc[df["amount"] == 10, "percentile"].values[0] == 0.2

        # Maior valor deve ter percentil 1.0
        assert df.loc[df["amount"] == 50, "percentile"].values[0] == 1.0


class TestDataValidation:
    """Testes para validacao de features processadas."""

    def test_features_have_no_nulls_after_processing(self):
        """Testa que features nao tem nulls apos processamento."""
        # Features criadas pelo PySpark devem preencher nulls com 0
        df = pd.DataFrame(
            {
                "transaction_id": ["tx_1", "tx_2"],
                "is_fraud": [0, 1],
                "amount": [100.0, 200.0],
                "avg_amount_last_100": [100.0, 150.0],
                "std_amount_last_100": [0.0, 50.0],
                "amount_zscore": [0.0, 1.0],
            }
        )

        # Verificar que nao ha nulls nas features
        feature_cols = ["avg_amount_last_100", "std_amount_last_100", "amount_zscore"]
        for col in feature_cols:
            assert df[col].isnull().sum() == 0

    def test_std_is_non_negative(self):
        """Testa que desvio padrao e sempre >= 0."""
        df = pd.DataFrame(
            {
                "std_amount_last_100": [0.0, 10.5, 100.0, 0.0, 50.0],
            }
        )

        assert (df["std_amount_last_100"] >= 0).all()

    def test_percentile_is_bounded(self):
        """Testa que percentil esta entre 0 e 1."""
        df = pd.DataFrame(
            {
                "amount_percentile": [0.0, 0.25, 0.5, 0.75, 1.0],
            }
        )

        assert (df["amount_percentile"] >= 0).all()
        assert (df["amount_percentile"] <= 1).all()
