"""Testes para o modulo de ingestao de dados."""

import pandas as pd
from pathlib import Path
import tempfile
import sys

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestion


class TestDataIngestion:
    """Testes para a classe DataIngestion."""

    def test_standardize_columns_rename_class(self):
        """Testa renomeacao de 'class' para 'is_fraud'."""
        ingestion = DataIngestion()

        df = pd.DataFrame({"Class": [0, 1, 0], "Amount": [100, 200, 150]})

        result = ingestion._standardize_columns(df)

        assert "is_fraud" in result.columns
        assert "class" not in result.columns

    def test_standardize_columns_rename_time(self):
        """Testa renomeacao de 'time' para 'time_elapsed'."""
        ingestion = DataIngestion()

        df = pd.DataFrame({"Time": [0, 100, 200], "Amount": [100, 200, 150]})

        result = ingestion._standardize_columns(df)

        assert "time_elapsed" in result.columns
        assert "time" not in result.columns

    def test_standardize_columns_adds_transaction_id(self):
        """Testa adicao de transaction_id."""
        ingestion = DataIngestion()

        df = pd.DataFrame({"Amount": [100, 200, 150]})

        result = ingestion._standardize_columns(df)

        assert "transaction_id" in result.columns
        assert all(result["transaction_id"].str.startswith("tx_"))

    def test_get_stats(self):
        """Testa calculo de estatisticas."""
        ingestion = DataIngestion()

        df = pd.DataFrame(
            {
                "is_fraud": [0, 0, 1, 0, 0],
                "amount": [100, 200, 150, 300, 250],
                "transaction_id": ["tx_1", "tx_2", "tx_3", "tx_4", "tx_5"],
            }
        )

        stats = ingestion.get_stats(df)

        assert stats["total_transactions"] == 5
        assert stats["total_fraud"] == 1
        assert stats["fraud_percentage"] == 20.0
        assert stats["avg_amount"] == 200.0
        assert stats["max_amount"] == 300.0

    def test_save_raw_creates_parquet(self):
        """Testa salvamento em parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ingestion = DataIngestion(raw_data_path=tmpdir)

            df = pd.DataFrame(
                {
                    "is_fraud": [0, 1],
                    "amount": [100, 200],
                    "transaction_id": ["tx_1", "tx_2"],
                }
            )

            output_path = ingestion.save_raw(df, "test_data")

            assert Path(output_path).exists()
            loaded = pd.read_parquet(output_path)
            assert len(loaded) == 2

    def test_simulate_streaming_data_creates_new_ids(self):
        """Testa que dados simulados tem novos IDs."""
        ingestion = DataIngestion()

        base_df = pd.DataFrame(
            {
                "is_fraud": [0, 0, 1, 0, 0] * 100,
                "amount": [100, 200, 150, 300, 250] * 100,
                "transaction_id": [f"tx_{i}" for i in range(500)],
            }
        )

        simulated = ingestion.simulate_streaming_data(base_df, n_samples=50)

        assert len(simulated) == 50
        # IDs devem ser diferentes
        assert not set(simulated["transaction_id"]).intersection(
            set(base_df["transaction_id"])
        )


class TestDataIngestionEdgeCases:
    """Testes de casos extremos."""

    def test_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        ingestion = DataIngestion()

        df = pd.DataFrame({"Class": [], "Amount": []})
        result = ingestion._standardize_columns(df)

        assert "is_fraud" in result.columns
        assert len(result) == 0

    def test_columns_lowercase(self):
        """Testa que colunas sao convertidas para lowercase."""
        ingestion = DataIngestion()

        df = pd.DataFrame({"AMOUNT": [100], "CLASS": [0]})
        result = ingestion._standardize_columns(df)

        # Todas as colunas devem ser lowercase
        assert all(col == col.lower() for col in result.columns)
