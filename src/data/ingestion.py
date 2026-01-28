"""
Modulo de ingestao de dados.
Suporta CSV (Kaggle) e simulacao de APIs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class DataIngestion:
    """Classe para ingestao de dados de multiplas fontes"""

    def __init__(self, raw_data_path: str = "data/raw"):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Carrega dados do CSV do Kaggle"""

        print(f"Lendo CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Padronizar colunas
        df = self._standardize_columns(df)

        print(f"[OK] {len(df)} registros carregados do CSV")
        return df

    def load_from_api(
        self, api_url: str, params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Carrega dados de uma API REST.
        Exemplo com API publica de transacoes (simulacao).
        """
        print(f"Consultando API: {api_url}")

        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Assumindo que a resposta e uma lista de transacoes
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "transactions" in data:
                df = pd.DataFrame(data["transactions"])
            else:
                raise ValueError("Formato de resposta nao suportado")

            df = self._standardize_columns(df)
            print(f"[OK] {len(df)} registros carregados da API")
            return df

        except requests.RequestException as e:
            print(f"Erro na API: {e}")
            raise

    def simulate_streaming_data(
        self, base_df: pd.DataFrame, n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Simula dados de streaming (novas transacoes).
        Util para testar monitoramento de drift.
        """
        print(f"Simulando {n_samples} novas transacoes...")

        # Amostrar do dataset base
        sample = base_df.sample(n=n_samples, replace=True).copy()

        # Adicionar ruido para simular variacao
        numeric_cols = sample.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if col not in ["is_fraud"]:
                noise = sample[col].std() * 0.1
                sample[col] += np.random.normal(0, noise, len(sample))

        # Novos IDs e timestamps
        sample["transaction_id"] = [
            f"tx_{uuid.uuid4().hex[:12]}" for _ in range(len(sample))
        ]
        sample["ingested_at"] = datetime.now().isoformat()

        print(f"[OK] {len(sample)} transacoes simuladas")
        return sample

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de colunas"""

        # Lowercase
        df.columns = df.columns.str.lower()

        # Renomear colunas conhecidas
        rename_map = {"class": "is_fraud", "time": "time_elapsed", "amt": "amount"}

        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        # Adicionar ID se nao existir
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [
                f"tx_{uuid.uuid4().hex[:12]}" for _ in range(len(df))
            ]

        return df

    def save_raw(self, df: pd.DataFrame, filename: str) -> str:
        """Salva dados brutos em Parquet"""

        output_path = self.raw_data_path / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"[OK] Dados salvos em: {output_path}")
        return str(output_path)

    def get_stats(self, df: pd.DataFrame) -> dict:
        """Retorna estatisticas basicas dos dados"""

        return {
            "total_transactions": len(df),
            "total_fraud": int(df["is_fraud"].sum()),
            "fraud_percentage": round(100.0 * df["is_fraud"].mean(), 3),
            "avg_amount": round(df["amount"].mean(), 2),
            "max_amount": round(df["amount"].max(), 2),
            "columns": list(df.columns),
        }


def main():
    """Funcao principal de ingestao"""

    ingestion = DataIngestion()

    csv_path = Path("data/raw/creditcard.csv")

    if csv_path.exists():
        # Carregar CSV
        df = ingestion.load_from_csv(str(csv_path))

        # Salvar em formato otimizado
        ingestion.save_raw(df, "transactions")

        # Estatisticas
        stats = ingestion.get_stats(df)
        print("\nEstatisticas:")
        for key, value in stats.items():
            if key != "columns":
                print(f"  {key}: {value}")

        # Simular dados de streaming
        new_data = ingestion.simulate_streaming_data(df, n_samples=1000)
        ingestion.save_raw(new_data, "transactions_new")

    else:
        print(f"Arquivo nao encontrado: {csv_path}")
        print("Baixe o dataset em: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\nInstrucoes:")
        print("1. Acesse o link acima e faca login no Kaggle")
        print("2. Clique em 'Download' para baixar o arquivo creditcard.csv")
        print("3. Coloque o arquivo em: data/raw/creditcard.csv")
        print("4. Execute novamente: make ingest")


if __name__ == "__main__":
    main()
