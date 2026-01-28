"""
Feature Store baseado em arquivos Parquet.
Gerencia acesso aos dados processados para treino e inferencia.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


class FeatureStore:
    """Feature Store simples baseado em Parquet."""

    def __init__(self, features_path: str = "data/processed/features"):
        """
        Inicializa o Feature Store.

        Args:
            features_path: Caminho para o diretorio de features
        """
        self.features_path = Path(features_path)

    def _load_features(self) -> pd.DataFrame:
        """Carrega features do Parquet."""
        # Verificar arquivo direto ou dentro de diretorio
        if self.features_path.is_file():
            parquet_path = self.features_path
        else:
            parquet_path = self.features_path / "features.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Features nao encontradas em: {parquet_path}\n"
                "Execute primeiro: make features"
            )

        return pd.read_parquet(parquet_path)

    def get_training_data(
        self, sample_frac: float = 1.0, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Retorna dados para treino com estratificacao.

        Args:
            sample_frac: Fracao dos dados a amostrar (0-1)
            random_state: Seed para reproducibilidade

        Returns:
            DataFrame com dados de treino
        """
        df = self._load_features()

        # Estratificacao: manter proporcao de fraudes
        if sample_frac < 1.0:
            fraud = df[df["is_fraud"] == 1]
            non_fraud = df[df["is_fraud"] == 0].sample(
                frac=sample_frac, random_state=random_state
            )

            # Manter todas as fraudes (classe minoritaria)
            df = pd.concat([fraud, non_fraud])

        return df.sample(frac=1, random_state=random_state)  # Shuffle

    def get_balanced_data(
        self, ratio: int = 10, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Retorna dados balanceados (1:ratio).

        Args:
            ratio: Proporcao de nao-fraude para cada fraude
            random_state: Seed para reproducibilidade

        Returns:
            DataFrame balanceado
        """
        df = self._load_features()

        fraud = df[df["is_fraud"] == 1]
        n_fraud = len(fraud)

        non_fraud = df[df["is_fraud"] == 0].sample(
            n=min(n_fraud * ratio, len(df[df["is_fraud"] == 0])),
            random_state=random_state,
        )

        return pd.concat([fraud, non_fraud]).sample(frac=1, random_state=random_state)

    def get_inference_data(self, transaction_ids: List[str]) -> pd.DataFrame:
        """
        Retorna dados para inferencia por ID de transacao.

        Args:
            transaction_ids: Lista de IDs de transacao

        Returns:
            DataFrame com features das transacoes
        """
        df = self._load_features()
        return df[df["transaction_id"].isin(transaction_ids)]

    def get_feature_columns(self) -> List[str]:
        """
        Retorna lista de colunas de features (exclui ID, target, etc).

        Returns:
            Lista de nomes de colunas de features
        """
        df = self._load_features()

        exclude_cols = ["transaction_id", "is_fraud", "created_at", "ingested_at"]
        return [col for col in df.columns if col not in exclude_cols]

    def get_feature_stats(self) -> pd.DataFrame:
        """
        Retorna estatisticas das features.

        Returns:
            DataFrame com estatisticas descritivas
        """
        df = self._load_features()

        stats = df.describe().T
        stats["null_count"] = df.isnull().sum()
        stats["null_pct"] = (stats["null_count"] / len(df) * 100).round(2)

        return stats

    def get_class_distribution(self) -> dict:
        """
        Retorna distribuicao das classes.

        Returns:
            Dicionario com contagens e percentuais
        """
        df = self._load_features()

        total = len(df)
        fraud_count = int(df["is_fraud"].sum())
        non_fraud_count = total - fraud_count

        return {
            "total": total,
            "fraud": fraud_count,
            "non_fraud": non_fraud_count,
            "fraud_pct": round(100 * fraud_count / total, 4),
            "imbalance_ratio": round(non_fraud_count / fraud_count, 1)
            if fraud_count > 0
            else 0,
        }

    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple:
        """
        Divide dados em treino, validacao e teste com estratificacao.

        Args:
            test_size: Fracao para teste
            val_size: Fracao para validacao (do restante)
            random_state: Seed para reproducibilidade

        Returns:
            Tupla (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        df = self._load_features()

        # Primeiro split: treino+val vs teste
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["is_fraud"],
            random_state=random_state,
        )

        # Segundo split: treino vs validacao
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            stratify=train_val["is_fraud"],
            random_state=random_state,
        )

        return train, val, test

    def get_reference_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Retorna dados de referencia para monitoramento de drift.

        Args:
            sample_size: Tamanho da amostra

        Returns:
            DataFrame com dados de referencia
        """
        df = self._load_features()

        if len(df) <= sample_size:
            return df

        # Amostragem estratificada
        fraud = df[df["is_fraud"] == 1]
        non_fraud = df[df["is_fraud"] == 0]

        fraud_sample_size = min(len(fraud), int(sample_size * 0.2))
        non_fraud_sample_size = sample_size - fraud_sample_size

        fraud_sample = fraud.sample(n=fraud_sample_size, random_state=42)
        non_fraud_sample = non_fraud.sample(n=non_fraud_sample_size, random_state=42)

        return pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1)


def main():
    """Funcao principal para demonstrar uso do Feature Store."""
    store = FeatureStore()

    try:
        print("=" * 50)
        print("FEATURE STORE")
        print("=" * 50)

        # Distribuicao das classes
        print("\n--- Distribuicao das Classes ---")
        dist = store.get_class_distribution()
        print(f"Total de registros: {dist['total']:,}")
        print(f"Fraudes: {dist['fraud']:,} ({dist['fraud_pct']:.3f}%)")
        print(f"Nao-fraudes: {dist['non_fraud']:,}")
        print(f"Razao de desbalanceamento: 1:{dist['imbalance_ratio']}")

        # Dados de treino
        print("\n--- Dados de Treino ---")
        train_df = store.get_training_data()
        print(f"Total: {len(train_df):,} registros")
        print(f"Features: {len(store.get_feature_columns())} colunas")

        # Dados balanceados
        print("\n--- Dados Balanceados (1:10) ---")
        balanced_df = store.get_balanced_data(ratio=10)
        print(f"Total: {len(balanced_df):,} registros")
        balanced_fraud_pct = 100 * balanced_df["is_fraud"].mean()
        print(f"Fraudes: {balanced_fraud_pct:.1f}%")

        # Estatisticas
        print("\n--- Estatisticas das Features (primeiras 5) ---")
        stats = store.get_feature_stats()
        print(stats.head())

        print("\n" + "=" * 50)
        print("[OK] Feature Store funcionando!")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"[ERRO] {e}")


if __name__ == "__main__":
    main()
