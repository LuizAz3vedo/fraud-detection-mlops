"""
Feature Engineering com PySpark.
Utiliza Window Functions para criar features temporais.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path


class FeatureEngineer:
    """Classe para feature engineering com PySpark."""

    def __init__(self):
        """Inicializa SparkSession."""
        self.spark = (
            SparkSession.builder.appName("FraudFeatureEngineering")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )

        self.spark.sparkContext.setLogLevel("WARN")

    def load_data(self, path: str):
        """
        Carrega dados do CSV ou Parquet.

        Args:
            path: Caminho para o arquivo de dados

        Returns:
            Spark DataFrame com dados carregados
        """
        path_obj = Path(path)

        if path_obj.suffix == ".csv":
            df = self.spark.read.csv(path, header=True, inferSchema=True)
        elif path_obj.suffix == ".parquet":
            df = self.spark.read.parquet(path)
        else:
            raise ValueError(f"Formato nao suportado: {path_obj.suffix}")

        # Padronizar nomes de colunas (lowercase)
        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower())

        # Renomear colunas conhecidas
        if "class" in df.columns:
            df = df.withColumnRenamed("class", "is_fraud")
        if "time" in df.columns:
            df = df.withColumnRenamed("time", "time_elapsed")

        # Adicionar ID unico se nao existir
        if "transaction_id" not in df.columns:
            df = df.withColumn(
                "transaction_id",
                F.concat(F.lit("tx_"), F.monotonically_increasing_id()),
            )

        print(f"[OK] Dados carregados: {df.count()} registros")
        return df

    def create_features(self, df):
        """
        Cria features usando Window Functions.

        Features criadas:
        - avg_amount_last_100: Media do valor nas ultimas 100 transacoes
        - std_amount_last_100: Desvio padrao do valor nas ultimas 100 transacoes
        - tx_count_last_50: Contagem de transacoes nas ultimas 50
        - max_amount_last_100: Valor maximo nas ultimas 100 transacoes
        - min_amount_last_100: Valor minimo nas ultimas 100 transacoes
        - amount_deviation: Desvio do valor em relacao a media de 1000 transacoes
        - amount_percentile: Percentil do valor da transacao
        - amount_change: Mudanca em relacao a transacao anterior
        - amount_ratio: Razao em relacao a transacao anterior
        - amount_zscore: Z-score do valor

        Args:
            df: Spark DataFrame com dados

        Returns:
            Spark DataFrame com features adicionadas
        """
        print("Criando features com Window Functions...")

        # Definir janelas temporais
        window_50 = Window.orderBy("time_elapsed").rowsBetween(-50, 0)
        window_100 = Window.orderBy("time_elapsed").rowsBetween(-100, 0)
        window_1000 = Window.orderBy("time_elapsed").rowsBetween(-1000, 0)
        window_all = Window.orderBy("amount")
        window_lag = Window.orderBy("time_elapsed")

        # Features de agregacao temporal (janela de 100)
        df = df.withColumn("avg_amount_last_100", F.avg("amount").over(window_100))

        df = df.withColumn("std_amount_last_100", F.stddev("amount").over(window_100))

        df = df.withColumn("max_amount_last_100", F.max("amount").over(window_100))

        df = df.withColumn("min_amount_last_100", F.min("amount").over(window_100))

        # Features de agregacao temporal (janela de 50)
        df = df.withColumn("tx_count_last_50", F.count("*").over(window_50))

        # Desvio do valor medio (janela de 1000)
        df = df.withColumn(
            "amount_deviation", F.col("amount") - F.avg("amount").over(window_1000)
        )

        # Percentil do valor (global)
        df = df.withColumn("amount_percentile", F.percent_rank().over(window_all))

        # Features de mudanca (LAG)
        df = df.withColumn("prev_amount", F.lag("amount", 1).over(window_lag))

        df = df.withColumn("amount_change", F.col("amount") - F.col("prev_amount"))

        df = df.withColumn(
            "amount_ratio",
            F.when(F.col("prev_amount") != 0, F.col("amount") / F.col("prev_amount"))
            .otherwise(1.0),
        )

        # Z-score
        df = df.withColumn(
            "amount_zscore",
            F.when(
                F.col("std_amount_last_100") > 0,
                (F.col("amount") - F.col("avg_amount_last_100"))
                / F.col("std_amount_last_100"),
            ).otherwise(0.0),
        )

        # Remover coluna temporaria
        df = df.drop("prev_amount")

        # Preencher nulls com 0
        feature_cols = [
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

        for col in feature_cols:
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

        print(f"[OK] {len(feature_cols)} features criadas")
        return df

    def save_features(self, df, output_path: str):
        """
        Salva features em formato Parquet.

        Usa pandas para salvar (mais confiavel no Windows).

        Args:
            df: Spark DataFrame com features
            output_path: Caminho para salvar
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Converter para pandas e salvar (evita problemas com Hadoop no Windows)
        pdf = df.toPandas()
        parquet_file = Path(output_path) / "features.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(parquet_file, index=False)
        print(f"[OK] Features salvas em: {parquet_file}")

    def get_feature_stats(self, df) -> dict:
        """
        Retorna estatisticas das features criadas.

        Args:
            df: Spark DataFrame com features

        Returns:
            Dicionario com estatisticas
        """
        feature_cols = [
            "avg_amount_last_100",
            "std_amount_last_100",
            "amount_zscore",
            "amount_percentile",
        ]

        stats = {}
        for col in feature_cols:
            if col in df.columns:
                col_stats = df.select(
                    F.mean(col).alias("mean"),
                    F.stddev(col).alias("std"),
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                ).collect()[0]

                stats[col] = {
                    "mean": float(col_stats["mean"]) if col_stats["mean"] else 0,
                    "std": float(col_stats["std"]) if col_stats["std"] else 0,
                    "min": float(col_stats["min"]) if col_stats["min"] else 0,
                    "max": float(col_stats["max"]) if col_stats["max"] else 0,
                }

        return stats

    def run(self, input_path: str, output_path: str):
        """
        Pipeline completo de feature engineering.

        Args:
            input_path: Caminho dos dados de entrada
            output_path: Caminho para salvar features

        Returns:
            Spark DataFrame com features
        """
        print("=" * 50)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 50)

        # Carregar dados
        df = self.load_data(input_path)
        print(f"Total de registros: {df.count()}")

        # Criar features
        df_features = self.create_features(df)

        # Estatisticas
        print("\nEstatisticas das features:")
        stats = self.get_feature_stats(df_features)
        for feature, values in stats.items():
            print(f"  {feature}: mean={values['mean']:.2f}, std={values['std']:.2f}")

        # Salvar
        self.save_features(df_features, output_path)

        print("=" * 50)
        print("[OK] Feature engineering concluido!")
        print("=" * 50)

        return df_features

    def stop(self):
        """Para a SparkSession."""
        self.spark.stop()


def main():
    """Funcao principal."""
    engineer = FeatureEngineer()

    try:
        # Verificar se existe dados raw
        csv_path = Path("data/raw/creditcard.csv")
        parquet_path = Path("data/raw/transactions.parquet")

        if parquet_path.exists():
            input_path = str(parquet_path)
        elif csv_path.exists():
            input_path = str(csv_path)
        else:
            print("Dados nao encontrados!")
            print("Execute primeiro: make ingest")
            print("Ou baixe o dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            return

        output_path = "data/processed/features"

        engineer.run(input_path, output_path)

    finally:
        engineer.stop()


if __name__ == "__main__":
    main()
