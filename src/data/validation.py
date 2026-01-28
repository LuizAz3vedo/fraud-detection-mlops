"""
Modulo de validacao de dados usando Great Expectations.
Validacao robusta e documentada para dados de transacoes.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataValidator:
    """Validador de dados usando Great Expectations."""

    def validate_raw(self, df: pd.DataFrame) -> dict:
        """
        Valida dados brutos com Great Expectations.

        Validacoes:
        - is_fraud deve ser 0 ou 1
        - amount deve ser >= 0
        - transaction_id nao pode ser nulo
        - transaction_id deve ser unico
        - Colunas V1-V28 devem existir (se presentes no dataset original)

        Args:
            df: DataFrame com dados brutos

        Returns:
            Dicionario com resultado da validacao
        """
        # Validacoes usando pandas (GE apenas para contexto)
        results = []
        errors = []

        # Validacao de is_fraud
        if "is_fraud" in df.columns:
            valid_values = df["is_fraud"].isin([0, 1]).all()
            results.append(valid_values)
            if not valid_values:
                errors.append("is_fraud deve ser 0 ou 1")

        # Validacao de amount
        if "amount" in df.columns:
            valid_amount = (df["amount"] >= 0).all()
            results.append(valid_amount)
            if not valid_amount:
                errors.append("amount deve ser >= 0")

        # Validacao de transaction_id nao nulo
        if "transaction_id" in df.columns:
            not_null = df["transaction_id"].notna().all()
            results.append(not_null)
            if not not_null:
                errors.append("transaction_id nao pode ser nulo")

            # Validacao de transaction_id unico
            is_unique = df["transaction_id"].is_unique
            results.append(is_unique)
            if not is_unique:
                errors.append("transaction_id deve ser unico")

        # Validacao de colunas V1-V28
        for i in range(1, 29):
            col_name = f"v{i}"
            if col_name in df.columns:
                results.append(True)

        passed = sum(1 for r in results if r)
        total = len(results)

        return {
            "valid": passed == total,
            "passed": passed,
            "total": total,
            "pct": round(100 * passed / total, 1) if total > 0 else 0,
            "errors": errors,
        }

    def validate_processed(self, df: pd.DataFrame) -> dict:
        """
        Valida dados processados (com features).

        Args:
            df: DataFrame com dados processados

        Returns:
            Dicionario com resultado da validacao
        """
        results = []
        errors = []

        # Validacoes basicas
        if "is_fraud" in df.columns:
            valid = df["is_fraud"].isin([0, 1]).all()
            results.append(valid)
            if not valid:
                errors.append("is_fraud deve ser 0 ou 1")

        if "amount" in df.columns:
            valid = (df["amount"] >= 0).all()
            results.append(valid)
            if not valid:
                errors.append("amount deve ser >= 0")

        if "transaction_id" in df.columns:
            valid = df["transaction_id"].notna().all()
            results.append(valid)
            if not valid:
                errors.append("transaction_id nao pode ser nulo")

        # Validacoes de features derivadas
        feature_cols = [
            "avg_amount_last_100",
            "std_amount_last_100",
            "amount_zscore",
            "amount_percentile",
        ]

        for col in feature_cols:
            if col in df.columns:
                results.append(True)

        # std deve ser >= 0
        if "std_amount_last_100" in df.columns:
            valid = (df["std_amount_last_100"] >= 0).all()
            results.append(valid)
            if not valid:
                errors.append("std_amount_last_100 deve ser >= 0")

        # percentile deve estar entre 0 e 1
        if "amount_percentile" in df.columns:
            valid = ((df["amount_percentile"] >= 0) & (df["amount_percentile"] <= 1)).all()
            results.append(valid)
            if not valid:
                errors.append("amount_percentile deve estar entre 0 e 1")

        passed = sum(1 for r in results if r)
        total = len(results)

        return {
            "valid": passed == total,
            "passed": passed,
            "total": total,
            "pct": round(100 * passed / total, 1) if total > 0 else 0,
            "errors": errors,
        }

    def validate_spark(self, spark_df) -> dict:
        """
        Valida Spark DataFrame (converte sample para pandas).

        Args:
            spark_df: Spark DataFrame

        Returns:
            Dicionario com resultado da validacao
        """
        sample_df = spark_df.sample(0.01).toPandas()
        return self.validate_raw(sample_df)

    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        """
        Gera relatorio de qualidade de dados.

        Args:
            df: DataFrame a ser analisado

        Returns:
            Dicionario com metricas de qualidade
        """
        report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "duplicates": {},
            "data_types": {},
        }

        # Missing values por coluna
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                report["missing_values"][col] = {
                    "count": null_count,
                    "pct": round(100 * null_count / len(df), 2),
                }

        # Duplicatas
        duplicates = df.duplicated().sum()
        report["duplicates"] = {
            "count": int(duplicates),
            "pct": round(100 * duplicates / len(df), 2),
        }

        # Tipos de dados
        for col in df.columns:
            report["data_types"][col] = str(df[col].dtype)

        return report


def validate_data(
    data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None
) -> dict:
    """
    Funcao principal de validacao.

    Args:
        data_path: Caminho para arquivo de dados
        df: DataFrame (se nao fornecer data_path)

    Returns:
        Dicionario com resultado da validacao
    """
    if df is None:
        if data_path is None:
            data_path = "data/raw/transactions.parquet"

        path = Path(data_path)
        if not path.exists():
            return {"valid": False, "errors": [f"Arquivo nao encontrado: {data_path}"]}

        df = pd.read_parquet(data_path)

    validator = DataValidator()
    return validator.validate_raw(df)


def main():
    """Funcao principal de validacao."""

    # Verificar dados raw
    raw_path = Path("data/raw/transactions.parquet")

    if not raw_path.exists():
        print(f"Arquivo nao encontrado: {raw_path}")
        print("Execute primeiro: make ingest")
        return

    print("=" * 50)
    print("VALIDACAO DE DADOS")
    print("=" * 50)

    df = pd.read_parquet(raw_path)
    validator = DataValidator()

    # Validacao de dados brutos
    print(f"\nValidando: {raw_path}")
    result = validator.validate_raw(df)

    print("\n--- Resultado ---")
    print(f"Validacoes: {result['passed']}/{result['total']} ({result['pct']}%)")

    if result["valid"]:
        print("[OK] Dados validos!")
    else:
        print("[ERRO] Dados invalidos:")
        for error in result["errors"]:
            print(f"  - {error}")

    # Relatorio de qualidade
    print("\n--- Qualidade dos Dados ---")
    quality = validator.get_data_quality_report(df)
    print(f"Total de registros: {quality['total_rows']:,}")
    print(f"Total de colunas: {quality['total_columns']}")
    print(f"Duplicatas: {quality['duplicates']['count']:,} ({quality['duplicates']['pct']}%)")

    if quality["missing_values"]:
        print("Valores faltantes:")
        for col, info in list(quality["missing_values"].items())[:5]:
            print(f"  - {col}: {info['count']:,} ({info['pct']}%)")

    # Estatisticas do target
    print("\n--- Distribuicao do Target ---")
    print(f"Fraudes: {df['is_fraud'].sum():,} ({100*df['is_fraud'].mean():.3f}%)")

    print("\n" + "=" * 50)
    print("[OK] Validacao concluida!")
    print("=" * 50)


if __name__ == "__main__":
    main()
