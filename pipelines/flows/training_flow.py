"""
Pipeline de treinamento orquestrado com Prefect.
Inclui validacao, feature engineering, treino e monitoramento.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from prefect import flow, task
from prefect.logging import get_run_logger

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@task(name="validate_data", retries=2, retry_delay_seconds=10)
def validate_data_task() -> Dict[str, Any]:
    """Task de validacao de dados."""
    logger = get_run_logger()
    logger.info("Iniciando validacao de dados...")

    from src.data.validation import DataValidator
    import pandas as pd

    validator = DataValidator()

    # Carregar e validar dados raw
    raw_path = Path("data/raw/transactions.parquet")
    if not raw_path.exists():
        raise FileNotFoundError(f"Dados nao encontrados: {raw_path}")

    df = pd.read_parquet(raw_path)
    result = validator.validate_raw(df)

    logger.info(f"Validacao: {result['passed']}/{result['total']} ({result['pct']}%)")

    if not result["valid"]:
        logger.warning(f"Erros de validacao: {result['errors']}")

    return result


@task(name="create_features", retries=1)
def create_features_task() -> str:
    """Task de feature engineering."""
    logger = get_run_logger()
    logger.info("Iniciando feature engineering...")

    from src.features.engineering import FeatureEngineer

    engineer = FeatureEngineer()

    try:
        # Verificar dados de entrada
        csv_path = Path("data/raw/creditcard.csv")
        parquet_path = Path("data/raw/transactions.parquet")

        if parquet_path.exists():
            input_path = str(parquet_path)
        elif csv_path.exists():
            input_path = str(csv_path)
        else:
            raise FileNotFoundError("Dados de entrada nao encontrados")

        output_path = "data/processed/features"

        # Executar feature engineering
        engineer.run(input_path, output_path)

        logger.info(f"Features salvas em: {output_path}")
        return output_path

    finally:
        engineer.stop()


@task(name="train_model", retries=1)
def train_model_task(
    use_balanced: bool = True,
    balance_ratio: int = 10,
) -> Dict[str, Any]:
    """Task de treinamento do modelo."""
    logger = get_run_logger()
    logger.info("Iniciando treinamento do modelo...")

    from src.training.train import FraudModelTrainer

    trainer = FraudModelTrainer()

    results = trainer.train(
        use_balanced=use_balanced,
        balance_ratio=balance_ratio,
        test_size=0.2,
    )

    logger.info(f"Modelo treinado - F1: {results['metrics']['f1_score']:.4f}")
    logger.info(f"MLflow Run ID: {results['run_id']}")

    return results


@task(name="detect_drift", retries=1)
def detect_drift_task() -> Dict[str, Any]:
    """Task de deteccao de drift."""
    logger = get_run_logger()
    logger.info("Verificando drift nos dados...")

    from src.monitoring.drift import DriftMonitor

    monitor = DriftMonitor()

    # Simular dados de producao para demonstracao
    production_data = monitor.simulate_production_drift(
        drift_factor=1.2, sample_size=5000
    )

    result = monitor.detect_drift(production_data)

    if result["drift_detected"]:
        logger.warning(
            f"DRIFT DETECTADO! {result['n_drifted_columns']} colunas afetadas"
        )
    else:
        logger.info("Dados estaveis, sem drift significativo")

    return result


@task(name="notify_completion")
def notify_completion_task(
    validation_result: Dict[str, Any],
    training_result: Dict[str, Any],
    drift_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Task de notificacao de conclusao."""
    logger = get_run_logger()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "validation": {
            "valid": validation_result["valid"],
            "passed": validation_result["passed"],
            "total": validation_result["total"],
        },
        "training": {
            "f1_score": training_result["metrics"]["f1_score"],
            "precision": training_result["metrics"]["precision"],
            "recall": training_result["metrics"]["recall"],
            "roc_auc": training_result["metrics"]["roc_auc"],
            "run_id": training_result["run_id"],
        },
        "drift": {
            "detected": drift_result["drift_detected"],
            "drifted_columns": drift_result["n_drifted_columns"],
        },
    }

    logger.info("=" * 50)
    logger.info("PIPELINE CONCLUIDO")
    logger.info("=" * 50)
    logger.info(f"Validacao: {'OK' if summary['validation']['valid'] else 'FALHOU'}")
    logger.info(f"F1-Score: {summary['training']['f1_score']:.4f}")
    logger.info(f"Drift: {'DETECTADO' if summary['drift']['detected'] else 'OK'}")

    return summary


@flow(name="fraud-detection-training-pipeline")
def training_pipeline(
    skip_features: bool = False,
    use_balanced: bool = True,
    balance_ratio: int = 10,
    check_drift: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline completo de treinamento de modelo de fraude.

    Args:
        skip_features: Pular feature engineering (usar features existentes)
        use_balanced: Usar dados balanceados no treino
        balance_ratio: Razao de balanceamento
        check_drift: Verificar drift nos dados

    Returns:
        Resumo da execucao do pipeline
    """
    logger = get_run_logger()
    logger.info("Iniciando pipeline de treinamento...")

    # 1. Validar dados
    validation_result = validate_data_task()

    if not validation_result["valid"]:
        logger.error("Validacao falhou! Abortando pipeline.")
        raise ValueError(f"Dados invalidos: {validation_result['errors']}")

    # 2. Feature Engineering (opcional)
    if not skip_features:
        features_path = create_features_task()
        logger.info(f"Features criadas em: {features_path}")

    # 3. Treinar modelo
    training_result = train_model_task(
        use_balanced=use_balanced,
        balance_ratio=balance_ratio,
    )

    # 4. Verificar drift (opcional)
    if check_drift:
        drift_result = detect_drift_task()
    else:
        drift_result = {"drift_detected": False, "n_drifted_columns": 0}

    # 5. Notificar conclusao
    summary = notify_completion_task(
        validation_result=validation_result,
        training_result=training_result,
        drift_result=drift_result,
    )

    return summary


def main():
    """Funcao principal para executar o pipeline."""
    print("=" * 50)
    print("PREFECT TRAINING PIPELINE")
    print("=" * 50)

    # Executar pipeline
    result = training_pipeline(
        skip_features=True,  # Usar features existentes
        use_balanced=True,
        balance_ratio=10,
        check_drift=True,
    )

    print("\n--- Resultado Final ---")
    print(f"Validacao: {'OK' if result['validation']['valid'] else 'FALHOU'}")
    print(f"F1-Score: {result['training']['f1_score']:.4f}")
    print(f"ROC-AUC: {result['training']['roc_auc']:.4f}")
    print(f"Drift: {'DETECTADO' if result['drift']['detected'] else 'OK'}")
    print(f"MLflow Run: {result['training']['run_id']}")


if __name__ == "__main__":
    main()
