"""
Modulo de treinamento com MLflow tracking.
Treina modelo de deteccao de fraudes com XGBoost.
"""

import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.features.store import FeatureStore


class FraudModelTrainer:
    """Treinador de modelos de deteccao de fraudes."""

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "fraud-detection",
    ):
        """
        Inicializa o treinador.

        Args:
            mlflow_tracking_uri: URI do servidor MLflow (None = local)
            experiment_name: Nome do experimento no MLflow
        """
        self.experiment_name = experiment_name
        self.feature_store = FeatureStore()
        self.model = None
        self.feature_columns = None

        # Configurar MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Retorna colunas de features para treino.

        Args:
            df: DataFrame com todas as colunas

        Returns:
            Lista de nomes de colunas de features
        """
        exclude_cols = [
            "transaction_id",
            "is_fraud",
            "created_at",
            "ingested_at",
            "time_elapsed",
        ]
        return [col for col in df.columns if col not in exclude_cols]

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula metricas de avaliacao.

        Args:
            y_true: Labels verdadeiros
            y_pred: Predicoes
            y_prob: Probabilidades

        Returns:
            Dicionario com metricas
        """
        # Metricas basicas
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC-ROC e AUC-PR
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        # Matriz de confusao
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    def train(
        self,
        test_size: float = 0.2,
        use_balanced: bool = True,
        balance_ratio: int = 10,
        random_state: int = 42,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Treina o modelo com MLflow tracking.

        Args:
            test_size: Fracao para teste
            use_balanced: Usar dados balanceados
            balance_ratio: Razao de balanceamento (nao-fraude:fraude)
            random_state: Seed para reproducibilidade
            model_params: Parametros do XGBoost

        Returns:
            Dicionario com metricas e informacoes do treino
        """
        print("=" * 50)
        print("TRAINING PIPELINE")
        print("=" * 50)

        # Parametros padrao do XGBoost otimizados para dados desbalanceados
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 10,  # Peso para classe minoritaria
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": "aucpr",
        }

        if model_params:
            default_params.update(model_params)

        # Carregar dados
        print("\n--- Carregando dados ---")
        if use_balanced:
            df = self.feature_store.get_balanced_data(
                ratio=balance_ratio, random_state=random_state
            )
            print(f"Dados balanceados: {len(df):,} registros (ratio 1:{balance_ratio})")
        else:
            df = self.feature_store.get_training_data()
            print(f"Dados completos: {len(df):,} registros")

        # Preparar features
        self.feature_columns = self._get_feature_columns(df)
        X = df[self.feature_columns]
        y = df["is_fraud"]

        print(f"Features: {len(self.feature_columns)} colunas")
        print(f"Target: {y.sum():,} fraudes ({100*y.mean():.2f}%)")

        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        print(f"\nTreino: {len(X_train):,} | Teste: {len(X_test):,}")

        # Treinar com MLflow tracking
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parametros
            mlflow.log_params(default_params)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("use_balanced", use_balanced)
            mlflow.log_param("balance_ratio", balance_ratio)
            mlflow.log_param("n_features", len(self.feature_columns))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size_n", len(X_test))

            # Treinar modelo
            print("\n--- Treinando modelo ---")
            self.model = XGBClassifier(**default_params)
            self.model.fit(X_train, y_train)

            # Predicoes
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]

            # Calcular metricas
            metrics = self._calculate_metrics(y_test, y_pred, y_prob)

            # Log metricas
            mlflow.log_metrics(metrics)

            # Feature importance
            importance = pd.DataFrame({
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_
            }).sort_values("importance", ascending=False)

            # Log feature importance como artifact
            importance_path = Path("models") / "feature_importance.csv"
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))

            # Log modelo
            mlflow.sklearn.log_model(self.model, "model")

            # Salvar modelo localmente
            model_path = Path("models") / "fraud_model.joblib"
            joblib.dump(self.model, model_path)
            print(f"\n[OK] Modelo salvo em: {model_path}")

            # Salvar feature columns
            feature_cols_path = Path("models") / "feature_columns.joblib"
            joblib.dump(self.feature_columns, feature_cols_path)

            run_id = mlflow.active_run().info.run_id

        # Exibir resultados
        print("\n--- Resultados ---")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

        print("\nMatriz de Confusao:")
        print(f"  TP: {metrics['true_positives']:,} | FP: {metrics['false_positives']:,}")
        print(f"  FN: {metrics['false_negatives']:,} | TN: {metrics['true_negatives']:,}")

        print("\n--- Top 5 Features ---")
        for _, row in importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        print("\n" + "=" * 50)
        print(f"[OK] Treino concluido! Run ID: {run_id}")
        print("=" * 50)

        return {
            "metrics": metrics,
            "feature_importance": importance.to_dict("records"),
            "run_id": run_id,
            "model_path": str(model_path),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicoes.

        Args:
            X: DataFrame com features

        Returns:
            Array com predicoes
        """
        if self.model is None:
            raise ValueError("Modelo nao treinado. Execute train() primeiro.")
        return self.model.predict(X[self.feature_columns])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades.

        Args:
            X: DataFrame com features

        Returns:
            Array com probabilidades
        """
        if self.model is None:
            raise ValueError("Modelo nao treinado. Execute train() primeiro.")
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def load_model(self, model_path: str = "models/fraud_model.joblib"):
        """
        Carrega modelo salvo.

        Args:
            model_path: Caminho do modelo
        """
        self.model = joblib.load(model_path)
        feature_cols_path = Path(model_path).parent / "feature_columns.joblib"
        if feature_cols_path.exists():
            self.feature_columns = joblib.load(feature_cols_path)
        print(f"[OK] Modelo carregado de: {model_path}")


def main():
    """Funcao principal de treinamento."""
    trainer = FraudModelTrainer()

    # Treinar com dados balanceados
    results = trainer.train(
        use_balanced=True,
        balance_ratio=10,
        test_size=0.2,
    )

    print("\nMLflow UI: mlflow ui --port 5001")
    print(f"Run ID: {results['run_id']}")


if __name__ == "__main__":
    main()
