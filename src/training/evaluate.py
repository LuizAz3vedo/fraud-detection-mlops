"""
Modulo de avaliacao de modelos.
Gera relatorios detalhados de performance.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

from src.features.store import FeatureStore


class ModelEvaluator:
    """Avaliador de modelos de deteccao de fraudes."""

    def __init__(
        self,
        model_path: str = "models/fraud_model.joblib",
        feature_cols_path: str = "models/feature_columns.joblib",
    ):
        """
        Inicializa o avaliador.

        Args:
            model_path: Caminho do modelo
            feature_cols_path: Caminho das colunas de features
        """
        self.model_path = Path(model_path)
        self.feature_cols_path = Path(feature_cols_path)
        self.model = None
        self.feature_columns = None
        self.feature_store = FeatureStore()

    def load_model(self):
        """Carrega o modelo e feature columns."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado: {self.model_path}\n"
                "Execute primeiro: make train"
            )

        self.model = joblib.load(self.model_path)

        if self.feature_cols_path.exists():
            self.feature_columns = joblib.load(self.feature_cols_path)
        else:
            # Inferir das features
            self.feature_columns = self.feature_store.get_feature_columns()

        print(f"[OK] Modelo carregado: {self.model_path}")

    def evaluate(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste.

        Args:
            test_size: Fracao para teste
            random_state: Seed para reproducibilidade

        Returns:
            Dicionario com resultados da avaliacao
        """
        if self.model is None:
            self.load_model()

        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # Carregar dados de teste
        _, _, test_df = self.feature_store.split_data(
            test_size=test_size, random_state=random_state
        )

        X_test = test_df[self.feature_columns]
        y_test = test_df["is_fraud"]

        print(f"\nDados de teste: {len(test_df):,} registros")
        print(f"Fraudes no teste: {y_test.sum():,} ({100*y_test.mean():.3f}%)")

        # Predicoes
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Classification report
        print("\n--- Classification Report ---")
        report = classification_report(
            y_test, y_pred, target_names=["Normal", "Fraude"], output_dict=True
        )
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

        # Metricas de ranking
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC:  {pr_auc:.4f}")

        # Matriz de confusao
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("\n--- Matriz de Confusao ---")
        print("                 Pred Normal  Pred Fraude")
        print(f"Real Normal      {tn:>10,}  {fp:>10,}")
        print(f"Real Fraude      {fn:>10,}  {tp:>10,}")

        # Analise de thresholds
        print("\n--- Analise de Thresholds ---")
        thresholds_analysis = self._analyze_thresholds(y_test, y_prob)

        for t in [0.3, 0.5, 0.7]:
            if t in thresholds_analysis:
                ta = thresholds_analysis[t]
                print(
                    f"Threshold {t:.1f}: "
                    f"Precision={ta['precision']:.3f}, "
                    f"Recall={ta['recall']:.3f}, "
                    f"F1={ta['f1']:.3f}"
                )

        # Curvas para plots
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_prob
        )

        print("\n" + "=" * 50)
        print("[OK] Avaliacao concluida!")
        print("=" * 50)

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "thresholds_analysis": thresholds_analysis,
            "curves": {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                "pr": {
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist(),
                },
            },
        }

    def _analyze_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[float, Dict[str, float]]:
        """
        Analisa performance em diferentes thresholds.

        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades

        Returns:
            Dicionario com metricas por threshold
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        results = {}

        for threshold in np.arange(0.1, 1.0, 0.1):
            y_pred = (y_prob >= threshold).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            results[round(threshold, 1)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return results

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
    ) -> float:
        """
        Encontra threshold otimo para uma metrica.

        Args:
            y_true: Labels verdadeiros
            y_prob: Probabilidades
            metric: Metrica a otimizar (f1, precision, recall)

        Returns:
            Threshold otimo
        """
        analysis = self._analyze_thresholds(y_true, y_prob)

        best_threshold = 0.5
        best_value = 0

        for threshold, metrics in analysis.items():
            if metrics[metric] > best_value:
                best_value = metrics[metric]
                best_threshold = threshold

        return best_threshold

    def generate_report(self, output_path: str = "reports/evaluation_report.json"):
        """
        Gera relatorio de avaliacao em JSON.

        Args:
            output_path: Caminho para salvar o relatorio
        """
        import json

        results = self.evaluate()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Converter arrays numpy para listas
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        results_serializable = convert_to_serializable(results)

        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        print(f"[OK] Relatorio salvo em: {output_path}")


def main():
    """Funcao principal de avaliacao."""
    evaluator = ModelEvaluator()
    evaluator.evaluate()
    evaluator.generate_report()


if __name__ == "__main__":
    main()
