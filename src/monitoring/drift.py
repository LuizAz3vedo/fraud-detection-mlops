"""
Modulo de monitoramento de drift com Evidently.
Detecta data drift e model drift em producao.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.features.store import FeatureStore


class DriftMonitor:
    """Monitor de drift usando testes estatisticos."""

    def __init__(self, reports_path: str = "reports"):
        """
        Inicializa o monitor.

        Args:
            reports_path: Caminho para salvar relatorios
        """
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.feature_store = FeatureStore()

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Retorna colunas de features numericas."""
        exclude_cols = [
            "transaction_id",
            "is_fraud",
            "created_at",
            "ingested_at",
            "time_elapsed",
        ]
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        return [col for col in numeric_cols if col not in exclude_cols]

    def get_reference_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Retorna dados de referencia para comparacao.

        Args:
            sample_size: Tamanho da amostra

        Returns:
            DataFrame com dados de referencia
        """
        return self.feature_store.get_reference_data(sample_size)

    def _ks_test(self, ref_col: pd.Series, curr_col: pd.Series) -> tuple:
        """
        Realiza teste Kolmogorov-Smirnov para detectar drift.

        Args:
            ref_col: Coluna de referencia
            curr_col: Coluna atual

        Returns:
            Tupla (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(ref_col.dropna(), curr_col.dropna())
        return statistic, p_value

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Detecta drift entre dados atuais e de referencia.

        Usa teste Kolmogorov-Smirnov para cada feature.

        Args:
            current_data: Dados atuais (producao)
            reference_data: Dados de referencia (treino)
            threshold: Threshold de p-value para deteccao de drift

        Returns:
            Dicionario com resultados do drift
        """
        if reference_data is None:
            reference_data = self.get_reference_data()

        print("=" * 50)
        print("DRIFT DETECTION")
        print("=" * 50)

        # Identificar colunas de features
        feature_columns = self._get_feature_columns(reference_data)

        print(f"\nDados de referencia: {len(reference_data):,} registros")
        print(f"Dados atuais: {len(current_data):,} registros")
        print(f"Features analisadas: {len(feature_columns)}")

        # Testar drift em cada coluna
        drift_results = {}
        drifted_columns = []

        for col in feature_columns:
            if col in current_data.columns:
                stat, p_value = self._ks_test(reference_data[col], current_data[col])
                drift_detected = p_value < threshold

                drift_results[col] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": bool(drift_detected),
                }

                if drift_detected:
                    drifted_columns.append(col)

        n_drifted = len(drifted_columns)
        n_columns = len(feature_columns)
        drift_share = n_drifted / n_columns if n_columns > 0 else 0
        dataset_drift = drift_share > 0.5  # Drift se mais de 50% das colunas

        print("\n--- Resultados ---")
        print(f"Drift detectado: {'SIM' if dataset_drift else 'NAO'}")
        print(f"Colunas com drift: {n_drifted}/{n_columns} ({100*drift_share:.1f}%)")

        if drifted_columns:
            print(f"\nColunas com drift (p < {threshold}):")
            for col in drifted_columns[:10]:  # Top 10
                info = drift_results[col]
                print(f"  - {col}: p-value={info['p_value']:.4f}, stat={info['statistic']:.4f}")

        # Salvar resultado JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_path / f"drift_result_{timestamp}.json"
        result = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": bool(dataset_drift),
            "drift_share": float(drift_share),
            "n_drifted_columns": n_drifted,
            "n_total_columns": n_columns,
            "drifted_columns": drifted_columns,
            "reference_size": len(reference_data),
            "current_size": len(current_data),
            "threshold": threshold,
            "column_results": drift_results,
        }

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[OK] Resultado salvo: {json_path}")

        # Gerar report HTML simples
        html_path = self._generate_html_report(result, timestamp)
        print(f"[OK] Relatorio HTML: {html_path}")

        print("=" * 50)
        print("[OK] Analise de drift concluida!")
        print("=" * 50)

        return result

    def _generate_html_report(self, result: Dict, timestamp: str) -> str:
        """Gera relatorio HTML simples."""
        html_path = self.reports_path / f"drift_report_{timestamp}.html"

        drift_status = "DRIFT DETECTADO" if result["drift_detected"] else "SEM DRIFT"
        status_color = "#dc3545" if result["drift_detected"] else "#28a745"

        rows = ""
        for col, info in result.get("column_results", {}).items():
            color = "#dc3545" if info["drift_detected"] else "#28a745"
            rows += f"""
            <tr>
                <td>{col}</td>
                <td>{info['p_value']:.4f}</td>
                <td>{info['statistic']:.4f}</td>
                <td style="color: {color}">{"Sim" if info['drift_detected'] else "Nao"}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .status {{ padding: 20px; border-radius: 5px; color: white; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Drift Detection Report</h1>
            <p>Timestamp: {result['timestamp']}</p>

            <div class="status" style="background-color: {status_color}">
                <h2>{drift_status}</h2>
                <p>Colunas com drift: {result['n_drifted_columns']}/{result['n_total_columns']} ({100*result['drift_share']:.1f}%)</p>
            </div>

            <h3>Resumo</h3>
            <ul>
                <li>Dados de referencia: {result['reference_size']:,} registros</li>
                <li>Dados atuais: {result['current_size']:,} registros</li>
                <li>Threshold: {result['threshold']}</li>
            </ul>

            <h3>Resultados por Coluna</h3>
            <table>
                <tr>
                    <th>Coluna</th>
                    <th>P-Value</th>
                    <th>KS Statistic</th>
                    <th>Drift</th>
                </tr>
                {rows}
            </table>
        </body>
        </html>
        """

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(html_path)

    def simulate_production_drift(
        self,
        drift_factor: float = 2.0,
        sample_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Simula dados de producao com drift para testes.

        Args:
            drift_factor: Fator de drift (multiplicador)
            sample_size: Tamanho da amostra

        Returns:
            DataFrame com dados simulados
        """
        reference = self.get_reference_data(sample_size)

        # Simular drift em algumas colunas
        production = reference.copy()

        # Adicionar drift no amount
        if "amount" in production.columns:
            production["amount"] = production["amount"] * drift_factor + np.random.normal(
                0, 50, len(production)
            )

        # Adicionar drift em algumas features V
        drift_cols = ["v1", "v2", "v3", "v14", "v17"]
        for col in drift_cols:
            if col in production.columns:
                production[col] = production[col] + np.random.normal(
                    drift_factor, 0.5, len(production)
                )

        return production


def main():
    """Funcao principal de monitoramento."""
    monitor = DriftMonitor()

    print("Simulando dados de producao com drift...")
    production_data = monitor.simulate_production_drift(drift_factor=1.5)

    print("\nDetectando drift...")
    result = monitor.detect_drift(production_data)

    if result["drift_detected"]:
        print("\n[ALERTA] Drift significativo detectado!")
        print("Recomendacao: Retreinar o modelo com dados mais recentes.")
    else:
        print("\n[OK] Dados estaveis, sem drift significativo.")


if __name__ == "__main__":
    main()
