# Fraud Detection MLOps Pipeline

![CI Pipeline](https://github.com/LuizAz3vedo/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Pipeline completo de Machine Learning para detecao de fraudes em transacoes financeiras, implementando as melhores praticas de MLOps.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRAUD DETECTION MLOPS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   [CSV Kaggle] → [Ingestao] → [Features PySpark] → [Treino XGBoost]         │
│                       ↓              ↓                    ↓                  │
│                  [Validacao]    [Feature Store]      [MLflow]               │
│                      GE            Parquet           Tracking                │
│                                                          ↓                   │
│                                                    [Modelo .joblib]          │
│                                                          ↓                   │
│   [Cliente] ←→ [API FastAPI] ←→ [Modelo] ←→ [Monitoramento Drift]          │
│                    :8000                          (scipy KS-test)            │
│                                                          ↓                   │
│                                                   [Relatorios HTML]          │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    PREFECT (Orquestracao)                           │   │
│   │  [validate] → [features] → [train] → [drift] → [notify]            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Componente | Tecnologia | Descricao |
|------------|------------|-----------|
| Processamento | **PySpark** | Feature engineering com Window Functions |
| Validacao | **Great Expectations** | Validacao de qualidade dos dados |
| Treinamento | **XGBoost** | Modelo de classificacao |
| Tracking | **MLflow** | Registro de experimentos e metricas |
| Serving | **FastAPI** | API REST para predicoes |
| Monitoramento | **Scipy (KS-test)** | Deteccao de data drift |
| Orquestracao | **Prefect** | Pipeline automatizado |
| Containerizacao | **Docker** | Infraestrutura containerizada |

## Resultados do Modelo

| Metrica | Valor |
|---------|-------|
| Precision | **94.62%** |
| Recall | **89.80%** |
| F1-Score | **92.15%** |
| ROC-AUC | **99.60%** |
| PR-AUC | **97.48%** |

**Top 5 Features mais importantes:**
1. v14 (32.37%)
2. v10 (14.74%)
3. v4 (3.92%)
4. v8 (2.64%)
5. v17 (2.38%)

## Quick Start

### 1. Setup

```bash
# Clonar repositorio
git clone <repo>
cd fraud-detection-mlops

# Criar ambiente virtual e instalar dependencias
make setup

# Ativar ambiente (Windows)
venv\Scripts\activate
```

### 2. Baixar Dataset

1. Acesse: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Baixe e extraia `creditcard.csv` para `data/raw/`

### 3. Executar Pipeline Completo

```bash
# Ingerir dados
make ingest

# Validar dados
make validate

# Criar features com PySpark
set HADOOP_HOME=<path>\hadoop
make features

# Treinar modelo
make train

# Rodar API
make api
```

### 4. Testar a API

```bash
# Em outro terminal, testar predicao
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 149.62, "v14": -2.5}'

# Ou usar o demo client
python demo/client.py
```

## Estrutura do Projeto

```
fraud-detection-mlops/
├── src/
│   ├── data/
│   │   ├── ingestion.py      # Carrega e padroniza dados
│   │   └── validation.py     # Valida com Great Expectations
│   ├── features/
│   │   ├── engineering.py    # Feature engineering com PySpark
│   │   └── store.py          # Feature Store baseado em Parquet
│   ├── training/
│   │   ├── train.py          # Treinamento com MLflow tracking
│   │   └── evaluate.py       # Avaliacao detalhada do modelo
│   ├── serving/
│   │   └── api.py            # API FastAPI para predicoes
│   └── monitoring/
│       └── drift.py          # Deteccao de drift com KS-test
├── pipelines/
│   └── flows/
│       └── training_flow.py  # Pipeline Prefect completo
├── demo/
│   └── client.py             # Cliente demo para testar API
├── docker/
│   ├── docker-compose.yml    # Infraestrutura Docker
│   └── Dockerfile.api        # Container da API
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_training.py
│   ├── test_api.py
│   └── test_monitoring.py
├── data/
│   ├── raw/                  # Dados brutos
│   └── processed/            # Features processadas
├── models/                   # Modelos salvos (.joblib)
├── reports/                  # Relatorios de drift (HTML/JSON)
├── Makefile                  # Comandos automatizados
└── requirements.txt          # Dependencias Python
```

## Comandos Disponiveis

```bash
# Setup
make setup          # Criar ambiente e instalar dependencias

# Data Pipeline
make ingest         # Ingerir dados do CSV
make validate       # Validar dados com Great Expectations
make features       # Criar features com PySpark

# ML Pipeline
make train          # Treinar modelo com MLflow
make evaluate       # Avaliar modelo

# Serving
make api            # Rodar API FastAPI (localhost:8000)

# Monitoramento
make drift          # Verificar drift nos dados

# Orquestracao
make prefect-run    # Executar pipeline completo com Prefect

# Testes
make test           # Rodar testes unitarios
make lint           # Verificar codigo com ruff

# Infraestrutura Docker
make infra-up       # Subir containers
make infra-down     # Parar containers
```

## API Endpoints

| Endpoint | Metodo | Descricao |
|----------|--------|-----------|
| `/` | GET | Informacoes da API |
| `/health` | GET | Health check |
| `/predict` | POST | Predicao individual |
| `/predict/batch` | POST | Predicao em batch (max 1000) |
| `/model/info` | GET | Informacoes do modelo |

### Exemplo de Request

```json
POST /predict
{
    "amount": 149.62,
    "v1": -1.359807,
    "v14": -0.5278
}
```

### Exemplo de Response

```json
{
    "transaction_id": "tx_20260127120000",
    "is_fraud": false,
    "fraud_probability": 0.12,
    "risk_level": "low",
    "timestamp": "2026-01-27T12:00:00"
}
```

## Features Criadas (PySpark)

| Feature | Descricao |
|---------|-----------|
| `avg_amount_last_100` | Media do valor nas ultimas 100 transacoes |
| `std_amount_last_100` | Desvio padrao nas ultimas 100 transacoes |
| `max_amount_last_100` | Valor maximo nas ultimas 100 transacoes |
| `min_amount_last_100` | Valor minimo nas ultimas 100 transacoes |
| `tx_count_last_50` | Contagem de transacoes nas ultimas 50 |
| `amount_deviation` | Desvio do valor em relacao a media de 1000 transacoes |
| `amount_percentile` | Percentil do valor da transacao |
| `amount_change` | Mudanca em relacao a transacao anterior |
| `amount_ratio` | Razao em relacao a transacao anterior |
| `amount_zscore` | Z-score do valor |

## Monitoramento de Drift

O sistema usa o teste Kolmogorov-Smirnov para detectar mudancas na distribuicao dos dados:

```bash
# Executar analise de drift
make drift
```

Output:
```
DRIFT DETECTION
==================================================
Dados de referencia: 10,000 registros
Dados atuais: 5,000 registros
Features analisadas: 39

--- Resultados ---
Drift detectado: NAO
Colunas com drift: 18/39 (46.2%)
```

Relatorios salvos em `reports/`:
- `drift_report_YYYYMMDD_HHMMSS.html` - Relatorio visual
- `drift_result_YYYYMMDD_HHMMSS.json` - Dados estruturados

## Pipeline Prefect

O pipeline orquestrado executa todas as etapas automaticamente:

```bash
make prefect-run
```

Etapas:
1. **validate_data** - Valida qualidade dos dados
2. **create_features** - Feature engineering (opcional)
3. **train_model** - Treina modelo com MLflow
4. **detect_drift** - Verifica drift nos dados
5. **notify_completion** - Gera resumo final

## Dataset

- **Fonte**: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Registros**: 284.807 transacoes
- **Fraudes**: 492 (0.172%)
- **Features**: 28 anonimizadas (PCA) + Time + Amount

## Requisitos

- Python 3.10+
- Java 11+ (para PySpark)
- Docker (opcional, para infraestrutura)
- Make (GnuWin32 no Windows)

## Status das Fases

- [x] Fase 1: Setup & Ingestao
- [x] Fase 2: Feature Engineering (PySpark)
- [x] Fase 3: Treinamento (MLflow)
- [x] Fase 4: API Serving (FastAPI)
- [x] Fase 5: Monitoramento (Drift Detection)
- [x] Fase 6: Orquestracao (Prefect)
- [x] Fase 7: CI/CD (GitHub Actions)

## Autor

Projeto de portfolio para demonstrar habilidades em MLOps e Data Science.

## Licenca

MIT
