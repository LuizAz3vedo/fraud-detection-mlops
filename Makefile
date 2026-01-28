# Makefile - Fraud Detection MLOps Pipeline
# Compativel com Windows (GnuWin32 Make)

.PHONY: setup infra-up infra-down ingest features train api test pipeline clean help

# ============================================
# HELP
# ============================================

help:
	@echo Comandos disponiveis:
	@echo.
	@echo   SETUP
	@echo     make setup        - Criar ambiente virtual e instalar dependencias
	@echo.
	@echo   DATA PIPELINE
	@echo     make ingest       - Ingerir dados do CSV
	@echo     make validate     - Validar dados
	@echo     make features     - Criar features com PySpark
	@echo.
	@echo   ML PIPELINE
	@echo     make train        - Treinar modelo
	@echo.
	@echo   SERVING
	@echo     make api          - Rodar API localmente
	@echo.
	@echo   TESTES
	@echo     make test         - Rodar testes
	@echo     make lint         - Verificar codigo com ruff
	@echo.
	@echo   PIPELINE COMPLETO
	@echo     make pipeline     - Rodar pipeline completo

# ============================================
# SETUP
# ============================================

setup:
	python -m venv venv
	venv\Scripts\python.exe -m pip install --upgrade pip
	venv\Scripts\pip.exe install -r requirements.txt
	if not exist data\raw mkdir data\raw
	if not exist data\processed mkdir data\processed
	if not exist models mkdir models
	if not exist reports mkdir reports
	if not exist docs\adr mkdir docs\adr
	@echo.
	@echo [OK] Setup completo!
	@echo Ative o ambiente: venv\Scripts\activate

# ============================================
# INFRAESTRUTURA
# ============================================

infra-up:
	cd docker && docker-compose up -d
	@echo [OK] Servicos iniciados

infra-down:
	cd docker && docker-compose down

infra-logs:
	cd docker && docker-compose logs -f

# ============================================
# DATA PIPELINE
# ============================================

ingest:
	venv\Scripts\python.exe -m src.data.ingestion

validate:
	venv\Scripts\python.exe -m src.data.validation

features:
	venv\Scripts\python.exe -m src.features.engineering

# ============================================
# ML PIPELINE
# ============================================

train:
	venv\Scripts\python.exe -m src.training.train

evaluate:
	venv\Scripts\python.exe -m src.training.evaluate

# ============================================
# SERVING
# ============================================

api:
	venv\Scripts\uvicorn.exe src.serving.api:app --reload --port 8000

# ============================================
# ORQUESTRACAO (PREFECT)
# ============================================

prefect-server:
	venv\Scripts\prefect.exe server start

prefect-run:
	venv\Scripts\python.exe -m pipelines.flows.training_flow

# ============================================
# MONITORAMENTO
# ============================================

drift:
	venv\Scripts\python.exe -m src.monitoring.drift

# ============================================
# TESTES
# ============================================

test:
	venv\Scripts\pytest.exe tests/ -v

test-cov:
	venv\Scripts\pytest.exe tests/ -v --cov=src --cov-report=html

lint:
	venv\Scripts\ruff.exe check src/ tests/

lint-fix:
	venv\Scripts\ruff.exe check src/ tests/ --fix

# ============================================
# PIPELINE COMPLETO
# ============================================

pipeline: ingest validate features train
	@echo [OK] Pipeline executado!

# ============================================
# LIMPEZA
# ============================================

clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	if exist htmlcov rmdir /s /q htmlcov
	if exist .coverage del .coverage
	@echo [OK] Limpo!
