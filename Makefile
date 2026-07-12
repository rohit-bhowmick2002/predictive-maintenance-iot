.PHONY: help install train evaluate dashboard test clean all

# ============================================================
# Predictive Maintenance IIoT — Makefile
# ============================================================

PYTHON := python3
PIP := pip
STREAMLIT := streamlit
PYTEST := pytest
MLFLOW := mlflow

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install dev dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov hypothesis black ruff mypy

download-data: ## Download Azure PdM dataset
	@echo "Downloading telemetry data..."
	wget -nc -P data/raw/ https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_telemetry.csv
	wget -nc -P data/raw/ https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_errors.csv
	wget -nc -P data/raw/ https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_maint.csv
	wget -nc -P data/raw/ https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_failures.csv
	wget -nc -P data/raw/ https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_machines.csv
	@echo "✅ Data downloaded to data/raw/"

preprocess: ## Run data preprocessing & feature engineering
	$(PYTHON) src/data/preprocessor.py
	$(PYTHON) src/data/feature_engine.py

train: preprocess ## Train all models
	$(PYTHON) src/models/train.py --model all

train-xgboost: preprocess ## Train XGBoost only
	$(PYTHON) src/models/train.py --model xgboost

train-lstm: preprocess ## Train LSTM only
	$(PYTHON) src/models/train.py --model lstm

evaluate: ## Run evaluation suite
	$(PYTHON) src/evaluation/metrics.py
	$(PYTHON) src/evaluation/cost_analysis.py
	$(PYTHON) src/evaluation/visualization.py

dashboard: ## Launch Streamlit dashboard
	$(STREAMLIT) run src/dashboard/app.py --server.port 8501

api: ## Start FastAPI prediction server
	uvicorn src.dashboard.api:app --host 0.0.0.0 --port 8000 --reload

test: ## Run unit tests
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-smoke: ## Run smoke tests only
	$(PYTEST) tests/ -v -m "smoke"

lint: ## Run linters
	black src/ tests/ --check
	ruff check src/ tests/

format: ## Auto-format code
	black src/ tests/
	ruff check src/ tests/ --fix

mlflow-ui: ## Launch MLflow tracking UI
	$(MLFLOW) ui --host 0.0.0.0 --port 5000

all: train evaluate test ## Run full pipeline (train + evaluate + test)

clean: ## Clean generated files
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf reports/figures/*
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .ruff_cache
	@echo "✅ Cleaned."

setup: download-data install preprocess ## Full setup from scratch
	@echo "✅ Full setup complete. Run 'make train' to train models."
