# Developer convenience targets. Run `make` or `make help` to list them.
.DEFAULT_GOAL := help

.PHONY: help install install-dev data train evaluate predict api dashboard test lint typecheck format docker clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies and the package
	pip install -r requirements.txt && pip install --no-deps -e .

install-dev: ## Install dev dependencies and the package
	pip install -r requirements-dev.txt && pip install --no-deps -e .

data: ## Download the raw dataset
	python -m credit_risk.data.ingestion

train: ## Train the model end-to-end
	python -m credit_risk.models.train

evaluate: ## Evaluate the saved model on the test split
	python -m credit_risk.models.evaluate --split test

predict: ## Score a demo applicant
	python -m credit_risk.models.predict

api: ## Run the FastAPI service
	uvicorn app.api:app --host 0.0.0.0 --port 8000

dashboard: ## Run the Streamlit dashboard
	streamlit run app/dashboard.py

test: ## Run the test suite
	pytest -q

lint: ## Lint with ruff
	ruff check src app tests

typecheck: ## Type-check with mypy
	mypy src/credit_risk

format: ## Auto-fix lint issues
	ruff check --fix src app tests

docker: ## Build and start the stack
	docker compose up --build

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ reports/experiments/*.json
