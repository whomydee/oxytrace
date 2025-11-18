# Virtual environment paths
VENV = .venv
PYTHON = poetry run python
POETRY = poetry

.PHONY: venv install setup

venv:
	poetry config virtualenvs.in-project true
	poetry env use python3
	@echo "Virtual environment configured for Poetry"

install:
	poetry install

setup: venv install
	@echo "Setup complete! Poetry environment created and packages installed."


#################################################################################
# Formatting checks #############################################################

check-isort: ## Checks if .py files are formatted with isort
	@echo "Checking isort formatting(without update)"
	$(PYTHON) -m isort --check --diff oxytrace/

check-black: ## Checks if .py files are formatted with black
	@echo "Checking black formatting(without change)"
	$(PYTHON) -m black --config pyproject.toml --check --diff oxytrace/


check-format: check-isort check-black ## Checks all formatting issues


#################################################################################
# Formatting fixes ##############################################################

format-isort: ## Fixes .py files with isort
	@echo "Fixing isort formatting issues"
	$(PYTHON) -m isort oxytrace/

format-black: ## Fixes .py files with black
	@echo "Fixing black formatting issues"
	$(PYTHON) -m black --config pyproject.toml oxytrace/

format-unused-imports: ## Fixes unused imports and unused variables
	@echo "Removing unused imports"
	$(PYTHON) -m autoflake -i --remove-all-unused-imports --recursive oxytrace/

format: format-unused-imports format-isort format-black ## Fixes all formatting issues


#################################################################################
# Linting checks ################################################################

lint-flake8: ## Checks if .py files follow flake8
	@echo "Checking flake8 errors"
	$(PYTHON) -m flake8 oxytrace/ --exclude=.venv

lint-pylint: ## Checks if .py files follow pylint
	@echo "Checking pylint errors"
	$(PYTHON) -m pylint oxytrace/
lint-pylint-with-report-txt: ## Checks if .py files follow pylint and generates pylint-output.txt
	@echo "Checking pylint errors and generating pylint-output.txt"
	set -o pipefail && $(PYTHON) -m pylint oxytrace/ | tee pylint-output.txt

check-lint: lint-flake8 lint-pylint ## Checks all linting issues


#################################################################################
# Data & Model Management #######################################################

download-data: ## Download dataset from Google Drive
	@echo "Downloading dataset..."
	$(PYTHON) -c "from oxytrace.core.utils.dataset import DatasetManager; DatasetManager.download_dataset()"

train-detector: ## Train anomaly detector (10% data, quick mode)
	@echo "Training anomaly detector..."
	$(PYTHON) oxytrace/cli/train_detector.py --data-percent 10

train-detector-full: ## Train anomaly detector on full dataset
	@echo "Training anomaly detector on full dataset..."
	$(PYTHON) oxytrace/cli/train_detector.py --data-percent 100

train-forecaster: ## Train forecaster on full dataset
	@echo "Training forecaster..."
	$(PYTHON) oxytrace/cli/train_forecaster.py

train: train-detector train-forecaster ## Train all models (detector + forecaster)

train-full: train-detector-full train-forecaster ## Train all models on full dataset


#################################################################################
# Prediction & Evaluation #######################################################

demo: ## Run anomaly detection demo (5% data)
	@echo "Running demo workflow..."
	$(PYTHON) -m oxytrace.cli.main --demo

predict: ## Predict anomalies from input file (input/input_for_anomaly.py)
	@echo "Predicting anomalies from input file..."
	$(PYTHON) -m oxytrace.cli.main --predict

predict-custom: ## Predict with custom input file (usage: make predict-custom INPUT=path/to/file.py)
	@echo "Predicting anomalies from $(INPUT)..."
	$(PYTHON) -m oxytrace.cli.main --predict --input-file $(INPUT)

forecast: ## Generate 7-day forecast (uses existing model or trains if needed)
	@echo "Generating 7-day forecast..."
	$(PYTHON) -m oxytrace.cli.main --forecast

forecast-retrain: ## Retrain forecaster and generate 7-day forecast
	@echo "Retraining and generating 7-day forecast..."
	$(PYTHON) -m oxytrace.cli.main --forecast --train

forecast-14d: ## Generate 14-day forecast
	@echo "Generating 14-day forecast..."
	$(PYTHON) -m oxytrace.cli.main --forecast --horizon 14

evaluate: ## Evaluate forecaster performance
	@echo "Evaluating forecaster..."
	$(PYTHON) oxytrace/cli/evaluate_forecaster.py


#################################################################################
# Cleanup #######################################################################

clean-models: ## Remove trained models
	@echo "Removing trained models..."
	rm -rf artifacts/anomaly_detector artifacts/forecaster

clean-outputs: ## Remove output files
	@echo "Removing outputs..."
	rm -rf outputs/

clean: clean-models clean-outputs ## Clean all generated files


#################################################################################
# Development ###################################################################

notebook: ## Launch Jupyter notebook
	@echo "Launching Jupyter notebook..."
	$(PYTHON) -m jupyter notebook notebooks/

#################################################################################
# Help ##########################################################################

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-25s %s\n", $$1, $$2}'

