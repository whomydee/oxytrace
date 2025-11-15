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
