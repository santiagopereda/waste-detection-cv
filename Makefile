#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Install Local Packages
reinstall_package:
	@pip uninstall -y src || :
	@pip install -e .

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py