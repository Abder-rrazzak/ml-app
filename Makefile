.PHONY: setup install lint test train predict clean run-notebook

# Setup virtual environment and install dependencies
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

# Install dependencies (requires virtual environment)
install:
	./venv/bin/pip install -r requirements.txt

# Lint code
lint:
	./venv/bin/flake8 src/ tests/ --max-line-length=88

# Run tests
test:
	PYTHONPATH=. ./venv/bin/pytest tests/ -v

# Train model
train:
	PYTHONPATH=. ./venv/bin/python -m src.train

# Make prediction (example usage)
predict:
	PYTHONPATH=. ./venv/bin/python -m src.predict --features "5.1,3.5,1.4,0.2"

# Clean generated files
clean:
	rm -rf models/
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete

# Run jupyter notebook
run-notebook:
	./venv/bin/jupyter notebook notebooks/