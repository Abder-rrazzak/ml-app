# ML App - Iris Classification Project

A production-ready machine learning project for classifying Iris flowers using scikit-learn and Python 3.11.

## Project Structure

```
ml-app/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── notebooks/
│   └── 01_data_analysis.ipynb  # Exploratory data analysis
├── src/
│   ├── __init__.py       # Package initialization
│   ├── data_loader.py    # Data loading utilities
│   ├── model.py          # ML model definitions
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction script
│   └── utils.py          # Utility functions
├── tests/
│   └── test_model.py     # Unit tests
├── requirements.txt      # Python dependencies
├── Makefile             # Build automation
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd ml-app
```

2. Setup virtual environment and install dependencies:
```bash
make setup
```

Alternatively, you can set up manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the Iris classification model:
```bash
make train
```

This will:
- Load the Iris dataset
- Split data into train/test sets
- Train a Logistic Regression model
- Evaluate performance
- Save the model to `models/iris_model.pkl`

### Making Predictions

Make predictions on new data:
```bash
make predict
```

Or with custom features:
```bash
python -m src.predict --features "5.1,3.5,1.4,0.2"
```

Expected output:
```
Input features: [5.1, 3.5, 1.4, 0.2]
Predicted class: setosa
```

### Running Tests

Execute unit tests:
```bash
make test
```

### Code Linting

Check code style:
```bash
make lint
```

### Data Analysis

Launch Jupyter notebook for exploratory data analysis:
```bash
make run-notebook
```

### Cleaning

Remove generated files:
```bash
make clean
```

## Model Details

- **Algorithm**: Logistic Regression
- **Dataset**: Iris flower dataset (150 samples, 4 features, 3 classes)
- **Features**: sepal length, sepal width, petal length, petal width
- **Classes**: setosa, versicolor, virginica

## Development

The project follows Python best practices:
- PEP8 code style
- Modular architecture
- Unit testing with pytest
- CLI interface with click
- Comprehensive documentation

## Requirements

- Python 3.11+
- Linux/WSL environment
- Dependencies listed in `requirements.txt`

## Example Commands

```bash
# Setup virtual environment and install dependencies
make setup

# Train model
make train

# Make prediction
make predict
# Or with custom features:
./venv/bin/python -m src.predict --features "6.2,3.4,5.4,2.3"

# Run tests
make test

# Clean up
make clean
```