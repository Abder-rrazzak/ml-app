"""Training script for the ML model."""

import os

import click

from .data_loader import load_iris_data, split_data
from .model import IrisClassifier


@click.command()
@click.option(
    "--model-path",
    default="models/iris_model.pkl",
    help="Path to save the trained model",
)
def train(model_path):
    """Train the Iris classification model."""
    click.echo("Loading Iris dataset...")
    df, target_names = load_iris_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    click.echo("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    click.echo("Training model...")
    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    click.echo("Evaluating model...")
    accuracy, report = classifier.evaluate(X_test, y_test)
    click.echo(f"Accuracy: {accuracy:.4f}")
    click.echo(f"Classification Report:\n{report}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    click.echo(f"Saving model to {model_path}...")
    classifier.save(model_path)
    click.echo("Training completed successfully!")


if __name__ == "__main__":
    train()
