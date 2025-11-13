"""Prediction script for the ML model."""

import click
import numpy as np

from .model import IrisClassifier


@click.command()
@click.option(
    "--model-path", default="models/iris_model.pkl", help="Path to the trained model"
)
@click.option("--features", required=True, help="Comma-separated feature values")
def predict(model_path, features):
    """Make predictions using the trained model."""
    try:
        # Parse features
        feature_values = [float(x.strip()) for x in features.split(",")]
        if len(feature_values) != 4:
            raise ValueError("Expected 4 features")

        click.echo("Loading trained model...")
        classifier = IrisClassifier()
        classifier.load(model_path)

        # Make prediction
        X = np.array([feature_values])
        prediction = classifier.predict(X)

        # Map prediction to class name
        class_names = ["setosa", "versicolor", "virginica"]
        predicted_class = class_names[prediction[0]]

        click.echo(f"Input features: {feature_values}")
        click.echo(f"Predicted class: {predicted_class}")

    except FileNotFoundError:
        click.echo(f"Error: Model file not found at {model_path}")
        click.echo("Please train the model first using: make train")
    except ValueError as e:
        click.echo(f"Error: {e}")
        click.echo("Please provide 4 comma-separated numeric values")
    except Exception as e:
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    predict()
