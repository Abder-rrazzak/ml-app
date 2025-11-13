"""Machine learning model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


class IrisClassifier:
    """Logistic Regression classifier for Iris dataset."""

    def __init__(self, random_state=42):
        """Initialize the classifier."""
        self.model = LogisticRegression(random_state=random_state, max_iter=200)
        self.is_trained = False

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report

    def save(self, filepath):
        """Save the trained model."""
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
