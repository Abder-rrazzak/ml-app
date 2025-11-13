"""
Tests pour l'API FastAPI.

Ce module teste tous les endpoints de l'API pour s'assurer
qu'ils fonctionnent correctement et retournent les bonnes réponses.
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.model import IrisClassifier


# Fixture pour le client de test FastAPI
@pytest.fixture
def client():
    """
    Crée un client de test pour l'API FastAPI.

    Returns:
        TestClient: Client pour tester l'API
    """
    return TestClient(app)


# Fixture pour un modèle mocké
@pytest.fixture
def mock_model():
    """
    Crée un modèle mocké pour les tests.

    Returns:
        Mock: Modèle simulé avec les méthodes nécessaires
    """
    model = Mock(spec=IrisClassifier)
    model.is_trained = True
    model.predict.return_value = np.array([0])  # Prédiction setosa
    model.model = Mock()
    model.model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
    return model


class TestHealthEndpoint:
    """Tests pour l'endpoint de santé /health"""

    def test_health_endpoint_success(self, client, mock_model):
        """Test que l'endpoint de santé retourne un statut healthy."""
        with patch("src.api.main.model", mock_model):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            # Vérifier la structure de la réponse
            assert "status" in data
            assert "version" in data
            assert "timestamp" in data
            assert "model_loaded" in data

            # Vérifier les valeurs
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_health_endpoint_model_not_loaded(self, client):
        """Test l'endpoint de santé quand le modèle n'est pas chargé."""
        with patch("src.api.main.model", None):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

    def test_health_endpoint_model_error(self, client):
        """Test l'endpoint de santé quand le modèle génère une erreur."""
        mock_model = Mock()
        mock_model.is_trained = True
        mock_model.predict.side_effect = Exception("Erreur modèle")

        with patch("src.api.main.model", mock_model):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests pour l'endpoint de prédiction /predict"""

    def test_predict_single_flower_success(self, client, mock_model):
        """Test de prédiction réussie pour une seule fleur."""
        with patch("src.api.main.model", mock_model):
            # Données de test valides
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }

            response = client.post("/predict", json=test_data)

            assert response.status_code == 200
            data = response.json()

            # Vérifier la structure de la réponse
            assert "predictions" in data
            assert "count" in data
            assert "model_version" in data

            # Vérifier les valeurs
            assert len(data["predictions"]) == 1
            assert data["predictions"][0] == "setosa"
            assert data["count"] == 1

    def test_predict_multiple_flowers_success(self, client, mock_model):
        """Test de prédiction pour plusieurs fleurs."""
        # Configurer le mock pour retourner plusieurs prédictions
        mock_model.predict.return_value = np.array([0, 1, 2])

        with patch("src.api.main.model", mock_model):
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    },
                    {
                        "sepal_length": 6.2,
                        "sepal_width": 3.4,
                        "petal_length": 5.4,
                        "petal_width": 2.3,
                    },
                    {
                        "sepal_length": 5.9,
                        "sepal_width": 3.0,
                        "petal_length": 4.2,
                        "petal_width": 1.5,
                    },
                ]
            }

            response = client.post("/predict", json=test_data)

            assert response.status_code == 200
            data = response.json()

            assert len(data["predictions"]) == 3
            assert data["predictions"] == ["setosa", "versicolor", "virginica"]
            assert data["count"] == 3

    def test_predict_with_request_id(self, client, mock_model):
        """Test de prédiction avec un ID de requête."""
        with patch("src.api.main.model", mock_model):
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ],
                "request_id": "test_request_123",
            }

            response = client.post("/predict", json=test_data)

            assert response.status_code == 200
            data = response.json()

            assert data["request_id"] == "test_request_123"

    def test_predict_invalid_features_negative_values(self, client):
        """Test avec des valeurs négatives (invalides)."""
        test_data = {
            "features": [
                {
                    "sepal_length": -1.0,  # Valeur négative invalide
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }

        response = client.post("/predict", json=test_data)

        assert response.status_code == 422  # Erreur de validation

    def test_predict_invalid_features_out_of_range(self, client):
        """Test avec des valeurs hors limites."""
        test_data = {
            "features": [
                {
                    "sepal_length": 15.0,  # Valeur trop élevée
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }

        response = client.post("/predict", json=test_data)

        assert response.status_code == 422  # Erreur de validation

    def test_predict_missing_features(self, client):
        """Test avec des caractéristiques manquantes."""
        test_data = {
            "features": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    # petal_length et petal_width manquants
                }
            ]
        }

        response = client.post("/predict", json=test_data)

        assert response.status_code == 422  # Erreur de validation

    def test_predict_empty_features_list(self, client):
        """Test avec une liste vide de caractéristiques."""
        test_data = {"features": []}  # Liste vide

        response = client.post("/predict", json=test_data)

        assert response.status_code == 422  # Erreur de validation

    def test_predict_too_many_features(self, client):
        """Test avec trop de fleurs (> 100)."""
        # Créer une liste de 101 fleurs
        features = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ] * 101

        test_data = {"features": features}

        response = client.post("/predict", json=test_data)

        assert response.status_code == 422  # Erreur de validation

    def test_predict_model_not_loaded(self, client):
        """Test quand le modèle n'est pas chargé."""
        with patch("src.api.main.model", None):
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }

            response = client.post("/predict", json=test_data)

            assert response.status_code == 503  # Service unavailable

    def test_predict_model_error(self, client):
        """Test quand le modèle génère une erreur."""
        mock_model = Mock()
        mock_model.is_trained = True
        mock_model.predict.side_effect = Exception("Erreur de prédiction")

        with patch("src.api.main.model", mock_model):
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }

            response = client.post("/predict", json=test_data)

            assert response.status_code == 500  # Erreur interne


class TestModelInfoEndpoint:
    """Tests pour l'endpoint d'informations du modèle /model/info"""

    def test_model_info_success(self, client, mock_model):
        """Test de récupération des informations du modèle."""
        # Configurer le mock
        mock_model.model.get_params.return_value = {
            "C": 1.0,
            "max_iter": 200,
            "solver": "lbfgs",
        }

        with patch("src.api.main.model", mock_model):
            response = client.get("/model/info")

            assert response.status_code == 200
            data = response.json()

            # Vérifier la structure de la réponse
            assert "model_type" in data
            assert "model_version" in data
            assert "is_trained" in data
            assert "features" in data
            assert "classes" in data

            # Vérifier les valeurs
            assert data["model_type"] == "LogisticRegression"
            assert data["is_trained"] is True
            assert len(data["features"]) == 4
            assert len(data["classes"]) == 3

    def test_model_info_model_not_loaded(self, client):
        """Test des informations quand le modèle n'est pas chargé."""
        with patch("src.api.main.model", None):
            response = client.get("/model/info")

            assert response.status_code == 503  # Service unavailable


class TestAPIDocumentation:
    """Tests pour la documentation automatique de l'API"""

    def test_openapi_schema_accessible(self, client):
        """Test que le schéma OpenAPI est accessible."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        # Vérifier que c'est un schéma OpenAPI valide
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_page_accessible(self, client):
        """Test que la page de documentation Swagger est accessible."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_page_accessible(self, client):
        """Test que la page ReDoc est accessible."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIPerformance:
    """Tests de performance de l'API"""

    def test_predict_response_time(self, client, mock_model):
        """Test que les prédictions sont rapides."""
        import time

        with patch("src.api.main.model", mock_model):
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }

            start_time = time.time()
            response = client.post("/predict", json=test_data)
            end_time = time.time()

            assert response.status_code == 200
            # La prédiction devrait prendre moins de 1 seconde
            assert (end_time - start_time) < 1.0

    def test_concurrent_predictions(self, client, mock_model):
        """Test de prédictions concurrentes."""
        import concurrent.futures

        def make_prediction():
            test_data = {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    }
                ]
            }
            return client.post("/predict", json=test_data)

        with patch("src.api.main.model", mock_model):
            # Lancer 10 requêtes concurrentes
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_prediction) for _ in range(10)]
                responses = [future.result() for future in futures]

            # Toutes les requêtes devraient réussir
            for response in responses:
                assert response.status_code == 200
