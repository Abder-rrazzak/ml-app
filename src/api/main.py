"""
API FastAPI principale pour servir le modèle de classification Iris.

Cette API expose des endpoints REST pour :
- Faire des prédictions sur de nouvelles données
- Vérifier la santé de l'API
- Obtenir des informations sur le modèle
"""

import logging
import os
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..model import IrisClassifier
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Configuration du logging pour tracer les requêtes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI avec métadonnées
app = FastAPI(
    title="Iris Classification API",
    description="""
    API REST pour la classification des fleurs Iris.
    
    Cette API utilise un modèle de régression logistique entraîné
    sur le dataset Iris pour classifier les fleurs en trois espèces :
    - Iris Setosa
    - Iris Versicolor  
    - Iris Virginica
    
    ## Utilisation
    
    1. Envoyez une requête POST à `/predict` avec les caractéristiques de la fleur
    2. Recevez la prédiction de l'espèce avec les probabilités
    3. Utilisez `/health` pour vérifier que l'API fonctionne
    """,
    version="1.0.0",
    contact={
        "name": "Abder Rrazzak",
        "email": "abder.rrazzak@example.com",
        "url": "https://github.com/abder-rrazzak/ml-app",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour stocker le modèle chargé
model: IrisClassifier = None
model_version = "1.0.0"


@app.on_event("startup")
async def startup_event():
    """
    Événement de démarrage de l'API.
    
    Charge le modèle ML au démarrage pour éviter de le recharger
    à chaque requête (améliore les performances).
    """
    global model
    
    try:
        logger.info("Démarrage de l'API Iris Classification...")
        
        # Chemin vers le modèle entraîné
        model_path = os.getenv("MODEL_PATH", "models/iris_model.pkl")
        
        # Chargement du modèle
        model = IrisClassifier()
        model.load(model_path)
        
        logger.info(f"Modèle chargé avec succès depuis {model_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        # En production, on pourrait arrêter l'API si le modèle ne charge pas
        raise


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de santé",
    description="Endpoint pour vérifier que l'API et le modèle fonctionnent correctement"
)
async def health_check():
    """
    Endpoint de vérification de santé.
    
    Utilisé par les systèmes de monitoring (Kubernetes, Docker, etc.)
    pour vérifier que l'API est opérationnelle.
    """
    try:
        # Vérifier que le modèle est chargé et fonctionnel
        model_loaded = model is not None and model.is_trained
        
        # Test rapide du modèle avec des données factices
        if model_loaded:
            import numpy as np
            test_data = np.array([[5.0, 3.0, 1.5, 0.2]])
            _ = model.predict(test_data)
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=model_loaded
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de santé : {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=False
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction de classification",
    description="Classifie une ou plusieurs fleurs Iris selon leurs caractéristiques",
    responses={
        400: {"model": ErrorResponse, "description": "Données d'entrée invalides"},
        500: {"model": ErrorResponse, "description": "Erreur interne du serveur"},
    }
)
async def predict(request: PredictionRequest):
    """
    Endpoint principal pour les prédictions.
    
    Prend en entrée les caractéristiques d'une ou plusieurs fleurs
    et retourne les prédictions de classification.
    
    Args:
        request: Requête contenant les caractéristiques des fleurs
        
    Returns:
        PredictionResponse: Prédictions avec métadonnées
        
    Raises:
        HTTPException: En cas d'erreur de validation ou de prédiction
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None or not model.is_trained:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non disponible"
            )
        
        logger.info(f"Requête de prédiction reçue : {len(request.features)} fleur(s)")
        
        # Convertir les données Pydantic en format numpy pour le modèle
        import numpy as np
        
        features_array = np.array([
            [
                flower.sepal_length,
                flower.sepal_width,
                flower.petal_length,
                flower.petal_width
            ]
            for flower in request.features
        ])
        
        # Faire les prédictions
        predictions_numeric = model.predict(features_array)
        
        # Mapper les prédictions numériques vers les noms des classes
        class_names = ['setosa', 'versicolor', 'virginica']
        predictions = [class_names[pred] for pred in predictions_numeric]
        
        # Optionnel : calculer les probabilités si le modèle le supporte
        probabilities = None
        try:
            if hasattr(model.model, 'predict_proba'):
                proba_array = model.model.predict_proba(features_array)
                probabilities = proba_array.tolist()
        except Exception as e:
            logger.warning(f"Impossible de calculer les probabilités : {e}")
        
        # Construire la réponse
        response = PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            count=len(predictions),
            request_id=request.request_id,
            model_version=model_version
        )
        
        logger.info(f"Prédictions effectuées avec succès : {predictions}")
        
        return response
        
    except HTTPException:
        # Re-lever les HTTPException sans les modifier
        raise
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne : {str(e)}"
        )


@app.get(
    "/model/info",
    summary="Informations sur le modèle",
    description="Retourne des informations sur le modèle ML utilisé"
)
async def model_info():
    """
    Endpoint pour obtenir des informations sur le modèle.
    
    Utile pour le debugging et le monitoring du modèle en production.
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non disponible"
            )
        
        # Informations sur le modèle
        info = {
            "model_type": "LogisticRegression",
            "model_version": model_version,
            "is_trained": model.is_trained,
            "features": [
                "sepal_length",
                "sepal_width", 
                "petal_length",
                "petal_width"
            ],
            "classes": ["setosa", "versicolor", "virginica"],
            "sklearn_version": "1.3.0+",
        }
        
        # Ajouter des paramètres du modèle si disponibles
        if hasattr(model.model, 'get_params'):
            info["model_params"] = model.model.get_params()
        
        return info
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des infos modèle : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne : {str(e)}"
        )


# Gestionnaire d'erreur global pour les exceptions non gérées
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Gestionnaire global pour toutes les exceptions non gérées.
    
    Assure qu'aucune erreur ne remonte sans être loggée
    et que l'utilisateur reçoit toujours une réponse structurée.
    """
    logger.error(f"Exception non gérée : {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Erreur interne du serveur",
            error_code="INTERNAL_ERROR",
            details=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


def main():
    """
    Point d'entrée principal pour lancer l'API.
    
    Utilisé par le script de démarrage et les tests.
    """
    # Configuration depuis les variables d'environnement
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Démarrage de l'API sur {host}:{port}")
    
    # Lancement du serveur Uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=debug,  # Rechargement automatique en mode debug
        log_level="info" if not debug else "debug"
    )


if __name__ == "__main__":
    main()