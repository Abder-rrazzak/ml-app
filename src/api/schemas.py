"""
Schémas Pydantic pour la validation des données API.

Ce module définit les modèles de données utilisés par l'API
pour valider les entrées et formater les sorties.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class IrisFeatures(BaseModel):
    """
    Modèle Pydantic pour les caractéristiques d'une fleur Iris.
    
    Utilise la validation automatique de Pydantic pour s'assurer
    que les données d'entrée sont correctes et dans les bonnes plages.
    """
    
    # Longueur du sépale en centimètres
    sepal_length: float = Field(
        ...,  # Champ obligatoire
        ge=0.0,  # Greater or equal (>=) à 0
        le=10.0,  # Less or equal (<=) à 10
        description="Longueur du sépale en centimètres",
        example=5.1
    )
    
    # Largeur du sépale en centimètres
    sepal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Largeur du sépale en centimètres",
        example=3.5
    )
    
    # Longueur du pétale en centimètres
    petal_length: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Longueur du pétale en centimètres",
        example=1.4
    )
    
    # Largeur du pétale en centimètres
    petal_width: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Largeur du pétale en centimètres",
        example=0.2
    )
    
    @validator('*', pre=True)
    def validate_numeric_fields(cls, v):
        """
        Validateur personnalisé pour s'assurer que tous les champs
        sont des nombres valides et positifs.
        """
        if not isinstance(v, (int, float)):
            raise ValueError('Toutes les mesures doivent être des nombres')
        if v < 0:
            raise ValueError('Toutes les mesures doivent être positives')
        return float(v)


class PredictionRequest(BaseModel):
    """
    Modèle pour une requête de prédiction.
    
    Peut contenir une ou plusieurs fleurs à classifier.
    """
    
    # Liste des caractéristiques des fleurs à classifier
    features: List[IrisFeatures] = Field(
        ...,
        min_items=1,  # Au moins une fleur
        max_items=100,  # Maximum 100 fleurs par requête
        description="Liste des caractéristiques des fleurs à classifier"
    )
    
    # Métadonnées optionnelles pour la requête
    request_id: Optional[str] = Field(
        None,
        description="Identifiant unique pour tracer la requête",
        example="req_123456"
    )


class PredictionResponse(BaseModel):
    """
    Modèle pour la réponse de prédiction.
    
    Contient les prédictions et les métadonnées associées.
    """
    
    # Classes prédites pour chaque fleur
    predictions: List[str] = Field(
        ...,
        description="Classes prédites (setosa, versicolor, virginica)"
    )
    
    # Probabilités de chaque classe (optionnel)
    probabilities: Optional[List[List[float]]] = Field(
        None,
        description="Probabilités pour chaque classe"
    )
    
    # Nombre de prédictions effectuées
    count: int = Field(
        ...,
        description="Nombre de prédictions dans cette réponse"
    )
    
    # Identifiant de la requête (si fourni)
    request_id: Optional[str] = Field(
        None,
        description="Identifiant de la requête originale"
    )
    
    # Version du modèle utilisé
    model_version: str = Field(
        ...,
        description="Version du modèle utilisé pour la prédiction",
        example="1.0.0"
    )


class HealthResponse(BaseModel):
    """
    Modèle pour la réponse de santé de l'API.
    
    Utilisé par les endpoints de monitoring pour vérifier
    que l'API fonctionne correctement.
    """
    
    # Statut de l'API
    status: str = Field(
        ...,
        description="Statut de l'API",
        example="healthy"
    )
    
    # Version de l'API
    version: str = Field(
        ...,
        description="Version de l'API",
        example="1.0.0"
    )
    
    # Timestamp de la vérification
    timestamp: str = Field(
        ...,
        description="Timestamp de la vérification de santé",
        example="2023-11-13T12:00:00Z"
    )
    
    # Statut du modèle ML
    model_loaded: bool = Field(
        ...,
        description="Indique si le modèle ML est chargé et prêt"
    )


class ErrorResponse(BaseModel):
    """
    Modèle standardisé pour les réponses d'erreur.
    
    Fournit des informations détaillées sur les erreurs
    pour faciliter le debugging.
    """
    
    # Message d'erreur principal
    error: str = Field(
        ...,
        description="Message d'erreur principal"
    )
    
    # Code d'erreur spécifique
    error_code: str = Field(
        ...,
        description="Code d'erreur pour identification programmatique"
    )
    
    # Détails supplémentaires (optionnel)
    details: Optional[str] = Field(
        None,
        description="Détails supplémentaires sur l'erreur"
    )
    
    # Timestamp de l'erreur
    timestamp: str = Field(
        ...,
        description="Timestamp de l'erreur"
    )