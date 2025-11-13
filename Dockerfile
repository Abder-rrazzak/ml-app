# Dockerfile multi-stage pour optimiser la taille de l'image finale
# Stage 1: Image de base avec Python et dépendances système

FROM python:3.11-slim as base

# Métadonnées de l'image Docker
LABEL maintainer="Abder Rrazzak <abder.rrazzak@example.com>"
LABEL description="API ML pour la classification des fleurs Iris"
LABEL version="1.0.0"

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r mlapp && useradd -r -g mlapp mlapp

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    # Outils de build pour les packages Python avec extensions C
    build-essential \
    # Git pour le versioning du code
    git \
    # Outils de nettoyage
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Installation des dépendances Python
FROM base as dependencies

# Répertoire de travail
WORKDIR /app

# Copier les fichiers de configuration des dépendances
COPY requirements.txt pyproject.toml ./

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Image finale de production
FROM dependencies as production

# Copier le code source
COPY src/ ./src/
COPY configs/ ./configs/

# Créer les répertoires nécessaires
RUN mkdir -p models data/raw data/processed logs

# Changer la propriété des fichiers vers l'utilisateur mlapp
RUN chown -R mlapp:mlapp /app

# Basculer vers l'utilisateur non-root
USER mlapp

# Port exposé par l'API FastAPI
EXPOSE 8000

# Variables d'environnement pour l'application
ENV MODEL_PATH=/app/models/iris_model.pkl \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MLFLOW_TRACKING_URI=file:./mlruns

# Commande par défaut : lancer l'API
CMD ["python", "-m", "src.api.main"]

# Stage 4: Image de développement avec outils supplémentaires
FROM dependencies as development

# Installer les dépendances de développement
RUN pip install -e ".[dev,docs,viz,mlops]"

# Copier tous les fichiers (y compris les tests)
COPY . .

# Installer les hooks pre-commit
RUN pre-commit install || true

# Changer la propriété vers mlapp
RUN chown -R mlapp:mlapp /app

USER mlapp

# Port pour Jupyter (en plus de l'API)
EXPOSE 8000 8888

# Commande par défaut en mode développement
CMD ["python", "-m", "src.api.main"]