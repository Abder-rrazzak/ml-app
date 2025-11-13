# Makefile professionnel pour le projet ML Iris Classification
# Fournit des commandes standardisées pour le développement, les tests et le déploiement

# Variables de configuration
PYTHON := python3
PIP := pip
VENV_DIR := venv
SRC_DIR := src
TEST_DIR := tests
MODEL_DIR := models
DATA_DIR := data
DOCS_DIR := docs

# Détection de l'OS pour les commandes spécifiques
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	ACTIVATE := source $(VENV_DIR)/bin/activate
	PYTHON_VENV := $(VENV_DIR)/bin/python
	PIP_VENV := $(VENV_DIR)/bin/pip
else ifeq ($(UNAME_S),Darwin)
	ACTIVATE := source $(VENV_DIR)/bin/activate
	PYTHON_VENV := $(VENV_DIR)/bin/python
	PIP_VENV := $(VENV_DIR)/bin/pip
else
	ACTIVATE := $(VENV_DIR)\Scripts\activate
	PYTHON_VENV := $(VENV_DIR)\Scripts\python
	PIP_VENV := $(VENV_DIR)\Scripts\pip
endif

# Couleurs pour les messages
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Cibles principales (affichées dans l'aide)
.PHONY: help setup install dev-install lint format test test-cov train train-advanced predict api clean clean-all docker-build docker-run docs mlflow-ui jupyter security audit pre-commit

# Cible par défaut : afficher l'aide
help: ## Afficher cette aide
	@echo "$(BLUE)🚀 Commandes disponibles pour le projet ML Iris:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)📋 Workflow recommandé:$(NC)"
	@echo "  1. make setup          # Configuration initiale"
	@echo "  2. make dev-install    # Installation complète"
	@echo "  3. make train          # Entraînement du modèle"
	@echo "  4. make test           # Tests"
	@echo "  5. make api            # Lancer l'API"

## === CONFIGURATION ET INSTALLATION ===

setup: ## Configuration rapide de l'environnement
	@echo "$(BLUE)🔧 Configuration de l'environnement...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install -e .
	@echo "$(GREEN)✅ Environnement configuré$(NC)"

dev-setup: ## Configuration complète pour le développement
	@echo "$(BLUE)🛠️  Configuration développement...$(NC)"
	./scripts/setup_dev.sh
	@echo "$(GREEN)✅ Environnement de développement prêt$(NC)"

install: ## Installation des dépendances de base
	$(PIP_VENV) install -e .

dev-install: ## Installation complète avec outils de développement
	$(PIP_VENV) install -e ".[dev,docs,viz,mlops]"
	$(PIP_VENV) install pre-commit
	pre-commit install

## === QUALITÉ DU CODE ===

lint: ## Vérification du style de code avec flake8
	@echo "$(BLUE)🔍 Vérification du code...$(NC)"
	$(PYTHON_VENV) -m flake8 $(SRC_DIR)/ $(TEST_DIR)/ --max-line-length=88 --statistics
	@echo "$(GREEN)✅ Code vérifié$(NC)"

format: ## Formatage automatique du code avec Black et isort
	@echo "$(BLUE)🎨 Formatage du code...$(NC)"
	$(PYTHON_VENV) -m black $(SRC_DIR)/ $(TEST_DIR)/ --line-length=88
	$(PYTHON_VENV) -m isort $(SRC_DIR)/ $(TEST_DIR)/ --profile=black
	@echo "$(GREEN)✅ Code formaté$(NC)"

type-check: ## Vérification des types avec MyPy
	@echo "$(BLUE)🔬 Vérification des types...$(NC)"
	$(PYTHON_VENV) -m mypy $(SRC_DIR)/ --ignore-missing-imports

security: ## Analyse de sécurité avec Bandit
	@echo "$(BLUE)🔒 Analyse de sécurité...$(NC)"
	$(PYTHON_VENV) -m bandit -r $(SRC_DIR)/ -f json -o security-report.json
	$(PYTHON_VENV) -m bandit -r $(SRC_DIR)/

audit: ## Audit des vulnérabilités des dépendances
	@echo "$(BLUE)🛡️  Audit des dépendances...$(NC)"
	$(PIP_VENV) audit

pre-commit: ## Exécuter tous les hooks pre-commit
	@echo "$(BLUE)🪝 Exécution des hooks pre-commit...$(NC)"
	pre-commit run --all-files

## === TESTS ===

test: ## Exécuter les tests unitaires
	@echo "$(BLUE)🧪 Exécution des tests...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m pytest $(TEST_DIR)/ -v

test-cov: ## Tests avec couverture de code
	@echo "$(BLUE)📊 Tests avec couverture...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m pytest $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)📈 Rapport de couverture: htmlcov/index.html$(NC)"

test-parallel: ## Tests en parallèle pour plus de rapidité
	@echo "$(BLUE)⚡ Tests en parallèle...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m pytest $(TEST_DIR)/ -v -n auto

test-watch: ## Tests en mode watch (redémarrage automatique)
	@echo "$(BLUE)👀 Tests en mode watch...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m pytest-watch $(TEST_DIR)/

## === MACHINE LEARNING ===

train: ## Entraînement basique du modèle
	@echo "$(BLUE)🎯 Entraînement du modèle...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m src.train
	@echo "$(GREEN)✅ Modèle entraîné$(NC)"

train-advanced: ## Entraînement avancé avec MLflow
	@echo "$(BLUE)🚀 Entraînement avancé avec MLflow...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m src.train_advanced

predict: ## Faire une prédiction d'exemple
	@echo "$(BLUE)🔮 Prédiction d'exemple...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m src.predict --features "5.1,3.5,1.4,0.2"

predict-custom: ## Prédiction avec paramètres personnalisés (usage: make predict-custom FEATURES="6.2,3.4,5.4,2.3")
	@echo "$(BLUE)🔮 Prédiction personnalisée...$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m src.predict --features "$(FEATURES)"

mlflow-ui: ## Lancer l'interface MLflow
	@echo "$(BLUE)📊 Lancement de MLflow UI...$(NC)"
	@echo "$(YELLOW)🌐 Interface disponible sur: http://localhost:5000$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

## === API ET SERVICES ===

api: ## Lancer l'API FastAPI en mode développement
	@echo "$(BLUE)🌐 Lancement de l'API...$(NC)"
	@echo "$(YELLOW)📡 API disponible sur: http://localhost:8000$(NC)"
	@echo "$(YELLOW)📚 Documentation: http://localhost:8000/docs$(NC)"
	PYTHONPATH=. $(PYTHON_VENV) -m src.api.main

api-prod: ## Lancer l'API en mode production avec Gunicorn
	@echo "$(BLUE)🏭 Lancement API production...$(NC)"
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

api-test: ## Tester l'API avec des requêtes d'exemple
	@echo "$(BLUE)🧪 Test de l'API...$(NC)"
	curl -X GET "http://localhost:8000/health" | jq
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"features": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' | jq

## === DOCKER ===

docker-build: ## Construire l'image Docker
	@echo "$(BLUE)🐳 Construction de l'image Docker...$(NC)"
	docker build -t iris-ml-app:latest .

docker-run: ## Lancer le conteneur Docker
	@echo "$(BLUE)🚀 Lancement du conteneur...$(NC)"
	docker run -p 8000:8000 iris-ml-app:latest

docker-compose-up: ## Lancer tous les services avec Docker Compose
	@echo "$(BLUE)🐳 Lancement des services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Services démarrés:$(NC)"
	@echo "  • API: http://localhost:8000"
	@echo "  • MLflow: http://localhost:5000"

docker-compose-dev: ## Lancer en mode développement
	@echo "$(BLUE)🛠️  Mode développement...$(NC)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)✅ Services développement démarrés:$(NC)"
	@echo "  • API: http://localhost:8000"
	@echo "  • MLflow: http://localhost:5000"
	@echo "  • Jupyter: http://localhost:8888"

docker-compose-down: ## Arrêter tous les services
	docker-compose down

## === DOCUMENTATION ===

docs: ## Générer la documentation avec Sphinx
	@echo "$(BLUE)📚 Génération de la documentation...$(NC)"
	mkdir -p $(DOCS_DIR)
	$(PYTHON_VENV) -m sphinx-quickstart -q -p "ML Iris App" -a "Abder Rrazzak" $(DOCS_DIR)
	$(PYTHON_VENV) -m sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html
	@echo "$(GREEN)📖 Documentation: $(DOCS_DIR)/_build/html/index.html$(NC)"

docs-serve: ## Servir la documentation localement
	@echo "$(BLUE)🌐 Service de documentation...$(NC)"
	$(PYTHON_VENV) -m http.server 8080 -d $(DOCS_DIR)/_build/html

## === NOTEBOOKS ET ANALYSE ===

jupyter: ## Lancer Jupyter Lab
	@echo "$(BLUE)📓 Lancement de Jupyter Lab...$(NC)"
	@echo "$(YELLOW)🔗 Interface: http://localhost:8888$(NC)"
	$(PYTHON_VENV) -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

run-notebook: ## Lancer Jupyter Notebook (classique)
	@echo "$(BLUE)📔 Lancement de Jupyter Notebook...$(NC)"
	$(PYTHON_VENV) -m jupyter notebook notebooks/ --ip=0.0.0.0 --port=8888 --no-browser

notebook-convert: ## Convertir les notebooks en HTML
	@echo "$(BLUE)🔄 Conversion des notebooks...$(NC)"
	mkdir -p $(DOCS_DIR)/notebooks
	for notebook in notebooks/*.ipynb; do \
		$(PYTHON_VENV) -m jupyter nbconvert --to html --output-dir $(DOCS_DIR)/notebooks "$$notebook"; \
	done

## === NETTOYAGE ===

clean: ## Nettoyer les fichiers temporaires
	@echo "$(BLUE)🧹 Nettoyage des fichiers temporaires...$(NC)"
	rm -rf __pycache__/
	rm -rf $(SRC_DIR)/__pycache__/
	rm -rf $(TEST_DIR)/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-models: ## Supprimer les modèles entraînés
	@echo "$(YELLOW)⚠️  Suppression des modèles...$(NC)"
	rm -rf $(MODEL_DIR)/

clean-data: ## Supprimer les données traitées
	@echo "$(YELLOW)⚠️  Suppression des données traitées...$(NC)"
	rm -rf $(DATA_DIR)/processed/*

clean-all: clean clean-models ## Nettoyage complet (fichiers temporaires + modèles)
	@echo "$(BLUE)🧹 Nettoyage complet...$(NC)"
	rm -rf $(VENV_DIR)/
	rm -rf mlruns/
	rm -rf logs/
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

## === DÉPLOIEMENT ET CI/CD ===

build: ## Build complet du projet
	@echo "$(BLUE)🏗️  Build du projet...$(NC)"
	make clean
	make dev-install
	make lint
	make type-check
	make test-cov
	make train
	@echo "$(GREEN)✅ Build terminé avec succès$(NC)"

ci: ## Pipeline CI (utilisé par GitHub Actions)
	@echo "$(BLUE)🔄 Pipeline CI...$(NC)"
	make lint
	make type-check
	make security
	make test-cov
	@echo "$(GREEN)✅ Pipeline CI réussi$(NC)"

release: ## Préparer une release
	@echo "$(BLUE)🚀 Préparation de la release...$(NC)"
	make clean-all
	make setup
	make build
	make docker-build
	@echo "$(GREEN)✅ Release prête$(NC)"

## === MONITORING ET DEBUGGING ===

status: ## Afficher le statut du projet
	@echo "$(BLUE)📊 Statut du projet:$(NC)"
	@echo "  • Python: $$($(PYTHON) --version)"
	@echo "  • Environnement virtuel: $$(if [ -d $(VENV_DIR) ]; then echo '✅ Présent'; else echo '❌ Absent'; fi)"
	@echo "  • Modèle entraîné: $$(if [ -f $(MODEL_DIR)/iris_model.pkl ]; then echo '✅ Présent'; else echo '❌ Absent'; fi)"
	@echo "  • Tests: $$(if [ -d $(TEST_DIR) ]; then echo '✅ Configurés'; else echo '❌ Manquants'; fi)"

info: ## Informations détaillées sur l'environnement
	@echo "$(BLUE)ℹ️  Informations environnement:$(NC)"
	@echo "  • OS: $$(uname -s)"
	@echo "  • Architecture: $$(uname -m)"
	@echo "  • Python: $$($(PYTHON) --version)"
	@echo "  • Pip: $$($(PIP) --version)"
	@echo "  • Git: $$(git --version 2>/dev/null || echo 'Non installé')"
	@echo "  • Docker: $$(docker --version 2>/dev/null || echo 'Non installé')"

check-deps: ## Vérifier les dépendances obsolètes
	@echo "$(BLUE)🔍 Vérification des dépendances...$(NC)"
	$(PIP_VENV) list --outdated

update-deps: ## Mettre à jour les dépendances
	@echo "$(BLUE)⬆️  Mise à jour des dépendances...$(NC)"
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install -e ".[dev,docs,viz,mlops]" --upgrade

# Cible par défaut si aucune n'est spécifiée
.DEFAULT_GOAL := help