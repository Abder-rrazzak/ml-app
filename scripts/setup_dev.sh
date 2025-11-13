#!/bin/bash

# Script de configuration de l'environnement de développement
# Ce script automatise l'installation et la configuration complète du projet

set -e  # Arrêter le script en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher des messages colorés
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fonction pour vérifier si une commande existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Vérification des prérequis
check_prerequisites() {
    log_info "Vérification des prérequis..."
    
    # Vérifier Python 3.9+
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
            log_success "Python $PYTHON_VERSION détecté"
        else
            log_error "Python 3.9+ requis, version $PYTHON_VERSION détectée"
            exit 1
        fi
    else
        log_error "Python 3 non trouvé. Veuillez l'installer."
        exit 1
    fi
    
    # Vérifier pip
    if ! command_exists pip3; then
        log_error "pip3 non trouvé. Veuillez l'installer."
        exit 1
    fi
    
    # Vérifier Git
    if ! command_exists git; then
        log_warning "Git non trouvé. Certaines fonctionnalités seront limitées."
    fi
    
    # Vérifier Docker (optionnel)
    if command_exists docker; then
        log_success "Docker détecté"
    else
        log_warning "Docker non trouvé. Les fonctionnalités de containerisation seront indisponibles."
    fi
}

# Configuration de l'environnement virtuel
setup_virtual_environment() {
    log_info "Configuration de l'environnement virtuel..."
    
    # Supprimer l'ancien environnement s'il existe
    if [ -d "venv" ]; then
        log_warning "Suppression de l'ancien environnement virtuel..."
        rm -rf venv
    fi
    
    # Créer un nouvel environnement virtuel
    python3 -m venv venv
    log_success "Environnement virtuel créé"
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Mettre à jour pip
    pip install --upgrade pip
    log_success "pip mis à jour"
}

# Installation des dépendances
install_dependencies() {
    log_info "Installation des dépendances..."
    
    # Installer les dépendances principales
    pip install -e ".[dev,docs,viz,mlops]"
    log_success "Dépendances installées"
    
    # Installer les dépendances de développement supplémentaires
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        log_success "Dépendances de développement installées"
    fi
}

# Configuration des hooks pre-commit
setup_pre_commit() {
    log_info "Configuration des hooks pre-commit..."
    
    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Hooks pre-commit installés"
        
        # Exécuter pre-commit sur tous les fichiers
        log_info "Exécution de pre-commit sur tous les fichiers..."
        pre-commit run --all-files || log_warning "Certains hooks pre-commit ont échoué"
    else
        log_warning "pre-commit non installé. Hooks ignorés."
    fi
}

# Configuration de MLflow
setup_mlflow() {
    log_info "Configuration de MLflow..."
    
    # Créer le répertoire MLflow s'il n'existe pas
    mkdir -p mlruns
    
    # Définir les variables d'environnement MLflow
    export MLFLOW_TRACKING_URI="file:./mlruns"
    
    log_success "MLflow configuré (URI: $MLFLOW_TRACKING_URI)"
}

# Création des répertoires nécessaires
create_directories() {
    log_info "Création des répertoires nécessaires..."
    
    # Répertoires pour les données
    mkdir -p data/raw data/processed
    
    # Répertoires pour les modèles
    mkdir -p models
    
    # Répertoires pour les logs
    mkdir -p logs
    
    # Répertoires pour la documentation
    mkdir -p docs
    
    log_success "Répertoires créés"
}

# Test de l'installation
test_installation() {
    log_info "Test de l'installation..."
    
    # Test des imports Python
    python3 -c "
import sys
try:
    import numpy, pandas, sklearn, click, fastapi, mlflow
    print('✅ Tous les packages principaux sont importables')
except ImportError as e:
    print(f'❌ Erreur d\\'import: {e}')
    sys.exit(1)
"
    
    # Test de la syntaxe du code
    if command_exists flake8; then
        log_info "Vérification de la syntaxe du code..."
        flake8 src/ --max-line-length=88 --count --statistics || log_warning "Problèmes de style détectés"
    fi
    
    # Test des imports du projet
    PYTHONPATH=. python3 -c "
try:
    from src.model import IrisClassifier
    from src.data_loader import load_iris_data
    print('✅ Modules du projet importables')
except ImportError as e:
    print(f'❌ Erreur d\\'import du projet: {e}')
"
    
    log_success "Installation testée avec succès"
}

# Affichage des informations finales
show_final_info() {
    log_success "Configuration terminée avec succès!"
    echo ""
    echo "🚀 Commandes utiles:"
    echo "  • Activer l'environnement: source venv/bin/activate"
    echo "  • Entraîner le modèle: make train"
    echo "  • Lancer l'API: make api"
    echo "  • Lancer les tests: make test"
    echo "  • Interface MLflow: mlflow ui"
    echo "  • Lancer Jupyter: make run-notebook"
    echo ""
    echo "📚 Documentation:"
    echo "  • README: cat README.md"
    echo "  • API docs: http://localhost:8000/docs (après make api)"
    echo "  • MLflow UI: http://localhost:5000 (après mlflow ui)"
    echo ""
    echo "🐳 Docker:"
    echo "  • Build: docker-compose build"
    echo "  • Lancer: docker-compose up"
    echo "  • Mode dev: docker-compose --profile dev up"
}

# Fonction principale
main() {
    echo "🔧 Configuration de l'environnement de développement ML"
    echo "=================================================="
    echo ""
    
    # Vérifier que nous sommes dans le bon répertoire
    if [ ! -f "pyproject.toml" ]; then
        log_error "Ce script doit être exécuté depuis la racine du projet"
        exit 1
    fi
    
    # Exécuter toutes les étapes
    check_prerequisites
    setup_virtual_environment
    install_dependencies
    setup_pre_commit
    setup_mlflow
    create_directories
    test_installation
    show_final_info
}

# Gestion des arguments de ligne de commande
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Afficher cette aide"
        echo "  --skip-tests   Ignorer les tests d'installation"
        echo "  --minimal      Installation minimale (sans outils de dev)"
        echo ""
        exit 0
        ;;
    --skip-tests)
        SKIP_TESTS=true
        ;;
    --minimal)
        MINIMAL_INSTALL=true
        ;;
esac

# Exécuter le script principal
main