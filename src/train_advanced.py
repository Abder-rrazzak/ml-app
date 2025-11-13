"""
Script d'entraînement avancé avec tracking MLflow.

Ce script entraîne le modèle de classification Iris avec :
- Tracking complet des expériences via MLflow
- Validation croisée pour une évaluation robuste
- Sauvegarde automatique des artefacts
- Logging détaillé des métriques et paramètres
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

import click
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .data_loader import load_iris_data, split_data
from .model import IrisClassifier
from .mlflow_tracking import MLflowTracker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_mlflow(
    model_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
    experiment_name: str = "iris-classification"
) -> Dict[str, Any]:
    """
    Entraîne le modèle avec tracking MLflow complet.
    
    Args:
        model_path: Chemin pour sauvegarder le modèle
        random_state: Graine aléatoire pour la reproductibilité
        test_size: Proportion des données pour le test
        cv_folds: Nombre de folds pour la validation croisée
        experiment_name: Nom de l'expérience MLflow
        
    Returns:
        Dictionnaire avec les résultats de l'entraînement
    """
    # Initialiser le tracker MLflow
    tracker = MLflowTracker(experiment_name)
    
    # Générer un nom unique pour cette run
    run_name = f"iris_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Démarrer la run MLflow
        tracker.start_run(
            run_name=run_name,
            tags={
                "model_type": "LogisticRegression",
                "dataset": "iris",
                "training_type": "full_pipeline"
            }
        )
        
        logger.info("=== Début de l'entraînement avec MLflow ===")
        
        # 1. Chargement des données
        logger.info("Chargement du dataset Iris...")
        df, target_names = load_iris_data()
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 2. Division des données
        logger.info("Division des données train/test...")
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Logger les informations sur le dataset
        tracker.log_dataset_info(X_train, X_test, y_train, y_test)
        
        # 3. Initialisation et configuration du modèle
        logger.info("Initialisation du modèle...")
        classifier = IrisClassifier(random_state=random_state)
        
        # Logger les paramètres du modèle
        model_params = {
            "random_state": random_state,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "max_iter": 200,  # Paramètre du LogisticRegression
            "solver": "lbfgs"  # Solveur par défaut
        }
        
        # Ajouter les paramètres spécifiques du modèle sklearn
        if hasattr(classifier.model, 'get_params'):
            sklearn_params = classifier.model.get_params()
            model_params.update({f"sklearn_{k}": v for k, v in sklearn_params.items()})
        
        tracker.log_parameters(model_params)
        
        # 4. Validation croisée avant l'entraînement final
        logger.info(f"Validation croisée avec {cv_folds} folds...")
        
        # Utiliser StratifiedKFold pour maintenir la distribution des classes
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Scores de validation croisée
        cv_scores = cross_val_score(
            classifier.model, X_train, y_train, 
            cv=cv_strategy, scoring='accuracy'
        )
        
        # Logger les résultats de la validation croisée
        cv_metrics = {
            "cv_mean_accuracy": np.mean(cv_scores),
            "cv_std_accuracy": np.std(cv_scores),
            "cv_min_accuracy": np.min(cv_scores),
            "cv_max_accuracy": np.max(cv_scores)
        }
        
        tracker.log_metrics(cv_metrics)
        
        logger.info(f"Validation croisée - Accuracy: {cv_metrics['cv_mean_accuracy']:.4f} "
                   f"(±{cv_metrics['cv_std_accuracy']:.4f})")
        
        # 5. Entraînement final sur toutes les données d'entraînement
        logger.info("Entraînement du modèle final...")
        classifier.train(X_train, y_train)
        
        # 6. Évaluation sur les données de test
        logger.info("Évaluation sur les données de test...")
        accuracy, report = classifier.evaluate(X_test, y_test)
        
        # Logger les performances détaillées
        tracker.log_model_performance(
            classifier.model, X_test, y_test, class_names=target_names
        )
        
        # 7. Sauvegarde du modèle
        logger.info(f"Sauvegarde du modèle vers {model_path}...")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarder avec joblib (format standard)
        classifier.save(model_path)
        
        # Sauvegarder comme artefact MLflow
        tracker.log_model_artifact(classifier.model, "iris_classifier")
        
        # 8. Logger la version du code
        tracker.log_code_version()
        
        # 9. Métriques finales
        final_metrics = {
            "final_test_accuracy": accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_features": X_train.shape[1],
            "total_classes": len(target_names)
        }
        
        tracker.log_metrics(final_metrics)
        
        # 10. Résultats de l'entraînement
        results = {
            "model_path": model_path,
            "test_accuracy": accuracy,
            "cv_accuracy": cv_metrics['cv_mean_accuracy'],
            "cv_std": cv_metrics['cv_std_accuracy'],
            "classification_report": report,
            "run_id": None
        }
        
        logger.info("=== Entraînement terminé avec succès ===")
        logger.info(f"Accuracy finale: {accuracy:.4f}")
        logger.info(f"Validation croisée: {cv_metrics['cv_mean_accuracy']:.4f} ±{cv_metrics['cv_std_accuracy']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur during l'entraînement : {e}")
        raise
        
    finally:
        # Toujours fermer la run MLflow
        tracker.end_run()


@click.command()
@click.option(
    '--model-path',
    default='models/iris_model.pkl',
    help='Chemin pour sauvegarder le modèle entraîné'
)
@click.option(
    '--random-state',
    default=42,
    type=int,
    help='Graine aléatoire pour la reproductibilité'
)
@click.option(
    '--test-size',
    default=0.2,
    type=float,
    help='Proportion des données pour le test (0.0-1.0)'
)
@click.option(
    '--cv-folds',
    default=5,
    type=int,
    help='Nombre de folds pour la validation croisée'
)
@click.option(
    '--experiment-name',
    default='iris-classification',
    help='Nom de l\'expérience MLflow'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Affichage détaillé des logs'
)
def train_advanced(model_path, random_state, test_size, cv_folds, experiment_name, verbose):
    """
    Entraîne le modèle de classification Iris avec tracking MLflow.
    
    Ce script effectue un entraînement complet avec :
    - Validation croisée pour évaluer la robustesse
    - Tracking MLflow de tous les paramètres et métriques
    - Sauvegarde automatique du modèle et des artefacts
    - Logging détaillé pour le debugging
    """
    try:
        # Configuration du niveau de logging
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Mode verbose activé")
        
        # Validation des paramètres
        if not (0.0 < test_size < 1.0):
            raise click.BadParameter("test-size doit être entre 0.0 et 1.0")
        
        if cv_folds < 2:
            raise click.BadParameter("cv-folds doit être >= 2")
        
        # Affichage des paramètres
        click.echo("=== Configuration de l'entraînement ===")
        click.echo(f"Modèle: {model_path}")
        click.echo(f"Random state: {random_state}")
        click.echo(f"Test size: {test_size}")
        click.echo(f"CV folds: {cv_folds}")
        click.echo(f"Expérience MLflow: {experiment_name}")
        click.echo("")
        
        # Lancement de l'entraînement
        results = train_with_mlflow(
            model_path=model_path,
            random_state=random_state,
            test_size=test_size,
            cv_folds=cv_folds,
            experiment_name=experiment_name
        )
        
        # Affichage des résultats
        click.echo("=== Résultats de l'entraînement ===")
        click.echo(f"✅ Modèle sauvegardé: {results['model_path']}")
        click.echo(f"📊 Accuracy test: {results['test_accuracy']:.4f}")
        click.echo(f"🔄 Validation croisée: {results['cv_accuracy']:.4f} ±{results['cv_std']:.4f}")
        
        click.echo("\n📈 Rapport de classification détaillé:")
        click.echo(results['classification_report'])
        
        # Conseils pour la suite
        click.echo("\n💡 Prochaines étapes:")
        click.echo("• Visualiser les résultats: mlflow ui")
        click.echo("• Tester l'API: make api")
        click.echo("• Faire des prédictions: make predict")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {e}")
        click.echo(f"❌ Erreur: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    train_advanced()