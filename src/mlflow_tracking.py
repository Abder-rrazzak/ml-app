"""
Module MLflow pour le tracking des expériences ML.

Ce module intègre MLflow pour suivre :
- Les paramètres des modèles
- Les métriques de performance
- Les artefacts (modèles, graphiques)
- Les versions des datasets
"""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration du logging
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Classe pour gérer le tracking MLflow des expériences ML.

    Centralise toutes les opérations MLflow pour maintenir
    la cohérence et faciliter la maintenance.
    """

    def __init__(self, experiment_name: str = "iris-classification"):
        """
        Initialise le tracker MLflow.

        Args:
            experiment_name: Nom de l'expérience MLflow
        """
        self.experiment_name = experiment_name

        # Configuration MLflow depuis les variables d'environnement
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Créer ou récupérer l'expérience
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(
                    f"Expérience créée : {experiment_name} (ID: {experiment_id})"
                )
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Expérience existante : {experiment_name} (ID: {experiment_id})"
                )

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Erreur lors de la configuration MLflow : {e}")
            raise

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """
        Démarre une nouvelle run MLflow.

        Args:
            run_name: Nom optionnel pour la run
            tags: Tags optionnels pour la run
        """
        # Tags par défaut
        default_tags = {
            "model_type": "LogisticRegression",
            "dataset": "iris",
            "framework": "scikit-learn",
        }

        if tags:
            default_tags.update(tags)

        # Démarrer la run
        mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Run MLflow démarrée : {mlflow.active_run().info.run_id}")

    def log_parameters(self, params: Dict[str, Any]):
        """
        Log les paramètres du modèle.

        Args:
            params: Dictionnaire des paramètres à logger
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"Paramètres loggés : {list(params.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du logging des paramètres : {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log les métriques de performance.

        Args:
            metrics: Dictionnaire des métriques à logger
            step: Étape optionnelle (pour les métriques temporelles)
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"Métriques loggées : {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du logging des métriques : {e}")

    def log_model_performance(self, model, X_test, y_test, class_names=None):
        """
        Log les métriques de performance complètes du modèle.

        Args:
            model: Modèle entraîné
            X_test: Données de test
            y_test: Labels de test
            class_names: Noms des classes (optionnel)
        """
        try:
            # Prédictions
            y_pred = model.predict(X_test)

            # Métriques de base
            accuracy = accuracy_score(y_test, y_pred)

            # Logger les métriques
            metrics = {"accuracy": accuracy, "test_samples": len(y_test)}

            # Ajouter les métriques par classe si possible
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                # Log des probabilités moyennes par classe
                for i, class_name in enumerate(class_names or range(y_proba.shape[1])):
                    metrics[f"mean_proba_class_{class_name}"] = np.mean(y_proba[:, i])

            self.log_metrics(metrics)

            # Rapport de classification détaillé
            report = classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True
            )

            # Logger les métriques par classe
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict):
                    for metric_name, value in class_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{class_name}_{metric_name}", value)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)

            # Sauvegarder la matrice de confusion comme artefact
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)),
            )
            plt.title("Matrice de Confusion")
            plt.ylabel("Vraie Classe")
            plt.xlabel("Classe Prédite")

            # Sauvegarder le graphique
            confusion_matrix_path = "confusion_matrix.png"
            plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
            mlflow.log_artifact(confusion_matrix_path)
            plt.close()

            # Nettoyer le fichier temporaire
            if os.path.exists(confusion_matrix_path):
                os.remove(confusion_matrix_path)

            logger.info("Métriques de performance loggées avec succès")

        except Exception as e:
            logger.error(f"Erreur lors du logging des performances : {e}")

    def log_dataset_info(self, X_train, X_test, y_train, y_test):
        """
        Log les informations sur le dataset.

        Args:
            X_train, X_test, y_train, y_test: Données d'entraînement et de test
        """
        try:
            # Informations sur les données
            dataset_info = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1],
                "classes": len(np.unique(y_train)),
                "train_test_ratio": len(X_train) / (len(X_train) + len(X_test)),
            }

            # Distribution des classes
            unique, counts = np.unique(y_train, return_counts=True)
            for class_idx, count in zip(unique, counts):
                dataset_info[f"train_class_{class_idx}_count"] = count

            unique, counts = np.unique(y_test, return_counts=True)
            for class_idx, count in zip(unique, counts):
                dataset_info[f"test_class_{class_idx}_count"] = count

            self.log_parameters(dataset_info)
            logger.info("Informations dataset loggées")

        except Exception as e:
            logger.error(f"Erreur lors du logging du dataset : {e}")

    def log_model_artifact(self, model, model_name: str = "iris_classifier"):
        """
        Log le modèle comme artefact MLflow.

        Args:
            model: Modèle à sauvegarder
            model_name: Nom du modèle
        """
        try:
            # Sauvegarder le modèle avec MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                registered_model_name=model_name,
                signature=None,  # Peut être ajouté pour la validation des entrées
                input_example=None,  # Exemple d'entrée pour la documentation
            )

            logger.info(f"Modèle {model_name} sauvegardé comme artefact")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")

    def log_code_version(self):
        """
        Log la version du code (commit Git si disponible).
        """
        try:
            import subprocess

            # Récupérer le hash du commit Git
            try:
                git_commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    )
                    .decode("utf-8")
                    .strip()
                )

                mlflow.log_param("git_commit", git_commit)

                # Récupérer la branche Git
                git_branch = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode("utf-8")
                    .strip()
                )

                mlflow.log_param("git_branch", git_branch)

                logger.info(f"Version du code loggée : {git_commit[:8]} ({git_branch})")

            except subprocess.CalledProcessError:
                logger.warning("Impossible de récupérer les informations Git")

        except Exception as e:
            logger.error(f"Erreur lors du logging de la version : {e}")

    def end_run(self):
        """
        Termine la run MLflow active.
        """
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("Run MLflow terminée")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la run : {e}")

    def get_best_model(self, metric_name: str = "accuracy", ascending: bool = False):
        """
        Récupère le meilleur modèle basé sur une métrique.

        Args:
            metric_name: Nom de la métrique pour le classement
            ascending: True pour ordre croissant, False pour décroissant

        Returns:
            Informations sur la meilleure run
        """
        try:
            # Rechercher les runs de l'expérience
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            )

            if len(runs) > 0:
                best_run = runs.iloc[0]
                logger.info(
                    f"Meilleure run trouvée : {best_run['run_id']} "
                    f"({metric_name}: {best_run[f'metrics.{metric_name}']})"
                )
                return best_run
            else:
                logger.warning("Aucune run trouvée dans l'expérience")
                return None

        except Exception as e:
            logger.error(f"Erreur lors de la recherche du meilleur modèle : {e}")
            return None
