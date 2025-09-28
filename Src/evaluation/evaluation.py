import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Configuration des chemins (ancrée sur la racine du projet)
# __file__ -> .../src/evaluation/evaluation.py
# parents[0] = .../src/evaluation
# parents[1] = .../src
# parents[2] = .../projet3 (racine)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_INPUT_DIR = DATA_DIR / "models_data"  # là où se trouvent train/test
MODELS_DIR = PROJECT_ROOT / "models"        # là où sont sauvegardés les .pkl
ARTIFACTS_DIR = MODELS_INPUT_DIR / "artifacts"

# Création des répertoires si nécessaire
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Chemins des fichiers
train_path = MODELS_INPUT_DIR / "train.parquet"
test_path  = MODELS_INPUT_DIR / "test.parquet"

# Chargement des modèles
try:
    logreg = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    dtree = joblib.load(MODELS_DIR / "decision_tree.pkl")
    rf    = joblib.load(MODELS_DIR / "random_forest.pkl")
    print("✅ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {str(e)}")
    sys.exit(1)

# Chargement des données
try:
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    
    print(f"✅ Données chargées avec succès depuis {train_path} et {test_path}")
    print(f"📊 Dimensions - Train: {train_df.shape}, Test: {test_df.shape}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement des données: {str(e)}")
    sys.exit(1)

def save_plot(plt_obj, filename, dpi=300):
    """Sauvegarde un graphique dans le dossier des artefacts."""
    plot_path = ARTIFACTS_DIR / filename
    plt_obj.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {plot_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Affiche et sauvegarde une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Matrice de Confusion - {model_name}')
    save_plot(plt, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')

def plot_roc_curve(y_true, y_proba, model_name):
    """Affiche et sauvegarde une courbe ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend(loc="lower right")
    save_plot(plt, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')

def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle avec différentes métriques et génère des visualisations.
    
    Args:
        model: Le modèle entraîné à évaluer
        X_test: Features de test
        y_test: Labels vrais
        model_name: Nom du modèle pour l'affichage
        
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
    """
    print(f"\n{'='*60}")
    print(f"📊 ÉVALUATION DU MODÈLE: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Probabilités (si disponibles)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques de classification
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Ajout de l'AUC si disponible
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_test, y_proba)
    
    # Affichage du rapport de classification
    print("\nRapport de classification:")
    print("-" * 50)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Affichage des métriques
    print("\nMétriques d'évaluation:")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")
    
    # Visualisations
    if y_proba is not None:
        plot_roc_curve(y_test, y_proba, model_name)
    
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return metrics

def optimize_random_forest(X_train, y_train, X_test, y_test):
    """Optimise un modèle Random Forest avec GridSearchCV."""
    print("\n" + "="*60)
    print("🔍 OPTIMISATION: RANDOM FOREST")
    print("="*60)
    
    # Grille d'hyperparamètres
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Création du modèle
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Recherche par grille
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    print("Démarrage de la recherche par grille...")
    grid_search.fit(X_train, y_train)
    
    # Meilleurs paramètres
    print("\nMeilleurs paramètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Évaluation du meilleur modèle
    best_rf = grid_search.best_estimator_
    metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest Optimisé")
    
    # Sauvegarde du modèle optimisé
    model_path = MODELS_DIR / "random_forest_optimized.pkl"
    joblib.dump(best_rf, model_path)
    print(f"\n✅ Modèle optimisé sauvegardé: {model_path}")
    
    return best_rf, metrics

def optimize_logistic_regression(X_train, y_train, X_test, y_test):
    """Optimise un modèle de régression logistique avec GridSearchCV."""
    print("\n" + "="*60)
    print("🔍 OPTIMISATION: RÉGRESSION LOGISTIQUE")
    print("="*60)
    
    # Création du pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
    ])
    
    # Grille d'hyperparamètres
    param_grid = [
        {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__class_weight': ['balanced', None]
        },
        {
            'classifier__penalty': ['l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'newton-cg', 'sag'],
            'classifier__class_weight': ['balanced', None]
        }
    ]
    
    # Recherche par grille
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    print("Démarrage de la recherche par grille...")
    grid_search.fit(X_train, y_train)
    
    # Meilleurs paramètres
    print("\nMeilleurs paramètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Évaluation du meilleur modèle
    best_lr = grid_search.best_estimator_
    metrics = evaluate_model(best_lr, X_test, y_test, "Régression Logistique Optimisée")
    
    # Sauvegarde du modèle optimisé
    model_path = MODELS_DIR / "logistic_regression_optimized.pkl"
    joblib.dump(best_lr, model_path)
    print(f"\n✅ Modèle optimisé sauvegardé: {model_path}")
    
    return best_lr, metrics

def optimize_decision_tree(X_train, y_train, X_test, y_test):
    """Optimise un arbre de décision avec GridSearchCV."""
    print("\n" + "="*60)
    print("🔍 OPTIMISATION: ARBRE DE DÉCISION")
    print("="*60)
    
    # Grille d'hyperparamètres
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', None]
    }
    
    # Création du modèle
    dt = DecisionTreeClassifier(random_state=42)
    
    # Recherche par grille
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    print("Démarrage de la recherche par grille...")
    grid_search.fit(X_train, y_train)
    
    # Meilleurs paramètres
    print("\nMeilleurs paramètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Évaluation du meilleur modèle
    best_dt = grid_search.best_estimator_
    metrics = evaluate_model(best_dt, X_test, y_test, "Arbre de Décision Optimisé")
    
    # Visualisation de l'arbre (pour les petits arbres)
    if best_dt.tree_.max_depth <= 5:  # On ne visualise que les arbres pas trop profonds
        try:
            from sklearn.tree import plot_tree
            plt.figure(figsize=(20, 10))
            plot_tree(best_dt, feature_names=X_train.columns, 
                     class_names=['No Churn', 'Churn'],
                     filled=True, rounded=True)
            save_plot(plt, 'decision_tree_structure.png')
        except Exception as e:
            print(f"⚠️ Impossible de visualiser l'arbre: {str(e)}")
    
    # Sauvegarde du modèle optimisé
    model_path = MODELS_DIR / "decision_tree_optimized.pkl"
    joblib.dump(best_dt, model_path)
    print(f"\n✅ Modèle optimisé sauvegardé: {model_path}")
    
    return best_dt, metrics

def main():
    print("\n" + "="*80)
    print("🎯 DÉMARRAGE DE L'ÉVALUATION ET DE L'OPTIMISATION")
    print("="*80)
    
    # Évaluation des modèles de base
    print("\n" + "="*60)
    print("📈 ÉVALUATION DES MODÈLES DE BASE")
    print("="*60)
    
    models = {
        "Régression Logistique": logreg,
        "Arbre de Décision": dtree,
        "Forêt Aléatoire": rf
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Affichage des résultats
    print("\n" + "="*80)
    print("📊 RÉCAPITULATIF DES PERFORMANCES (MODÈLES DE BASE)")
    print("="*80)
    results_df = pd.DataFrame(results).T
    print("\nMétriques sur l'ensemble de test:")
    print("-" * 50)
    print(results_df)
    
    # Sauvegarde des résultats
    results_path = ARTIFACTS_DIR / "base_models_results.csv"
    results_df.to_csv(results_path)
    print(f"\n📊 Résultats sauvegardés: {results_path}")
    
    # Optimisation des modèles
    print("\n" + "="*80)
    print("⚙️  DÉMARRAGE DE L'OPTIMISATION DES MODÈLES")
    print("="*80)
    
    optimized_results = {}
    
    # 1. Optimisation de la Forêt Aléatoire
    best_rf, rf_metrics = optimize_random_forest(X_train, y_train, X_test, y_test)
    optimized_results["Random Forest Optimisé"] = rf_metrics
    
    # 2. Optimisation de la Régression Logistique
    best_lr, lr_metrics = optimize_logistic_regression(X_train, y_train, X_test, y_test)
    optimized_results["Régression Logistique Optimisée"] = lr_metrics
    
    # 3. Optimisation de l'Arbre de Décision
    best_dt, dt_metrics = optimize_decision_tree(X_train, y_train, X_test, y_test)
    optimized_results["Arbre de Décision Optimisé"] = dt_metrics
    
    # Affichage des résultats optimisés
    print("\n" + "="*80)
    print("🏆 RÉCAPITULATIF DES PERFORMANCES (MODÈLES OPTIMISÉS)")
    print("="*80)
    optimized_results_df = pd.DataFrame(optimized_results).T
    print("\nMétriques sur l'ensemble de test:")
    print("-" * 50)
    print(optimized_results_df)
    
    # Sauvegarde des résultats optimisés
    optimized_results_path = ARTIFACTS_DIR / "optimized_models_results.csv"
    optimized_results_df.to_csv(optimized_results_path)
    print(f"\n📊 Résultats optimisés sauvegardés: {optimized_results_path}")
    
    # Comparaison avant/après optimisation
    print("\n" + "="*80)
    print("🔄 COMPARAISON AVANT/APRÈS OPTIMISATION")
    print("="*80)
    
    # Création d'un DataFrame combiné
    all_results = pd.concat([results_df, optimized_results_df])
    
    # Tri par AUC (si disponible) ou par F1
    sort_by = 'auc' if 'auc' in all_results.columns else 'f1'
    all_results = all_results.sort_values(by=sort_by, ascending=False)
    
    print("\nClassement des modèles (du meilleur au moins bon):")
    print("-" * 70)
    print(all_results[[sort_by]].round(4).sort_values(by=sort_by, ascending=False))
    
    # Sauvegarde de la comparaison
    comparison_path = ARTIFACTS_DIR / "model_comparison.csv"
    all_results.to_csv(comparison_path)
    print(f"\n📊 Comparaison des modèles sauvegardée: {comparison_path}")
    
    # Fin du script
    print("\n" + "="*80)
    print("🎉 ÉVALUATION ET OPTIMISATION TERMINÉES AVEC SUCCÈS !")
    print("="*80)
    print(f"📂 Modèles optimisés sauvegardés dans: {MODELS_DIR.absolute()}")
    print(f"📊 Résultats et graphiques: {ARTIFACTS_DIR.absolute()}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
