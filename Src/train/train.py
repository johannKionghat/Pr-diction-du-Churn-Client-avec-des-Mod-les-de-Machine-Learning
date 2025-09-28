import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration des chemins (ancrés sur la racine du projet)
# __file__ -> .../src/train/train.py
# parents[0] = .../src/train
# parents[1] = .../src
# parents[2] = .../projet3  (racine)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models_data"  # dossiers d'entrée (train/test) et artefacts
MODELS_SAVE_DIR = PROJECT_ROOT / "models"  # dossiers de sortie pour les modèles (.pkl)
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Création des répertoires si nécessaire
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

# Chemins des fichiers d'entrée
train_path = MODELS_DIR / "train.parquet"
test_path = MODELS_DIR / "test.parquet"

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

def save_model_plot(plt_obj, filename, dpi=300):
    """Sauvegarde un graphique dans le dossier des artefacts."""
    plot_path = ARTIFACTS_DIR / filename
    plt_obj.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 Graphique sauvegardé: {plot_path}")
    plt.close()

def evaluate_model(model, X, y_true, model_name):
    """Évalue un modèle et retourne les métriques."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Métriques de classification
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC si disponible
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    # Affichage du rapport
    print(f"\n{'-'*50}")
    print(f"📊 ÉVALUATION DU MODÈLE: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
    if 'auc' in metrics:
        print(f"AUC: {metrics['auc']:.4f}")
    
    # Courbe ROC si disponible
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        save_model_plot(plt, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
    
    return metrics

def train_and_evaluate_models():
    """Entraîne et évalue plusieurs modèles de classification."""
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DES MODÈLES")
    print(f"📊 Dimensions des données - Entraînement: {X_train.shape}, Test: {X_test.shape}")
    print("="*80 + "\n")
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # 1. Régression logistique
    print("\n" + "="*50)
    print("ENTRAÎNEMENT: RÉGRESSION LOGISTIQUE")
    print("="*50)
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    log_reg.fit(X_train, y_train)
    results['logistic_regression'] = evaluate_model(log_reg, X_test, y_test, "Régression Logistique")
    
    # 2. Arbre de décision
    print("\n" + "="*50)
    print("ENTRAÎNEMENT: ARBRE DE DÉCISION")
    print("="*50)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    results['decision_tree'] = evaluate_model(dt, X_test, y_test, "Arbre de Décision")
    
    # Importance des caractéristiques pour l'arbre de décision
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title('Importance des caractéristiques - Arbre de Décision')
    save_model_plot(plt, "feature_importance_decision_tree.png")
    
    # 3. Forêt aléatoire
    print("\n" + "="*50)
    print("ENTRAÎNEMENT: Random Forest")
    print("="*50)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    results['random_forest'] = evaluate_model(rf, X_test, y_test, "Forêt Aléatoire")
    
    # Importance des caractéristiques pour la forêt aléatoire
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title('Importance des caractéristiques - Forêt Aléatoire')
    save_model_plot(plt, "feature_importance_random_forest.png")
    
    # Affichage des résultats
    print("\n" + "="*80)
    print("📈 RÉCAPITULATIF DES PERFORMANCES")
    print("="*80)
    results_df = pd.DataFrame(results).T
    print("\nMétriques sur l'ensemble de test:")
    print("-" * 50)
    print(results_df)
    
    # Sauvegarde des modèles
    print("\n" + "="*50)
    print("💾 SAUVEGARDE DES MODÈLES")
    print("="*50)
    
    models = {
        'logistic_regression': log_reg,
        'decision_tree': dt,
        'random_forest': rf
    }
    
    for name, model in models.items():
        model_path = MODELS_SAVE_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")
    
    # Sauvegarde des résultats
    results_path = ARTIFACTS_DIR / "training_results.csv"
    results_df.to_csv(results_path)
    print(f"\n📊 Résultats sauvegardés: {results_path}")
    
    return models

def main():
    try:
        models = train_and_evaluate_models()
        print("\n" + "="*80)
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        print("="*80)
        print(f"📂 Modèles sauvegardés dans: {MODELS_SAVE_DIR.absolute()}")
        print(f"📊 Résultats et graphiques: {ARTIFACTS_DIR.absolute()}")
        print("="*80 + "\n")
        return models
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()








    
