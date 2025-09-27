import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = "../models"

# Load pre-trained models
logreg = joblib.load(os.path.join(MODELS_DIR, "log_reg.pkl"))
dtree = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

# Load data
train_df = pd.read_parquet("../data/models/train.parquet")
test_df  = pd.read_parquet("../data/models/test.parquet")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]

X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model's performance using various metrics.
    
    Args:
        model: The trained model to evaluate
        X_test: Test features
        y_test: True labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score
    )
    
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Determine pos_label based on data type
    pos_label = 1 if y_test.unique().dtype in ["int64", "int32"] or set(y_test.unique()) == {0, 1} else "Yes"
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
        "F1": f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
    }
    
    # Add ROC AUC if probabilities are available
    if y_proba is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, y_proba)
    
    return metrics

def main():

    # --- Comparaison des modèles ---
    results = {}
    results["Logistic Regression"] = evaluate_model(logreg, X_test, y_test)
    results["Decision Tree"] = evaluate_model(dtree, X_test, y_test)
    results["Random Forest"] = evaluate_model(rf, X_test, y_test)

    results_df = pd.DataFrame(results).T
    print(results_df)

    # Optimisation des hyperparamètres
    # on va faire un GridSearchCV pour chaque modèle.
    # Avec RandoForest:

    # --- Grille d’hyperparamètres pour RandomForest ---
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    # --- GridSearch sur l’ensemble d’entraînement ---
    from sklearn.model_selection import GridSearchCV
    
    # Create a fresh Random Forest model for grid search
    rf_for_grid = RandomForestClassifier(random_state=42)
    
    grid_search_rf = GridSearchCV(
        estimator=rf_for_grid,
        param_grid=param_grid,
        scoring="roc_auc",  # on optimise l'AUC
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search_rf.fit(X_train, y_train)

    # --- Meilleurs hyperparamètres trouvés ---
    print("Best params RF:", grid_search_rf.best_params_)
    print("Best AUC (CV) RF:", grid_search_rf.best_score_)

    # --- Évaluation finale sur le jeu de test ---
    best_rf = grid_search_rf.best_estimator_

    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    from sklearn.metrics import classification_report, roc_auc_score
    print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))
    print("Final ROC AUC on Test:", roc_auc_score(y_test, y_proba))

    # --- Sauvegarde du modèle optimisé ---
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "random_forest_optimized.pkl"))

    # Avec LogisticRegression
    param_grid = [
        # liblinear
        {"logreg__penalty": ["l1", "l2"],
        "logreg__C": [0.01, 0.1, 1, 10, 100],
        "logreg__solver": ["liblinear"],
        "logreg__max_iter": [100, 500, 1000]},
        
        # lbfgs
        {"logreg__penalty": ["l2", None],
        "logreg__C": [0.01, 0.1, 1, 10, 100],
        "logreg__solver": ["lbfgs"],
        "logreg__max_iter": [100, 500, 1000]},
        
        # saga
        {"logreg__penalty": ["l1", "l2", None],
        "logreg__C": [0.01, 0.1, 1, 10, 100],
        "logreg__solver": ["saga"],
        "logreg__max_iter": [100, 500, 1000]},
        
        # elasticnet avec saga (il faut l1_ratio)
        {"logreg__penalty": ["elasticnet"],
        "logreg__C": [0.01, 0.1, 1, 10, 100],
        "logreg__solver": ["saga"],
        "logreg__l1_ratio": [0.1, 0.5, 0.7, 0.9],
        "logreg__max_iter": [500, 1000]}
    ]


    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(random_state=42))
    ])

    grid_search_logreg = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search_logreg.fit(X_train, y_train)

    best_logreg = grid_search_logreg.best_estimator_
    print("Best params Logistic Regression:", grid_search_logreg.best_params_)
    print("Best AUC Logistic Regression:", grid_search_logreg.best_score_)

    joblib.dump(best_logreg, os.path.join(MODELS_DIR, "logreg_optimized.pkl"))

    # Avec DecisionTreeClassifier:
    from sklearn.model_selection import GridSearchCV


    # Grille réaliste et efficace
    param_grid = {
        "max_depth": [3, 5, 7, 10, 15],          # Limite la profondeur pour éviter l'overfitting
        "min_samples_split": [2, 5, 10, 20],     # Nombre minimum d'échantillons pour diviser un nœud
        "min_samples_leaf": [1, 2, 4, 10],       # Nombre minimum d'échantillons dans une feuille
        "criterion": ["gini", "entropy"]         # Critère de séparation
    }

    grid_search_dtree = GridSearchCV(
        estimator=dtree,
        param_grid=param_grid,
        scoring="roc_auc",   # On optimise l'AUC
        cv=5,                # 5-fold cross validation
        n_jobs=-1,           # utiliser tous les cœurs
        verbose=2
    )

    grid_search_dtree.fit(X_train, y_train)

    best_dtree = grid_search_dtree.best_estimator_
    print("Best params Decision Tree:", grid_search_dtree.best_params_)
    print("Best AUC Decision Tree:", grid_search_dtree.best_score_)

    joblib.dump(best_dtree, os.path.join(MODELS_DIR, "decision_tree_optimized.pkl"))

    # Comparaison finale & justification
    # tableau avant vs après optimisation :
    results_opt = {}
    results_opt["Random Forest Optimized"] = evaluate_model(best_rf, X_test, y_test)
    results_opt["Logistic Regression Optimized"] = evaluate_model(best_logreg, X_test, y_test)
    results_opt["Decision Tree Optimized"] = evaluate_model(best_dtree, X_test, y_test)


    final_results = pd.concat([results_df, pd.DataFrame(results_opt).T])
    print(final_results)

    # Graphique comparatif des modèles pour visualiser leurs performances en termes de ROC AUC et F1 score :
    # Noms des modèles
    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Random Forest Optimized",
        "Logistic Regression Optimized",
        "Decision Tree Optimized"
    ]

    # F1 scores
    f1_scores = [0.606061, 0.598870, 0.572280, 0.580451, 0.602305, 0.598870]

    # ROC AUC scores
    roc_auc_scores = [0.841918, 0.830286, 0.840162, 0.842345, 0.841158, 0.830286]

    x = np.arange(len(models))  # positions des modèles
    width = 0.35  # largeur des barres

    fig, ax = plt.subplots(figsize=(12,6))

    # Barres F1
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='skyblue')

    # Barres ROC AUC
    bars2 = ax.bar(x + width/2, roc_auc_scores, width, label='ROC AUC', color='salmon')

    # Titres et labels
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des modèles pour prédire le churn')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    # Ajouter les valeurs sur chaque barre
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

