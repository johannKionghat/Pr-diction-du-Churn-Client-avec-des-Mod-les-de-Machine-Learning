import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

train_df = pd.read_parquet("../data/models/train.parquet")
test_df  = pd.read_parquet("../data/models/test.parquet")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]

X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

def main():

    # Régression logistique
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

    # Evaluation
    print(classification_report(y_test, y_pred_lr))
    print("AUC:", roc_auc_score(y_test, y_proba_lr))


    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
    plt.plot(fpr, tpr, label="Logistic Regression (AUC=%.3f)" % roc_auc_score(y_test, y_proba_lr))
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


    # Arbre de décision
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_dt))
    print("AUC:", roc_auc_score(y_test, y_proba_dt))

    # Importance des features
    feat_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Features (Decision Tree)")
    plt.show()
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred_rf))
    print("AUC:", roc_auc_score(y_test, y_proba_rf))

    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Features (Random Forest)")
    plt.show()
    
    # Comparaison des modèles
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [
            (y_pred_lr == y_test).mean(),
            (y_pred_dt == y_test).mean(),
            (y_pred_rf == y_test).mean()
        ],
        "AUC": [
            roc_auc_score(y_test, y_proba_lr),
            roc_auc_score(y_test, y_proba_dt),
            roc_auc_score(y_test, y_proba_rf)
        ]
    })

    print(results)

    # Sauvegarde des modèles
    os.makedirs("../models", exist_ok=True)
    joblib.dump(log_reg, "../models/log_reg.pkl")
    joblib.dump(dt, "../models/decision_tree.pkl")
    joblib.dump(rf, "../models/random_forest.pkl")

    print("Modèles sauvegardés dans ../models")

if __name__ == "__main__":
    main()








    
