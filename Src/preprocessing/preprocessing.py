# -----------------------
# Préparation finale des données (train/test/result)
# -----------------------
import os
import json
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# --- Configuration des chemins ---
from pathlib import Path

# Chemins de base
# __file__ -> .../src/preprocessing/preprocessing.py
# parents[0] = .../src/preprocessing
# parents[1] = .../src
# parents[2] = .../projet3  (racine du projet)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"  # data/ à la racine du projet
MODELS_DIR = DATA_DIR / "models_data"  # DÉPLACÉ: sous data/models
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Création des répertoires si nécessaire
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[INFO] MODELS_DIR: {MODELS_DIR}")

# --- Paramètres ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"

# Chargement des données (robuste)
processed_parquet = PROCESSED_DIR / "telco_customer_churn_processed.parquet"
processed_csv = PROCESSED_DIR / "telco_customer_churn_processed.csv"
print(f"[INFO] Emplacement attendu (parquet) : {processed_parquet}")
print(f"[INFO] Emplacement alternatif (csv)   : {processed_csv}")

if processed_parquet.exists():
    df_original = pd.read_parquet(processed_parquet)
elif processed_csv.exists():
    print("[WARN] Parquet introuvable, chargement du CSV alternatif...")
    df_original = pd.read_csv(processed_csv)
else:
    raise FileNotFoundError(
        f"Fichiers introuvables:\n - {processed_parquet}\n - {processed_csv}\n"
        "Assurez-vous d'avoir exécuté le pipeline ETL ou de placer le fichier au bon emplacement."
    )
df = df_original.copy()

def main():
   
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} absent du dataframe")

    # Si Churn est 'Yes'/'No' (category), on le mappe en 0/1
    if df[TARGET].dtype.name in ("category","object"):
        df[TARGET] = df[TARGET].map({'Yes':1, 'No':0}).astype(int)

    # --- 2) Features lists (robuste) ---
    # On choisit numeric et categorical en fonction des types actuels du dataframe
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # retirer la cible
    numeric_features = [c for c in numeric_features if c != TARGET]

    categorical_features = df.select_dtypes(include=['category','object']).columns.tolist()
    # certains 'object' peuvent être des IDs -> s'assurer qu'on n'inclut pas customerID si présent
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')

    # Afficher listes pour vérification (optionnel)
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # --- 3) FEATURE ENGINEERING simple (optionnel mais pro) ---
    # Exemple : average monthly charge (évite multicolinéarité TotalCharges ~ MonthlyCharges*tenure)
    # On crée AverageChargesPerMonth = TotalCharges / tenure (protéger division par 0)
    if set(['TotalCharges','tenure']).issubset(df.columns):
        df['AvgChargesPerMonth'] = df.apply(
            lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'], axis=1
        )
        # ajouter à numeric_features
        if 'AvgChargesPerMonth' not in numeric_features:
            numeric_features.append('AvgChargesPerMonth')

    # --- 4) Split train/test (stratified) ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train churn %:\n", y_train.value_counts(normalize=True))
    print("Test churn %:\n", y_test.value_counts(normalize=True))

    # --- 5) Preprocessing pipeline (fit ONLY on train) ---
    # numeric transformer: imputer + scaler
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # categorical transformer: imputer (rare) + one-hot (ignore unseen)
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sparse_output=False to get a dense array
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # on a tout listé
    )

    # Fit sur X_train et transformer
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # --- 6) Recover feature names to build DataFrame back (pro) ---
    # Note: get_feature_names_out requires sklearn >= 1.0
    num_features_out = numeric_features
    cat_features_out = []
    if len(categorical_features) > 0:
        cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_out = list(cat_ohe.get_feature_names_out(categorical_features))
    feature_names = list(num_features_out) + cat_features_out

    # Créer DataFrames avec index d'origine (pratique pour mapping)
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df  = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    # Joindre la target
    train_df = X_train_df.copy()
    train_df[TARGET] = y_train

    test_df = X_test_df.copy()
    test_df[TARGET] = y_test

    # --- 7) Créer template results (vide) pour stocker prédictions futures ---
    results_df = X_test_df.copy()
    results_df['true'] = y_test
    results_df['predicted'] = np.nan
    results_df['predicted_proba'] = np.nan

    # --- 8) Sauvegarder fichiers (parquet si possible) ---
    # Parquet est recommandé (plus compact, types préservés). Si pyarrow absent -> fallback CSV.
    def save_dataframe(df_obj, path):
        try:
            df_obj.to_parquet(path, index=True)
            print(f"Saved: {path}")
            return True
        except Exception as e:
            # fallback en CSV si échec avec parquet
            csv_path = str(path).replace('.parquet', '.csv')
            df_obj.to_csv(csv_path, index=True)
            print(f"Parquet failed, saved CSV: {csv_path} (error: {e})")
            return False

    # Chemins complets pour les fichiers de sortie
    train_path = MODELS_DIR / "train.parquet"
    test_path = MODELS_DIR / "test.parquet"
    results_path = MODELS_DIR / "results_template.parquet"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    # Sauvegarde des données
    save_dataframe(train_df, train_path)
    save_dataframe(test_df, test_path)
    save_dataframe(results_df, results_path)

    # --- 9) Sauvegarder préprocesseur et métadonnées ---
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    metadata = {
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_rows_total": df.shape[0],
        "n_features_raw": X.shape[1],
        "n_features_processed": len(feature_names),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_churn_distribution": y_train.value_counts().to_dict(),
        "test_churn_distribution": y_test.value_counts().to_dict(),
        "feature_names": feature_names,
        "output_paths": {
            "train_data": str(train_path.absolute()),
            "test_data": str(test_path.absolute()),
            "results_template": str(results_path.absolute()),
            "preprocessor": str(preprocessor_path.absolute()),
            "metadata": str(metadata_path.absolute())
        }
    }
    # Sauvegarde des métadonnées
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # --- 10) Optionnel: sauvegarder mapping customerID si tu l'as ---
    if 'customerID' in df_original.columns:
        customer_map_path = ARTIFACTS_DIR / "test_customerID_map.csv"
        df_original[['customerID']].loc[X_test.index].to_csv(customer_map_path)
        print(f"Saved customer ID mapping to {customer_map_path}")

    # Fin
    print("\n" + "="*80)
    print(f"Préparation terminée à {datetime.datetime.now().isoformat()}")
    print("Fichiers créés dans:")
    print(f"- Données: {MODELS_DIR.absolute()}")
    print(f"- Artefacts: {ARTIFACTS_DIR.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()

