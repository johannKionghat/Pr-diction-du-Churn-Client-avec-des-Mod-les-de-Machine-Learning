"""
Module pour la gestion des chemins du projet.
"""
from pathlib import Path

# Racine du projet (dossier contenant le dossier Src)
PROJECT_ROOT = Path(__file__).parent.parent

# Chemins des dossiers
data_dir = PROJECT_ROOT / 'Data'
raw_data_dir = data_dir / 'raw'
processed_data_dir = data_dir / 'processed'
models_dir = data_dir / 'models'

# Chemins des fichiers
raw_data_file = raw_data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
processed_data_file = processed_data_dir / 'telco_customer_churn_processed.parquet'
processed_csv_file = processed_data_dir / 'telco_customer_churn_processed.csv'

# Création des dossiers si nécessaire
for directory in [data_dir, raw_data_dir, processed_data_dir, models_dir]:
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Affiche les chemins pour vérification
    print("Chemins du projet :")
    print(f"Racine du projet : {PROJECT_ROOT}")
    print(f"Dossier des données : {data_dir}")
    print(f"Dossier des données brutes : {raw_data_dir}")
    print(f"Dossier des données transformées : {processed_data_dir}")
    print(f"Dossier des modèles : {models_dir}")
    print(f"\nFichiers :")
    print(f"Données brutes : {raw_data_file}")
    print(f"Données transformées (Parquet) : {processed_data_file}")
    print(f"Données transformées (CSV) : {processed_csv_file}")
