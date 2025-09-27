import pandas as pd
import yaml
from pathlib import Path
from load import Load

# Définition des chemins
import os
from pathlib import Path

# Chemin vers le dossier du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Chemins des dossiers
data_dir = PROJECT_ROOT / 'Data'
raw_data_dir = data_dir / 'raw'
processed_data_dir = data_dir / 'processed'

# Chemins des fichiers
raw_data_file = raw_data_dir / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
processed_data_file = processed_data_dir / 'telco_customer_churn_processed.parquet'
processed_csv_file = processed_data_dir / 'telco_customer_churn_processed.csv'

# Chargement de la configuration
def load_config():
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def transform_data(df):
    """
    Effectue les transformations nécessaires sur les données.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes
        
    Returns:
        pd.DataFrame: DataFrame transformé
    """
    try:
        # Faire une copie pour éviter les modifications inattendues
        df_transformed = df.copy()
        
        # 1) Conversion de TotalCharges en numérique
        print("\n1) Conversion de TotalCharges en numérique...")
        df_transformed['TotalCharges'] = pd.to_numeric(df_transformed['TotalCharges'], errors='coerce')
        
        # 2) Vérifier les NaN
        print("2) Vérification des valeurs manquantes...")
        nan_count = df_transformed['TotalCharges'].isnull().sum()
        print(f"Nombre de NaN dans TotalCharges: {nan_count}")
        
        # 3) Corriger les NaN -> clients avec tenure=0 n'ont pas encore de facture
        print("3) Correction des valeurs manquantes...")
        df_transformed.loc[(df_transformed['tenure'] == 0) & (df_transformed['TotalCharges'].isnull()), 'TotalCharges'] = 0.0
        
        # Vérification après correction
        remaining_nan = df_transformed['TotalCharges'].isnull().sum()
        print(f"Nombre de NaN restants après correction: {remaining_nan}")
        
        # 4) Supprimer la colonne customerID (identifiant inutile pour la prédiction)
        if 'customerID' in df_transformed.columns:
            print("4) Suppression de la colonne customerID...")
            df_transformed = df_transformed.drop(columns=['customerID'])
        
        # 5) Convertir les colonnes catégorielles en type 'category'
        print("5) Conversion des colonnes catégorielles...")
        for col in df_transformed.select_dtypes(include='object').columns:
            df_transformed[col] = df_transformed[col].astype('category')
        
        # Affichage des informations finales
        print("\n" + "="*50)
        print("INFORMATIONS APRÈS TRANSFORMATION")
        print("="*50)
        df_transformed.info()
        
        return df_transformed
        
    except Exception as e:
        print(f"Erreur lors de la transformation des données : {str(e)}")
        raise

def main():
    try:
        # Chargement de la configuration
        config = load_config()
        
        # Initialisation de la classe Load
        loader = Load()
        
        # Chargement des données
        print(f"\nChargement des données depuis : {raw_data_file}")
        df = loader.load_data(raw_data_file)
        
        # Transformation des données
        print("\n" + "="*50)
        print("DÉBUT DE LA TRANSFORMATION DES DONNÉES")
        print("="*50)
        df_transformed = transform_data(df)

        loader.save_processed_data(df_transformed)
        
        print("\n" + "="*50)
        print(f"TRANSFORMATION TERMINÉE AVEC SUCCÈS")
        print("="*50)
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution du script de transformation : {str(e)}")
        raise

if __name__ == "__main__":
    main()