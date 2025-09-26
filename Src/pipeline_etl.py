"""
Script principal pour exécuter le pipeline ETL complet.

Ce script orchestre l'exécution des différentes étapes du pipeline ETL :
1. Extraction des données brutes
2. Transformation des données
3. Chargement des données transformées

Chaque étape est gérée par un module séparé pour une meilleure maintenabilité.
"""

import argparse
from pathlib import Path
import yaml
from extract import extract_data, setup_kaggle
from transform import transform_data
from load import Load

def load_config():
    """Charge la configuration à partir du fichier config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_etl_pipeline(skip_extraction=False):
    """
    Exécute le pipeline ETL complet.
    
    Args:
        skip_extraction (bool): Si True, saute l'étape d'extraction.
    """
    try:
        # Chargement de la configuration
        config = load_config()
        data_config = config['data']
        
        # Chemins des fichiers
        base_dir = Path('..') / data_config['dirs']['base']
        raw_dir = base_dir / data_config['dirs']['raw']
        raw_file = raw_dir / data_config['files']['raw_data']
        
        # Étape 1: Extraction des données
        if not skip_extraction:
            print("\n" + "="*50)
            print("ÉTAPE 1: EXTRACTION DES DONNÉES")
            print("="*50)
            if not setup_kaggle():
                raise Exception("Configuration de kagglehub échouée. Veuillez installer kagglehub avec 'pip install kagglehub'")
            raw_file = Path(extract_data())
        else:
            print("\n" + "="*50)
            print("ÉTAPE 1: EXTRACTION DES DONNÉES (IGNORÉE)")
            print("="*50)
            print(f"Utilisation du fichier existant : {raw_file}")
        
        # Vérification que le fichier brut existe
        if not raw_file.exists():
            raise FileNotFoundError(f"Le fichier de données brutes est introuvable : {raw_file}")
        
        # Étape 2: Transformation des données
        print("\n" + "="*50)
        print("ÉTAPE 2: TRANSFORMATION DES DONNÉES")
        print("="*50)
        
        # Initialisation du chargeur
        loader = Load()
        
        # Chargement des données brutes
        print(f"Chargement des données depuis : {raw_file}")
        df_raw = loader.load_data(str(raw_file))
        
        # Transformation des données
        df_transformed = transform_data(df_raw)
        
        # Étape 3: Chargement des données transformées
        print("\n" + "="*50)
        print("ÉTAPE 3: CHARGEMENT DES DONNÉES")
        print("="*50)
        
        # Sauvegarde des données transformées
        processed_file = data_config['files']['processed_data']
        output_path = loader.save_processed_data(df_transformed, processed_file)
        
        print("\n" + "="*50)
        print("PIPELINE ETL TERMINÉ AVEC SUCCÈS")
        print("="*50)
        print(f"Données brutes : {raw_file}")
        print(f"Données transformées : {output_path}")
        
    except Exception as e:
        print(f"\nERREUR LORS DE L'EXÉCUTION DU PIPELINE ETL: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Exécute le pipeline ETL complet.')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Ignore l\'étape d\'extraction des données')
    
    args = parser.parse_args()
    
    # Exécution du pipeline
    run_etl_pipeline(skip_extraction=args.skip_extraction)
