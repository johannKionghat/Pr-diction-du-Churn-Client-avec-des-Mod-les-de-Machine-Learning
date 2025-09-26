import os
import yaml
import pandas as pd
from pathlib import Path
import kagglehub

# Chargement de la configuration
def load_config():
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Chargement de la configuration
config = load_config()
data_config = config['data']

def setup_kaggle():
    """Vérifie que kagglehub est correctement configuré."""
    try:
        # Vérifie que kagglehub est installé et configuré
        import kagglehub
        return True
    except Exception as e:
        print(f"Erreur lors de la configuration de kagglehub : {str(e)}")
        return False

def extract_data():
    """
    Fonction principale pour extraire les données brutes.
    Télécharge le dataset Telco Customer Churn depuis Kaggle Hub s'il n'existe pas déjà.
    Retourne le chemin vers le fichier de données brutes.
    """
    try:
        # Définit les chemins à partir de la configuration
        base_dir = Path('..') / data_config['dirs']['base']
        raw_dir = base_dir / data_config['dirs']['raw']
        raw_file = raw_dir / data_config['files']['raw_data']
        
        # Vérifie si le fichier existe déjà
        if raw_file.exists():
            print(f"Le fichier de données brutes existe déjà : {raw_file}")
            return str(raw_file)
        
        # Crée les dossiers s'ils n'existent pas
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Télécharge le dataset avec kagglehub
        print(f"Téléchargement du dataset {data_config['kaggle']['dataset']} depuis Kaggle Hub...")
        
        # Télécharge le dataset et obtient le chemin du répertoire de téléchargement
        dataset_path = kagglehub.dataset_download(data_config['kaggle']['dataset'])
        
        # Vérifie que le répertoire de téléchargement existe
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Le répertoire de téléchargement {dataset_path} est introuvable.")
            
        # Recherche le fichier CSV dans le répertoire téléchargé
        csv_files = list(Path(dataset_path).glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("Aucun fichier CSV trouvé dans le dataset téléchargé.")
            
        # Prend le premier fichier CSV trouvé (normalement il n'y en a qu'un)
        source_csv = csv_files[0]
        
        # Copie le fichier vers le dossier de destination
        import shutil
        shutil.copy2(source_csv, raw_file)
        
        print(f"\nLe dataset a été téléchargé avec succès dans : {raw_file}")
        return str(raw_file)
        
    except Exception as e:
        print(f"\nErreur lors du téléchargement du dataset : {str(e)}")
        raise

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        # Vérifie la configuration
        if not setup_kaggle():
            raise Exception("Configuration de kagglehub échouée. Veuillez installer kagglehub avec 'pip install kagglehub'")
            
        # Extrait les données
        raw_data_path = extract_data()
        print(f"\nExtraction terminée. Données brutes disponibles à : {raw_data_path}")
            
    except Exception as e:
        print(f"\nUne erreur s'est produite : {str(e)}")
        print("\n" + "="*50)
        print("ERREUR DE CONFIGURATION KAGGLE HUB")
        print("="*50)
        print("\nAssurez-vous que :")
        print("1. Vous avez un compte Kaggle (https://www.kaggle.com/)")
        print(f"2. Vous avez accepté les conditions d'utilisation du dataset : https://www.kaggle.com/{data_config['kaggle']['dataset']}")
        print("3. Le package kagglehub est installé (pip install kagglehub)")
        print("4. Vous êtes authentifié avec la commande 'kagglehub configure'")
        
        print("\n" + "="*50)
        print("POUR VOUS AUTHENTIFIER :")
        print("="*50)
        print("1. Installez kagglehub : pip install kagglehub")
        print("2. Exécutez : kagglehub configure")
        print("3. Suivez les instructions pour vous connecter avec votre compte Kaggle")
        print("\n" + "="*50)
