import os
import yaml
import pandas as pd
from pathlib import Path
import kagglehub
import sys

# Chemins ancrés sur la racine du projet (src/etl -> src -> projet)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Création des répertoires si nécessaire
os.makedirs(RAW_DATA_FILE.parent, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

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
        # Vérifier si le fichier existe déjà
        if RAW_DATA_FILE.exists():
            print(f"Le fichier de données brutes existe déjà : {RAW_DATA_FILE}")
            return str(RAW_DATA_FILE)

        # Vérifier la configuration Kaggle
        if not setup_kaggle():
            raise Exception("Configuration Kaggle manquante ou incorrecte.")

        print("Téléchargement du dataset depuis Kaggle Hub...")
        
        # S'assurer que le dossier de destination existe
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Télécharger le dataset via KaggleHub (datasets)
        # IMPORTANT: utiliser dataset_download avec un handle de dataset valide
        dataset_handles = [
            "blastchar/telco-customer-churn",
            "yasserh/telco-customer-churn",
            "pavansubhasht/telco-customer-churn",
        ]

        last_err = None
        for handle in dataset_handles:
            try:
                # Tente d'écrire directement le fichier ciblé
                kagglehub.dataset_download(handle, path=str(RAW_DATA_FILE))
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Sinon télécharge tout le dataset dans le dossier
                try:
                    kagglehub.dataset_download(handle, path=str(RAW_DATA_FILE.parent))
                    last_err = None
                    break
                except Exception as e2:
                    last_err = e2
                    continue

        # Fallback HTTP si KaggleHub échoue
        if last_err is not None:
            try:
                import urllib.request
                fallback_urls = [
                    "https://raw.githubusercontent.com/blastchar/telco-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
                    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
                ]
                for url in fallback_urls:
                    try:
                        print(f"Kaggle indisponible, tentative de téléchargement direct: {url}")
                        urllib.request.urlretrieve(url, str(RAW_DATA_FILE))
                        last_err = None
                        break
                    except Exception:
                        continue
                if last_err is not None:
                    raise last_err
            except Exception:
                raise last_err
        
        # Vérifier que le fichier a bien été téléchargé
        if not RAW_DATA_FILE.exists():
            # Essayer de trouver le fichier téléchargé avec un nom différent
            csv_files = list(RAW_DATA_FILE.parent.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError("Le fichier n'a pas été téléchargé correctement.")
            
            # Renommer le fichier téléchargé vers le nom attendu
            import shutil
            source_file = csv_files[0]
            shutil.move(source_file, RAW_DATA_FILE)
        
        print(f"Téléchargement terminé. Fichier enregistré : {RAW_DATA_FILE}")
        return str(RAW_DATA_FILE)
        
    except Exception as e:
        print(f"\nErreur lors du téléchargement du dataset : {str(e)}")
        raise

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        # Vérifie la configuration
        if not setup_kaggle():
            raise Exception("Configuration de kagglehub échouée. Veuillez installer kagglehub avec 'pip install kagglehub'")
            
        # Exécute l'extraction
        print("Démarrage de l'extraction des données...")
        data_file = extract_data()
        
        # Affiche un message de succès
        print(f"\nExtraction terminée avec succès !")
        print(f"Fichier de données : {data_file}")
        
    except Exception as e:
        print(f"\nERREUR : {str(e)}")
        print("\nDétails de l'erreur :")
        import traceback
        traceback.print_exc()
        
        print("\nConseils de dépannage :")
        print("1. Vérifiez que vous êtes connecté à Internet")
        print("2. Vérifiez que kagglehub est correctement configuré")
        print("3. Vérifiez que vous avez les permissions nécessaires pour écrire dans le dossier de destination")
        print("\n" + "="*50)
        print("POUR VOUS AUTHENTIFIER :")
        print("="*50)
        print("1. Installez kagglehub : pip install kagglehub")
        print("2. Exécutez : kagglehub configure")
        print("3. Suivez les instructions pour vous connecter avec votre compte Kaggle")
        print("\n" + "="*50)
