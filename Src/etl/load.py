import os
import pandas as pd
import yaml
from pathlib import Path
import sys

# Détection du répertoire racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Définition des chemins relatifs à la racine du projet
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

class Load:
    def __init__(self):
        # Initialisation des chemins
        self.RAW_DATA_DIR = Path(RAW_DATA_DIR)
        self.PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)
        
        # Création des répertoires s'ils n'existent pas
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path):
        """
        Charge les données depuis un fichier (CSV ou Parquet) et retourne un DataFrame pandas.
        
        Args:
            file_path (str): Chemin vers le fichier à charger (.csv ou .parquet)
        
        Returns:
            pd.DataFrame: DataFrame contenant les données chargées
        """
        try:
            # Vérification de l'extension du fichier
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.parquet':
                # Chargement d'un fichier Parquet
                df = pd.read_parquet(file_path)
            else:
                # Chargement d'un fichier CSV
                df = pd.read_csv(file_path)
            
            # Affichage des informations
            print("\n" + "="*50)
            print(f"CHARGEMENT DEPUIS : {file_path}")
            print("="*50)
            
            print("\nAPERÇU DES DONNÉES")
            print("-"*50)
            print(df.head())
            
            print("\nINFORMATIONS SUR LE DATASET")
            print("-"*50)
            print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
            print("\nTypes de données :")
            print(df.dtypes)
            
            return df
            
        except Exception as e:
            print(f"Erreur lors du chargement des données : {str(e)}")
            raise

    def save_processed_data(self, df, output_file=None):
        """
        Sauvegarde un DataFrame dans le dossier de données traitées au format CSV ou Parquet.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            output_file (str or Path, optional): Chemin du fichier de sortie. 
                Si non spécifié, utilise un chemin par défaut.
                Peut être un chemin relatif ou absolu.
            
        Returns:
            str: Chemin absolu vers le fichier sauvegardé
        """
        try:
            # Déterminer le répertoire de sortie
            output_dir = Path(self.PROCESSED_DATA_DIR)
            
            # Créer le répertoire s'il n'existe pas
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Déterminer le nom du fichier de sortie
            if output_file is None:
                output_file = output_dir / 'telco_customer_churn_processed.parquet'
            else:
                # Convertir en Path si c'est une chaîne
                output_file = Path(output_file)
                # Si c'est un chemin relatif, le rendre absolu par rapport au répertoire de sortie
                if not output_file.is_absolute():
                    output_file = output_dir / output_file.name
            
            # S'assurer que le répertoire parent existe
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le fichier
            output_path = str(output_file.absolute())
            
            # Déterminer le format en fonction de l'extension
            if str(output_file).lower().endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:  # Par défaut, utiliser parquet
                df.to_parquet(output_path, index=False)
            
            # Vérifier que le fichier a été créé
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Le fichier {output_path} n'a pas pu être créé")
                
            print(f"\n[SUCCÈS] Données sauvegardées avec succès : {output_path}")
            return output_path
            
        except Exception as e:
            import traceback
            print(f"[ERREUR] Erreur lors de la sauvegarde des données : {str(e)}")
            print(traceback.format_exc())
            raise

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        load = Load()
        
        # Chemins en dur comme demandé
        raw_file = Path("../../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        
        # Chargement des données
        print(f"Chargement des données depuis : {raw_file}")
        df = load.load_data(raw_file)
        
        # Affichage d'un message de confirmation
        print(f"\nChargement terminé. Le dataset contient {len(df)} lignes et {len(df.columns)} colonnes.")
        
    except Exception as e:
        print(f"\nErreur lors du chargement des données : {str(e)}")