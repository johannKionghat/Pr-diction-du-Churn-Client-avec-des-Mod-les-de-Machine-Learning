import pandas as pd
import yaml
from pathlib import Path

class Load:
    def __init__(self):
        pass
    
    def load_config(self):
        """Charge la configuration depuis le fichier config.yaml"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

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
        Sauvegarde un DataFrame dans le dossier de données traitées au format CSV.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            output_file (str, optional): Chemin du fichier de sortie. Si non spécifié, utilise un chemin par défaut.
            
        Returns:
            str: Chemin vers le fichier sauvegardé
        """
        try:
            # Définit le chemin de sortie par défaut si non spécifié
            current_dir = Path(__file__).parent.absolute()
            project_root = current_dir.parent
            output_dir = project_root / 'data' / 'processed'
            
            print(f"\n[DEBUG] Dossier de sortie : {output_dir}")
            
            if output_file is None:
                output_file = output_dir / 'telco_customer_churn_processed.parquet'
            else:
                output_file = output_dir / Path(output_file).name
            
            print(f"[DEBUG] Fichier de sortie : {output_file}")
            
            # Crée le dossier parent si nécessaire
            output_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Dossier de sortie créé : {output_file.parent}")
            
            # Sauvegarde le fichier
            output_path = str(output_file.absolute())
            print(f"[DEBUG] Tentative de sauvegarde dans : {output_path}")
            df.to_parquet(output_path, index=False)
            
            # Vérifie que le fichier a été créé
            if Path(output_path).exists():
                print(f"[DEBUG] Fichier sauvegardé avec succès : {output_path}")
            else:
                print("[ERREUR] Le fichier n'a pas été créé !")
            
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
        
        # Chemin vers les données brutes
        config = load.load_config()
        data_config = config['data']
        base_dir = Path('..') / data_config['dirs']['base']
        raw_dir = base_dir / data_config['dirs']['raw']
        raw_file = raw_dir / data_config['files']['raw_data']
        
        # Chargement des données
        print(f"Chargement des données depuis : {raw_file}")
        df = load.load_data(raw_file)
        
        # Affichage d'un message de confirmation
        print(f"\nChargement terminé. Le dataset contient {len(df)} lignes et {len(df.columns)} colonnes.")
        
    except Exception as e:
        print(f"\nErreur lors du chargement des données : {str(e)}")