"""
Pipeline ETL complet avec Prefect.

Étapes :
1. Extraction des données brutes
2. Transformation des données
3. Chargement des données transformées
"""

import argparse
from pathlib import Path
import yaml
from prefect import flow, task, get_run_logger
from extract import extract_data, setup_kaggle
from transform import transform_data
from load import Load


def load_config():
    """Charge la configuration à partir du fichier config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# ----------------------
# Étapes (tasks Prefect)
# ----------------------

@task
def extraction(config, skip_extraction=False):
    """Étape 1 : Extraction des données"""
    logger = get_run_logger()
    data_config = config['data']

    base_dir = Path('..') / data_config['dirs']['base']
    raw_dir = base_dir / data_config['dirs']['raw']
    raw_file = raw_dir / data_config['files']['raw_data']

    if skip_extraction:
        logger.info("Extraction ignorée. Utilisation du fichier existant : %s", raw_file)
    else:
        logger.info("Démarrage de l'extraction...")
        if not setup_kaggle():
            raise Exception("Configuration de kagglehub échouée. Installez-le avec `pip install kagglehub`.")
        raw_file = Path(extract_data())
        logger.info("Extraction terminée. Fichier obtenu : %s", raw_file)

    if not raw_file.exists():
        raise FileNotFoundError(f"Fichier brut introuvable : {raw_file}")

    return raw_file


@task
def transformation(raw_file, config):
    """Étape 2 : Transformation des données"""
    logger = get_run_logger()
    logger.info("Transformation en cours...")

    loader = Load()
    df_raw = loader.load_data(str(raw_file))
    df_transformed = transform_data(df_raw)

    logger.info("Transformation terminée.")
    return df_transformed


@task
def chargement(df_transformed, config):
    """Étape 3 : Chargement des données transformées"""
    logger = get_run_logger()
    logger.info("Chargement des données transformées...")

    loader = Load()
    processed_file = config['data']['files']['processed_data']
    output_path = loader.save_processed_data(df_transformed, processed_file)

    logger.info("Données sauvegardées dans : %s", output_path)
    return output_path


# ----------------------
# Flow principal Prefect
# ----------------------

@flow(name="Pipeline ETL complet")
def etl_pipeline(skip_extraction: bool = False):
    """Orchestre l’exécution du pipeline ETL avec Prefect"""
    logger = get_run_logger()

    try:
        logger.info("Chargement de la configuration...")
        config = load_config()

        logger.info("=== ÉTAPE 1 : EXTRACTION ===")
        raw_file = extraction(config, skip_extraction)

        logger.info("=== ÉTAPE 2 : TRANSFORMATION ===")
        df_transformed = transformation(raw_file, config)

        logger.info("=== ÉTAPE 3 : CHARGEMENT ===")
        output_path = chargement(df_transformed, config)

        logger.info("=== PIPELINE TERMINÉ ✅ ===")
        logger.info("📂 Données brutes : %s", raw_file)
        logger.info("📂 Données transformées : %s", output_path)

    except Exception as e:
        logger.error("ERREUR lors de l'exécution du pipeline : %s", str(e))
        raise


# ----------------------
# Exécution en CLI
# ----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exécute le pipeline ETL complet avec Prefect.")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Ignore l'étape d'extraction des données")

    args = parser.parse_args()
    etl_pipeline(skip_extraction=args.skip_extraction)
