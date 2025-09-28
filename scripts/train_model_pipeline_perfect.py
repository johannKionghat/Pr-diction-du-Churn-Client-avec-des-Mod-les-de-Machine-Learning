"""
train_model_pipeline_perfect.py
--------------------------------
Automatise l'exécution des trois étapes critiques du projet ML :
1. Préparation des données (preprocessing.py)
2. Entraînement des modèles (train.py)
3. Évaluation et optimisation des modèles (evaluation.py)

Chaque étape est appelée séquentiellement avec gestion des logs et des chemins absolus.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent  # Racine du projet
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = DATA_DIR / "models" / "artifacts"

# Création des répertoires si nécessaire
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def run_step(script_path, description):
    """
    Exécute un script Python avec gestion des erreurs et journalisation.
    
    Args:
        script_path (str or Path): Chemin relatif ou absolu vers le script à exécuter
        description (str): Description de l'étape en cours
    """
    # Convertir en chemin absolu si ce n'est pas déjà le cas
    if not os.path.isabs(script_path):
        script_path = SRC_DIR / script_path
    
    # Vérifier que le fichier existe
    if not script_path.exists():
        print(f"❌ ERREUR: Le fichier {script_path} n'existe pas")
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"🚀 DÉMARRAGE : {description}")
    print(f"📂 Script: {script_path}")
    print("="*80)

    try:
        # Exécuter le script avec le bon working directory
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,  # S'exécute depuis la racine du projet
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✅ TERMINÉ : {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERREUR pendant {description}")
        print("-"*40 + " STDOUT " + "-"*40)
        print(e.stdout)
        print("\n" + "-"*40 + " STDERR " + "-"*40)
        print(e.stderr)
        print("="*80 + "\n")
        return False

def main():
    print("\n" + "="*80)
    print("🎯 DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT")
    print(f"📂 Répertoire de travail: {BASE_DIR}")
    print(f"🕒 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # --- 1) Préprocessing ---
    print("\n" + "="*80)
    print("🔧 ÉTAPE 1: PRÉPARATION DES DONNÉES")
    print("="*80)
    if not run_step("preprocessing/preprocessing.py", "Préparation des données"):
        print("❌ Le pipeline s'est arrêté en raison d'une erreur lors de la préparation des données.")
        sys.exit(1)

    # --- 2) Entraînement ---
    print("\n" + "="*80)
    print("🤖 ÉTAPE 2: ENTRAÎNEMENT DES MODÈLES")
    print("="*80)
    if not run_step("train/train.py", "Entraînement des modèles"):
        print("❌ Le pipeline s'est arrêté en raison d'une erreur lors de l'entraînement.")
        sys.exit(1)

    # --- 3) Évaluation & Optimisation ---
    print("\n" + "="*80)
    print("📊 ÉTAPE 3: ÉVALUATION ET OPTIMISATION")
    print("="*80)
    if not run_step("evaluation/evaluation.py", "Évaluation et optimisation des modèles"):
        print("❌ Le pipeline s'est arrêté en raison d'une erreur lors de l'évaluation.")
        sys.exit(1)

    # Fin du pipeline
    print("\n" + "="*80)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("="*80)
    print(f"📂 Modèles sauvegardés dans: {MODELS_DIR.absolute()}")
    print(f"📊 Métadonnées et artefacts: {ARTIFACTS_DIR.absolute()}")
    print(f"🕒 Durée totale: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
    print("="*80 + "\n")

if __name__ == "__main__":
    start_time = datetime.now()
    main()
