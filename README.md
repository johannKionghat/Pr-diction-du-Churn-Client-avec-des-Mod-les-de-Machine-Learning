# Projet 3 – Prédiction du Churn Client avec des Modèles de Machine Learning  

## 📌 Contexte
Vous êtes embauché en tant que **data scientist** au sein d'une entreprise de télécommunications.  
Votre mission est de développer un modèle de Machine Learning capable de **prédire le churn des clients** (résiliation d’abonnement).  

Le projet inclut :
- L’exploration et la préparation des données,
- L’implémentation et la comparaison de différents modèles (régression logistique, arbres de décision, Random Forest),
- L’optimisation des modèles,
- La présentation des résultats via visualisations et documentation.

---

## 🎯 Objectifs
1. Collecter et préparer les données nécessaires à la prédiction du churn.  
2. Développer et comparer plusieurs modèles de Machine Learning.  
3. Évaluer et optimiser les performances des modèles avec validation et tuning des hyperparamètres.  
4. Visualiser les résultats pour faciliter l’interprétation.  
5. Documenter le processus complet et présenter les conclusions.  

---

## ⚙️ Contraintes
- Travail en autonomie.  
- Documentation complète et claire.  
- Gestion du code avec **Git & GitHub** (commits conventionnels, README détaillé, bonne structure).  
- Livraison dans les délais sous forme d’une archive `.zip`.  

---

## 🆓 Libertés
- Choix des bibliothèques Python (Pandas, Scikit-learn, Seaborn, etc.).  
- Choix des algorithmes (justifiés).  
- Possibilité de proposer des pistes de déploiement (API Flask/FastAPI, dashboard, etc.).  
- Améliorations possibles du pipeline d’apprentissage.  

---

## 🚀 Étapes du projet
### 1. Collecte & Préparation des Données
- Utilisation du dataset public **Telco Customer Churn** de Kaggle.  
- Nettoyage : gestion des valeurs manquantes, encodage des variables catégorielles, normalisation des données numériques.  

### 2. Exploration des Données (EDA)
- Analyse statistique descriptive.  
- Visualisations (Seaborn, Matplotlib).  
- Sélection des caractéristiques pertinentes (âge, type de contrat, services, etc.).  

### 3. Développement des Modèles
- **Régression Logistique** : baseline.  
- **Arbre de Décision** : interprétation des règles.  
- **Random Forest** : amélioration des performances.  

### 4. Évaluation & Optimisation
- Métriques : Précision, Rappel, F1-score, AUC.  
- Optimisation des hyperparamètres (Grid Search, Cross-Validation).  
- Comparaison des modèles.  

### 5. Visualisation & Présentation
- Visualisation des performances (matrice de confusion, ROC, importance des features).  
- Création de dashboards interactifs (Plotly, Bokeh).  
- Rapport et présentation finale.  

### 6. Documentation & Gestion de Projet
- Documentation de chaque étape.  
- Gestion avec GitHub (commits réguliers, issues si besoin, branches).  
- Livrable final sous format `.zip`.  

---

## 📂 Données
Dataset : [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)  
Téléchargement via CLI :  
```bash
curl -L -o ~/Downloads/telco-customer-churn.zip \
https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn
