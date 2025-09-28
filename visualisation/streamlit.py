import os
from pathlib import Path
import json
import io

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import joblib


# =============================
# Configuration & Utilities
# =============================
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_paths():
    """Return key project paths relative to this file."""
    base_dir = Path(__file__).resolve().parent.parent  # project root
    data_dir = base_dir / "data"
    processed_dir = data_dir / "processed"
    models_data_dir = data_dir / "models_data"
    artifacts_dir = models_data_dir / "artifacts"
    models_dir = base_dir / "models"
    raw_dir = data_dir / "raw"
    return {
        "base": base_dir,
        "data": data_dir,
        "processed": processed_dir,
        "models_data": models_data_dir,
        "artifacts": artifacts_dir,
        "models": models_dir,
        "raw": raw_dir,
    }


PATHS = get_paths()


@st.cache_data(show_spinner=False)
def load_metadata():
    meta_path = PATHS["artifacts"] / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_parquet(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_csv(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def load_joblib(p: Path):
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure engineered features expected by the preprocessor exist.
    - AvgChargesPerMonth = TotalCharges / tenure (fallback to MonthlyCharges if tenure is 0/NaN)
    - Cast key numeric columns to numeric types.
    """
    df = df.copy()
    # Ensure numeric types
    for col in ["tenure", "TotalCharges", "MonthlyCharges"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create engineered feature if missing
    if "AvgChargesPerMonth" not in df.columns:
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            denom = df["tenure"].replace(0, np.nan)
            avg = df["TotalCharges"] / denom
            if "MonthlyCharges" in df.columns:
                avg = avg.fillna(df["MonthlyCharges"])  # fallback when tenure==0 or NaN
            df["AvgChargesPerMonth"] = avg.fillna(0.0)
        elif "MonthlyCharges" in df.columns:
            df["AvgChargesPerMonth"] = df["MonthlyCharges"].astype(float)
        else:
            df["AvgChargesPerMonth"] = 0.0

    return df


@st.cache_data(show_spinner=False)
def list_images(pattern_contains: str) -> list[Path]:
    imgs = []
    for p in PATHS["artifacts"].glob("*.png"):
        if pattern_contains.lower() in p.name.lower():
            imgs.append(p)
    return sorted(imgs)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    p = PATHS["artifacts"] / "model_comparison.csv"
    return load_csv(p)


@st.cache_data(show_spinner=False)
def load_results_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in [
        "training_results.csv",
        "base_models_results.csv",
        "optimized_models_results.csv",
    ]:
        df = load_csv(PATHS["artifacts"] / name)
        if df is not None and df.shape[0] > 0:
            tables[name] = df
    return tables


@st.cache_data(show_spinner=False)
def load_processed_dataset() -> pd.DataFrame | None:
    return load_parquet(PATHS["processed"] / "telco_customer_churn_processed.parquet")


@st.cache_data(show_spinner=False)
def load_train_test() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train = load_parquet(PATHS["models_data"] / "train.parquet")
    test = load_parquet(PATHS["models_data"] / "test.parquet")
    return train, test


@st.cache_data(show_spinner=False)
def load_raw_head() -> pd.DataFrame | None:
    # For building input forms from original schema
    raw_csv = PATHS["raw"] / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_csv(raw_csv)
    if df is not None:
        # Clean TotalCharges that might be non-numeric in raw
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df.head(100)
    return None


@st.cache_resource(show_spinner=False)
def load_models_and_preprocessor():
    models = {}
    for p in PATHS["models"].glob("*.pkl"):
        try:
            models[p.stem] = joblib.load(p)
        except Exception:
            pass
    preprocessor = load_joblib(PATHS["artifacts"] / "preprocessor.joblib")
    return models, preprocessor


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.title("📉 Churn Dashboard")
    st.caption("Visualisation interactive des performances, explications et prédictions")
    meta = load_metadata()
    if meta:
        st.markdown("**Configuration d'entraînement**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random State", value=str(meta.get("random_state")))
            st.metric("Test Size", value=str(meta.get("test_size")))
        with col2:
            st.metric("Features (raw)", value=str(meta.get("n_features_raw")))
            st.metric("Features (processed)", value=str(meta.get("n_features_processed")))
        st.markdown("---")
        st.caption("Distribution du churn (train / test)")
        tr = meta.get("train_churn_distribution", {})
        te = meta.get("test_churn_distribution", {})
        if tr and te:
            d = pd.DataFrame({
                "set": ["train", "train", "test", "test"],
                "churn": ["0", "1", "0", "1"],
                "count": [tr.get("0", 0), tr.get("1", 0), te.get("0", 0), te.get("1", 0)],
            })
            fig = px.bar(d, x="set", y="count", color="churn", barmode="group", title="Distribution Churn")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")


# =============================
# Header
# =============================
st.markdown("""
# 📊 Tableau de Bord - Prédiction de Churn

Ce dashboard consolide les résultats clés pour comprendre, expliquer et exploiter les prédictions de churn.
Chaque section décrit le rôle des graphiques pour la prise de décision.
""")


# =============================
# Tabs Layout
# =============================
tab_overview, tab_perf, tab_curves, tab_importance, tab_predict, tab_data, tab_report = st.tabs([
    "🏁 Vue d'ensemble",
    "📈 Performances modèles",
    "📉 ROC & Matrices",
    "🧠 Importance features",
    "🔮 Prédictions",
    "🗂️ Données",
    "📝 Rapport",
])


# =============================
# Overview Tab
# =============================
with tab_overview:
    st.subheader("Objectif")
    st.write(
        "Ce projet vise à prédire le churn des clients afin d'identifier les profils à risque et orienter les actions de rétention."
    )

    st.subheader("Ce que vous trouverez ici")
    st.markdown(
        """
        - **Performances globales**: comparaison des modèles (accuracy, precision, recall, F1, AUC).
        - **Courbes ROC et matrices de confusion**: compréhension fine des erreurs et du compromis entre rappel et précision.
        - **Importance des caractéristiques**: interprétation des facteurs influençant le churn.
        - **Prédictions**: prédire au cas par cas ou par lot, avec probabilités de churn.
        - **Exploration des données**: aperçu des jeux d'entraînement et de test.
        """
    )

    train_df, test_df = load_train_test()
    col1, col2 = st.columns(2)
    with col1:
        if train_df is not None:
            st.metric("Taille train", value=f"{train_df.shape[0]} lignes / {train_df.shape[1]} colonnes")
    with col2:
        if test_df is not None:
            st.metric("Taille test", value=f"{test_df.shape[0]} lignes / {test_df.shape[1]} colonnes")

    comp = load_model_comparison()
    if comp is not None and comp.shape[0] > 0:
        st.markdown("---")
        st.markdown("### Meilleures performances (AUC)")
        comp_sorted = comp.rename(columns={comp.columns[0]: "modèle"}).sort_values("auc", ascending=False)
        fig = px.bar(comp_sorted, x="modèle", y="auc", color="auc", title="AUC par modèle", text="auc")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(xaxis_title="Modèle", yaxis_title="AUC")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        with st.expander("Pourquoi l'AUC est importante ?"):
            st.write(
                """
                L'**AUC (Area Under the ROC Curve)** mesure la capacité du modèle à distinguer les churners des non-churners.
                Plus l'AUC est élevée, meilleure est la séparation—utile pour fixer des seuils d'action marketing.
                """
            )


# =============================
# Performance Tab
# =============================
with tab_perf:
    st.subheader("Comparaison des modèles")
    comp = load_model_comparison()
    if comp is None or comp.shape[0] == 0:
        st.info("Aucune table de comparaison trouvée.")
    else:
        dfc = comp.copy()
        dfc = dfc.rename(columns={dfc.columns[0]: "modèle"})
        st.dataframe(dfc.set_index("modèle"), use_container_width=True)

        metric = st.selectbox("Sélectionnez une métrique à visualiser", ["accuracy", "precision", "recall", "f1", "auc"], index=4)
        fig = px.bar(dfc.sort_values(metric, ascending=False), x="modèle", y=metric, color=metric,
                     title=f"{metric.upper()} par modèle", text=metric)
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        with st.expander("Comment interpréter ces métriques ?"):
            st.write(
                """
                - **Accuracy**: proportion de prédictions correctes (globalement).
                - **Precision**: quand le modèle prédit churn, la proportion correcte.
                - **Recall**: proportion de churners correctement détectés (sensibilité).
                - **F1**: compromis précision/rappel—utile avec classes déséquilibrées.
                - **AUC**: capacité de séparation globale.
                """
            )

    st.markdown("---")
    st.subheader("Autres résultats")
    tables = load_results_tables()
    if tables:
        for name, df in tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Aucun tableau de résultats additionnel détecté.")


# =============================
# ROC & Confusion Tab
# =============================
with tab_curves:
    st.subheader("Courbes ROC")
    roc_imgs = list_images("roc_curve")
    if roc_imgs:
        cols = st.columns(2)
        for i, p in enumerate(roc_imgs):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)
        with st.expander("Rôle des courbes ROC"):
            st.write(
                """
                Les **courbes ROC** illustrent le compromis entre **taux de vrais positifs** et **taux de faux positifs** pour tous les seuils.
                Elles permettent de choisir un seuil adapté selon le coût d'une action de rétention versus le coût d'un churn non détecté.
                """
            )
    else:
        st.info("Aucune courbe ROC trouvée dans artifacts.")

    st.markdown("---")
    st.subheader("Matrices de Confusion")
    cm_imgs = list_images("confusion_matrix")
    if cm_imgs:
        cols = st.columns(2)
        for i, p in enumerate(cm_imgs):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)
        with st.expander("Comment lire une matrice de confusion ?"):
            st.write(
                """
                La **matrice de confusion** détaille les **vrais positifs**, **faux positifs**, **vrais négatifs** et **faux négatifs**.
                Elle permet d'identifier si le modèle manque trop de churners (FN élevés) ou déclenche trop d'alertes inutiles (FP élevés).
                """
            )
    else:
        st.info("Aucune matrice de confusion trouvée dans artifacts.")


# =============================
# Feature Importance Tab
# =============================
with tab_importance:
    st.subheader("Importance des caractéristiques")
    imp_imgs = [
        PATHS["artifacts"] / "feature_importance_decision_tree.png",
        PATHS["artifacts"] / "feature_importance_random_forest.png",
    ]
    showed = False
    cols = st.columns(2)
    idx = 0
    for p in imp_imgs:
        if p.exists():
            with cols[idx % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)
            showed = True
            idx += 1
    if not showed:
        st.info("Aucune importance de feature disponible sous forme d'image.")

    with st.expander("Pourquoi c'est utile ?"):
        st.write(
            """
            L'importance des features permet d'**expliquer** les prédictions et d'identifier les **leviers d'action** (ex: contrat, services, facturation).
            Ces insights aident à concevoir des offres ciblées et à prioriser les variables à améliorer.
            """
        )


# =============================
# Predictions Tab
# =============================
with tab_predict:
    st.subheader("Prédictions de churn")
    models, preprocessor = load_models_and_preprocessor()

    if not models:
        st.warning("Aucun modèle trouvé dans le dossier models/.")
    if preprocessor is None:
        st.warning("Preprocessor introuvable (artifacts/preprocessor.joblib). Les formulaires bruts pourraient ne pas fonctionner.")

    model_name = st.selectbox("Choisir un modèle", options=sorted(models.keys()) if models else [])
    selected_model = models.get(model_name) if model_name else None

    st.markdown("### 1) Prédiction unitaire (formulaire)")
    raw_head = load_raw_head()
    if raw_head is None:
        st.info("Dataset brut non disponible. Impossible de générer un formulaire convivial.")
    else:
        raw_cols = [c for c in raw_head.columns if c not in ("customerID",)]
        num_cols = [c for c in raw_cols if pd.api.types.is_numeric_dtype(raw_head[c])]
        cat_cols = [c for c in raw_cols if c not in num_cols]

        with st.form("single_pred_form"):
            st.caption("Renseignez les caractéristiques pour un client")
            cols = st.columns(3)
            values = {}
            for i, c in enumerate(num_cols):
                with cols[i % 3]:
                    series = raw_head[c].dropna()
                    default = float(series.median()) if not series.empty else 0.0
                    min_v = float(series.min()) if not series.empty else 0.0
                    max_v = float(series.max()) if not series.empty else 1.0
                    values[c] = st.number_input(c, value=default, min_value=min_v, max_value=max_v, step=0.1)
            for i, c in enumerate(cat_cols):
                with cols[(i + len(num_cols)) % 3]:
                    options = sorted([str(x) for x in raw_head[c].dropna().unique().tolist()]) if c in raw_head.columns else []
                    default = options[0] if options else ""
                    values[c] = st.selectbox(c, options=options, index=0 if options else None)

            submitted = st.form_submit_button("Prédire")
            if submitted:
                if selected_model is None:
                    st.error("Veuillez sélectionner un modèle.")
                else:
                    try:
                        df_input = pd.DataFrame([values])
                        df_input = add_engineered_features(df_input)
                        if preprocessor is not None:
                            X = preprocessor.transform(df_input)
                        else:
                            st.warning("Preprocessor manquant: tentative d'inférence directe (peut échouer).")
                            X = df_input  # fallback
                        proba = None
                        if hasattr(selected_model, "predict_proba"):
                            proba = selected_model.predict_proba(X)[:, 1]
                            pred = (proba >= 0.5).astype(int)
                        else:
                            pred = selected_model.predict(X)
                        colA, colB = st.columns(2)
                        with colA:
                            st.metric("Prédiction (0=Non,1=Oui)", int(pred[0]))
                        with colB:
                            if proba is not None:
                                st.metric("Probabilité de churn", f"{float(proba[0]):.3f}")
                    except Exception as e:
                        st.error(f"Erreur de prédiction: {e}")

    st.markdown("---")
    st.markdown("### 2) Prédiction par lot")
    st.caption("Chargez un CSV brut (schéma similaire au dataset original) OU utilisez le jeu de test prétraité.")
    uploaded = st.file_uploader("Uploader un CSV brut", type=["csv"])  # raw schema expected
    use_test = st.checkbox("Utiliser le test set (models_data/test.parquet)")

    if st.button("Lancer la prédiction par lot"):
        if selected_model is None:
            st.error("Veuillez sélectionner un modèle.")
        else:
            try:
                if uploaded is not None:
                    df_raw = pd.read_csv(uploaded)
                    if preprocessor is None:
                        st.error("Preprocessor requis pour convertir le brut en features du modèle.")
                    else:
                        # Ensure numeric and engineered features
                        for col in ["TotalCharges", "tenure", "MonthlyCharges"]:
                            if col in df_raw.columns:
                                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
                        df_raw = add_engineered_features(df_raw)
                        X = preprocessor.transform(df_raw)
                        y_proba = selected_model.predict_proba(X)[:, 1] if hasattr(selected_model, "predict_proba") else None
                        y_pred = (y_proba >= 0.5).astype(int) if y_proba is not None else selected_model.predict(X)
                        out = df_raw.copy()
                        out["churn_pred"] = y_pred
                        if y_proba is not None:
                            out["churn_proba"] = y_proba
                        st.success("Prédictions réalisées sur le fichier uploadé.")
                        st.dataframe(out.head(50), use_container_width=True)
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Télécharger les prédictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
                elif use_test:
                    test_df = load_parquet(PATHS["models_data"] / "test.parquet")
                    if test_df is None:
                        st.error("Test set introuvable.")
                    else:
                        # test.parquet est déjà pré-traité (contient X et y). On essaie d'identifier la cible si présente.
                        df = test_df.copy()
                        # Heuristic: if 'Churn' or 'churn' in columns -> target; otherwise no target column
                        target_col = None
                        for col in ["Churn", "churn", "target"]:
                            if col in df.columns:
                                target_col = col
                                break
                        feature_df = df.drop(columns=[target_col]) if target_col in df.columns else df
                        X = feature_df.values if hasattr(feature_df, "values") else feature_df
                        y_proba = selected_model.predict_proba(X)[:, 1] if hasattr(selected_model, "predict_proba") else None
                        y_pred = (y_proba >= 0.5).astype(int) if y_proba is not None else selected_model.predict(X)
                        out = df.copy()
                        out["churn_pred"] = y_pred
                        if y_proba is not None:
                            out["churn_proba"] = y_proba
                        st.success("Prédictions réalisées sur le test set.")
                        st.dataframe(out.head(50), use_container_width=True)
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Télécharger les prédictions (CSV)", data=csv, file_name="predictions_test.csv", mime="text/csv")
                else:
                    st.info("Veuillez uploader un CSV ou cocher l'option test set.")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction par lot: {e}")


# =============================
# Data Tab
# =============================
with tab_data:
    st.subheader("Aperçu des données")
    train_df, test_df = load_train_test()
    tabs = st.tabs(["Train", "Test", "Processed (global)"])
    with tabs[0]:
        if train_df is not None:
            st.dataframe(train_df.head(50), use_container_width=True)
        else:
            st.info("train.parquet introuvable")
    with tabs[1]:
        if test_df is not None:
            st.dataframe(test_df.head(50), use_container_width=True)
        else:
            st.info("test.parquet introuvable")
    with tabs[2]:
        processed = load_processed_dataset()
        if processed is not None:
            st.dataframe(processed.head(50), use_container_width=True)
        else:
            st.info("processed dataset introuvable")

    with st.expander("Pourquoi explorer les données ?"):
        st.write(
            """
            Explorer l'échantillon de données permet de **valider le schéma**, de **détecter des valeurs aberrantes** et de mieux comprendre les profils clients.
            """
        )


# =============================
# Report Tab
# =============================
with tab_report:
    st.subheader("Synthèse et Recommandations")
    st.markdown(
        """
        - **Objectif**: Identifier les clients à risque de churn pour cibler des actions de rétention.
        - **Modèle recommandé**: basé sur l'AUC et le compromis F1, privilégier le modèle le plus performant dans l'onglet Performances.
        - **Stratégie**:
          - Définir un **seuil d'intervention** à partir de la courbe ROC selon les coûts métiers.
          - Utiliser l'**importance des features** pour construire des **offres ciblées** (ex: engagement, options).
          - Mettre en place un **suivi régulier** des métriques pour éviter la dérive.
        - **Prochaines étapes**:
          - Collecter du feedback métier et ajuster le seuil.
          - Tester des modèles complémentaires (XGBoost, calibration des probabilités).
          - Déployer un scoring batch hebdomadaire et journaliser les performances.
        """
    )

    st.caption("Ce rapport est généré automatiquement à partir des résultats et artifacts présents dans le projet.")

