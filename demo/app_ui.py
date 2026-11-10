"""
Application de démonstration : Maintenance prédictive de capteurs IoT
Jeu de données : Intel Berkeley Research Lab (2004)
Replay de données historiques. Ce n'est pas une surveillance en temps réel.
"""

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT (STYLE SOBRE / ACADÉMIQUE)
# ==============================================================================
st.set_page_config(
    page_title="Maintenance prédictive IoT - Intel Berkeley Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS épuré : thème clair, sobre, type rapport de laboratoire
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* 1. Base App and Global Text */
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #F9FAFB !important;
        color: #1F2937 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #F9FAFB !important;
    }

    /* 2. Sidebar Full Light Styling */
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"], section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
        color: #1F2937 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1F2937 !important;
    }

    [data-testid="stSidebar"] small, [data-testid="stSidebar"] .stCaption {
        color: #6B7280 !important;
    }

    /* 3. Streamlit Native Inputs, Selectboxes, Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] span {
        color: #1F2937 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
    }
    li[role="option"] {
        color: #1F2937 !important;
        background-color: #FFFFFF !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #F3F4F6 !important;
        color: #111827 !important;
    }

    /* 4. Streamlit Sliders & Labels */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="slider"] div {
        color: #1F2937 !important;
    }

    /* 5. Checkboxes */
    [data-testid="stCheckbox"] label span {
        color: #1F2937 !important;
    }

    /* 6. Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
    }

    /* 7. Native Tables and DataFrames */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #FFFFFF !important;
    }

    /* 8. Headings and Body text */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #111827 !important;
    }
    p, span, li, .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: #1F2937 !important;
    }

    /* Bannière obligatoire */
    .mandatory-banner {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Cartouche en-tête académique */
    .report-header {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .report-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827 !important;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.01em;
    }
    .report-meta {
        font-size: 0.86rem;
        color: #4B5563 !important;
        margin: 0;
    }

    /* Cartes d'indicateurs sobres */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6B7280 !important;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #6B7280 !important;
        margin-top: 0.2rem;
    }

    /* Badges d'état */
    .status-normal {
        display: inline-block;
        padding: 4px 10px;
        background-color: #ECFDF5;
        color: #065F46 !important;
        border: 1px solid #A7F3D0;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-alert {
        display: inline-block;
        padding: 4px 10px;
        background-color: #FEF2F2;
        color: #991B1B !important;
        border: 1px solid #FECACA;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Panneau d'explication */
    .explanation-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 0.75rem;
    }
    .explanation-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #111827 !important;
        margin-bottom: 0.5rem;
    }

    /* Onglets et formulaires */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F3F4F6 !important;
        border-radius: 6px !important;
        padding: 3px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #4B5563 !important;
        padding: 6px 14px !important;
        font-size: 0.88rem !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
    }

    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 0.85rem;
    }
    th, td {
        border: 1px solid #E5E7EB;
        padding: 6px 10px;
        text-align: left;
    }
    th {
        background-color: #F9FAFB;
        color: #374151 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CHARGEMENT STRICT DES FICHIERS DE L'ÉTUDE (AVEC VÉRIFICATION ET CACHE)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "demo" else Path(__file__).resolve().parent

PATHS = {
    "model": BASE_DIR / "models" / "best_model_intel.joblib",
    "model_alt": BASE_DIR / "best_model_intel.joblib",
    "final_results": BASE_DIR / "reports" / "results" / "intel_final_results.json",
    "oof": BASE_DIR / "reports" / "results" / "intel_oof_predictions.parquet",
    "features": BASE_DIR / "data" / "processed" / "features_intel.parquet",
    "onsets": BASE_DIR / "data" / "processed" / "mote_onsets.csv",
    "split": BASE_DIR / "data" / "processed" / "split_nodes.json",
}

@st.cache_resource(show_spinner=False)
def load_study_resources():
    """Charge le modèle, les métadonnées et résultats de l'étude."""
    # 1. Modèle
    model_path = PATHS["model"] if PATHS["model"].exists() else PATHS["model_alt"]
    if not model_path.exists():
        st.error(f"Fichier manquant : {PATHS['model']} ou {PATHS['model_alt']}")
        st.stop()
    bundle = joblib.load(model_path)
    
    # 2. Résultats finaux
    if not PATHS["final_results"].exists():
        st.error(f"Fichier manquant : {PATHS['final_results']}")
        st.stop()
    with open(PATHS["final_results"], "r", encoding="utf-8") as f:
        final_results = json.load(f)

    # 3. Répartition dev/test
    if not PATHS["split"].exists():
        st.error(f"Fichier manquant : {PATHS['split']}")
        st.stop()
    with open(PATHS["split"], "r", encoding="utf-8") as f:
        split_nodes = json.load(f)

    # 4. Onsets de panne
    if not PATHS["onsets"].exists():
        st.error(f"Fichier manquant : {PATHS['onsets']}")
        st.stop()
    onsets_df = pd.read_csv(PATHS["onsets"], index_col="moteid", parse_dates=["first_ts", "last_ts", "t_onset", "t_lowv"])

    return bundle, final_results, split_nodes, onsets_df

@st.cache_data(show_spinner=False)
def load_oof_predictions():
    """Charge les prédictions hors-échantillon (out-of-fold) des nœuds de dév."""
    if not PATHS["oof"].exists():
        st.error(f"Fichier manquant : {PATHS['oof']}")
        st.stop()
    return pd.read_parquet(PATHS["oof"])

@st.cache_data(show_spinner=False)
def load_node_features(mote_id: int):
    """Charge les 48 features pour un nœud spécifique depuis le fichier Parquet."""
    if not PATHS["features"].exists():
        st.error(f"Fichier manquant : {PATHS['features']}")
        st.stop()
    df = pd.read_parquet(PATHS["features"], filters=[("moteid", "=", int(mote_id))])
    if "timestamp" not in df.columns:
        df = df.reset_index()
    return df.sort_values("timestamp").reset_index(drop=True)

# Initialisation des données
bundle, final_results, split_nodes, onsets_df = load_study_resources()
model = bundle["model"]
FEATS = bundle["features"]
H_PARAM = int(final_results.get("horizon_h", 24))
THR_NODE = float(final_results.get("threshold", 0.1078787))
THR_ROW_OOF = float(bundle.get("threshold_oof", 0.9372587))

DEV_NODES = sorted(split_nodes.get("dev_nodes", []))
TEST_NODES = sorted(split_nodes.get("test_nodes", []))
ALL_NODES = sorted(int(m) for m in onsets_df.index)
USABLE_NODES = sorted(DEV_NODES + TEST_NODES)
UNUSABLE_NODES = sorted(set(ALL_NODES) - set(USABLE_NODES))

# ==============================================================================
# LOGIQUE MÉTIER STRICTE DE L'ÉTUDE (REPRODUCTION FORMELLE DE L'ÉVALUATION)
# ==============================================================================
H1 = np.timedelta64(1, "h")
GAP = np.timedelta64(12, "h")
MIN_LEAD_H = 6
MAX_LEAD_H = 168

def compute_episodes(alert_timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Regroupe les minutes d'alerte espacées de moins de 12 h en épisodes."""
    if len(alert_timestamps) == 0:
        empty = np.array([], dtype="datetime64[ns]")
        return empty, empty
    br = np.where(np.diff(alert_timestamps) > GAP)[0]
    starts = np.r_[alert_timestamps[0], alert_timestamps[br + 1]]
    ends = np.r_[alert_timestamps[br], alert_timestamps[-1]]
    return starts, ends

def evaluate_node_strict(ts_arr: np.ndarray, al_arr: np.ndarray, onset_val: Any) -> Dict[str, Any]:
    """
    Règle stricte d'évaluation d'un nœud selon le protocole de recherche :
    - Alerte utile : épisode débutant entre 168 h et 6 h avant la panne.
    - Fausse alerte : épisode débutant plus de 168 h avant la panne.
    """
    st_ep, en_ep = compute_episodes(al_arr)
    if pd.notna(onset_val):
        on = np.datetime64(onset_val)
        lead_st = (on - st_ep) / H1
        useful = lead_st[(lead_st >= MIN_LEAD_H) & (lead_st <= MAX_LEAD_H)]
        late = ((lead_st >= 0) & (lead_st < MIN_LEAD_H)).any()
        n_fa = int((lead_st > MAX_LEAD_H).sum())
        detected = len(useful) > 0
        lead = float(useful.max()) if detected else np.nan
        outcome = "detecte" if detected else ("tardif" if late else "manque")
        end_normal = min(on - np.timedelta64(MAX_LEAD_H, "h"), ts_arr[-1])
    else:
        n_fa = len(st_ep)
        lead = np.nan
        outcome = "sain"
        end_normal = ts_arr[-1]
    
    normal_days = max((end_normal - ts_arr[0]) / np.timedelta64(1, "D"), 0.0) if len(ts_arr) > 0 else 0.0
    return {
        "outcome": outcome,
        "lead_h": lead,
        "false_alarms": n_fa,
        "normal_days": normal_days,
        "n_episodes": len(st_ep),
    }

def get_node_telemetry_and_scores(mote_id: int) -> Tuple[pd.DataFrame, str]:
    """
    Fournit la série temporelle exacte et le score sans fuite :
    - Pour les nœuds de développement : scores out-of-fold (modèle n'ayant jamais vu ce nœud).
    - Pour les nœuds de test verrouillés : inférence avec le modèle final.
    """
    is_dev = mote_id in DEV_NODES
    group_label = "Développement" if is_dev else ("Test (réservé)" if mote_id in TEST_NODES else "Non exploitable")
    
    if mote_id in UNUSABLE_NODES:
        return pd.DataFrame(), group_label

    if is_dev:
        oof_df = load_oof_predictions()
        sub_oof = oof_df[oof_df["moteid"] == mote_id].sort_values("timestamp").reset_index(drop=True)
        # Compléter avec les features physiques pour les graphiques secondaires
        feat_df = load_node_features(mote_id)
        merged = pd.merge(sub_oof, feat_df, on=["timestamp", "moteid"], how="inner", suffixes=("", "_feat"))
        merged["_score"] = merged["LogisticRegression"]
        return merged, group_label
    else:
        feat_df = load_node_features(mote_id)
        if len(feat_df) == 0:
            return pd.DataFrame(), group_label
        X = feat_df[FEATS]
        scores = model.predict_proba(X)[:, 1]
        feat_df["_score"] = scores
        return feat_df, group_label

def compute_top_contributions(row: pd.Series, top_n: int = 5) -> pd.Series:
    """Calcule l'influence locale des variables (coefficient x variable standardisée en log-odds)."""
    sc = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    x_vals = row[FEATS].values.astype(float)
    z = (x_vals - sc.mean_) / sc.scale_
    contrib = z * lr.coef_[0]
    s = pd.Series(contrib, index=FEATS)
    return s.sort_values(key=np.abs, ascending=False).head(top_n)

# ==============================================================================
# 1. BANNIÈRE OBLIGATOIRE ET EN-TÊTE
# ==============================================================================
st.markdown(
    """
    <div class="mandatory-banner">
        Replay de données historiques (Intel Berkeley lab, 2004). Ce n'est pas une surveillance en temps réel.
    </div>
    <div class="report-header">
        <h1 class="report-title">Maintenance prédictive de capteurs IoT — Démonstrateur scientifique</h1>
        <p class="report-meta">
            Étude expérimentale sur 54 nœuds sans fil Mica2Dot • Grille temporelle de 10 minutes • Horizon de prédiction : 24 heures
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# BARRE LATÉRALE : SÉLECTEUR DE NŒUD ET PARAMÈTRES
# ==============================================================================
with st.sidebar:
    st.markdown("### Sélection du capteur")

    # Construction des étiquettes de sélection avec tag d'appartenance
    def format_node_option(nid: int) -> str:
        if nid in DEV_NODES:
            return f"Nœud {nid:02d} [Développement]"
        elif nid in TEST_NODES:
            return f"Nœud {nid:02d} [Test - réservé]"
        else:
            return f"Nœud {nid:02d} [Non exploitable]"

    selected_node_str = st.selectbox(
        "Nœud capteur",
        options=ALL_NODES,
        format_func=format_node_option,
        index=ALL_NODES.index(7) if 7 in ALL_NODES else 0,
        help="41 nœuds de développement (scores hors-échantillon), 10 nœuds de test verrouillés, 3 sans données.",
    )
    current_node = int(selected_node_str)
    if current_node in DEV_NODES:
        st.caption("Scores hors-échantillon (validation croisée par nœud)")
    elif current_node in TEST_NODES:
        st.caption("Modèle final, nœud jamais vu à l'entraînement")

    st.markdown("---")
    st.markdown("### Paramètres d'évaluation")
    st.markdown(
        f"""
        - **Seuil d'alerte (nœud) :** `{THR_NODE:.4f}`
        - **Horizon de prédiction :** `{H_PARAM} h`
        - **Délai utile :** `6 h à 168 h` avant panne
        - **Séparation épisodes :** `12 h`
        """
    )
    with st.expander("Note technique sur les seuils"):
        st.markdown(
            f"""
            - **Seuil d'alerte au niveau nœud :** `{THR_NODE:.4f}` (sélectionné pour respecter le budget de 0,02 fausse alerte par nœud-jour).
            - **Seuil F1 au niveau ligne (minute) :** `{THR_ROW_OOF:.3f}` (optimisé sur F1 hors-échantillon, précision 88,8 %, rappel 81,6 %).
            Ce dernier n'est pas le seuil opérationnel d'alerte.
            """
        )

# Récupération des données du nœud sélectionné
df_node, node_group = get_node_telemetry_and_scores(current_node)
node_onset_raw = onsets_df.loc[current_node, "t_onset"] if current_node in onsets_df.index else pd.NaT
node_fail_type = onsets_df.loc[current_node, "fail_type"] if current_node in onsets_df.index else "inconnu"

# ==============================================================================
# ONGLET PRINCIPAL DE NAVIGATION
# ==============================================================================
tab_replay, tab_manual, tab_fleet, tab_limits = st.tabs([
    "Replay historique (Mode B)",
    "Simulation manuelle (Mode A)",
    "Tableau de la flotte",
    "Limites de l'étude",
])

# ==============================================================================
# ONGLET 1 : REPLAY HISTORIQUE (MODE B)
# ==============================================================================
with tab_replay:
    if current_node in DEV_NODES:
        score_source = "Scores hors-échantillon (validation croisée par nœud)"
    elif current_node in TEST_NODES:
        score_source = "Modèle final, nœud jamais vu à l'entraînement"
    else:
        score_source = ""

    if len(df_node) == 0:
        ft_label = "valeurs impossibles" if node_fail_type == "garbage" else ("silence" if node_fail_type == "silent" else node_fail_type)
        st.warning(
            f"Le nœud {current_node} fait partie des 3 nœuds sans enregistrements exploitables dans le jeu traité. "
            f"Type de panne documenté : {ft_label}."
        )
    else:
        # Évaluation stricte du nœud
        ts_full = pd.to_datetime(df_node["timestamp"]).values
        scores_full = df_node["_score"].values
        alerts_full = ts_full[scores_full >= THR_NODE]
        eval_dict = evaluate_node_strict(ts_full, alerts_full, node_onset_raw)

        # En-tête du nœud et source des scores
        st.markdown(
            f"""
            <div style="margin-bottom: 0.75rem;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #111827;">Nœud {current_node:02d}</span>
                <span style="font-size: 0.88rem; color: #4B5563; margin-left: 8px;">— Groupe : <b>{node_group}</b> • <i>{score_source}</i></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Case à cocher pour afficher le résultat rétrospectif (en pleine largeur pour éviter toute troncature)
        show_hindsight = st.checkbox(
            "Afficher le résultat rétrospectif",
            value=False,
            help="Affiche le statut d'évaluation, le délai d'alerte et l'instant réel de panne.",
        )

        if not show_hindsight:
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 12px 16px; margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #1F2937; margin-bottom: 4px;">
                        Résultat masqué (rétrospectif). Cochez la case pour l'afficher.
                    </div>
                    <div style="font-size: 0.85rem; color: #6B7280;">
                        Groupe : {node_group} • {score_source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Indicateurs rétrospectifs du nœud (visibles uniquement si la case est cochée)
            c_stat, c_lead, c_fa, c_type = st.columns(4)
            with c_stat:
                outcome_fr = {
                    "detecte": "Détecté",
                    "tardif": "Tardif",
                    "manque": "Manqué",
                    "sain": "Aucune panne",
                    "aucune donnée": "Non exploitable",
                }.get(eval_dict["outcome"], eval_dict["outcome"])
                color_badge = "status-normal" if eval_dict["outcome"] == "detecte" else "status-alert"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Statut de détection</div>
                        <div style="margin-top: 4px;"><span class="{color_badge}">{outcome_fr}</span></div>
                        <div class="metric-caption">Groupe : {node_group}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_lead:
                lead_val = f"{eval_dict['lead_h']:.1f} h" if pd.notna(eval_dict["lead_h"]) else "—"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Délai d'alerte</div>
                        <div class="metric-value">{lead_val}</div>
                        <div class="metric-caption">temps entre le début de l'alerte et la panne</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_fa:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Fausses alertes</div>
                        <div class="metric-value">{eval_dict['false_alarms']}</div>
                        <div class="metric-caption">Épisodes débutant > 168 h avant panne</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_type:
                ft_label = "Valeurs impossibles" if node_fail_type == "garbage" else ("Silence" if node_fail_type == "silent" else node_fail_type)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Type de panne</div>
                        <div class="metric-value" style="font-size: 1.15rem;">{ft_label}</div>
                        <div class="metric-caption">Définition protocole</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Contrôles de replay temporel
        time_idx = st.slider(
            "Pas de temps de rejeu (t)",
            min_value=0,
            max_value=len(df_node) - 1,
            value=int(len(df_node) * 0.75),
            step=1,
            format="Pas %d",
            help="Déplacez le curseur pour afficher uniquement les données collectées jusqu'à cet instant.",
        )

        current_row = df_node.iloc[time_idx]
        current_time = pd.to_datetime(current_row["timestamp"])
        current_score = float(current_row["_score"])
        is_alert_now = current_score >= THR_NODE

        # État instantané à t
        st_color = "status-alert" if is_alert_now else "status-normal"
        st_label = "ALERTE" if is_alert_now else "NORMAL"
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <b>Instant t :</b> <code>{current_time.strftime('%Y-%m-%d %H:%M:%S')}</code> • 
                    <b>Score de risque :</b> <code>{current_score:.4f}</code> (seuil : <code>{THR_NODE:.4f}</code>)
                </div>
                <div><span class="{st_color}">{st_label}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Filtrage strict jusqu'à t (aucune observation future)
        df_visible = df_node.iloc[:time_idx + 1]

        # Tracé des courbes synchronisées (Plotly)
        if HAS_PLOTLY:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.55, 0.22, 0.23],
                subplot_titles=(
                    "Score de risque au cours du temps",
                    "Tension batterie (V)",
                    "Température ambiante (°C)"
                )
            )

            # Ligne de score
            fig.add_trace(
                go.Scatter(
                    x=df_visible["timestamp"],
                    y=df_visible["_score"],
                    mode="lines",
                    name="Score de risque",
                    line=dict(color="#1E3A8A", width=1.5),
                ),
                row=1, col=1
            )

            # Ligne de seuil
            fig.add_trace(
                go.Scatter(
                    x=[df_visible["timestamp"].iloc[0], df_visible["timestamp"].iloc[-1]],
                    y=[THR_NODE, THR_NODE],
                    mode="lines",
                    name=f"Seuil d'alerte ({THR_NODE:.4f})",
                    line=dict(color="#DC2626", width=1.2, dash="dash"),
                ),
                row=1, col=1
            )

            # Marqueurs d'alerte passés
            alerts_sub = df_visible[df_visible["_score"] >= THR_NODE]
            if len(alerts_sub) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=alerts_sub["timestamp"],
                        y=alerts_sub["_score"],
                        mode="markers",
                        name="Points en alerte",
                        marker=dict(color="#DC2626", size=5),
                    ),
                    row=1, col=1
                )

            # Information rétrospective (optionnelle)
            if show_hindsight and pd.notna(node_onset_raw):
                on_dt = pd.to_datetime(node_onset_raw)
                w_start = on_dt - pd.Timedelta(hours=24)
                
                # Zone 24h avant panne
                fig.add_vrect(
                    x0=w_start, x1=on_dt,
                    fillcolor="rgba(220, 38, 38, 0.08)",
                    layer="below", line_width=0,
                    annotation_text="Fenêtre 24 h avant panne",
                    annotation_position="top left",
                    row=1, col=1
                )
                # Ligne de panne
                fig.add_vline(
                    x=on_dt,
                    line_width=1.5,
                    line_dash="dot",
                    line_color="#991B1B",
                    annotation_text="Panne réelle",
                    annotation_position="bottom right",
                    row=1, col=1
                )

            # Sous-graphique Tension
            v_col = "voltage" if "voltage" in df_visible.columns else ("volt_mean_1h" if "volt_mean_1h" in df_visible.columns else None)
            if v_col:
                fig.add_trace(
                    go.Scatter(
                        x=df_visible["timestamp"],
                        y=df_visible[v_col],
                        mode="lines",
                        name="Tension (V)",
                        line=dict(color="#059669", width=1.0),
                    ),
                    row=2, col=1
                )

            # Sous-graphique Température
            t_col = "temp_mean_1h" if "temp_mean_1h" in df_visible.columns else None
            if t_col:
                fig.add_trace(
                    go.Scatter(
                        x=df_visible["timestamp"],
                        y=df_visible[t_col],
                        mode="lines",
                        name="Température (°C)",
                        line=dict(color="#D97706", width=1.0),
                    ),
                    row=3, col=1
                )

            fig.update_layout(
                height=520,
                margin=dict(l=40, r=20, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#1F2937", size=11),
                showlegend=False,
            )
            fig.update_yaxes(title_text="Score", range=[0.0, 1.02], gridcolor="#F3F4F6", row=1, col=1)
            fig.update_yaxes(title_text="Volts", gridcolor="#F3F4F6", row=2, col=1)
            fig.update_yaxes(title_text="°C", gridcolor="#F3F4F6", row=3, col=1)
            fig.update_xaxes(gridcolor="#F3F4F6")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df_visible.set_index("timestamp")[["_score"]])

        # 6. Panneau d'explication locale à l'instant t
        st.markdown("### Facteurs explicatifs à l'instant t")
        top_contribs = compute_top_contributions(current_row, top_n=5)
        
        c_expl1, c_expl2 = st.columns([1.2, 0.8])
        with c_expl1:
            st.markdown(
                """
                <div class="explanation-box">
                    <div class="explanation-title">Top 5 des contributions locales (log-odds)</div>
                """,
                unsafe_allow_html=True,
            )
            for feat_name, val in top_contribs.items():
                raw_v = current_row[feat_name]
                dir_txt = "augmente le risque" if val > 0 else "diminue le risque"
                color_txt = "#B91C1C" if val > 0 else "#047857"
                st.markdown(
                    f"- <code>{feat_name}</code> = <b>{raw_v:.3f}</b> : "
                    f"<span style='color:{color_txt}; font-weight:600;'>{dir_txt} ({val:+.2f})</span>",
                    unsafe_allow_html=True
                )
            st.markdown(
                """
                    <div style="font-size:0.78rem; color:#6B7280; margin-top:8px; border-top:1px solid #E5E7EB; padding-top:6px;">
                        <i>Influence sur le score (log-odds), pas une cause physique ; le score n'est pas une probabilité calibrée.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_expl2:
            st.markdown(
                f"""
                <div class="explanation-box">
                    <div class="explanation-title">Recommandation protocolaire</div>
                    <div style="font-size:0.88rem; color:#374151; margin-top:6px;">
                        {'Planifier un remplacement de pile sous 24 h (score >= seuil).' if is_alert_now else 'Aucune action requise (score en dessous du seuil opérationnel).'}
                    </div>
                    <div style="font-size:0.78rem; color:#6B7280; margin-top:10px;">
                        Rappel : à l'échelle du nœud, le modèle statistique ne surpasse pas significativement la règle heuristique basée sur la tension.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# ONGLET 2 : SIMULATION MANUELLE (MODE A)
# ==============================================================================
with tab_manual:
    st.markdown("### Exploration interactive à partir d'un vecteur réel")
    st.write(
        "Le modèle de régression logistique requiert l'ensemble des 48 descripteurs temporels. "
        "Pour éviter d'inventer des données, la simulation part d'une **ligne réelle observée** "
        "(nœud et date choisis) et permet de modifier localement quelques variables clés. "
        "Les 43 autres variables demeurent issues de l'enregistrement réel sélectionné."
    )

    if len(df_node) == 0:
        st.warning("Veuillez sélectionner un nœud possédant des enregistrements exploitables.")
    else:
        # Sélection de la date de référence
        obs_idx = st.slider(
            "Point de départ réel (nœud courant)",
            min_value=0,
            max_value=len(df_node) - 1,
            value=int(len(df_node) * 0.5),
            step=1,
            format="Observation %d",
            key="sim_obs_slider",
        )
        base_row = df_node.iloc[obs_idx].copy()
        base_time = pd.to_datetime(base_row["timestamp"])

        st.caption(f"Vecteur de référence issu du nœud {current_node} au {base_time.strftime('%Y-%m-%d %H:%M:%S')}")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_volt = st.slider("Tension instantanée (voltage, V)", 2.00, 3.00, float(base_row.get("voltage", 2.60)), 0.01)
            sim_v6h = st.slider("Moyenne 6 h de tension (volt_mean_6h, V)", 2.00, 3.00, float(base_row.get("volt_mean_6h", 2.60)), 0.01)
            sim_t1h = st.slider("Moyenne 1 h de température (temp_mean_1h, °C)", 10.0, 45.0, float(base_row.get("temp_mean_1h", 22.0)), 0.5)
        with sim_col2:
            sim_t24h = st.slider("Moyenne 24 h de température (temp_mean_24h, °C)", 10.0, 45.0, float(base_row.get("temp_mean_24h", 22.0)), 0.5)
            sim_h24h = st.slider("Moyenne 24 h d'humidité (hum_mean_24h, %)", 10.0, 95.0, float(base_row.get("hum_mean_24h", 40.0)), 1.0)
            st.info("Les 43 autres caractéristiques sont préservées de la ligne réelle choisie.")

        # Construction du vecteur modifié
        sim_vector = base_row.copy()
        sim_vector["voltage"] = sim_volt
        sim_vector["volt_mean_6h"] = sim_v6h
        sim_vector["temp_mean_1h"] = sim_t1h
        sim_vector["temp_mean_24h"] = sim_t24h
        sim_vector["hum_mean_24h"] = sim_h24h

        X_sim = pd.DataFrame([sim_vector[FEATS].astype(float)])
        sim_score = float(model.predict_proba(X_sim)[0, 1])
        sim_alert = sim_score >= THR_NODE

        res_box_color = "#FEF2F2" if sim_alert else "#ECFDF5"
        res_border = "#FECACA" if sim_alert else "#A7F3D0"
        res_txt_color = "#991B1B" if sim_alert else "#065F46"
        res_label = "ALERTE DÉCLENCHÉE (seuil dépassé)" if sim_alert else "ÉTAT NORMAL (sous le seuil)"

        # Notice d'avertissement explicite
        st.markdown(
            """
            <div style="background-color: #FEF3C7; border: 1px solid #FCD34D; color: #92400E; border-radius: 6px; padding: 12px 14px; margin-top: 1rem; font-size: 0.88rem; line-height: 1.45;">
                <b>Scénario hypothétique :</b> seules 5 variables sont modifiées. Les variables liées (par exemple volt_min_6h ou volt_low_share_24h) ne sont pas recalculées, donc une combinaison peut être physiquement incohérente. Le score obtenu n'est pas la prédiction d'un capteur réel.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background-color: {res_box_color}; border: 1px solid {res_border}; border-radius: 6px; padding: 14px; margin-top: 0.75rem;">
                <div style="font-size:0.8rem; text-transform:uppercase; color:{res_txt_color}; font-weight:700;">Résultat de l'inférence</div>
                <div style="font-size:1.6rem; font-weight:700; color:{res_txt_color}; font-family:'JetBrains Mono', monospace; margin: 4px 0;">
                    Score de risque : {sim_score:.4f}
                </div>
                <div style="font-size:0.9rem; color:{res_txt_color}; font-weight:600;">
                    {res_label} • Seuil nœud : {THR_NODE:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Facteurs explicatifs du score simulé")
        sim_contribs = compute_top_contributions(sim_vector, top_n=5)
        for fn, cval in sim_contribs.items():
            st.markdown(
                f"- ` {fn} ` = **{float(sim_vector[fn]):.3f}** : "
                f"{'augmente le risque' if cval > 0 else 'diminue le risque'} ({cval:+.2f} en log-odds)"
            )
        st.caption("Influence sur le score (log-odds), pas une cause physique ; le score n'est pas une probabilité calibrée.")

# ==============================================================================
# ONGLET 3 : TABLEAU DE LA FLOTTE (VALIDATION DES FAITS CONTRÔLÉS)
# ==============================================================================
with tab_fleet:
    st.markdown("### Synthèse globale des 54 nœuds de l'étude")

    # Calcul complet de la flotte
    @st.cache_data(show_spinner=False)
    def compute_fleet_table():
        rows = []
        oof_all = load_oof_predictions()
        feats_all = pd.read_parquet(PATHS["features"])
        if "timestamp" not in feats_all.columns:
            feats_all = feats_all.reset_index()

        for m in ALL_NODES:
            ft = onsets_df.loc[m, "fail_type"] if m in onsets_df.index else "inconnu"
            onset_val = onsets_df.loc[m, "t_onset"] if m in onsets_df.index else pd.NaT

            if m in UNUSABLE_NODES:
                rows.append({
                    "node": m,
                    "split": "Non exploitable",
                    "fail_type": ft,
                    "outcome": "aucune donnée",
                    "lead_h": np.nan,
                    "false_alarms": 0,
                    "normal_days": 0.0,
                })
                continue

            if m in DEV_NODES:
                sub = oof_all[oof_all["moteid"] == m].sort_values("timestamp")
                ts_arr = pd.to_datetime(sub["timestamp"]).values
                sc_arr = sub["LogisticRegression"].values
                al_arr = ts_arr[sc_arr >= THR_NODE]
                res = evaluate_node_strict(ts_arr, al_arr, onset_val)
                rows.append({"node": m, "split": "Développement", "fail_type": ft, **res})
            else:
                sub = feats_all[feats_all["moteid"] == m].sort_values("timestamp")
                if len(sub) == 0:
                    rows.append({
                        "node": m, "split": "Test", "fail_type": ft, "outcome": "aucune donnée",
                        "lead_h": np.nan, "false_alarms": 0, "normal_days": 0.0
                    })
                    continue
                ts_arr = pd.to_datetime(sub["timestamp"]).values
                sc_arr = model.predict_proba(sub[FEATS])[:, 1]
                al_arr = ts_arr[sc_arr >= THR_NODE]
                res = evaluate_node_strict(ts_arr, al_arr, onset_val)
                rows.append({"node": m, "split": "Test", "fail_type": ft, **res})

        return pd.DataFrame(rows)

    df_fleet = compute_fleet_table()

    # Résumés Développement et Test (faits autorisés)
    dev_f = df_fleet[df_fleet["split"] == "Développement"]
    test_f = df_fleet[df_fleet["split"] == "Test"]

    col_d, col_t = st.columns(2)
    with col_d:
        n_det_d = (dev_f["outcome"] == "detecte").sum()
        pct_d = 100 * n_det_d / len(dev_f)
        med_lead_d = dev_f["lead_h"].median()
        fa_rate_d = dev_f["false_alarms"].sum() / dev_f["normal_days"].sum()
        missed_d = dev_f[dev_f["outcome"] != "detecte"]["node"].tolist()

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nœuds de développement ({len(dev_f)} nœuds, scores OOF)</div>
                <div class="metric-value">{n_det_d}/{len(dev_f)} détectés ({pct_d:.1f} %)</div>
                <div style="font-size:0.85rem; color:#374151; margin-top:4px;">
                    • Délai d'anticipation médian : <b>{med_lead_d:.1f} h</b><br/>
                    • Taux de fausses alertes : <b>{fa_rate_d:.3f} / nœud-jour</b><br/>
                    • Nœuds non détectés : <b>{missed_d}</b> (panne silence 20, 50 et nœud 16)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t:
        n_det_t = (test_f["outcome"] == "detecte").sum()
        pct_t = 100 * n_det_t / len(test_f)
        med_lead_t = test_f["lead_h"].median()
        fa_rate_t = test_f["false_alarms"].sum() / test_f["normal_days"].sum()
        missed_t = test_f[test_f["outcome"] != "detecte"]["node"].tolist()

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nœuds de test verrouillés ({len(test_f)} nœuds, modèle final)</div>
                <div class="metric-value">{n_det_t}/{len(test_f)} détectés ({pct_t:.1f} %)</div>
                <div style="font-size:0.85rem; color:#374151; margin-top:4px;">
                    • Délai d'anticipation médian : <b>{med_lead_t:.1f} h</b><br/>
                    • Taux de fausses alertes : <b>{fa_rate_t:.3f} / nœud-jour</b><br/>
                    • Nœuds non détectés : <b>{missed_t}</b> (panne silence nœud 8)<br/>
                    • Métriques ligne (minute) : PR-AUC <b>0.686</b>, ROC-AUC <b>0.987</b>, taux base <b>6,0 %</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Détail par nœud individuel")
    view_table = df_fleet[["node", "split", "fail_type", "outcome", "lead_h", "false_alarms"]].copy()
    view_table.columns = ["Nœud", "Groupe", "Type de panne", "Résultat", "Délai (h)", "Fausses alertes"]
    view_table["Type de panne"] = view_table["Type de panne"].replace({
        "garbage": "Valeurs impossibles",
        "silent": "Silence",
    })
    view_table["Résultat"] = view_table["Résultat"].replace({
        "detecte": "Détecté", "manque": "Manqué", "tardif": "Tardif", "aucune donnée": "Non exploitable"
    })
    view_table["Délai (h)"] = view_table["Délai (h)"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    st.dataframe(
        view_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Nœud": st.column_config.NumberColumn("Nœud", width="small"),
            "Groupe": st.column_config.TextColumn("Groupe", width="medium"),
            "Type de panne": st.column_config.TextColumn("Type de panne", width="medium"),
            "Résultat": st.column_config.TextColumn("Résultat", width="medium"),
            "Délai (h)": st.column_config.TextColumn("Délai (h)", width="small"),
            "Fausses alertes": st.column_config.NumberColumn("Fausses alertes", width="medium"),
        },
    )

# ==============================================================================
# ONGLET 4 : LIMITES DE L'ÉTUDE (SECTIONS FORMELLES)
# ==============================================================================
with tab_limits:
    st.markdown("### Limites méthodologiques et opérationnelles de l'étude")

    st.markdown(
        """
        1. **Échantillon de test réduit :** L'évaluation finale repose sur **10 nœuds de test uniquement**, 
           ce qui limite la portée statistique des conclusions et justifie une grande prudence sur la généralisation.
        
        2. **Cause de défaillance unique :** Toutes les pannes documentées dans ce jeu de données partagent 
           la même origine physique : l'épuisement progressif de la pile d'alimentation (piles bouton lithium/alcalines).
        
        3. **Environnement intérieur contrôlé :** Le réseau a été déployé dans les bureaux du laboratoire d'Intel Berkeley, 
           et non en milieu extérieur, industriel ou urbain. La dynamique thermique et hygrométrique reflète celle d'un bâtiment climatisé.
        
        4. **Définition de panne construite a posteriori :** La condition de défaillance a été formalisée à partir des données 
           (au moins 50 % de valeurs aberrantes de température hors 0–45 °C ou d'humidité hors 0–100 % sur 60 mesures consécutives, 
           ou arrêt définitif d'émission).
        
        5. **Incapacité à prédire les pannes silencieuses :** Les nœuds cessant brutalement d'émettre sans phase préalable de dégradation 
           (nœuds 20 et 50 en développement, nœud 8 en test) ne peuvent être anticipés par le modèle statistique.
        
        6. **Absence de supériorité nette sur la règle simple au niveau nœud :**
           Une règle heuristique élémentaire (proportion des dernières 24 h avec une tension < 2,4 V) détecte **95,1 %** 
           des nœuds de développement et **9/10** nœuds de test.
           **Le modèle de régression logistique ne bat pas clairement cette règle simple au niveau nœud.**
           Son gain de lift à 24 h (passant de 12,1 à 17,8) provient principalement des descripteurs de température et d'humidité 
           qui capturent les premiers symptômes de la panne elle-même. Sans ces variables environnementales, le lift retombe à 11,9.
        """
    )
