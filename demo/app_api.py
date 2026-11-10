"""
API FastAPI pour le modèle de maintenance prédictive (Intel Berkeley Lab)
Replay et inférence sur données historiques. Ce n'est pas une surveillance en temps réel.
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "demo" else Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model_intel.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "best_model_intel.joblib"
RESULTS_PATH = BASE_DIR / "reports" / "results" / "intel_final_results.json"

# Données d'état
bundle = None
model = None
FEATS = []
THR_NODE = 0.1078787
THR_ROW_OOF = 0.9372587

def load_study_model():
    global bundle, model, FEATS, THR_NODE, THR_ROW_OOF
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        FEATS = bundle["features"]
        THR_ROW_OOF = float(bundle.get("threshold_oof", 0.9372587))
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            res_dict = json.load(f)
            THR_NODE = float(res_dict.get("threshold", 0.1078787))

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_study_model()
    yield

app = FastAPI(
    title="API Maintenance Prédictive IoT (Intel Berkeley Lab)",
    description="Service d'évaluation du score de risque sur les 48 variables de l'étude historique.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeatureVectorInput(BaseModel):
    moteid: int = Field(..., description="Identifiant du capteur (1 à 54)")
    features: Dict[str, float] = Field(..., description="Dictionnaire des 48 descripteurs temporels")

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    moteid: int
    risk_score: float
    threshold_node: float
    threshold_row_f1: float
    alert: bool
    status: str
    top_contributions_log_odds: Dict[str, float]
    note: str

@app.get("/")
def root():
    return {
        "service": "Maintenance prédictive IoT - Replay historique",
        "dataset": "Intel Berkeley Research Lab (2004)",
        "model": "Régression Logistique (StandardScaler + LogisticRegression)",
        "threshold_node_level": round(THR_NODE, 4),
        "threshold_row_oof": round(THR_ROW_OOF, 4),
        "horizon_h": 24,
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_features": len(FEATS),
        "threshold_node": round(THR_NODE, 4),
        "threshold_row_f1": round(THR_ROW_OOF, 4),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: FeatureVectorInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non initialisé.")
    
    missing = [f for f in FEATS if f not in payload.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Caractéristiques manquantes ({len(missing)}): {missing[:5]}")
    
    x_vals = np.array([[payload.features[f] for f in FEATS]], dtype=float)
    score = float(model.predict_proba(x_vals)[0, 1])
    is_alert = score >= THR_NODE

    # Top contributions locales
    sc = model.named_steps["standardscaler"]
    lr = model.named_steps["logisticregression"]
    z = (x_vals[0] - sc.mean_) / sc.scale_
    contrib = z * lr.coef_[0]
    top_indices = np.argsort(np.abs(contrib))[::-1][:5]
    top_dict = {FEATS[i]: round(float(contrib[i]), 4) for i in top_indices}

    return PredictionResponse(
        moteid=payload.moteid,
        risk_score=round(score, 4),
        threshold_node=round(THR_NODE, 4),
        threshold_row_f1=round(THR_ROW_OOF, 4),
        alert=is_alert,
        status="ALERTE" if is_alert else "NORMAL",
        top_contributions_log_odds=top_dict,
        note="Influence sur le score (log-odds), pas une cause physique ; le score n'est pas une probabilité calibrée.",
    )
