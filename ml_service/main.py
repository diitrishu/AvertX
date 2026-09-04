"""
ml_service/main.py — Lightweight ML inference microservice.

Exposes a single endpoint: POST /predict
Called by the main backend (backend/main.py) instead of loading the model
in-process. This separates the ~350 MB ML stack from the ~150 MB API stack,
letting both run within free-tier 512 MB limits.

Deploy this as a SEPARATE Render web service with:
  Build:  pip install -r ml_service/requirements.txt
  Start:  uvicorn ml_service.main:app --host 0.0.0.0 --port $PORT
  Root:   (repo root)
"""

import os, json
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = FastAPI(title="SIF-Guard ML Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

ARTIFACTS = Path(__file__).parent.parent / "model_artifacts"

# ── Load ONNX encoder (no torch needed) ─────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.onnx_encoder import OnnxSentenceEncoder

encoder  = OnnxSentenceEncoder(ARTIFACTS / "onnx_encoder")
clf_sif  = joblib.load(ARTIFACTS / "sif_classifier.joblib")
clf_rule = joblib.load(ARTIFACTS / "rule_classifier.joblib")

with open(ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

SIF_THRESHOLD = metadata["sif_threshold"]
RULE_SEVERITY = metadata["rule_severity"]
SIF_CLASSES   = metadata["sif_classes"]   # ["No", "Yes"]
RULE_CLASSES  = metadata["rule_classes"]

print("[ml_service] Model loaded — ready.")


class PredictRequest(BaseModel):
    report_text: str
    site: str = "Unknown"
    activity: str = "Unknown"


class PredictResponse(BaseModel):
    sif_potential: str
    confidence: float
    life_saving_rule: str
    rule_confidence: float
    risk_score: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    emb = encoder.encode([req.report_text], convert_to_numpy=True)

    # SIF
    sif_proba   = clf_sif.predict_proba(emb)[0]
    sif_prob_yes = float(sif_proba[1])
    sif_label   = "Yes" if sif_prob_yes >= SIF_THRESHOLD else "No"

    # Rule
    if sif_label == "Yes":
        rule_proba   = clf_rule.predict_proba(emb)[0]
        rule_idx     = int(np.argmax(rule_proba))
        rule_pred    = RULE_CLASSES[rule_idx]
        rule_conf    = float(rule_proba[rule_idx])
    else:
        rule_pred, rule_conf = "N/A", 0.0

    # Risk score (simple: confidence * 60 + severity * 40)
    severity  = RULE_SEVERITY.get(rule_pred, 0.5)
    risk      = int(round(np.clip(100 * (0.60 * sif_prob_yes + 0.40 * severity), 0, 100)))

    return PredictResponse(
        sif_potential=sif_label,
        confidence=round(sif_prob_yes, 4),
        life_saving_rule=rule_pred,
        rule_confidence=round(rule_conf, 4),
        risk_score=risk,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml"}
