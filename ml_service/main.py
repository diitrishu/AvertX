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

IMPORTANT -- this service must stay behind the same three-way output
contract as backend/model.py (classify_sif / classify_rule). An earlier
version of this file reimplemented raw threshold logic directly
(`"Yes" if sif_prob_yes >= SIF_THRESHOLD else "No"`) instead of importing
sif_output_contract.py / rule_output_contract.py, which silently brought
back the exact problem those contracts exist to prevent: a low/mid
-confidence guess (e.g. 44.8% on unrelated text) rendered as a confident
"Yes" instead of NEEDS_REVIEW. Any future change to this file must keep
calling classify_sif / classify_rule, not reimplement their logic here.
"""

import os, sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.onnx_encoder import OnnxSentenceEncoder
from backend.sif_output_contract import classify_sif
from backend.rule_output_contract import classify_rule

app = FastAPI(title="SIF-Guard ML Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

ARTIFACTS = Path(__file__).parent.parent / "model_artifacts"

# ── Load ONNX encoder (no torch needed) ─────────────────────────────
encoder  = OnnxSentenceEncoder(ARTIFACTS / "onnx_encoder")
clf_sif  = joblib.load(ARTIFACTS / "sif_classifier.joblib")
clf_rule = joblib.load(ARTIFACTS / "rule_classifier.joblib")

import json
with open(ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

# Same as backend/model.py: RULE_SEVERITY drives the risk-score formula's
# severity term. SIF_THRESHOLD is NOT used for the sif_potential decision
# any more (classify_sif's own thresholds are) -- kept only because
# compute_risk_score below reads RULE_SEVERITY, matching backend/model.py.
RULE_SEVERITY = metadata["rule_severity"]

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
    top_contributing_phrases: list = []
    sif_reasons: list = []
    sif_model_name: str = ""
    sif_model_version: str = ""
    sif_decision_thresholds: dict = {}
    sif_prediction_timestamp: str = ""
    rule_needs_review: bool = False
    rule_review_reason: str | None = None
    rule_runner_up: str | None = None
    rule_runner_up_confidence: float | None = None
    rule_model_name: str | None = None
    rule_model_version: str | None = None
    rule_prediction_timestamp: str | None = None


def compute_risk_score(sif_confidence: float, predicted_rule: str) -> int:
    """
    Mirrors backend/model.py::compute_risk_score's weighting (confidence
    0.60 / recurrence 0.25 / severity 0.15) so risk scores mean the same
    thing whether the app is running local-dev (in-process model.py) or
    this deployed microservice. recurrence_factor is always 0 here --
    this stateless service has no access to the training CSV that builds
    the recurrence table (see backend/main.py's startup comment), so
    that 25% of the weight simply contributes nothing rather than being
    silently redistributed into a different-looking formula.
    """
    confidence = float(np.clip(sif_confidence, 0, 1))
    recurrence_factor = 0.0  # unavailable in this stateless service
    severity = RULE_SEVERITY.get(predicted_rule, 0.5)
    risk = 100.0 * (0.60 * confidence + 0.25 * recurrence_factor + 0.15 * severity)
    return int(round(np.clip(risk, 0, 100)))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # ── SIF: three-way contract (Yes / No / NEEDS_REVIEW) ──────────────
    # No LIME here (that stack isn't loaded in this low-memory service --
    # see backend/main.py's proxy-path explain_prediction stub), so
    # classify_sif's reasons are probability-band-only, same as any call
    # made without lime_phrases.
    sif_prediction = classify_sif(
        report_text=req.report_text,
        classifier=clf_sif,
        encoder=encoder,
    )
    sif_label = sif_prediction.sif_potential
    sif_prob_yes = sif_prediction.confidence

    # ── Rule: three-way contract, run whenever SIF isn't a confident "No" ──
    if sif_label in ("Yes", "NEEDS_REVIEW"):
        rule_prediction = classify_rule(
            report_text=req.report_text,
            classifier=clf_rule,
            encoder=encoder,
        )
        rule_pred = rule_prediction.rule_prediction
        rule_conf = rule_prediction.confidence
    else:
        rule_prediction = None
        rule_pred, rule_conf = "N/A", 0.0

    risk = compute_risk_score(sif_prob_yes, rule_pred)

    result = {
        "sif_potential": sif_label,
        "confidence": round(sif_prob_yes, 4),
        "life_saving_rule": rule_pred,
        "rule_confidence": round(rule_conf, 4),
        "risk_score": risk,
        "top_contributing_phrases": [],
    }
    result.update(sif_prediction.as_dict())

    if rule_prediction is not None:
        result["rule_needs_review"] = rule_prediction.needs_review
        result["rule_review_reason"] = rule_prediction.review_reason
        result["rule_runner_up"] = rule_prediction.runner_up_rule
        result["rule_runner_up_confidence"] = rule_prediction.runner_up_confidence
        result["rule_model_name"] = rule_prediction.model_name
        result["rule_model_version"] = rule_prediction.model_version
        result["rule_prediction_timestamp"] = rule_prediction.timestamp
    else:
        result["rule_needs_review"] = False
        result["rule_review_reason"] = None
        result["rule_runner_up"] = None
        result["rule_runner_up_confidence"] = None
        result["rule_model_name"] = None
        result["rule_model_version"] = None
        result["rule_prediction_timestamp"] = None

    return PredictResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml"}
