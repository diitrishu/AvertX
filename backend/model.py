"""
Model loading, prediction, LIME explanation, and risk-score logic.
Loads saved artifacts from model_artifacts/ -- no retraining.

On first run (e.g. Render deploy), if embedding_model_ref.joblib is absent,
the encoder is downloaded directly from HuggingFace and cached via
sentence-transformers' own cache (~/.cache/torch/sentence_transformers).
This means the 87 MB joblib file does not need to be committed to git.
"""

import os, json
import numpy as np
import joblib
from lime.lime_text import LimeTextExplainer

from backend.sif_output_contract import classify_sif
from backend.rule_output_contract import classify_rule, NEEDS_REVIEW as RULE_NEEDS_REVIEW

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

# ── Load everything at import time (module-level singletons) ──────────
print("[model] Loading artifacts from", os.path.abspath(ARTIFACTS_DIR))

_encoder_path = os.path.join(ARTIFACTS_DIR, "embedding_model_ref.joblib")
if os.path.exists(_encoder_path):
    encoder = joblib.load(_encoder_path)
    print("[model] Loaded encoder from joblib cache")
else:
    from sentence_transformers import SentenceTransformer
    with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json")) as _f:
        _meta = json.load(_f)
    _model_id = _meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"[model] embedding_model_ref.joblib not found — downloading {_model_id} from HuggingFace ...")
    encoder = SentenceTransformer(_model_id)
    print("[model] Encoder downloaded OK")

clf_sif  = joblib.load(os.path.join(ARTIFACTS_DIR, "sif_classifier.joblib"))
clf_rule = joblib.load(os.path.join(ARTIFACTS_DIR, "rule_classifier.joblib"))

with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

# NOTE: SIF_THRESHOLD (a single cut point) is kept only for backward
# compatibility / logging. It is NOT used to decide sif_potential anymore --
# that decision now goes through backend.sif_output_contract.classify_sif,
# which uses two thresholds (THRESHOLD_LOW / THRESHOLD_HIGH) and can return
# "NEEDS_REVIEW" instead of forcing Yes/No. See that module's docstring for
# why a single threshold was replaced (Phase 11/12, benchmark_results.json,
# threshold_tradeoff_summary.md).
SIF_THRESHOLD = metadata["sif_threshold"]
RULE_SEVERITY = metadata["rule_severity"]

# ── LIME explainer ───────────────────────────────────────────────────
def _lime_predict_proba(texts):
    embs = encoder.encode(list(texts), convert_to_numpy=True)
    return clf_sif.predict_proba(embs)

lime_explainer = LimeTextExplainer(
    class_names=["No SIF", "Yes SIF"],
    random_state=42,
    split_expression=r"\W+",
)

def explain_prediction(report_text: str, num_samples: int = 200,
                       num_features: int = 6) -> list:
    """Returns [(word, weight), ...] sorted by |weight| desc."""
    exp = lime_explainer.explain_instance(
        report_text, _lime_predict_proba,
        num_samples=num_samples, num_features=num_features, labels=(1,),
    )
    return exp.as_list(label=1)

# ── Recurrence table (built from training CSV) ───────────────────────
EXCLUDE = {"Unmapped", "None", "none", "nan"}
recurrence_counts = {}
max_recurrence = 1

def build_recurrence_table(df):
    """Call once at startup with the training DataFrame."""
    global recurrence_counts, max_recurrence
    syn = df[df["source"] == "synthetic"]
    for _, row in syn.iterrows():
        key = (row["site"], row["activity"], row["life_saving_rule"])
        recurrence_counts[key] = recurrence_counts.get(key, 0) + 1
    max_recurrence = max(recurrence_counts.values()) if recurrence_counts else 1
    print(f"[model] Recurrence table: {len(recurrence_counts)} keys, max={max_recurrence}")

# ── Risk score ───────────────────────────────────────────────────────
def compute_risk_score(sif_confidence: float, predicted_rule: str,
                       site: str, activity: str) -> int:
    confidence = float(np.clip(sif_confidence, 0, 1))
    key = (site, activity, predicted_rule)
    raw = recurrence_counts.get(key, 0)
    recurrence_factor = min(raw / max_recurrence, 1.0) if max_recurrence > 0 else 0.0
    severity = RULE_SEVERITY.get(predicted_rule, 0.5)
    risk = 100.0 * (0.60 * confidence + 0.25 * recurrence_factor + 0.15 * severity)
    return int(round(np.clip(risk, 0, 100)))

# ── Main predict function ────────────────────────────────────────────
def predict(report_text: str, site: str = "Unknown",
            activity: str = "Unknown") -> dict:
    # ── SIF: three-way contract (Yes / No / NEEDS_REVIEW) ──────────────
    # LIME is computed first so classify_sif can fold the top contributing
    # phrases into its `reasons` list -- same phrases the API/UI already show,
    # no duplicate/second explanation method introduced.
    try:
        phrases = explain_prediction(report_text)
    except Exception:
        phrases = []

    sif_prediction = classify_sif(
        report_text=report_text,
        classifier=clf_sif,
        encoder=encoder,
        lime_phrases=phrases,
    )
    sif_label = sif_prediction.sif_potential          # "Yes" | "No" | "NEEDS_REVIEW"
    sif_prob_yes = sif_prediction.confidence

    # ── Rule: three-way contract, run whenever SIF isn't a confident "No" ──
    # A NEEDS_REVIEW SIF call may still turn out to be a real SIF event, so a
    # human reviewer benefits from seeing the model's best guess at *which*
    # Life-Saving Rule applies, clearly marked with its own confidence/review
    # status -- it is never silently upgraded to a confident classification.
    if sif_label in ("Yes", "NEEDS_REVIEW"):
        rule_prediction = classify_rule(
            report_text=report_text,
            classifier=clf_rule,
            encoder=encoder,
        )
        rule_pred = rule_prediction.rule_prediction    # rule name | "NEEDS_REVIEW"
        rule_conf = rule_prediction.confidence
    else:
        rule_prediction = None
        rule_pred, rule_conf = "N/A", 0.0

    # compute_risk_score expects a rule *label* to key the recurrence table;
    # NEEDS_REVIEW / "N/A" simply won't match any recorded key (raw=0), which
    # is the correct, conservative behaviour -- no fabricated recurrence signal.
    risk = compute_risk_score(sif_prob_yes, rule_pred, site, activity)

    result = {
        "sif_potential": sif_label,
        "confidence": round(sif_prob_yes, 4),
        "risk_score": risk,
        "life_saving_rule": rule_pred,
        "rule_confidence": round(rule_conf, 4),
        "top_contributing_phrases": [(w, round(float(wt), 4)) for w, wt in phrases],
    }

    # ── Additional contract fields (additive only -- nothing above is removed
    # or renamed, so existing consumers of predict() keep working unchanged) ──
    result.update(sif_prediction.as_dict())
    # as_dict() also sets "sif_potential" and "confidence" again with the same
    # values already set above -- harmless, kept for clarity that they come
    # from the contract.

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

    return result

print("[model] Ready.")
