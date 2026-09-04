#!/usr/bin/env python3
"""
SIH26165 -- SIF Precursor Detection Model Trainer
==================================================
Oil India Limited -- Serious Injury / Fatality precursor detection from
safety reports using semantic embeddings + logistic regression.

Trains two classifiers on sentence-transformer embeddings:
  1. Binary SIF classifier  (sif_potential: Yes / No) -- recall-first threshold
  2. Life-Saving Rule classifier (9 IOGP rules, trained on SIF-Yes only)

Also provides:
  - LIME-based word-level explainability for individual predictions
  - Composite SIF Risk Score (0-100) combining confidence, recurrence, rule severity

Embedding model: sentence-transformers/all-MiniLM-L6-v2  (384-dim)

Usage:
    py -3.12 train_model.py
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, average_precision_score
)
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

#                                 CONFIG                                
DATA_PATH       = "C:/Users/hp/Desktop/SIF/oil_safety_reports_merged.csv"
OUTPUT_DIR      = "C:/Users/hp/Desktop/SIF/model_artifacts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
MAX_ITER_SIF    = 1000
MAX_ITER_RULE   = 500

#                                 RISK SCORE FORMULA                    
# risk_score = round(100 * (
#     0.60 * classifier_confidence
#   + 0.25 * recurrence_factor
#   + 0.15 * rule_severity_weight
# ))
#
# Components:
#   confidence        -- P(SIF=Yes) from logistic regression, clamped [0,1]
#   recurrence_factor -- normalised count of training rows sharing the same
#                       (site, activity, predicted_rule) tuple, [0,1]
#   rule_severity     -- fixed severity per IOGP rule (see dict below)
#
# Rule severity rationale:
#   1.0  -- Rules where barrier failure   fatality is the direct mechanism:
#          Energy Isolation (electrocution/pressurised release),
#          Confined Space (H S / asphyxiation),
#          Hot Work (flash-back / explosion),
#          Line of Fire (struck-by / engulfment)
#   0.85 -- Working at Height (serious fall, but survival more likely with PPE),
#          Safe Mechanical Lifting (crush risk, but usually slower mechanism)
#   0.70 -- Work Authorisation (latent control gap, outcome depends on context),
#          Driving (vehicle incidents vary widely in severity),
#          Bypassing Safety Controls (indirect; severity depends on what was bypassed)

RULE_SEVERITY = {
    "Energy Isolation":         1.00,
    "Confined Space":           1.00,
    "Hot Work":                 1.00,
    "Line of Fire":             1.00,
    "Working at Height":        0.85,
    "Safe Mechanical Lifting":  0.85,
    "Work Authorisation":       0.70,
    "Driving":                  0.70,
    "Bypassing Safety Controls": 0.70,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

#                                 DEVICE                                
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"[OK] Device: {DEVICE}")

#                                 LOAD DATA                            
print(f"\n{'='*65}")
print("STEP 1 -- Loading dataset")
print(f"{'='*65}")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"\n  sif_potential distribution:")
for val, cnt in df["sif_potential"].value_counts().items():
    print(f"    {val}: {cnt}  ({cnt/len(df)*100:.1f}%)")

#                                 ENCODE TEXT                          
print(f"\n{'='*65}")
print(f"STEP 2 -- Encoding report_text  ({EMBEDDING_MODEL})")
print(f"{'='*65}")

t0 = time.time()
encoder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
texts = df["report_text"].fillna("").tolist()
embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True)
print(f"  Shape: {embeddings.shape}  |  Time: {time.time()-t0:.1f}s")

#                                 SIF CLASSIFIER                       
print(f"\n{'='*65}")
print("STEP 3a -- Binary SIF Classifier  (recall-first threshold)")
print(f"{'='*65}")

y_sif = df["sif_potential"].map({"Yes": 1, "No": 0}).values

X_tr, X_te, y_tr, y_te = train_test_split(
    embeddings, y_sif, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_sif
)
print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

clf_sif = LogisticRegression(
    class_weight="balanced", max_iter=MAX_ITER_SIF,
    C=1.0, solver="lbfgs", random_state=RANDOM_STATE,
)
clf_sif.fit(X_tr, y_tr)

#    probabilities on test set   
y_proba = clf_sif.predict_proba(X_te)[:, 1]

#    precision-recall curve   choose threshold   
precisions, recalls, thresholds = precision_recall_curve(y_te, y_proba)
# Find threshold that maximises recall while precision >= some floor.
# We target recall >= 0.85 and pick the threshold with best F1 among those.
target_recall = 0.85
valid = recalls[:-1] >= target_recall  # thresholds array is len-1
if valid.any():
    # Among thresholds meeting target recall, pick the one with highest F1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )
    f1_scores[~valid] = -1
    best_idx = np.argmax(f1_scores)
    CHOSEN_THRESHOLD = float(thresholds[best_idx])
    chosen_prec = precisions[best_idx]
    chosen_rec  = recalls[best_idx]
else:
    # Fallback: pick threshold closest to recall=target_recall
    diffs = np.abs(recalls[:-1] - target_recall)
    best_idx = np.argmin(diffs)
    CHOSEN_THRESHOLD = float(thresholds[best_idx])
    chosen_prec = precisions[best_idx]
    chosen_rec  = recalls[best_idx]

print(f"\n  Precision-Recall analysis:")
print(f"    Target recall:       {target_recall}")
print(f"    Chosen threshold:    {CHOSEN_THRESHOLD:.3f}")
print(f"    At this threshold:   Precision={chosen_prec:.3f}  Recall={chosen_rec:.3f}")
print(f"    (Threshold chosen to maximise F1 while recall >= {target_recall})")

#    Evaluate at DEFAULT 0.5 threshold   
y_pred_default = (y_proba >= 0.5).astype(int)
print(f"\n     Classification Report @ threshold 0.50 (default)   ")
print(classification_report(
    y_te, y_pred_default, target_names=["No SIF", "Yes SIF"], digits=3
))
print(f"  Confusion Matrix @ 0.50:")
cm_def = confusion_matrix(y_te, y_pred_default)
print(f"    TN={cm_def[0,0]:>4d}  FP={cm_def[0,1]:>4d}")
print(f"    FN={cm_def[1,0]:>4d}  TP={cm_def[1,1]:>4d}")

#    Evaluate at CHOSEN threshold   
y_pred_chosen = (y_proba >= CHOSEN_THRESHOLD).astype(int)
print(f"\n     Classification Report @ threshold {CHOSEN_THRESHOLD:.3f} (chosen)   ")
print(classification_report(
    y_te, y_pred_chosen, target_names=["No SIF", "Yes SIF"], digits=3
))
print(f"  Confusion Matrix @ {CHOSEN_THRESHOLD:.3f}:")
cm_ch = confusion_matrix(y_te, y_pred_chosen)
print(f"    TN={cm_ch[0,0]:>4d}  FP={cm_ch[0,1]:>4d}")
print(f"    FN={cm_ch[1,0]:>4d}  TP={cm_ch[1,1]:>4d}")

#                                 RULE CLASSIFIER                      
print(f"\n{'='*65}")
print("STEP 3b -- Life-Saving Rule Classifier")
print(f"         (SIF=Yes, excluding Unmapped / None / nan)")
print(f"{'='*65}")

EXCLUDE = {"Unmapped", "None", "none", "nan"}
mask = (
    (df["sif_potential"] == "Yes")
    & (~df["life_saving_rule"].isin(EXCLUDE))
    & (df["life_saving_rule"].notna())
)
df_rule = df[mask].copy()
X_emb_r = embeddings[mask.values]
y_rule  = df_rule["life_saving_rule"].values

rule_classes = sorted(df_rule["life_saving_rule"].unique())
print(f"  Rows: {len(df_rule)}  |  Classes ({len(rule_classes)}): {rule_classes}")

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_emb_r, y_rule, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_rule
)
print(f"  Train: {len(X_tr_r)}  |  Test: {len(X_te_r)}")

clf_rule = LogisticRegression(
    class_weight="balanced", max_iter=MAX_ITER_RULE,
    C=1.0, solver="lbfgs", multi_class="multinomial",
    random_state=RANDOM_STATE,
)
clf_rule.fit(X_tr_r, y_tr_r)

y_pred_rule = clf_rule.predict(X_te_r)
print(f"\n     Rule Classifier: Test Set Results   ")
print(classification_report(y_te_r, y_pred_rule, digits=3))

#                                 RECURRENCE TABLE                     
print(f"\n{'='*65}")
print("STEP 4 -- Building recurrence lookup table")
print(f"{'='*65}")

# For each (site, activity, predicted_rule) triple, count occurrences in training data.
# We use the actual life_saving_rule column for known rows and the predicted rule
# for others.  Normalise 0 1 across the dataset.
# NOTE: real_ihm_stefanini rows have site/activity from a different domain,
# so we only count synthetic rows for recurrence.
syn_mask = df["source"] == "synthetic"

def _compute_recurrence(row):
    """Return the count of training rows sharing site+activity+rule."""
    rule_val = row["life_saving_rule"]
    if rule_val in EXCLUDE or pd.isna(rule_val):
        # For unmapped rows, we use 'Any' as placeholder -- low recurrence
        rule_val = "__any__"
    return ((df["site"] == row["site"]) &
            (df["activity"] == row["activity"]) &
            (df["life_saving_rule"].fillna("__any__") == rule_val)).sum()

# Pre-build the recurrence lookup for speed (site, activity)   rule   count
recurrence_counts = {}
for _, row in df[syn_mask].iterrows():
    key = (row["site"], row["activity"], row["life_saving_rule"])
    recurrence_counts[key] = recurrence_counts.get(key, 0) + 1

max_recurrence = max(recurrence_counts.values()) if recurrence_counts else 1
print(f"  Max raw recurrence count: {max_recurrence}")
print(f"  Normalisation divisor:    {max_recurrence}")

#                                 SAVE ARTIFACTS                      
print(f"\n{'='*65}")
print("STEP 5 -- Saving model artifacts")
print(f"{'='*65}")

joblib.dump(clf_sif,  os.path.join(OUTPUT_DIR, "sif_classifier.joblib"))
joblib.dump(clf_rule, os.path.join(OUTPUT_DIR, "rule_classifier.joblib"))
joblib.dump(encoder,  os.path.join(OUTPUT_DIR, "embedding_model_ref.joblib"))

metadata = {
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dim": int(embeddings.shape[1]),
    "sif_threshold": CHOSEN_THRESHOLD,
    "sif_classes": ["No", "Yes"],
    "rule_classes": rule_classes,
    "rule_severity": RULE_SEVERITY,
    "risk_score_weights": {
        "confidence": 0.60,
        "recurrence": 0.25,
        "rule_severity": 0.15,
    },
    "training_rows_total": len(df),
    "training_rows_sif": len(X_tr),
    "training_rows_rule": len(X_tr_r),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "device": DEVICE,
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

for fname in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024
    print(f"  {fname:45s}  {sz:>7.1f} KB")

#                                 LIME EXPLAINER                      
print(f"\n{'='*65}")
print("STEP 6 -- Initialising LIME text explainer")
print(f"{'='*65}")

# We create a LIME explainer that wraps the full embed   classify pipeline.
# LIME needs a predict_proba function that takes raw strings and returns
# class probabilities for ALL classes (binary SIF).

_lime_encoder = encoder
_lime_clf     = clf_sif
_lime_threshold = CHOSEN_THRESHOLD

def _lime_predict_proba(texts):
    """
    LIME-compatible predict_proba: takes list of raw strings,
    returns np.array of shape (n, 2) with P(No SIF), P(SIF=Yes).
    """
    embs = _lime_encoder.encode(list(texts), convert_to_numpy=True)
    return _lime_clf.predict_proba(embs)

lime_explainer = LimeTextExplainer(
    class_names=["No SIF", "Yes SIF"],
    random_state=RANDOM_STATE,
    split_expression=r"\W+",  # word-level tokenization
)
print("  [OK] LimeTextExplainer ready")

def explain_prediction(report_text: str, num_samples: int = 200,
                       num_features: int = 6) -> list:
    """
    Generate LIME word-level explanation for a single report.

    Returns:
        List of (word, weight) tuples, sorted by absolute weight descending.
        Positive weight   pushes toward SIF=Yes.
        Negative weight   pushes toward No SIF.
    """
    exp = lime_explainer.explain_instance(
        report_text,
        _lime_predict_proba,
        num_samples=num_samples,
        num_features=num_features,
        labels=(1,),  # explain class 1 = SIF=Yes
    )
    # exp.as_list(label=1) returns [(word, weight), ...]
    return exp.as_list(label=1)

print("  [OK] explain_prediction() ready  (num_samples=200, num_features=6)")

#                                 RISK SCORE                          
print(f"\n{'='*65}")
print("STEP 7 -- Risk Score function")
print(f"{'='*65}")

def compute_risk_score(sif_confidence: float, predicted_rule: str,
                       site: str, activity: str) -> int:
    """
    Composite SIF Risk Score on a 0-100 scale.

    Formula:
        risk = 100   (0.60   confidence
                     + 0.25   recurrence_factor
                     + 0.15   rule_severity)

    Where:
        confidence        -- P(SIF=Yes) from the logistic regression, clamped [0,1]
        recurrence_factor -- normalised count of training rows sharing the same
                            (site, activity, predicted_rule), range [0, 1]
        rule_severity     -- fixed severity weight for the predicted IOGP rule,
                            range [0.7, 1.0] (see RULE_SEVERITY dict)
    """
    confidence = float(np.clip(sif_confidence, 0, 1))

    # Recurrence factor: look up (site, activity, rule) count
    key = (site, activity, predicted_rule)
    raw_count = recurrence_counts.get(key, 0)
    recurrence_factor = min(raw_count / max_recurrence, 1.0) if max_recurrence > 0 else 0.0

    # Rule severity
    severity = RULE_SEVERITY.get(predicted_rule, 0.5)  # default 0.5 if unknown

    risk = 100.0 * (0.60 * confidence + 0.25 * recurrence_factor + 0.15 * severity)
    return int(round(np.clip(risk, 0, 100)))

print("  [OK] compute_risk_score() ready")
print("  Formula: risk = 100   (0.60 confidence + 0.25 recurrence + 0.15 severity)")

#                                 PREDICT FUNCTION                     
print(f"\n{'='*65}")
print("STEP 8 -- predict() function")
print(f"{'='*65}")

def predict(report_text: str, site: str = "Unknown",
            activity: str = "Unknown") -> dict:
    """
    Full SIF prediction pipeline for a single safety report.

    Args:
        report_text: Raw narrative text of the safety report.
        site:        Site/facility name (for recurrence lookup).
        activity:    Activity type (for recurrence lookup).

    Returns:
        {
            "sif_potential":           "Yes" | "No",
            "confidence":              float [0,1]  -- P(SIF=Yes),
            "risk_score":              int [0,100]  -- composite risk score,
            "life_saving_rule":        str          -- predicted IOGP rule or "N/A",
            "rule_confidence":         float [0,1],
            "top_contributing_phrases": [(phrase, weight), ...]  -- LIME explanation,
        }
    """
    emb = encoder.encode([report_text], convert_to_numpy=True)

    #    SIF prediction   
    sif_proba = clf_sif.predict_proba(emb)[0]
    sif_prob_yes = float(sif_proba[1])
    sif_label = "Yes" if sif_prob_yes >= CHOSEN_THRESHOLD else "No"
    sif_conf  = sif_prob_yes

    #    Rule prediction (only if SIF=Yes)   
    if sif_label == "Yes":
        rule_pred   = clf_rule.predict(emb)[0]
        rule_proba  = clf_rule.predict_proba(emb)[0]
        rule_conf   = float(rule_proba.max())
    else:
        rule_pred = "N/A"
        rule_conf = 0.0

    #    Risk score   
    risk = compute_risk_score(sif_conf, rule_pred, site, activity)

    #    LIME explanation   
    try:
        phrases = explain_prediction(report_text)
    except Exception:
        phrases = []

    return {
        "sif_potential":            sif_label,
        "confidence":               round(sif_conf, 4),
        "risk_score":               risk,
        "life_saving_rule":         rule_pred,
        "rule_confidence":          round(rule_conf, 4),
        "top_contributing_phrases": [(w, round(wt, 4)) for w, wt in phrases],
    }

print("  [OK] predict(report_text, site, activity) ready")

#                                 EXAMPLE PREDICTIONS                 
print(f"\n{'='*65}")
print("STEP 9 -- 5 Example Predictions (held-out test reports)")
print(f"{'='*65}")

# Pick 5 diverse test-set rows: 3 SIF-Yes, 2 SIF-No
np.random.seed(RANDOM_STATE + 1)
yes_idx = np.where(y_te == 1)[0]
no_idx  = np.where(y_te == 0)[0]
chosen  = np.concatenate([
    np.random.choice(yes_idx, size=min(3, len(yes_idx)), replace=False),
    np.random.choice(no_idx,  size=min(2, len(no_idx)),  replace=False),
])
np.random.shuffle(chosen)

for rank, test_i in enumerate(chosen, 1):
    emb = X_te[test_i]
    orig = np.where(np.all(embeddings == emb, axis=1))[0]
    if len(orig) == 0:
        continue
    row = df.iloc[orig[0]]

    result = predict(row["report_text"],
                     site=str(row["site"]),
                     activity=str(row["activity"]))

    print(f"\n     Example {rank}   ")
    print(f"  Report:  {row['report_id']}  |  Source: {row['source']}")
    print(f"  Text:    {str(row['report_text'])[:120]}...")
    print(f"  TRUE:    SIF={row['sif_potential']}  Rule={row['life_saving_rule']}")
    print(f"  PREDICT: SIF={result['sif_potential']}  "
          f"Conf={result['confidence']:.3f}  "
          f"Risk={result['risk_score']}/100  "
          f"Rule={result['life_saving_rule']}  "
          f"RuleConf={result['rule_confidence']:.3f}")
    if result["top_contributing_phrases"]:
        top_phrases = result["top_contributing_phrases"][:4]
        print(f"  Phrases: {top_phrases}")

#                                 DONE                                 
print(f"\n{'='*65}")
print("DONE -- All artifacts saved to:")
print(f"  {OUTPUT_DIR}")
print(f"  Chosen SIF threshold: {CHOSEN_THRESHOLD:.3f}")
print(f"{'='*65}")
print("\nBackend integration:")
print("  from train_model import predict, explain_prediction")
print("  result = predict('Worker fell from scaffold during maintenance',")
print("                    site='Jorjabar', activity='Maintenance')")
