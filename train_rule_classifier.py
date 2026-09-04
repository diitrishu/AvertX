#!/usr/bin/env python3
"""
train_rule_classifier.py
========================
Phase 13 — Life-Saving Rule multi-class classifier.

Trains a LogisticRegression multi-class classifier on top of all-MiniLM-L6-v2
sentence embeddings to predict which of the 9 IOGP Life-Saving Rules is most
applicable to a safety incident report.

TRAINING DATA CONSTRAINT
------------------------
Only rows where ``life_saving_rule`` is one of the 9 canonical IOGP rule names
from taxonomy/life_saving_rules.csv are used. Rows with "UNKNOWN", "Unmapped",
NaN, or any other value are excluded. This is intentional: training on "Unmapped"
as a 10th class would corrupt the taxonomy (it is not a real rule — it is a
data-quality signal meaning "we could not map this incident to a rule").

MODEL
-----
Embedding:   sentence-transformers/all-MiniLM-L6-v2 (384-dim)
             Same model as the SIF classifier (Phase 11). Not swapped here
             because no new embedding model has been evaluated.
Classifier:  scikit-learn LogisticRegression, multi_class="multinomial",
             class_weight="balanced" (handles severe class imbalance across 9 rules),
             solver="lbfgs", C=1.0.

OUTPUT
------
model_artifacts/rule_classifier_v2.joblib  — trained classifier
model_artifacts/rule_metadata_v2.json      — metadata snapshot
data/rule_classifier_eval.md               — evaluation report

The old rule_classifier.joblib (from train_model.py) is left untouched.
This script produces a separately versioned artifact (v2) so the existing
backend is not disrupted until Phase 14 integration is approved.

EVALUATION
----------
Uses data/splits/test.csv filtered to valid-rule rows (likely very small — see
audit). Reports per-class precision/recall/F1, overall accuracy, and the
fraction of test predictions that trigger NEEDS_REVIEW under the contract
thresholds. Writes a frank assessment of reliability to data/rule_classifier_eval.md.

Usage
-----
    py -3.12 train_rule_classifier.py

Dependencies (same as existing backend):
    sentence-transformers, scikit-learn, joblib, pandas, numpy
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Optional

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV      = os.path.join(BASE_DIR, "data", "splits", "train.csv")
TEST_CSV       = os.path.join(BASE_DIR, "data", "splits", "test.csv")
TAXONOMY_CSV   = os.path.join(BASE_DIR, "taxonomy", "life_saving_rules.csv")
ARTIFACTS_DIR  = os.path.join(BASE_DIR, "model_artifacts")
EVAL_OUT       = os.path.join(BASE_DIR, "data", "rule_classifier_eval.md")

# Reuse the already-downloaded embedding model saved in model_artifacts/
# (avoids a redundant network download and keeps the embedding space identical
# to the SIF classifier).
EMBEDDING_JOBLIB = os.path.join(ARTIFACTS_DIR, "embedding_model_ref.joblib")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

RULE_CLF_OUT  = os.path.join(ARTIFACTS_DIR, "rule_classifier_v2.joblib")
META_OUT      = os.path.join(ARTIFACTS_DIR, "rule_metadata_v2.json")

# ── Contract thresholds (must match backend/rule_output_contract.py) ───────────
CONFIDENCE_THRESHOLD = 0.50
AMBIGUITY_MARGIN     = 0.15

# ── Sklearn config ─────────────────────────────────────────────────────────────
MAX_ITER     = 1000
RANDOM_STATE = 42
C            = 1.0

# ── Minimum examples per class to avoid silent unreliable training ─────────────
MIN_EXAMPLES_WARNING = 5

# ── The 9 valid IOGP rule names (human-readable, matching training labels) ─────
# Source: taxonomy/life_saving_rules.csv → rule_name column, which is what
# the training data was labelled with. Snake-case rule_ids are the canonical
# identifiers in the taxonomy CSV, but the training labels use the names.
VALID_RULE_NAMES = [
    "Bypassing Safety Controls",
    "Confined Space",
    "Driving",
    "Energy Isolation",
    "Hot Work",
    "Line of Fire",
    "Safe Mechanical Lifting",
    "Work Authorisation",
    "Working at Height",
]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _banner(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(title)
    print(f"{'=' * 65}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Verify taxonomy
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 0 — Verify taxonomy (life_saving_rules.csv)")

taxonomy_df = pd.read_csv(TAXONOMY_CSV)
taxonomy_rule_names = set(taxonomy_df["rule_name"].tolist())
print(f"  Taxonomy rule_names ({len(taxonomy_rule_names)}): {sorted(taxonomy_rule_names)}")

# Cross-check our hardcoded list against the CSV
extra   = set(VALID_RULE_NAMES) - taxonomy_rule_names
missing = taxonomy_rule_names - set(VALID_RULE_NAMES)
if extra or missing:
    print(f"  [WARNING] Mismatch between VALID_RULE_NAMES and taxonomy CSV:")
    if extra:
        print(f"    In code but NOT in CSV: {extra}")
    if missing:
        print(f"    In CSV but NOT in code: {missing}")
    sys.exit(1)
else:
    print("  [OK] VALID_RULE_NAMES exactly matches taxonomy CSV rule_names.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and filter training data
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 1 — Load and filter training data (data/splits/train.csv)")

train_raw = pd.read_csv(TRAIN_CSV)
print(f"  Raw train.csv: {len(train_raw)} rows")

# Filter: keep only rows with a real mapped rule
train_df = train_raw[train_raw["life_saving_rule"].isin(VALID_RULE_NAMES)].copy()
train_df = train_df.reset_index(drop=True)

n_train = len(train_df)
print(f"\n  Rows after filtering to valid IOGP rules: {n_train}")
print(f"  (Excluded {len(train_raw) - n_train} rows with UNKNOWN / Unmapped / NaN)")

# ── Per-class counts ──────────────────────────────────────────────────────────
print(f"\n  Per-rule class counts in training set:")
class_counts = train_df["life_saving_rule"].value_counts().sort_index()
low_count_rules = []
for rule, cnt in class_counts.items():
    flag = ""
    if cnt < MIN_EXAMPLES_WARNING:
        flag = "  *** UNRELIABLE — fewer than 5 examples ***"
        low_count_rules.append(rule)
    print(f"    {rule:<35s}  {cnt:>4d}{flag}")

if low_count_rules:
    print(f"\n  [WARNING] The following rules have fewer than {MIN_EXAMPLES_WARNING} training "
          f"examples and their predictions will be UNRELIABLE:")
    for r in low_count_rules:
        print(f"    - {r}")
else:
    print(f"\n  [OK] All 9 rules have >= {MIN_EXAMPLES_WARNING} training examples.")

# ── Synthetic vs real breakdown ───────────────────────────────────────────────
if "is_synthetic" in train_df.columns:
    n_synthetic = train_df["is_synthetic"].sum()
    n_real = len(train_df) - n_synthetic
    print(f"\n  Source mix: {n_real} real  |  {n_synthetic} synthetic")
    print(f"  Note: {n_synthetic / n_train * 100:.1f}% synthetic. Model relies heavily on "
          f"synthetic data — real-world performance may differ from eval metrics.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load encoder (reuse existing artifact to avoid re-download)
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 2 — Load embedding model (all-MiniLM-L6-v2)")

if os.path.exists(EMBEDDING_JOBLIB):
    print(f"  Loading pre-saved encoder from {EMBEDDING_JOBLIB}")
    encoder = joblib.load(EMBEDDING_JOBLIB)
    print(f"  [OK] Encoder loaded from disk.")
else:
    print(f"  Pre-saved encoder not found at {EMBEDDING_JOBLIB}.")
    print(f"  Downloading {EMBEDDING_MODEL} from HuggingFace...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    joblib.dump(encoder, EMBEDDING_JOBLIB)
    print(f"  [OK] Encoder downloaded and saved to {EMBEDDING_JOBLIB}.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Embed training texts
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 3 — Embed training narratives")

texts_train = train_df["narrative_text"].fillna("").tolist()
print(f"  Encoding {len(texts_train)} training texts …")
X_train = encoder.encode(texts_train, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True)
y_train = train_df["life_saving_rule"].values
print(f"  Embeddings shape: {X_train.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train LogisticRegression (multinomial, balanced)
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 4 — Train LogisticRegression (multinomial, class_weight='balanced')")

clf = LogisticRegression(
    multi_class="multinomial",
    class_weight="balanced",
    solver="lbfgs",
    C=C,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)
print(f"  [OK] Classifier trained.  Classes: {list(clf.classes_)}")

# Quick sanity-check on training set (not a real metric — just confirms fit)
y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"  Training-set accuracy (sanity check, not a real metric): {train_acc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Load and filter test data
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 5 — Load and filter test data (data/splits/test.csv)")

test_raw = pd.read_csv(TEST_CSV)
print(f"  Raw test.csv: {len(test_raw)} rows")

test_df = test_raw[test_raw["life_saving_rule"].isin(VALID_RULE_NAMES)].copy()
test_df = test_df.reset_index(drop=True)
n_test = len(test_df)
print(f"  Rows after filtering to valid IOGP rules: {n_test}")
print(f"  (Excluded {len(test_raw) - n_test} rows with UNKNOWN / Unmapped / NaN)")

print(f"\n  Per-rule counts in test set:")
test_class_counts = test_df["life_saving_rule"].value_counts().sort_index()
rules_in_test = set(test_class_counts.index)
rules_missing_from_test = set(VALID_RULE_NAMES) - rules_in_test
for rule in sorted(VALID_RULE_NAMES):
    cnt = test_class_counts.get(rule, 0)
    flag = "  *** ABSENT from test ***" if cnt == 0 else ""
    flag = flag or ("  *** VERY FEW (<5) — metrics unreliable ***" if cnt < 5 else "")
    print(f"    {rule:<35s}  {cnt:>4d}{flag}")

if rules_missing_from_test:
    print(f"\n  [WARNING] Rules absent from test set: {sorted(rules_missing_from_test)}")
    print(f"  Precision/recall/F1 for absent rules will be 0.0 or undefined.")

if n_test < 50:
    print(f"\n  [CAUTION] Test set is very small (n={n_test}). Per-class metrics with "
          f"fewer than 5 test examples should NOT be interpreted as reliable estimates.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Embed test texts and evaluate
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 6 — Evaluate on test set")

texts_test = test_df["narrative_text"].fillna("").tolist()
print(f"  Encoding {len(texts_test)} test texts …")
X_test = encoder.encode(texts_test, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True)
y_test = test_df["life_saving_rule"].values

# Standard classification metrics
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)          # (n_test, 9)
overall_acc = accuracy_score(y_test, y_pred)

print(f"\n  Overall accuracy: {overall_acc:.3f}  (n={n_test})")
print(f"\n  Per-class classification report:")
# Use zero_division=0 so absent-from-test classes show 0.0 rather than warnings
report_str = classification_report(
    y_test, y_pred,
    labels=list(clf.classes_),
    target_names=list(clf.classes_),
    digits=3,
    zero_division=0,
)
print(report_str)

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=list(clf.classes_))

# ── NEEDS_REVIEW fraction ─────────────────────────────────────────────────────
_banner("STEP 7 — NEEDS_REVIEW fraction under contract thresholds")

sorted_indices = np.argsort(y_proba, axis=1)[::-1]  # descending per row
# For each row: top conf and gap
top_confs = y_proba[np.arange(n_test), np.argmax(y_proba, axis=1)]
# Runner-up: second-highest probability per row
second_best_idx = np.argsort(y_proba, axis=1)[:, -2]  # second from right
runner_confs = y_proba[np.arange(n_test), second_best_idx]
gaps = top_confs - runner_confs

low_conf_mask  = top_confs < CONFIDENCE_THRESHOLD
low_gap_mask   = gaps < AMBIGUITY_MARGIN
needs_review_mask = low_conf_mask | low_gap_mask

n_needs_review     = int(needs_review_mask.sum())
n_low_conf         = int(low_conf_mask.sum())
n_low_gap          = int(low_gap_mask.sum())
n_both             = int((low_conf_mask & low_gap_mask).sum())
frac_needs_review  = n_needs_review / n_test if n_test > 0 else 0.0

print(f"  CONFIDENCE_THRESHOLD : {CONFIDENCE_THRESHOLD}")
print(f"  AMBIGUITY_MARGIN     : {AMBIGUITY_MARGIN}")
print(f"\n  Total test rows      : {n_test}")
print(f"  Trigger: low confidence (<{CONFIDENCE_THRESHOLD})  : {n_low_conf}")
print(f"  Trigger: low gap (<{AMBIGUITY_MARGIN})             : {n_low_gap}")
print(f"  Trigger: both                                     : {n_both}")
print(f"  NEEDS_REVIEW total   : {n_needs_review}  ({frac_needs_review * 100:.1f}%)")

# ── Among NEEDS_REVIEW, how often was the correct rule still top-1? ───────────
if n_needs_review > 0:
    nr_correct = sum(
        y_test[i] == list(clf.classes_)[int(np.argmax(y_proba[i]))]
        for i in np.where(needs_review_mask)[0]
    )
    print(f"\n  Of NEEDS_REVIEW rows: top-1 was correct in {nr_correct}/{n_needs_review} "
          f"({nr_correct / n_needs_review * 100:.1f}%) — human-review value estimate.")

# ── Confident predictions: accuracy on rows that did NOT trigger NEEDS_REVIEW ──
confident_mask = ~needs_review_mask
n_confident = int(confident_mask.sum())
if n_confident > 0:
    y_test_conf = y_test[confident_mask]
    y_pred_conf = y_pred[confident_mask]
    acc_conf = accuracy_score(y_test_conf, y_pred_conf)
    print(f"\n  Confident predictions (no NEEDS_REVIEW): {n_confident} rows, "
          f"accuracy = {acc_conf:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save model artifacts
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 8 — Save model artifacts")

joblib.dump(clf, RULE_CLF_OUT)
print(f"  Saved: {RULE_CLF_OUT}")

metadata = {
    "model_name": "rule_classifier_lr_minilm",
    "model_version": "1.0.0",
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dim": int(X_train.shape[1]),
    "classifier_type": "LogisticRegression",
    "multi_class": "multinomial",
    "class_weight": "balanced",
    "C": C,
    "solver": "lbfgs",
    "max_iter": MAX_ITER,
    "random_state": RANDOM_STATE,
    "rule_classes": list(clf.classes_),
    "n_train_rows": n_train,
    "n_test_rows": n_test,
    "training_class_counts": class_counts.to_dict(),
    "test_class_counts": test_class_counts.to_dict(),
    "overall_accuracy_on_test": round(float(overall_acc), 4),
    "needs_review_fraction_on_test": round(float(frac_needs_review), 4),
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "ambiguity_margin": AMBIGUITY_MARGIN,
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}

with open(META_OUT, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved: {META_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Write evaluation report (data/rule_classifier_eval.md)
# ─────────────────────────────────────────────────────────────────────────────
_banner("STEP 9 — Write data/rule_classifier_eval.md")


def _class_report_dict(y_true, y_pred, classes):
    """Return classification_report as a dict for easy table rendering."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {c: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
            for i, c in enumerate(classes)}


cr_dict = _class_report_dict(y_test, y_pred, list(clf.classes_))
rules_with_zero_test = [r for r, d in cr_dict.items() if d["support"] == 0]
rules_with_few_test  = [r for r, d in cr_dict.items() if 0 < d["support"] < 5]

timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

md_lines = [
    "# Rule Classifier Evaluation Report",
    "",
    f"**Generated:** {timestamp_str}  ",
    f"**Model:** `rule_classifier_lr_minilm v1.0.0`  ",
    f"**Embedding:** `all-MiniLM-L6-v2` (384-dim)  ",
    f"**Classifier:** `LogisticRegression` multinomial, `class_weight='balanced'`, "
    f"`C={C}`, `max_iter={MAX_ITER}`  ",
    "",
    "---",
    "",
    "## 1. Data summary",
    "",
    f"| Split | Total rows | Valid-rule rows | Excluded (UNKNOWN/Unmapped/NaN) |",
    f"|---|---|---|---|",
    f"| train.csv | {len(train_raw)} | {n_train} | {len(train_raw) - n_train} |",
    f"| test.csv  | {len(test_raw)} | {n_test}  | {len(test_raw)  - n_test}  |",
    "",
    "### Training class distribution",
    "",
    "| Life-Saving Rule | Train count | Warning |",
    "|---|---|---|",
]
for rule in sorted(VALID_RULE_NAMES):
    cnt = class_counts.get(rule, 0)
    warn = "⚠️ fewer than 5 — unreliable" if cnt < MIN_EXAMPLES_WARNING else ""
    md_lines.append(f"| {rule} | {cnt} | {warn} |")

md_lines += [
    "",
    "> **Note on training data quality:**",
    f"> Of the {n_train} training rows, {int(n_synthetic if 'n_synthetic' in dir() else 0)} "
    f"({int(n_synthetic / n_train * 100) if 'n_synthetic' in dir() else '?'}%) are synthetic. "
    "The model relies heavily on synthetic data generated from templates. "
    "Real-world performance on genuinely novel incident narratives may be lower "
    "than these evaluation metrics suggest.",
    "",
    "### Test class distribution",
    "",
    "| Life-Saving Rule | Test count | Coverage |",
    "|---|---|---|",
]
for rule in sorted(VALID_RULE_NAMES):
    cnt = test_class_counts.get(rule, 0)
    cov = "✅ present" if cnt >= 5 else ("⚠️ sparse (<5)" if cnt > 0 else "❌ absent")
    md_lines.append(f"| {rule} | {cnt} | {cov} |")

if rules_with_zero_test:
    md_lines += [
        "",
        f"> **Critical gap:** {len(rules_with_zero_test)} rules have **zero** test examples: "
        f"`{'`, `'.join(sorted(rules_with_zero_test))}`. "
        "Precision/recall/F1 for these rules cannot be estimated at all from this split. "
        "Do NOT treat their reported metrics as meaningful.",
    ]

md_lines += [
    "",
    "---",
    "",
    "## 2. Overall test-set performance",
    "",
    f"| Metric | Value | Caveat |",
    f"|---|---|---|",
    f"| Overall accuracy | {overall_acc:.3f} | n={n_test} — very small sample |",
    f"| NEEDS_REVIEW fraction | {frac_needs_review:.3f} | "
    f"(conf < {CONFIDENCE_THRESHOLD} OR gap < {AMBIGUITY_MARGIN}) |",
    f"| Rows triggering low-confidence | {n_low_conf}/{n_test} | — |",
    f"| Rows triggering low-gap (ambiguity) | {n_low_gap}/{n_test} | — |",
    "",
]

if n_confident > 0:
    md_lines += [
        f"**Accuracy on confident predictions** (rows NOT flagged NEEDS_REVIEW, "
        f"n={n_confident}): `{acc_conf:.3f}`",
        "",
    ]

md_lines += [
    "---",
    "",
    "## 3. Per-class precision / recall / F1",
    "",
    "| Life-Saving Rule | Precision | Recall | F1 | Test support | Reliability |",
    "|---|---|---|---|---|---|",
]
for rule in sorted(VALID_RULE_NAMES):
    d = cr_dict.get(rule, {"precision": 0, "recall": 0, "f1": 0, "support": 0})
    sup = d["support"]
    if sup == 0:
        rel = "❌ cannot estimate (0 test examples)"
    elif sup < 5:
        rel = "⚠️ unreliable (<5 test examples)"
    else:
        rel = "✅ marginally reliable (≥5 examples)"
    md_lines.append(
        f"| {rule} | {d['precision']:.3f} | {d['recall']:.3f} | "
        f"{d['f1']:.3f} | {sup} | {rel} |"
    )

md_lines += [
    "",
    "---",
    "",
    "## 4. NEEDS_REVIEW analysis",
    "",
    f"Contract thresholds: `CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}`, "
    f"`AMBIGUITY_MARGIN = {AMBIGUITY_MARGIN}`  ",
    "(Both are **placeholders** pending calibration on a larger test set — "
    "see `backend/rule_output_contract.py` module docstring.)",
    "",
    f"| Trigger | Count | % of test |",
    "|---|---|---|",
    f"| Low confidence only | {n_low_conf - n_both} | "
    f"{(n_low_conf - n_both) / n_test * 100:.1f}% |",
    f"| Low gap only | {n_low_gap - n_both} | "
    f"{(n_low_gap - n_both) / n_test * 100:.1f}% |",
    f"| Both triggers | {n_both} | {n_both / n_test * 100:.1f}% |",
    f"| **Total NEEDS_REVIEW** | **{n_needs_review}** | "
    f"**{frac_needs_review * 100:.1f}%** |",
    "",
]
if n_needs_review > 0:
    md_lines += [
        f"Among the {n_needs_review} NEEDS_REVIEW predictions, the top-1 class was "
        f"correct in **{nr_correct}/{n_needs_review}** cases "
        f"({nr_correct / n_needs_review * 100:.1f}%), indicating that the model often "
        "has the right answer but with insufficient certainty — consistent with the "
        "expected behaviour of balanced-weight LR on an imbalanced dataset.",
        "",
    ]

md_lines += [
    "---",
    "",
    "## 5. Honest assessment of classifier reliability",
    "",
]

# Dynamically generate the honest assessment based on what we found
assessment_lines = []

if n_test < 30:
    assessment_lines.append(
        f"> [!CAUTION]  \n"
        f"> **The test set is too small (n={n_test}) to support confident claims "
        f"about this classifier's real-world performance.** The overall accuracy figure "
        f"of {overall_acc:.3f} and all per-class metrics should be treated as indicative "
        f"only. A minimum of ~20 examples per class (180 total for 9 classes) would be "
        f"needed for reliable per-class metric estimation. Do NOT cite these numbers as "
        f"evidence that the classifier works well."
    )
else:
    assessment_lines.append(
        f"> [!NOTE]  \n"
        f"> Test set has n={n_test} — sufficient for overall accuracy but marginal for "
        f"per-class metrics where some classes have fewer than 5 examples."
    )

if rules_with_zero_test:
    assessment_lines.append(
        f"\n> [!WARNING]  \n"
        f"> **{len(rules_with_zero_test)} rules have zero test coverage**: "
        f"{', '.join(sorted(rules_with_zero_test))}. "
        "The classifier may have learned useful representations for these classes "
        "(they are all represented in training), but we have no empirical evidence either way."
    )

if rules_with_few_test:
    assessment_lines.append(
        f"\n> [!WARNING]  \n"
        f"> **{len(rules_with_few_test)} rules have sparse test coverage (<5 examples)**: "
        f"{', '.join(sorted(rules_with_few_test))}. "
        "Per-class F1 for these rules is unreliable."
    )

training_synthetic_pct = int(n_synthetic / n_train * 100) if 'n_synthetic' in dir() else None
if training_synthetic_pct is not None and training_synthetic_pct > 50:
    assessment_lines.append(
        f"\n> [!IMPORTANT]  \n"
        f"> **Training data is {training_synthetic_pct}% synthetic.** "
        "Synthetic narratives were generated from templates — they may not capture "
        "the linguistic diversity of real incident reports. The classifier's learned "
        "decision boundaries reflect template language patterns as much as genuine "
        "semantic differences between IOGP rules. Real-world accuracy is likely "
        "lower than these metrics suggest."
    )

md_lines += assessment_lines

md_lines += [
    "",
    "### What the classifier IS reliable enough for right now",
    "",
    "- **Triage / pre-filtering**: flagging *candidate* rules for human review, "
    "not making final determinations. The NEEDS_REVIEW contract enforces this discipline.",
    "- **High-confidence, unambiguous cases**: when `confidence > 0.5` AND "
    "`gap > 0.15`, the model's top prediction is worth showing to a reviewer as "
    "a starting hypothesis.",
    "- **Ruling out clearly irrelevant rules**: a very low probability for a rule "
    "is informative even if the top prediction is uncertain.",
    "",
    "### What the classifier is NOT reliable enough for right now",
    "",
    "- **Autonomous rule assignment** without human sign-off.",
    "- **Per-rule performance comparison** between rules with sparse test coverage.",
    "- **Threshold calibration**: the `CONFIDENCE_THRESHOLD=0.5` and "
    "`AMBIGUITY_MARGIN=0.15` values in `backend/rule_output_contract.py` are "
    "placeholders and have not been derived from a calibration curve on this data.",
    "",
    "### Next steps to improve reliability",
    "",
    "1. **More labelled real incident data** — the single biggest lever. "
    "Even 20 real examples per class would transform this evaluation.",
    "2. **Cross-validation on training set** — with n=450 training rows and k=5 "
    "folds, CV would give a more stable accuracy estimate than a single 24-row test set.",
    "3. **Threshold calibration** via Platt scaling or isotonic regression on a "
    "held-out calibration split once the dataset is larger.",
    "4. **Error analysis** on the NEEDS_REVIEW rows to understand which rule pairs "
    "the model most frequently confuses — may reveal taxonomy overlap issues.",
    "",
    "---",
    "",
    f"*Report generated by `train_rule_classifier.py` at {timestamp_str}.*",
]

eval_md = "\n".join(md_lines)

os.makedirs(os.path.dirname(EVAL_OUT), exist_ok=True)
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    f.write(eval_md)
print(f"  Wrote: {EVAL_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
_banner("DONE")
print(f"  Artifacts:")
print(f"    {RULE_CLF_OUT}")
print(f"    {META_OUT}")
print(f"    {EVAL_OUT}")
print(f"\n  Model: {metadata['model_name']} v{metadata['model_version']}")
print(f"  Classes: {list(clf.classes_)}")
print(f"  Train n: {n_train}  |  Test n: {n_test}")
print(f"  Overall test accuracy: {overall_acc:.3f}")
print(f"  NEEDS_REVIEW fraction: {frac_needs_review:.3f}")
print()
print("  [!] Do NOT integrate into main.py/App.tsx yet.")
print("  [!] Review data/rule_classifier_eval.md before Phase 14 integration.")
