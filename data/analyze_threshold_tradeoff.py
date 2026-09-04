"""
analyze_threshold_tradeoff.py -- Phase 11 post-hoc diagnostic.

PURPOSE
-------
Phase 11's benchmark (benchmark_models.py) selected a threshold of 0.13 for
both models using validation-set sweep.  That threshold turned out to be a
near-degenerate "predict almost everyone positive" operating point: it achieved
perfect recall but at precision 0.352 / 0.347 on the test set, barely above
the 0.5 floor that was required on *validation*.

This script sweeps the full threshold range on the TEST SET to show whether
any usable operating point exists anywhere on the curve.

IMPORTANT METHODOLOGICAL WARNING (read before interpreting results):
  This analysis applies thresholds to the TEST SET directly.  That is
  intentionally different from how benchmark_models.py selected thresholds
  (validation-only), and it means the reference points reported here CANNOT
  be used to choose a production threshold -- doing so would be tuning and
  evaluating on the same data, which inflates every reported metric.
  The only legitimate purpose of this script is post-hoc diagnosis: helping
  a human judge whether Phase 11's models have ANY operating region that
  looks viable before deciding whether to iterate further.  It does not
  replace the validation-based threshold selection and must not be treated
  as if it does.

METHODOLOGY
-----------
- Retrains each classifier with the EXACT same settings as benchmark_models.py
  (same train data, same random_state=42, same class_weight='balanced', same
  LogisticRegression solver) so the models being examined are identical to the
  ones Phase 11 measured.  No new hyperparameter choices are made.
- Nothing is retrained with different settings or data.

OUTPUTS
-------
  data/threshold_tradeoff.csv          -- full curve (one row per model * threshold)
  data/threshold_tradeoff_summary.md   -- four reference points per model, plain prose

Run with:
    python data/analyze_threshold_tradeoff.py
"""

import io
import sys

# Force UTF-8 stdout so non-ASCII characters don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths and constants -- must match benchmark_models.py exactly
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(__file__).parent
SPLITS_DIR  = DATA_ROOT / "splits"
TRAIN_CSV   = SPLITS_DIR / "train.csv"
TEST_CSV    = SPLITS_DIR / "test.csv"
RESULTS_JSON = DATA_ROOT / "benchmark_results.json"

OUT_CSV  = DATA_ROOT / "threshold_tradeoff.csv"
OUT_MD   = DATA_ROOT / "threshold_tradeoff_summary.md"

# These MUST match benchmark_models.py -- same values reproduce the same models
RANDOM_SEED     = 42
THRESHOLD_MIN   = 0.05
THRESHOLD_MAX   = 0.95
THRESHOLD_STEP  = 0.01

# Reference-point thresholds to highlight
RECALL_FLOOR       = 0.90   # "lenient recall-priority" operating point
PRECISION_TARGET   = 0.70   # "moderately usable" reference
PRECISION_FLOOR    = 0.50   # used in Phase 11; check if ever exceeded here too

# ---------------------------------------------------------------------------
# Models -- identical to benchmark_models.py registry (SafetyBERT still skipped)
# ---------------------------------------------------------------------------

ACTIVE_MODELS = [
    {
        "id":    "minilm",
        "name":  "sentence-transformers/all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
    },
    {
        "id":    "bge_small",
        "name":  "BAAI/bge-small-en-v1.5",
        "hf_id": "BAAI/bge-small-en-v1.5",
    },
]

SKIPPED_MODELS = [
    {
        "id":   "safetybert",
        "name": "SafetyBERT",
        "skip_reason": (
            "No usable SafetyBERT found on HuggingFace with a clear license -- "
            "same finding as benchmark_models.py.  Not re-attempted here."
        ),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def load_split(path: Path) -> tuple[list[str], list[int]]:
    """Return (narratives, labels). label: 1=Yes, 0=No."""
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["narrative_text"])
            labels.append(1 if row["sif_potential"] == "Yes" else 0)
    return texts, labels


def sweep_thresholds(
    y_true: list[int],
    probs: np.ndarray,
) -> list[dict]:
    """
    For every threshold in [THRESHOLD_MIN, THRESHOLD_MAX] step THRESHOLD_STEP,
    compute precision, recall, F1, TP, FP, TN, FN on the given set.
    Returns list of dicts, one per threshold.
    """
    rows = []
    y_arr = np.array(y_true)
    for t_raw in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP):
        t = round(float(t_raw), 4)
        preds = (probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_arr == 0)))
        tn = int(np.sum((preds == 0) & (y_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_arr == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            "threshold": t,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return rows


def find_reference_points(curve: list[dict]) -> dict:
    """
    Extract four reference points from the full threshold curve:
      1. highest threshold where recall >= RECALL_FLOOR (0.90)
      2. threshold maximising F1
      3. threshold closest to PRECISION_TARGET (0.70) while recall is maximised
      4. lowest threshold where precision > PRECISION_FLOOR (0.50), if any
    """
    # 1. Highest threshold with recall >= 0.90
    high_recall_candidates = [r for r in curve if r["recall"] >= RECALL_FLOOR]
    if high_recall_candidates:
        rec90 = max(high_recall_candidates, key=lambda r: r["threshold"])
    else:
        rec90 = None

    # 2. Max F1
    max_f1_row = max(curve, key=lambda r: (r["f1"], -r["threshold"]))

    # 3. Closest to PRECISION_TARGET = 0.70, maximising recall at that precision level.
    #    Only consider rows where precision is within 0.05 of target.
    near_prec70 = [r for r in curve if abs(r["precision"] - PRECISION_TARGET) <= 0.05]
    if near_prec70:
        prec70 = max(near_prec70, key=lambda r: r["recall"])
        prec70_exact = False
    else:
        # Nobody within 0.05 of 0.70 -- find the single closest point
        prec70 = min(curve, key=lambda r: abs(r["precision"] - PRECISION_TARGET))
        prec70_exact = True  # "closest-but-still-bad" marker

    # 4. Whether precision ever exceeds 0.50
    exceeds_floor = [r for r in curve if r["precision"] > PRECISION_FLOOR]
    if exceeds_floor:
        # The one with the highest recall while precision > 0.50
        prec_floor_best = max(exceeds_floor, key=lambda r: (r["recall"], -r["threshold"]))
    else:
        prec_floor_best = None

    return {
        "recall_90":        rec90,
        "max_f1":           max_f1_row,
        "near_prec70":      prec70,
        "near_prec70_is_just_closest": prec70_exact,
        "prec_exceeds_floor": prec_floor_best,
    }


def fmt_row(r: dict) -> str:
    return (f"threshold={r['threshold']:.2f}  "
            f"precision={r['precision']:.4f}  "
            f"recall={r['recall']:.4f}  "
            f"F1={r['f1']:.4f}  "
            f"TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from sentence_transformers import SentenceTransformer

    section("PHASE 11 POST-HOC DIAGNOSTIC -- THRESHOLD TRADEOFF ANALYSIS")

    print()
    print("  !! METHODOLOGICAL WARNING:")
    print("  !! This script sweeps thresholds on the TEST SET.")
    print("  !! The reference points it reports CANNOT be used to select a")
    print("  !! production threshold -- that would be tuning on the test set.")
    print("  !! Its purpose is diagnostic only: showing whether ANY usable")
    print("  !! operating region exists, to inform human review of Phase 11.")

    # ------------------------------------------------------------------
    # Confirm we are using same settings as the original benchmark
    # ------------------------------------------------------------------
    section("CONFIRMING PARITY WITH benchmark_models.py")
    ref = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    print(f"  benchmark_results.json found: YES")
    print(f"  Original threshold selection: {ref['threshold_selection']}")
    print(f"  Hardware (original run): {ref['hardware']}")
    print()
    print(f"  Retraining with:")
    print(f"    random_state = {RANDOM_SEED}  (matches original)")
    print(f"    class_weight = 'balanced'     (matches original)")
    print(f"    solver       = 'lbfgs'        (matches original)")
    print(f"    max_iter     = 2000           (matches original)")
    print(f"    train data   = {TRAIN_CSV}")
    print()
    print("  NOTE: benchmark_models.py did not persist fitted classifiers")
    print("  or embeddings to disk.  This script retrains identically")
    print("  (same data + same seed) so the models are equivalent.")

    # ------------------------------------------------------------------
    # SafetyBERT skip notice
    # ------------------------------------------------------------------
    section("SAFETYBERT -- SKIPPED (same reason as benchmark_models.py)")
    for s in SKIPPED_MODELS:
        print(f"  {s['name']}: {s['skip_reason']}")

    # ------------------------------------------------------------------
    # Load splits
    # ------------------------------------------------------------------
    section("LOADING SPLITS")
    train_texts, train_labels = load_split(TRAIN_CSV)
    test_texts,  test_labels  = load_split(TEST_CSV)
    print(f"  train: {len(train_texts)} rows  "
          f"({sum(train_labels)} Yes / {len(train_labels)-sum(train_labels)} No)")
    print(f"  test : {len(test_texts)} rows  "
          f"({sum(test_labels)} Yes / {len(test_labels)-sum(test_labels)} No)")

    # ------------------------------------------------------------------
    # Per-model loop
    # ------------------------------------------------------------------
    all_curves: dict[str, list[dict]] = {}

    for spec in ACTIVE_MODELS:
        mid   = spec["id"]
        hf_id = spec["hf_id"]
        name  = spec["name"]

        section(f"MODEL: {name}")

        # Load embedding model
        print(f"  Loading {hf_id} ...")
        model = SentenceTransformer(hf_id)

        # Embed train, fit LR -- identical settings to benchmark_models.py
        print("  Embedding train set ...")
        train_embs = model.encode(train_texts, convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)

        print("  Fitting LogisticRegression ...")
        clf = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(train_embs, train_labels)

        # Embed test, get probabilities
        print("  Embedding test set ...")
        test_embs  = model.encode(test_texts, convert_to_numpy=True,
                                  show_progress_bar=False, batch_size=64)
        test_probs = clf.predict_proba(test_embs)[:, 1]

        # Sweep thresholds on test set
        print(f"  Sweeping {THRESHOLD_MIN:.2f} -> {THRESHOLD_MAX:.2f} "
              f"(step {THRESHOLD_STEP}) on test set ...")
        curve = sweep_thresholds(test_labels, test_probs)
        all_curves[mid] = curve

        # Cross-check: at the threshold originally chosen by benchmark_models.py,
        # our numbers should match benchmark_results.json exactly.
        orig = next(m for m in ref["models"] if m["model_id"] == mid)
        orig_thresh = orig["val_chosen_threshold"]
        our_row = next(r for r in curve if abs(r["threshold"] - orig_thresh) < 1e-5)
        parity_ok = (
            our_row["tp"] == orig["tp"] and
            our_row["fp"] == orig["fp"] and
            our_row["tn"] == orig["tn"] and
            our_row["fn"] == orig["fn"]
        )
        print()
        print(f"  Parity check at threshold={orig_thresh} "
              f"(originally chosen by benchmark_models.py):")
        print(f"    benchmark_results.json : "
              f"TP={orig['tp']} FP={orig['fp']} TN={orig['tn']} FN={orig['fn']}")
        print(f"    this run               : "
              f"TP={our_row['tp']} FP={our_row['fp']} "
              f"TN={our_row['tn']} FN={our_row['fn']}")
        print(f"    Match: {'YES -- models are identical' if parity_ok else 'NO -- INVESTIGATE BEFORE USING THESE RESULTS'}")

        # Print reference points
        ref_pts = find_reference_points(curve)

        print()
        print(f"  -- Reference points (test set) --")

        print(f"\n  [1] Highest threshold where recall >= {RECALL_FLOOR:.2f}:")
        if ref_pts["recall_90"]:
            print(f"    {fmt_row(ref_pts['recall_90'])}")
        else:
            print(f"    NO threshold in the sweep achieves recall >= {RECALL_FLOOR:.2f}")

        print(f"\n  [2] Threshold that maximises F1:")
        print(f"    {fmt_row(ref_pts['max_f1'])}")

        print(f"\n  [3] Closest to precision = {PRECISION_TARGET:.2f}:")
        p70 = ref_pts["near_prec70"]
        if ref_pts["near_prec70_is_just_closest"]:
            print(f"    CAVEAT: No threshold reaches precision = {PRECISION_TARGET:.2f} "
                  f"(closest-but-still-bad point shown below).")
            print(f"    {fmt_row(p70)}")
        else:
            print(f"    {fmt_row(p70)}")

        print(f"\n  [4] Whether precision EVER exceeds {PRECISION_FLOOR:.2f} "
              f"(the Phase 11 floor) on the test set:")
        if ref_pts["prec_exceeds_floor"]:
            pf = ref_pts["prec_exceeds_floor"]
            print(f"    YES -- best recall while precision > {PRECISION_FLOOR:.2f}:")
            print(f"    {fmt_row(pf)}")
        else:
            print(f"    NO -- precision NEVER exceeds {PRECISION_FLOOR:.2f} "
                  f"anywhere in the {THRESHOLD_MIN:.2f}-{THRESHOLD_MAX:.2f} sweep "
                  f"on the test set.")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    section("WRITING threshold_tradeoff.csv")

    CSV_FIELDS = ["model_id", "model_name", "threshold",
                  "precision", "recall", "f1",
                  "tp", "fp", "tn", "fn"]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for spec in ACTIVE_MODELS:
            mid  = spec["id"]
            name = spec["name"]
            for row in all_curves[mid]:
                writer.writerow({
                    "model_id":   mid,
                    "model_name": name,
                    **row,
                })
    print(f"  Wrote: {OUT_CSV}")

    n_rows = sum(len(c) for c in all_curves.values())
    print(f"  Rows: {n_rows}  ({len(all_curves)} models x {n_rows // len(all_curves)} thresholds)")

    # ------------------------------------------------------------------
    # Write Markdown summary
    # ------------------------------------------------------------------
    section("WRITING threshold_tradeoff_summary.md")

    md: list[str] = []

    md += [
        "# Phase 11 Threshold Tradeoff Analysis",
        "",
        "> **DIAGNOSTIC WARNING:** This analysis applies thresholds to the "
        "**test set directly** to show the full precision/recall curve. "
        "The reference points reported here **must never be used to select a "
        "production threshold** -- that would be evaluating and tuning on the "
        "same data, invalidating every reported number. "
        "Its purpose is to help a human judge whether Phase 11's models have "
        "ANY usable operating point, not to replace Phase 11's "
        "validation-based threshold selection.",
        "",
        "Generated by `data/analyze_threshold_tradeoff.py`.  "
        "Full curve data: `data/threshold_tradeoff.csv`.",
        "",
        "---",
        "",
    ]

    for spec in ACTIVE_MODELS:
        mid  = spec["id"]
        name = spec["name"]
        curve = all_curves[mid]
        ref_pts = find_reference_points(curve)
        orig = next(m for m in ref["models"] if m["model_id"] == mid)

        md.append(f"## {name}")
        md.append("")
        md.append(
            f"Phase 11 auto-selected threshold: **{orig['val_chosen_threshold']}** "
            f"(chosen on validation set, precision={orig['val_precision_at_threshold']}, "
            f"recall={orig['val_recall_at_threshold']}).  "
            f"On the test set at that threshold: precision={orig['precision']}, "
            f"recall={orig['recall']}, F1={orig['f1']}, "
            f"TP={orig['tp']}, FP={orig['fp']}, TN={orig['tn']}, FN={orig['fn']}."
        )
        md.append("")

        md.append("### Reference points on the test set")
        md.append("")

        # [1] Recall >= 0.90
        md.append(f"**1. Highest threshold where recall >= {RECALL_FLOOR:.2f}**")
        md.append("")
        if ref_pts["recall_90"]:
            r = ref_pts["recall_90"]
            md.append(
                f"Threshold **{r['threshold']:.2f}** achieves recall={r['recall']:.4f} "
                f"at precision={r['precision']:.4f} (F1={r['f1']:.4f}). "
                f"At this point the model flags {r['tp']+r['fp']} of "
                f"{len(test_texts)} test narratives as SIF-potential, "
                f"catching {r['tp']} of {r['tp']+r['fn']} true positives "
                f"and generating {r['fp']} false alarms."
            )
        else:
            md.append(
                f"**No threshold in the {THRESHOLD_MIN:.2f}-{THRESHOLD_MAX:.2f} sweep "
                f"achieves recall >= {RECALL_FLOOR:.2f} on the test set.**"
            )
        md.append("")

        # [2] Max F1
        md.append("**2. Threshold that maximises F1**")
        md.append("")
        r = ref_pts["max_f1"]
        md.append(
            f"Threshold **{r['threshold']:.2f}** gives the highest F1={r['f1']:.4f} "
            f"(precision={r['precision']:.4f}, recall={r['recall']:.4f}). "
            f"TP={r['tp']}, FP={r['fp']}, TN={r['tn']}, FN={r['fn']}."
        )
        md.append("")

        # [3] Near precision=0.70
        md.append(f"**3. Reference point nearest to precision = {PRECISION_TARGET:.2f}**")
        md.append("")
        p70 = ref_pts["near_prec70"]
        if ref_pts["near_prec70_is_just_closest"]:
            md.append(
                f"**No threshold reaches precision = {PRECISION_TARGET:.2f} "
                f"(or within 0.05 of it) on the test set.** "
                f"The closest achievable point is threshold={p70['threshold']:.2f}, "
                f"which gives precision={p70['precision']:.4f}, "
                f"recall={p70['recall']:.4f}, F1={p70['f1']:.4f}. "
                f"This is reported only for reference; "
                f"it does not meet the 0.70 target."
            )
        else:
            md.append(
                f"Threshold **{p70['threshold']:.2f}** is the best point "
                f"within 0.05 of precision={PRECISION_TARGET:.2f}: "
                f"precision={p70['precision']:.4f}, recall={p70['recall']:.4f}, "
                f"F1={p70['f1']:.4f}. "
                f"TP={p70['tp']}, FP={p70['fp']}, TN={p70['tn']}, FN={p70['fn']}."
            )
        md.append("")

        # [4] Precision > 0.50
        md.append(f"**4. Does precision ever exceed {PRECISION_FLOOR:.2f} on the test set?**")
        md.append("")
        if ref_pts["prec_exceeds_floor"]:
            pf = ref_pts["prec_exceeds_floor"]
            md.append(
                f"Yes. The best recall achievable while keeping precision above "
                f"{PRECISION_FLOOR:.2f} is at threshold={pf['threshold']:.2f}: "
                f"precision={pf['precision']:.4f}, recall={pf['recall']:.4f}, "
                f"F1={pf['f1']:.4f}. "
                f"TP={pf['tp']}, FP={pf['fp']}, TN={pf['tn']}, FN={pf['fn']}."
            )
        else:
            md.append(
                f"**No. Precision does not exceed {PRECISION_FLOOR:.2f} at any "
                f"threshold between {THRESHOLD_MIN:.2f} and {THRESHOLD_MAX:.2f} "
                f"on the test set.** This means the model cannot simultaneously "
                f"achieve meaningful positive predictive value and non-trivial "
                f"recall on this held-out real-only data. "
                f"This is a plain statement of fact from the curve; whether it "
                f"is acceptable or actionable is for a human reviewer to decide."
            )
        md.append("")
        md.append("---")
        md.append("")

    md += [
        "## Full curve",
        "",
        "All threshold x metric values are in `data/threshold_tradeoff.csv`.",
        "",
        "---",
        "",
        "*This document is a diagnostic artifact only.  "
        "No model selection recommendation is made here.*",
    ]

    with OUT_MD.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"  Wrote: {OUT_MD}")

    print()
    print("  Done.  No winner declared, no threshold selected.")
    print("  Review threshold_tradeoff_summary.md and threshold_tradeoff.csv")
    print("  alongside benchmark_results.json before Phase 18.")
    print()


if __name__ == "__main__":
    main()
