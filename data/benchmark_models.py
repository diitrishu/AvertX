"""
benchmark_models.py — Phase 11: SIF classifier embedding benchmark.

Trains a Logistic Regression head on top of three sentence-embedding models
using data/splits/train.csv, tunes a decision threshold on validation.csv,
and evaluates ONCE on test.csv.

Models compared (apples-to-apples -- same LR head on all three):
  1. sentence-transformers/all-MiniLM-L6-v2  (Apache 2.0, current production)
  2. SafetyBERT                               -- see LICENSE FINDING below
  3. BAAI/bge-small-en-v1.5                   (MIT)

LICENSE FINDING -- SafetyBERT (recorded before first use, per project policy):
  Two models appear on HuggingFace under the name "SafetyBERT":
    a) adanish91/safetybert  -- domain-adapted BertForMaskedLM (no classification
       head, no declared license). Architecture makes it unsuitable for the
       sentence-embedding + LR pattern used here without custom mean-pooling;
       license is absent, not just ambiguous.
    b) invalidexception/safetybert -- BertForSequenceClassification trained on
       ifmain/text-moderation (content moderation, not occupational/industrial
       safety). Wrong domain entirely. Also no declared license.
  Neither model meets the project's license-verification standard (same standard
  applied to every dataset in this project). Per the spec's explicit instruction:
  "If you cannot find a real, working SafetyBERT model on HuggingFace with a
  clear license, STOP and report that instead of substituting a guess or a
  different model under the same name."
  Model 2 slot is therefore marked SKIPPED with this finding in all outputs.
  The benchmark proceeds with models 1 and 3.

Outputs:
  data/benchmark_results.json   -- full per-model metrics
  data/benchmark_results.md     -- human-readable markdown table

Usage:
    python data/benchmark_models.py

Re-running produces identical metrics (fixed seed=42), aside from timing
numbers which vary with CPU load.
"""

import io
import sys

# Force UTF-8 stdout/stderr so non-ASCII in strings doesn't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import csv
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(__file__).parent
SPLITS_DIR  = DATA_ROOT / "splits"
TRAIN_CSV   = SPLITS_DIR / "train.csv"
VAL_CSV     = SPLITS_DIR / "validation.csv"
TEST_CSV    = SPLITS_DIR / "test.csv"
RESULTS_DIR = DATA_ROOT
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
RESULTS_MD   = RESULTS_DIR / "benchmark_results.md"

RANDOM_SEED = 42
THRESHOLD_MIN   = 0.05
THRESHOLD_MAX   = 0.95
THRESHOLD_STEP  = 0.01
PRECISION_FLOOR = 0.50   # minimum precision for threshold selection
N_CAL_BINS      = 10     # reliability diagram buckets

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry is either a real spec or a SKIPPED marker.
# SKIPPED entries are recorded in outputs with reason; they are not trained.

MODEL_SPECS = [
    {
        "id":    "minilm",
        "name":  "sentence-transformers/all-MiniLM-L6-v2",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "license": "Apache 2.0",
        "license_source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "skipped": False,
    },
    {
        "id":    "safetybert",
        "name":  "SafetyBERT",
        "hf_id": None,
        "license": None,
        "skipped": True,
        "skip_reason": (
            "No usable SafetyBERT model found on HuggingFace with a clear license. "
            "Two candidates exist: (1) adanish91/safetybert — BertForMaskedLM, no "
            "classification head, no declared license; (2) invalidexception/safetybert "
            "— BertForSequenceClassification trained on text-moderation data (wrong "
            "domain), no declared license. Neither meets the project's license-"
            "verification standard. Slot left empty per spec instructions."
        ),
    },
    {
        "id":    "bge_small",
        "name":  "BAAI/bge-small-en-v1.5",
        "hf_id": "BAAI/bge-small-en-v1.5",
        "license": "MIT",
        "license_source": "https://huggingface.co/BAAI/bge-small-en-v1.5",
        "skipped": False,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(path: Path) -> tuple[list[str], list[int]]:
    """Return (narratives, labels) from a split CSV. label: 1=Yes, 0=No."""
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["narrative_text"])
            labels.append(1 if row["sif_potential"] == "Yes" else 0)
    return texts, labels


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1_048_576


def disk_size_mb(hf_id: str) -> float:
    """Approximate on-disk size of a cached HuggingFace model (MB)."""
    import glob as _glob
    cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    # HF hub stores models in models--{org}--{name} directories
    slug = hf_id.replace("/", "--")
    candidates = list(cache_root.glob(f"models--{slug}/**/*"))
    total = sum(p.stat().st_size for p in candidates if p.is_file())
    if total == 0:
        # Try sentence-transformers cache location
        st_cache = Path(os.path.expanduser("~/.cache/torch/sentence_transformers"))
        slug2 = hf_id.replace("/", "_")
        candidates2 = list(st_cache.glob(f"{slug2}/**/*"))
        total = sum(p.stat().st_size for p in candidates2 if p.is_file())
    return total / 1_048_576 if total > 0 else float("nan")


def pick_threshold(
    y_true: list[int],
    probs: np.ndarray,
) -> tuple[float, float, float]:
    """
    Sweep thresholds from THRESHOLD_MIN to THRESHOLD_MAX.
    Pick the highest threshold where precision >= PRECISION_FLOOR
    (ensuring some floor) AND recall is maximized at that precision level.
    Returns (threshold, precision_at_threshold, recall_at_threshold).
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP)
    y_arr = np.array(y_true)
    best_thresh = THRESHOLD_MIN
    best_recall = 0.0
    best_precision = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_arr == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec >= PRECISION_FLOOR and rec > best_recall:
            best_recall = rec
            best_precision = prec
            best_thresh = float(t)

    return round(best_thresh, 4), round(best_precision, 4), round(best_recall, 4)


def evaluate_at_threshold(
    y_true: list[int],
    probs: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all test-set metrics at a given threshold."""
    y_arr  = np.array(y_true)
    preds  = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fn_rate   = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    fp_rate   = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # PR-AUC (threshold-independent)
    sk_prec, sk_rec, _ = precision_recall_curve(y_arr, probs)
    pr_auc = auc(sk_rec, sk_prec)

    # Brier score
    brier = brier_score_loss(y_arr, probs)

    # Reliability (calibration) bins
    cal_bins = []
    bin_edges = np.linspace(0.0, 1.0, N_CAL_BINS + 1)
    for i in range(N_CAL_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            cal_bins.append({
                "bin_low": round(float(lo), 2),
                "bin_high": round(float(hi), 2),
                "n": 0,
                "mean_predicted_prob": None,
                "actual_positive_rate": None,
            })
        else:
            cal_bins.append({
                "bin_low": round(float(lo), 2),
                "bin_high": round(float(hi), 2),
                "n": int(mask.sum()),
                "mean_predicted_prob": round(float(probs[mask].mean()), 4),
                "actual_positive_rate": round(float(y_arr[mask].mean()), 4),
            })

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "fn_rate":    round(fn_rate, 4),
        "fp_rate":    round(fp_rate, 4),
        "pr_auc":     round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "calibration_bins": cal_bins,
    }


def latency_ms(model, texts: list[str], n_warmup: int = 3) -> float:
    """
    Mean ms to embed + classify one narrative, measured on CPU.
    Warm-up runs first (cache/JIT effects), then 20 timed single-text passes.
    """
    clf_dummy = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                   class_weight="balanced")
    # Fit on two fake points so predict_proba works
    dummy_embs = model.encode(["dummy text a", "dummy text b"],
                               convert_to_numpy=True)
    clf_dummy.fit(dummy_embs, [0, 1])

    # Warm-up
    for t in texts[:n_warmup]:
        e = model.encode([t], convert_to_numpy=True)
        clf_dummy.predict_proba(e)

    # Timed runs (single text each)
    timed_texts = texts[:20]
    times = []
    for t in timed_texts:
        t0 = time.perf_counter()
        e = model.encode([t], convert_to_numpy=True)
        clf_dummy.predict_proba(e)
        times.append((time.perf_counter() - t0) * 1000)

    return round(float(np.mean(times)), 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch

    section("PHASE 11 — SIF CLASSIFIER EMBEDDING BENCHMARK")

    # Environment disclosure (required by spec)
    cuda_available = torch.cuda.is_available()
    print(f"\n  GPU available : {cuda_available}")
    if cuda_available:
        print(f"  GPU device    : {torch.cuda.get_device_name(0)}")
        print("  NOTE: Embeddings will run on GPU — latency numbers reflect GPU inference.")
    else:
        print("  NOTE: No GPU found. Embeddings run on CPU — latency numbers reflect CPU inference.")
    print(f"  CPU count     : {psutil.cpu_count()}")
    print(f"  RAM total     : {psutil.virtual_memory().total / 1_048_576:,.0f} MB")

    # Load splits
    section("LOADING SPLITS")
    train_texts, train_labels = load_split(TRAIN_CSV)
    val_texts,   val_labels   = load_split(VAL_CSV)
    test_texts,  test_labels  = load_split(TEST_CSV)
    print(f"  train : {len(train_texts):>5} rows  "
          f"({sum(train_labels)} Yes / {len(train_labels)-sum(train_labels)} No)")
    print(f"  val   : {len(val_texts):>5} rows  "
          f"({sum(val_labels)} Yes / {len(val_labels)-sum(val_labels)} No)")
    print(f"  test  : {len(test_texts):>5} rows  "
          f"({sum(test_labels)} Yes / {len(test_labels)-sum(test_labels)} No)")
    print()
    print("  IMPORTANT: Threshold is tuned on val, evaluated ONCE on test.")
    print("  Test set results are never examined until after threshold is fixed.")

    # -----------------------------------------------------------------------
    # License check and SafetyBERT skip notice (printed before any model use)
    # -----------------------------------------------------------------------
    section("LICENSE VERIFICATION (printed before any model is loaded)")
    for spec in MODEL_SPECS:
        if spec["skipped"]:
            print(f"\n  [{spec['id'].upper()}]  {spec['name']}")
            print(f"    STATUS  : SKIPPED")
            print(f"    REASON  : {spec['skip_reason']}")
        else:
            print(f"\n  [{spec['id'].upper()}]  {spec['name']}")
            print(f"    HF ID   : {spec['hf_id']}")
            print(f"    LICENSE : {spec['license']}")
            print(f"    SOURCE  : {spec['license_source']}")

    # -----------------------------------------------------------------------
    # Per-model benchmark loop
    # -----------------------------------------------------------------------
    all_results: dict = {}

    active_specs = [s for s in MODEL_SPECS if not s["skipped"]]
    skipped_specs = [s for s in MODEL_SPECS if s["skipped"]]

    for spec in active_specs:
        mid   = spec["id"]
        hf_id = spec["hf_id"]

        section(f"MODEL: {spec['name']}")

        result: dict = {
            "model_id":     mid,
            "model_name":   spec["name"],
            "hf_id":        hf_id,
            "license":      spec["license"],
            "license_source": spec.get("license_source"),
            "skipped":      False,
            "error":        None,
        }

        try:
            from sentence_transformers import SentenceTransformer

            # ── Load model ────────────────────────────────────────────────
            subsection("Loading model")
            rss_before_load = rss_mb()
            t_load_start = time.perf_counter()
            model = SentenceTransformer(hf_id)
            t_load = time.perf_counter() - t_load_start
            rss_after_load = rss_mb()
            load_delta_mb = rss_after_load - rss_before_load
            on_disk_mb = disk_size_mb(hf_id)
            print(f"  Load time       : {t_load:.1f}s")
            print(f"  RSS delta       : {load_delta_mb:+.1f} MB  (before={rss_before_load:.0f} MB, after={rss_after_load:.0f} MB)")
            print(f"  On-disk size    : {on_disk_mb:.1f} MB" if not np.isnan(on_disk_mb)
                  else "  On-disk size    : (cache path not found)")

            result["memory_load_rss_delta_mb"] = round(load_delta_mb, 1)
            result["memory_rss_after_load_mb"] = round(rss_after_load, 1)
            result["on_disk_mb"] = round(on_disk_mb, 1) if not np.isnan(on_disk_mb) else None

            # ── Embed train ───────────────────────────────────────────────
            subsection("Embedding training set")
            t0 = time.perf_counter()
            train_embs = model.encode(train_texts, convert_to_numpy=True,
                                      show_progress_bar=True, batch_size=64)
            t_train_embed = time.perf_counter() - t0
            print(f"  Shape           : {train_embs.shape}")
            print(f"  Time            : {t_train_embed:.1f}s")

            # ── Fit Logistic Regression ───────────────────────────────────
            subsection("Training Logistic Regression (class_weight='balanced')")
            clf = LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
                class_weight="balanced",
                solver="lbfgs",
            )
            t0 = time.perf_counter()
            clf.fit(train_embs, train_labels)
            t_fit = time.perf_counter() - t0
            print(f"  Fit time        : {t_fit:.2f}s")

            # ── Embed validation and get probabilities ────────────────────
            subsection("Embedding validation set + threshold selection")
            val_embs  = model.encode(val_texts, convert_to_numpy=True,
                                     show_progress_bar=False, batch_size=64)
            val_probs = clf.predict_proba(val_embs)[:, 1]

            chosen_thresh, prec_at_thresh, rec_at_thresh = pick_threshold(
                val_labels, val_probs
            )
            print(f"  Threshold sweep : {THRESHOLD_MIN:.2f} → {THRESHOLD_MAX:.2f}, "
                  f"step {THRESHOLD_STEP:.2f}")
            print(f"  Criterion       : max recall subject to precision >= {PRECISION_FLOOR:.2f}")
            print(f"  Chosen threshold: {chosen_thresh}")
            print(f"  Precision at threshold (val): {prec_at_thresh:.4f}")
            print(f"    Recall at threshold (val): {rec_at_thresh:.4f}")
            print()
            print("  Threshold is now FIXED. Test set will be evaluated exactly once below.")

            result["val_chosen_threshold"]      = chosen_thresh
            result["val_precision_at_threshold"] = prec_at_thresh
            result["val_recall_at_threshold"]    = rec_at_thresh

            # ── One-shot test evaluation ──────────────────────────────────
            subsection("Evaluating on TEST SET (one shot, threshold already fixed)")
            rss_before_infer = rss_mb()
            test_embs  = model.encode(test_texts, convert_to_numpy=True,
                                      show_progress_bar=False, batch_size=64)
            test_probs = clf.predict_proba(test_embs)[:, 1]
            rss_after_infer = rss_mb()
            infer_delta_mb = rss_after_infer - rss_before_infer

            metrics = evaluate_at_threshold(test_labels, test_probs, chosen_thresh)
            result.update(metrics)
            result["memory_inference_rss_delta_mb"] = round(infer_delta_mb, 1)

            print(f"  Threshold used  : {chosen_thresh}")
            print()
            print(f"  Confusion matrix:")
            print(f"                      Predicted No   Predicted Yes")
            print(f"    Actual No   :       {metrics['tn']:>6}         {metrics['fp']:>6}")
            print(f"    Actual Yes  :       {metrics['fn']:>6}         {metrics['tp']:>6}")
            print()
            print(f"  Precision     : {metrics['precision']:.4f}")
            print(f"  Recall        : {metrics['recall']:.4f}")
            print(f"  F1            : {metrics['f1']:.4f}")
            print(f"  PR-AUC        : {metrics['pr_auc']:.4f}")
            print()
            print(f"  False Negative rate : {metrics['fn_rate']:.4f}  "
                  f"({metrics['fn']} missed SIF-Yes out of {metrics['tp']+metrics['fn']})")
            print(f"  False Positive rate : {metrics['fp_rate']:.4f}  "
                  f"({metrics['fp']} false alarms out of {metrics['tn']+metrics['fp']})")
            print()
            print(f"  Brier score   : {metrics['brier_score']:.4f}  "
                  f"(lower is better; 0=perfect, 0.25=no-skill baseline)")

            subsection("Calibration (reliability bins)")
            print(f"  {'Bin':<14}  {'n':>5}  {'Mean pred prob':>14}  {'Actual pos rate':>15}")
            print(f"  {'-'*14}  {'-'*5}  {'-'*14}  {'-'*15}")
            for b in metrics["calibration_bins"]:
                if b["n"] == 0:
                    print(f"  [{b['bin_low']:.2f}, {b['bin_high']:.2f})  "
                          f"{'0':>5}  {'(empty)':>14}  {'(empty)':>15}")
                else:
                    print(f"  [{b['bin_low']:.2f}, {b['bin_high']:.2f})  "
                          f"{b['n']:>5}  {b['mean_predicted_prob']:>14.4f}  "
                          f"{b['actual_positive_rate']:>15.4f}")

            # ── Latency ───────────────────────────────────────────────────
            subsection("Latency (embed + classify one narrative, CPU)")
            lat_ms = latency_ms(model, test_texts)
            result["latency_mean_ms_per_text"] = lat_ms
            result["latency_measured_on_gpu"]  = cuda_available
            print(f"  Mean latency    : {lat_ms:.2f} ms / text")
            print(f"  GPU was {'used' if cuda_available else 'NOT used'} — "
                  f"this affects the number and is disclosed here")
            result["memory_inference_rss_delta_mb"] = round(infer_delta_mb, 1)
            print(f"  Inference RSS delta: {infer_delta_mb:+.1f} MB")

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            print(f"\n  !! ERROR during benchmarking: {err_msg}")
            print(tb)
            result["error"] = err_msg
            result["traceback"] = tb

        all_results[mid] = result

    # -----------------------------------------------------------------------
    # Record skipped models
    # -----------------------------------------------------------------------
    for spec in skipped_specs:
        all_results[spec["id"]] = {
            "model_id":   spec["id"],
            "model_name": spec["name"],
            "hf_id":      spec.get("hf_id"),
            "license":    spec.get("license"),
            "skipped":    True,
            "skip_reason": spec["skip_reason"],
        }

    # -----------------------------------------------------------------------
    # Summary table to stdout
    # -----------------------------------------------------------------------
    section("SUMMARY TABLE")
    header = (f"  {'Model':<30}  {'Thresh':>6}  {'Prec':>6}  {'Rec':>6}  "
              f"{'F1':>6}  {'PR-AUC':>6}  {'FN-rate':>7}  {'FP-rate':>7}  "
              f"{'Brier':>6}  {'Lat(ms)':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    order = ["minilm", "safetybert", "bge_small"]
    for mid in order:
        r = all_results.get(mid, {})
        if r.get("skipped"):
            print(f"  {'SafetyBERT':<30}  {'SKIPPED — see license finding above':}")
        elif r.get("error"):
            print(f"  {r.get('model_name','?'):<30}  ERROR: {r['error'][:50]}")
        else:
            print(
                f"  {r['model_name']:<30}  "
                f"{r.get('threshold', float('nan')):>6.3f}  "
                f"{r.get('precision', float('nan')):>6.4f}  "
                f"{r.get('recall', float('nan')):>6.4f}  "
                f"{r.get('f1', float('nan')):>6.4f}  "
                f"{r.get('pr_auc', float('nan')):>6.4f}  "
                f"{r.get('fn_rate', float('nan')):>7.4f}  "
                f"{r.get('fp_rate', float('nan')):>7.4f}  "
                f"{r.get('brier_score', float('nan')):>6.4f}  "
                f"{r.get('latency_mean_ms_per_text', float('nan')):>8.2f}"
            )

    print()
    print("  FN-rate = False Negative rate (fraction of SIF-Yes missed by the model).")
    print("  Lower FN-rate is higher priority given the project's cost asymmetry.")
    print()
    print("  NOTE: No winner is declared here. Model selection is Phase 18 (MODEL_COMPARISON.md),")
    print("  after a human reviews these numbers.")

    # -----------------------------------------------------------------------
    # Write JSON
    # -----------------------------------------------------------------------
    section("WRITING OUTPUTS")
    results_list = [all_results[mid] for mid in order if mid in all_results]

    with RESULTS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_by": "data/benchmark_models.py",
                "phase": "Phase 11",
                "splits": {
                    "train": str(TRAIN_CSV),
                    "validation": str(VAL_CSV),
                    "test": str(TEST_CSV),
                },
                "threshold_selection": {
                    "sweep_range": [THRESHOLD_MIN, THRESHOLD_MAX],
                    "step": THRESHOLD_STEP,
                    "criterion": (
                        f"maximize recall subject to precision >= {PRECISION_FLOOR}"
                    ),
                },
                "hardware": {
                    "gpu_available": cuda_available,
                    "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
                    "cpu_count": psutil.cpu_count(),
                },
                "models": results_list,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"  Wrote: {RESULTS_JSON}")

    # -----------------------------------------------------------------------
    # Write Markdown
    # -----------------------------------------------------------------------
    active_results = [all_results[mid] for mid in order
                      if mid in all_results and not all_results[mid].get("skipped")
                      and not all_results[mid].get("error")]

    md_lines = [
        "# SIF Classifier Embedding Benchmark — Phase 11",
        "",
        "Generated by `data/benchmark_models.py`. "
        "No winner is declared -- model selection is Phase 18 (`MODEL_COMPARISON.md`) "
        "after human review.",
        "",
        "## Environment",
        "",
        f"- GPU available: `{cuda_available}`"
        + (f" ({torch.cuda.get_device_name(0)})" if cuda_available else ""),
        f"- Inference device: `{'GPU' if cuda_available else 'CPU'}`",
        f"- Splits: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}",
        "",
        "## Threshold selection criterion",
        "",
        f"Sweep {THRESHOLD_MIN}-{THRESHOLD_MAX} in steps of {THRESHOLD_STEP}. "
        f"Pick threshold that maximises recall subject to precision >= {PRECISION_FLOOR}. "
        "Tuned on validation set only -- test set never seen during this step.",
        "",
        "## License verification",
        "",
    ]

    for spec in MODEL_SPECS:
        if spec["skipped"]:
            md_lines.append(f"- **{spec['name']}**: SKIPPED — {spec['skip_reason']}")
        else:
            md_lines.append(
                f"- **{spec['name']}** (`{spec['hf_id']}`): "
                f"{spec['license']} — {spec.get('license_source','')}"
            )
    md_lines.append("")

    # Main results table
    md_lines += [
        "## Test-set results",
        "",
        "| Model | Threshold | Precision | Recall | F1 | PR-AUC | "
        "FN-rate | FP-rate | Brier | Lat (ms) | On-disk (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for mid in order:
        r = all_results.get(mid, {})
        name = r.get("model_name", mid)
        if r.get("skipped"):
            md_lines.append(f"| {name} | SKIPPED | — | — | — | — | — | — | — | — | — |")
        elif r.get("error"):
            md_lines.append(f"| {name} | ERROR | — | — | — | — | — | — | — | — | — |")
        else:
            md_lines.append(
                f"| {name} "
                f"| {r.get('threshold',''):>5} "
                f"| {r.get('precision',''):>7} "
                f"| {r.get('recall',''):>7} "
                f"| {r.get('f1',''):>7} "
                f"| {r.get('pr_auc',''):>7} "
                f"| {r.get('fn_rate',''):>7} "
                f"| {r.get('fp_rate',''):>7} "
                f"| {r.get('brier_score',''):>7} "
                f"| {r.get('latency_mean_ms_per_text',''):>7} "
                f"| {r.get('on_disk_mb', '?')} |"
            )

    md_lines += [
        "",
        "**FN-rate** = False Negative rate = fraction of SIF-Yes cases missed. "
        "Lower is higher priority given the project's cost asymmetry.",
        "",
    ]

    # Val threshold context per model
    md_lines.append("## Threshold selection detail (validation set)")
    md_lines.append("")
    for mid in order:
        r = all_results.get(mid, {})
        if r.get("skipped") or r.get("error"):
            continue
        md_lines.append(
            f"- **{r['model_name']}**: threshold `{r['val_chosen_threshold']}` chosen -- "
            f"precision `{r['val_precision_at_threshold']}`, "
            f"recall `{r['val_recall_at_threshold']}` on validation set."
        )
    md_lines.append("")

    # Calibration tables
    md_lines.append("## Calibration (reliability bins, test set)")
    md_lines.append("")
    for mid in order:
        r = all_results.get(mid, {})
        if r.get("skipped") or r.get("error"):
            continue
        md_lines.append(f"### {r['model_name']}")
        md_lines.append("")
        md_lines.append("| Bin | n | Mean predicted prob | Actual positive rate |")
        md_lines.append("|---|---|---|---|")
        for b in r.get("calibration_bins", []):
            mpp = f"{b['mean_predicted_prob']:.4f}" if b["mean_predicted_prob"] is not None else "(empty)"
            apr = f"{b['actual_positive_rate']:.4f}" if b["actual_positive_rate"] is not None else "(empty)"
            md_lines.append(f"| [{b['bin_low']:.2f}, {b['bin_high']:.2f}) | {b['n']} | {mpp} | {apr} |")
        md_lines.append("")

    md_lines += [
        "---",
        "",
        "*This document is a measurement artifact only. "
        "No model selection recommendation is made here.*",
    ]

    with RESULTS_MD.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"  Wrote: {RESULTS_MD}")

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
