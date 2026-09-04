"""
backend/sif_output_contract.py
================================
Output contract for the SIF (Serious Injury or Fatality potential) binary
classifier. Phase 11's benchmark (data/benchmark_results.json) found that
neither candidate embedding model discriminates reliably at any threshold that
also keeps recall high on real, held-out data (best test precision ~0.35 at
100% recall — barely above the test set's base rate). Collapsing that
uncertainty into a forced Yes/No would present low-confidence guesses as
confident ones, which is exactly what this contract exists to prevent.

Design mirrors backend/rule_output_contract.py's discipline intentionally,
so both classifiers in this system are reviewed and consumed the same way.

Usage
-----
    from backend.sif_output_contract import classify_sif, SIFPrediction

    prediction: SIFPrediction = classify_sif(
        report_text="...",
        classifier=clf_sif,
        encoder=encoder,
    )
    if prediction.sif_potential == "NEEDS_REVIEW":
        # route to human reviewer — do NOT treat as "No"
        ...

Classifier contract
--------------------
``classifier`` must expose scikit-learn's ``predict_proba`` returning an
(n_samples, 2) matrix ordered [P(No), P(Yes)] — standard for a binary
LogisticRegression fit on labels {0: No, 1: Yes}.

``encoder`` must expose ``.encode(texts, convert_to_numpy=True)`` — standard
for sentence-transformers SentenceTransformer objects (matches model.py's
existing ``encoder`` singleton, no new dependency introduced).

Design decisions
-----------------
- Two thresholds, not one. THRESHOLD_LOW and THRESHOLD_HIGH are PLACEHOLDER
  defaults (0.35 / 0.70) pending human sign-off. data/threshold_tradeoff_summary.md
  (Phase 11 follow-up) is explicit that its reference points are a DIAGNOSTIC
  view of the test set and must never be used to select a production threshold
  (that would tune on the same data used to report performance). So this
  contract deliberately does NOT lift a number from that curve. What that
  analysis DOES establish, and what motivates keeping this band wide: on the
  gold test set, no threshold reaches 0.70 precision for MiniLM (closest is
  0.61 precision at 0.86 threshold, recall 0.29); precision only clears 0.50
  in a narrow window around threshold 0.72 (precision 0.506, recall 0.609);
  and the F1-maximizing point (threshold 0.56, F1 0.588) still means roughly
  2 in 5 flagged reports are false alarms. That is real evidence the model's
  raw probability is not reliably separable into a single confident cut —
  hence two placeholder thresholds and a deliberately wide "ask a human"
  middle band, rather than one placeholder threshold with a narrow one.
- probability < THRESHOLD_LOW  -> "No"
- probability >= THRESHOLD_HIGH -> "Yes"
- everything else               -> "NEEDS_REVIEW"
- reasons are built only from evidence the existing pipeline can actually
  produce (LIME top phrases from model.py's explain_prediction, plus the
  raw probability and which threshold band it fell into) — nothing here
  invents an explanation method that doesn't exist elsewhere in the codebase.
- NEEDS_REVIEW is intentionally not "Yes" for the purposes of critical-flag
  auto-raising (backend/main.py's _is_critical already only checks
  sif_potential == "Yes", so this contract doesn't change that behavior —
  a NEEDS_REVIEW report is neither auto-critical nor silently treated as safe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ── Sentinel ─────────────────────────────────────────────────────────────
NEEDS_REVIEW: str = "NEEDS_REVIEW"

# ── Thresholds — PLACEHOLDERS pending human calibration ────────────────────
# Deliberately NOT derived from data/threshold_tradeoff_summary.md — that
# document evaluates thresholds directly on the test set and explicitly warns
# its numbers must never be used to pick a production threshold (doing so
# would tune and report on the same data). It IS used here as evidence that a
# single confident cutoff is not supportable: on the gold test set no
# threshold reaches 0.70 precision, precision only exceeds 0.50 in a narrow
# window near 0.72 (recall drops to 0.61 there), and the best F1 (0.588 at
# threshold 0.56) still means ~2 in 5 flagged reports are false alarms.
# Deliberately wide apart, favoring more NEEDS_REVIEW rather than a forced
# binary call, given Phase 11's validation-based finding that this model's
# probability output does not reliably separate Yes from No on real
# narratives outside the training distribution.

#: Below this probability, predict "No" with reasonable confidence.
THRESHOLD_LOW: float = 0.35

#: At or above this probability, predict "Yes" with reasonable confidence.
THRESHOLD_HIGH: float = 0.70

MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2 + LogisticRegression"
MODEL_VERSION: str = "1.0.0"  # bump manually when model_artifacts/ is retrained


@dataclass
class SIFPrediction:
    sif_potential: str            # "Yes" | "No" | "NEEDS_REVIEW"
    confidence: float             # raw model P(Yes), 0-1, regardless of label
    reasons: list[str] = field(default_factory=list)
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    decision_thresholds_used: dict = field(
        default_factory=lambda: {"low": THRESHOLD_LOW, "high": THRESHOLD_HIGH}
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def as_dict(self) -> dict:
        """Flat dict for merging into the existing API response shape."""
        return {
            "sif_potential": self.sif_potential,
            "confidence": round(self.confidence, 4),
            "sif_reasons": self.reasons,
            "sif_model_name": self.model_name,
            "sif_model_version": self.model_version,
            "sif_decision_thresholds": self.decision_thresholds_used,
            "sif_prediction_timestamp": self.timestamp,
        }


def _band_reason(prob_yes: float) -> str:
    if prob_yes < THRESHOLD_LOW:
        return f"P(Yes)={prob_yes:.3f} below low threshold {THRESHOLD_LOW} -> No"
    if prob_yes >= THRESHOLD_HIGH:
        return f"P(Yes)={prob_yes:.3f} at/above high threshold {THRESHOLD_HIGH} -> Yes"
    return (
        f"P(Yes)={prob_yes:.3f} between thresholds "
        f"[{THRESHOLD_LOW}, {THRESHOLD_HIGH}) -> NEEDS_REVIEW "
        f"(Phase 11 found this range unreliable on real narratives)"
    )


def classify_sif(
    report_text: str,
    classifier,
    encoder,
    lime_phrases: Optional[list] = None,
) -> SIFPrediction:
    """
    Run the SIF binary classifier and wrap its output in the three-way
    contract. Does not retrain or modify the classifier — pure inference
    plus thresholding/packaging.

    ``lime_phrases``: optionally pass already-computed
    [(phrase, weight), ...] from model.explain_prediction so this function
    doesn't duplicate that (potentially slow) computation. If omitted,
    reasons will just contain the probability-band explanation.
    """
    if not report_text or not report_text.strip():
        raise ValueError("report_text is required and cannot be empty")

    emb = encoder.encode([report_text], convert_to_numpy=True)
    proba = classifier.predict_proba(emb)[0]
    prob_yes = float(np.clip(proba[1], 0.0, 1.0))

    if prob_yes < THRESHOLD_LOW:
        label = "No"
    elif prob_yes >= THRESHOLD_HIGH:
        label = "Yes"
    else:
        label = NEEDS_REVIEW

    reasons = [_band_reason(prob_yes)]
    if lime_phrases:
        top = sorted(lime_phrases, key=lambda x: abs(x[1]), reverse=True)[:3]
        for phrase, weight in top:
            direction = "toward Yes" if weight > 0 else "toward No"
            reasons.append(f"contributing phrase '{phrase}' ({direction}, weight={weight:.3f})")

    return SIFPrediction(
        sif_potential=label,
        confidence=prob_yes,
        reasons=reasons,
    )
