"""
backend/rule_output_contract.py
================================
Output contract for the Life-Saving Rule multi-class classifier.

Mirrors Phase 12's SIF output-contract discipline:
  - Structured dataclass (RulePrediction) with a defined schema.
  - NEEDS_REVIEW sentinel replaces the rule_id when the classifier is not
    confident enough to be trusted by downstream consumers.
  - Two independent NEEDS_REVIEW triggers (see module-level constants):
      1. Top-class confidence below CONFIDENCE_THRESHOLD  (low certainty)
      2. Gap between top-class and runner-up below AMBIGUITY_MARGIN
         (model is confident overall but genuinely torn between two rules —
         a different failure mode that deserves its own signal)

Usage
-----
    from backend.rule_output_contract import classify_rule, RulePrediction

    prediction: RulePrediction = classify_rule(
        report_text="Worker entered confined space without gas test",
        classifier=clf_rule,
        encoder=encoder,
        rule_classes=["Confined Space", "Energy Isolation", ...],
    )
    if prediction.rule_prediction == NEEDS_REVIEW:
        # route to human reviewer
        ...

Classifier contract
-------------------
``classifier`` must expose a scikit-learn-compatible ``predict_proba`` API
returning an (n_samples, n_classes) probability matrix, where classes are
ordered as ``classifier.classes_`` — standard for all sklearn estimators.

``encoder`` must expose ``.encode(texts, convert_to_numpy=True)`` — standard
for sentence-transformers SentenceTransformer objects.

The 9 valid IOGP Life-Saving Rule IDs (rule_ids match taxonomy CSV):
    bypassing_safety_controls, confined_space, driving, energy_isolation,
    hot_work, line_of_fire, safe_mechanical_lifting, work_authorisation,
    working_at_height

NEEDS_REVIEW is intentionally NOT one of those 9 IDs. It is the sentinel
returned when neither trigger condition is satisfied. Consumers must handle
it explicitly — it is not a valid IOGP rule label.

Design decisions
----------------
- Both thresholds are module-level constants documented as "placeholder pending
  human review" (identical language to Phase 12's SIF contract) — they carry
  no statistical guarantee until calibrated on a larger labelled dataset.
- runner_up_rule / runner_up_confidence are always computed and included in the
  dataclass regardless of NEEDS_REVIEW status — they are useful context for the
  human reviewer even when the classifier produces a confident, unambiguous call.
- model_version is kept as a free-form string, not a semver — the rule
  classifier is still in its first iteration and doesn't yet have a release
  cadence. Increment it manually when artifacts are retrained.
- timestamp is ISO-8601 UTC to the second, same as SIF contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# ── NEEDS_REVIEW sentinel ──────────────────────────────────────────────────────
NEEDS_REVIEW: str = "NEEDS_REVIEW"

# ── Thresholds — PLACEHOLDERS pending human calibration ───────────────────────
# Both values are flagged explicitly because they are NOT derived from a held-out
# calibration curve (the available test set is too small for reliable calibration
# — see data/rule_classifier_eval.md).  They should be revisited once a larger
# labelled test set with ≥20 examples per class is available.

#: Minimum top-class probability for a rule prediction to be trusted.
#: Below this → NEEDS_REVIEW (failure mode: low certainty).
CONFIDENCE_THRESHOLD: float = 0.50  # PLACEHOLDER — see module docstring

#: Minimum gap between top and runner-up probabilities.
#: Below this → NEEDS_REVIEW (failure mode: ambiguity between two specific rules).
AMBIGUITY_MARGIN: float = 0.15  # PLACEHOLDER — see module docstring

# ── Model identity constants ───────────────────────────────────────────────────
MODEL_NAME: str = "rule_classifier_lr_minilm"
MODEL_VERSION: str = "1.0.0"


# ── Output dataclass ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RulePrediction:
    """
    Structured output of the Life-Saving Rule classifier.

    Attributes
    ----------
    rule_prediction : str
        One of the 9 IOGP rule_ids from taxonomy/life_saving_rules.csv,
        or "NEEDS_REVIEW" when either confidence or ambiguity trigger fires.
        Note: the stored value is the human-readable rule name (e.g. "Hot Work"),
        not the snake_case rule_id — this matches the label encoding used in
        the training data and the existing model_metadata.json convention.
    confidence : float
        Probability assigned to the top predicted class, in [0, 1].
        Always the raw model probability regardless of NEEDS_REVIEW status.
    runner_up_rule : str or None
        Human-readable name of the second-highest-probability rule.
        None only if fewer than 2 classes exist (cannot happen with 9 classes,
        but included for defensive typing).
    runner_up_confidence : float or None
        Probability of the runner-up class, in [0, 1].
        None only if runner_up_rule is None.
    needs_review : bool
        True when either NEEDS_REVIEW trigger fired. Redundant with checking
        rule_prediction == NEEDS_REVIEW but included for ergonomics in
        boolean filter expressions.
    review_reason : str or None
        Human-readable explanation of why NEEDS_REVIEW was triggered.
        None when needs_review is False.
    model_name : str
        Stable identifier for the classifier family.
    model_version : str
        Version string; increment when artifacts are retrained.
    timestamp : str
        ISO-8601 UTC timestamp of prediction generation.
    """
    rule_prediction: str
    confidence: float
    runner_up_rule: Optional[str]
    runner_up_confidence: Optional[float]
    needs_review: bool
    review_reason: Optional[str]
    model_name: str
    model_version: str
    timestamp: str


# ── Core prediction function ───────────────────────────────────────────────────
def classify_rule(
    report_text: str,
    classifier,          # sklearn estimator with predict_proba
    encoder,             # sentence-transformers SentenceTransformer
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> RulePrediction:
    """
    Embed *report_text* and run the multi-class rule classifier.

    Returns a :class:`RulePrediction` dataclass. When ``rule_prediction``
    equals ``NEEDS_REVIEW``, route the report to a human reviewer before
    acting on the rule label.

    Parameters
    ----------
    report_text : str
        Raw safety report narrative. Empty/whitespace strings are handled
        gracefully — they produce low-confidence embeddings that almost
        always trigger NEEDS_REVIEW.
    classifier : sklearn estimator
        Trained LogisticRegression (or compatible) with ``.predict_proba``
        and ``.classes_`` attributes. Must be the model saved by
        ``train_rule_classifier.py``.
    encoder : SentenceTransformer
        Embedding model (all-MiniLM-L6-v2). Must match the model used
        during training.
    confidence_threshold : float
        Override the module-level ``CONFIDENCE_THRESHOLD``.
    ambiguity_margin : float
        Override the module-level ``AMBIGUITY_MARGIN``.

    Returns
    -------
    RulePrediction
    """
    if not report_text or not report_text.strip():
        warnings.warn(
            "classify_rule received empty or whitespace-only report_text; "
            "prediction will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # ── Embed ──────────────────────────────────────────────────────────────────
    emb = encoder.encode([report_text], convert_to_numpy=True)  # shape (1, 384)

    # ── Predict probabilities ──────────────────────────────────────────────────
    proba = classifier.predict_proba(emb)[0]        # shape (n_classes,)
    classes = list(classifier.classes_)             # ordered class labels

    # ── Rank classes ──────────────────────────────────────────────────────────
    sorted_idx = np.argsort(proba)[::-1]            # descending by probability
    top_idx = int(sorted_idx[0])
    top_rule = classes[top_idx]
    top_conf = float(proba[top_idx])

    if len(sorted_idx) >= 2:
        runner_idx = int(sorted_idx[1])
        runner_rule: Optional[str] = classes[runner_idx]
        runner_conf: Optional[float] = float(proba[runner_idx])
    else:
        runner_rule = None
        runner_conf = None

    # ── NEEDS_REVIEW logic ─────────────────────────────────────────────────────
    review_reasons: list[str] = []

    if top_conf < confidence_threshold:
        review_reasons.append(
            f"top-class confidence {top_conf:.3f} < threshold {confidence_threshold:.2f}"
        )

    gap = top_conf - (runner_conf or 0.0)
    if gap < ambiguity_margin:
        review_reasons.append(
            f"confidence gap {gap:.3f} < ambiguity margin {ambiguity_margin:.2f} "
            f"({top_rule!r} vs {runner_rule!r})"
        )

    needs_review = bool(review_reasons)
    review_reason = "; ".join(review_reasons) if review_reasons else None
    rule_out = NEEDS_REVIEW if needs_review else top_rule

    return RulePrediction(
        rule_prediction=rule_out,
        confidence=round(top_conf, 6),
        runner_up_rule=runner_rule,
        runner_up_confidence=round(runner_conf, 6) if runner_conf is not None else None,
        needs_review=needs_review,
        review_reason=review_reason,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ── Batch helper ───────────────────────────────────────────────────────────────
def classify_rules_batch(
    report_texts: list[str],
    classifier,
    encoder,
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    batch_size: int = 64,
) -> list[RulePrediction]:
    """
    Batch version of :func:`classify_rule`.

    Encodes all texts in one pass (more efficient than calling
    ``classify_rule`` in a loop), then applies the contract logic to
    each row individually.

    Parameters
    ----------
    report_texts : list[str]
        List of safety report narratives.
    batch_size : int
        Embedding batch size passed to ``encoder.encode``.

    Returns
    -------
    list[RulePrediction]
        One ``RulePrediction`` per input text, in the same order.
    """
    if not report_texts:
        return []

    embs = encoder.encode(report_texts, batch_size=batch_size, convert_to_numpy=True)
    probas = classifier.predict_proba(embs)          # (n, n_classes)
    classes = list(classifier.classes_)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results: list[RulePrediction] = []
    for proba in probas:
        sorted_idx = np.argsort(proba)[::-1]
        top_idx = int(sorted_idx[0])
        top_rule = classes[top_idx]
        top_conf = float(proba[top_idx])

        if len(sorted_idx) >= 2:
            runner_idx = int(sorted_idx[1])
            runner_rule: Optional[str] = classes[runner_idx]
            runner_conf: Optional[float] = float(proba[runner_idx])
        else:
            runner_rule = None
            runner_conf = None

        review_reasons: list[str] = []
        if top_conf < confidence_threshold:
            review_reasons.append(
                f"top-class confidence {top_conf:.3f} < threshold {confidence_threshold:.2f}"
            )
        gap = top_conf - (runner_conf or 0.0)
        if gap < ambiguity_margin:
            review_reasons.append(
                f"confidence gap {gap:.3f} < ambiguity margin {ambiguity_margin:.2f} "
                f"({top_rule!r} vs {runner_rule!r})"
            )

        needs_review = bool(review_reasons)
        results.append(RulePrediction(
            rule_prediction=NEEDS_REVIEW if needs_review else top_rule,
            confidence=round(top_conf, 6),
            runner_up_rule=runner_rule,
            runner_up_confidence=round(runner_conf, 6) if runner_conf is not None else None,
            needs_review=needs_review,
            review_reason="; ".join(review_reasons) if review_reasons else None,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            timestamp=now_str,
        ))

    return results
