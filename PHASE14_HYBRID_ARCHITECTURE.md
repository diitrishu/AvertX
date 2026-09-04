# Phase 14 — Hybrid (Non-Black-Box) Pipeline Architecture

## Why this document exists

Rule 10 says the AI is never the final safety authority, and rule 9 says
behavior never changes silently. A pure end-to-end neural classifier — text
in, "Yes/No" out, no visible reasoning — cannot satisfy either constraint on
its own: there is nothing for a reviewer to check, and nothing to show at
each step. This document records the pipeline actually built across Phases
4-13 and states plainly which parts are deterministic/auditable and which
are statistical, so a reviewer of this codebase (or an HSE auditor) can see
where a human judgment call is required.

## The pipeline, stage by stage

```
raw source file (OSHA / HSE / NIOSH / PHMSA / IHM / seed)
        |
        v
[1] build_master_schema.py / rebuild_master_fast.py   -- DETERMINISTIC
    per-source mapping functions, one per data source. Pure column
    renaming/reformatting -- no inference, no model. sif_potential and
    life_saving_rule are left UNKNOWN for every source except the
    human-labeled seed set (rule 3: never fabricate labels).
        |
        v
[2] build_cleaned.py                                   -- DETERMINISTIC
    adds is_duplicate_group_id, is_empty_or_malformed, has_possible_pii
    via regex/text-normalization. No model involved.
        |
        v
[3] extract_structured_fields.py (Phase 7)              -- DETERMINISTIC
    fills unsafe_act / unsafe_condition / barrier / potential_consequence
    via keyword/phrase matching ONLY (a fixed, auditable vocabulary from
    taxonomy/life_saving_rules.csv). No LLM, no embedding similarity.
    Every fill records the exact matched substring and its character
    offsets (unsafe_act_evidence_span/_start/_end, etc.) so a reviewer can
    see precisely what triggered the value -- this is the "show your work"
    layer for the fields that don't need statistical inference at all.
    Ambiguous matches (multiple categories hit) are left UNKNOWN and logged
    to extraction_ambiguous.csv rather than guessed.
        |
        v
[4] build_splits.py (Phase 10)                          -- DETERMINISTIC
    group-aware train/validation/test split, scoped to the 1,325-row
    labeled seed set only (see PHASE9_DECISION.md for why the 859K-row
    real corpus stays unlabeled/reserved rather than weakly labeled).
        |
        v
[5] embedding model (MiniLM) + LogisticRegression        -- STATISTICAL
    sif_classifier.joblib / rule_classifier.joblib, benchmarked in Phase 11
    against an honest, duplicate-free, real-only gold test set
    (benchmark_results.json/.md). This is the one genuinely "black-box"
    component -- a sentence embedding is not human-interpretable on its own.
        |
        v
[6] LIME explainer (backend/model.py::explain_prediction) -- STATISTICAL,
    but EXPLAINABLE. Perturbs the input text and fits a local linear model
    to show which words pushed the SIF probability up or down. This is what
    keeps stage [5] from being a pure black box in practice: every
    prediction ships with its top contributing phrases.
        |
        v
[7] sif_output_contract.py / rule_output_contract.py (Phase 12/13)
                                                          -- DETERMINISTIC
    wraps the raw model probability in a fixed decision rule (two
    thresholds, three outcomes: Yes / No / NEEDS_REVIEW). This is the
    layer that turns "the model is 52% confident" into a labeled, auditable
    decision with a machine-readable reason string -- and it is where
    "never let the AI become the final authority" is enforced structurally:
    NEEDS_REVIEW is not a UI suggestion, it's a value the frontend is
    required to render distinctly (Phase 12 fixed three places in App.tsx
    where it previously would have silently rendered as "safe").
        |
        v
[8] backend/main.py::_is_critical + reviewer workflow     -- DETERMINISTIC
    routing. A confident "Yes" above the risk threshold auto-flags for
    immediate review (critical=true, audit-logged). Every report --
    Yes, No, or NEEDS_REVIEW -- is saved and enters the same
    Submitted -> ... -> Closed lifecycle a human (Supervisor/HSE) drives via
    /reports/{id}/status. The model never closes a report itself.
```

## What "hybrid" means here, concretely

Two different kinds of components do two different jobs, and the codebase
keeps them visibly separate rather than blending them into one score:

- **Deterministic / rule-based** (stages 1-4, 7-8): auditable by reading the
  code, no training data dependency, produces the same output for the same
  input every time. This is where rules 1-3, 8, 11, 12 live structurally
  (provenance columns, UNKNOWN-by-default, evidence spans).
- **Statistical / learned** (stages 5-6): necessary because narrative text
  doesn't reduce to keyword rules alone (that's exactly what Phase 7's
  ambiguous-match log demonstrates — plenty of real narratives don't fit a
  fixed vocabulary). This is where rule 6 (never use synthetic data as the
  only evidence) and rule 15 (benchmark before replacing) apply, and where
  Phase 11's honest benchmark matters most, because it's the one place error
  can hide.

The explicit design choice is: never let stage 5-6 output reach the user or
the database without passing through stage 7's contract layer first. There
is no code path in backend/model.py::predict() that returns a raw
classifier label — every call goes through classify_sif / classify_rule.

## Known limitation, stated plainly

Stage 5's benchmark (Phase 11) found weak discrimination on real held-out
data (best test precision ~0.35-0.51 depending on threshold, base rate
0.347 — see data/threshold_tradeoff_summary.md). The hybrid design does not
fix this; it contains the consequence of it, by making "the model isn't
sure" a first-class, visible output instead of a forced guess. Improving
stage 5 itself needs more labeled real data (see PHASE9_DECISION.md and
Phase 15 below), not a different architecture.
