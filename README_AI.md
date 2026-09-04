# SIF-Guard — AI Component README (Phase 20)

This is the map of everything the AI/data side of this project actually
contains, what's verified, and what's still an open human decision. It is
written to be read by a person who was not in the room for any of the work
— an evaluator, a new team member, or Rishu six months from now.

## 1. What this system does, in one paragraph

Given a free-text safety incident narrative, the system estimates whether
it has Serious Injury or Fatality (SIF) *potential* (not the same as actual
harm — a near-miss with zero injuries can still be SIF-Yes), and if so,
which of the 9 IOGP Life-Saving Rules it most relates to. Both estimates
come with a confidence score, an explanation (top contributing phrases via
LIME), and — critically — the honest option to say "not confident enough,
route to a human" instead of forcing a guess. A person (Supervisor/HSE/
Admin) reviews and closes out every report; the AI never does.

## 2. Deliverables checklist (15 items)

| # | Deliverable | Status | Where |
|---|---|---|---|
| 1 | Dataset audit (real:synthetic ratio, PII, leakage risks, duplicates) | Done | `DATASET_AUDIT.md` |
| 2 | Verified external data source catalog with license/provenance | Done | `DATA_SOURCES.md`, `data/README.md` |
| 3 | Provenance-preserving unified schema across all sources | Done — 860,440 rows, 0 duplicate record_ids, verified | `data/processed/master_raw.csv`, `data/build_master_schema.py` / `data/rebuild_master_fast.py` |
| 4 | IOGP Life-Saving Rule taxonomy + legacy term mapping | Done — 1,917 terms mapped, 1,745 honestly Unmapped | `taxonomy/life_saving_rules.csv`, `taxonomy/legacy_term_mapping.csv` |
| 5 | Data-quality flags (duplicates, empty/malformed, possible PII) | Done — 15,510 dup groups, 66,017 empty/malformed, 26,354 possible-PII | `data/processed/safety_master.csv`, `data/build_cleaned.py` |
| 6 | Deterministic structured-field extraction (unsafe_act/condition/barrier/consequence) with quoted evidence spans | Done — completed during this session; conservative match rates (0.06%-23.7% depending on field) are expected from a strict keyword-only matcher, not a bug; 14,642 ambiguous matches logged for human review rather than guessed | `data/processed/safety_master_extracted.csv`, `data/extract_structured_fields.py`, `data/processed/extraction_ambiguous.csv` |
| 7 | Real:synthetic ratio & labeling-status decision, human-reviewed | Done | `PHASE9_DECISION.md` |
| 8 | Leakage-free, group-aware train/val/test split with a real-only, duplicate-free gold test set | Done — 927/199/199, verified zero overlap | `data/build_splits.py`, `data/splits/*.csv` |
| 9 | SIF classifier benchmark (multi-candidate, honest, on gold test set) | Done | `data/benchmark_models.py`, `data/benchmark_results.json/.md` |
| 10 | Threshold tradeoff analysis (diagnostic only, not for production tuning) | Done | `data/analyze_threshold_tradeoff.py`, `data/threshold_tradeoff.csv/.md` |
| 11 | Three-way (Yes/No/NEEDS_REVIEW) output contract — SIF classifier | Done, wired into `backend/model.py::predict()` and verified with mocked unit tests (see §4) | `backend/sif_output_contract.py` |
| 12 | Three-way output contract — Life-Saving Rule classifier | Done — contract wired in; underlying model swap (v1 vs v2) deliberately NOT made, see §5 | `backend/rule_output_contract.py`, `data/rule_classifier_eval.md` |
| 13 | Frontend correctness for the third state (NEEDS_REVIEW must never render as safe) | Done — `sifColor`, the result banner, the Life-Saving Rule badge, and the reports-table filter all fixed; TypeScript compiles clean | `frontend/src/App.tsx` |
| 14 | Explainability metadata persisted to the database (not just the API response) | Done — additive-only migration, original `report_text` untouched | `backend/database.py` (`insert_report`, `init_db` migrations) |
| 15 | Model registry, comparison doc, hybrid-architecture doc, this README | Done | `models/registry.json`, `models/README.md`, `MODEL_COMPARISON.md`, `PHASE14_HYBRID_ARCHITECTURE.md`, `PHASE15_PRECURSOR_DATA_PREP.md`, this file |

## 3. What changed in the backend/frontend this session, precisely

- `backend/sif_output_contract.py` — new file. Three-way SIF decision
  wrapper (`classify_sif`), placeholder thresholds (0.35/0.70) documented
  as pending human calibration, grounded in but not derived from the
  threshold-tradeoff diagnostic (which explicitly forbids being used to
  pick a production threshold).
- `backend/model.py` — `predict()` rewritten to call `classify_sif` /
  `classify_rule` instead of a raw `>= SIF_THRESHOLD` comparison. Every
  original return key (`sif_potential`, `confidence`, `risk_score`,
  `life_saving_rule`, `rule_confidence`, `top_contributing_phrases`) is
  still present with the same meaning — verified with a mocked-classifier
  test across three probability bands (No / NEEDS_REVIEW / Yes) that all
  six original keys survive every branch. New keys are additive
  (`sif_reasons`, `sif_model_name`, `sif_model_version`,
  `sif_decision_thresholds`, `sif_prediction_timestamp`,
  `rule_needs_review`, `rule_review_reason`, `rule_runner_up`,
  `rule_runner_up_confidence`, `rule_model_name`, `rule_model_version`,
  `rule_prediction_timestamp`).
- `backend/main.py` / `backend/database.py` — **inspected, not modified**
  for the critical-flagging and filtering logic, because every check there
  already uses exact `== "Yes"` comparisons (never `!= "No"`), so a
  `NEEDS_REVIEW` value was already safe by construction — confirmed by
  reading every `sif_potential` reference in both files. `database.py`
  *was* modified, additively, to persist the new explainability fields
  (see item 14 above).
- `frontend/src/App.tsx` — `sifColor()` now returns amber for
  `NEEDS_REVIEW` instead of falling through to the green/safe branch; the
  result banner's border color, badge text, and confidence-text color all
  route through `sifColor`/`sifLabel` instead of a hardcoded `=== "Yes"`
  ternary; the Life-Saving Rule badge gets the same amber treatment when
  its value is `NEEDS_REVIEW`; the reports filter dropdown now offers
  `NEEDS_REVIEW` as a filter option, not just Yes/No. `npx tsc --noEmit`
  passes clean after every edit.

## 4. How this was verified (given this session couldn't run the live app)

This session's device shell doesn't have the project's Python dependencies
installed (`joblib`, `sentence-transformers`, etc. aren't present there —
they live in whatever environment the user actually runs the FastAPI
backend from), so a true end-to-end request/response test wasn't possible
from here. What WAS done:
- `python3 -m py_compile` on every touched `.py` file (syntax-valid).
- A mocked-classifier unit test exercising `classify_sif` /
  `classify_rule` directly with fake `predict_proba`/`encode` objects,
  confirming the exact threshold boundaries (0.35/0.70) produce the
  correct label, and that the reason strings render correctly.
- A second mocked test that reproduces `model.py::predict()`'s actual
  merge logic line-for-line against fakes, confirming all 6 original
  dict keys survive all 3 branches (No / NEEDS_REVIEW / Yes).
- `npx tsc --noEmit -p .` on the frontend after every `App.tsx` edit —
  clean, no type errors.
- Direct `pandas` verification of `data/processed/safety_master_extracted.csv`
  (row count, per-source counts, and fill rates) rather than trusting the
  extraction script's own printed summary.

**What was not possible to verify from here:** an actual HTTP request
through a running `uvicorn` process, and therefore the exact JSON the
frontend receives at runtime. Recommend running the backend locally and
submitting one report per SIF probability band (low/mid/high) as a final
smoke test before considering Phase 12/13 fully closed.

## 5. Open decisions for a human (nothing here was decided unilaterally, per rule 10)

1. **SIF/rule NEEDS_REVIEW thresholds** (0.35/0.70 for SIF; 0.50 confidence
   / 0.15 ambiguity margin for rule) are placeholders. See
   `backend/sif_output_contract.py` and `backend/rule_output_contract.py`
   docstrings for the evidence available and why it doesn't support a
   precise number yet.
2. **`rule_classifier.joblib` (v1, in production) vs `rule_classifier_v2.joblib`
   (Phase 13, evaluated but not promoted)** — no like-for-like benchmark
   exists between them. See `models/registry.json`.
3. **`build_recurrence_table` uses only the 900 synthetic seed rows**,
   excluding the 425 real seed rows, for 25% of every report's risk score.
   Found this session, not fixed (changes existing risk scores). See
   `PHASE15_PRECURSOR_DATA_PREP.md`.
4. **Whether to ever weakly-label the 859,115-row real corpus** for SIF
   potential or Life-Saving Rule — explicitly declined for now in
   `PHASE9_DECISION.md`; revisit only with a human-reviewed labeling
   effort, not a heuristic.
5. **Physical model-artifact reorganization** into
   `models/{baseline_minilm,safetybert,best_model}/` — documented as the
   right end state in `models/README.md`, deliberately not done this
   session because it requires a matching `backend/model.py` path change
   that couldn't be smoke-tested from here (see §4).

## 6. Standing rules this document (and everything above) was written under

1. Never fabricate data. 2. Never fabricate dataset URLs. 3. Never
fabricate labels. 4. Never treat all accidents as SIF. 5. Never treat
actual injury severity as equivalent to SIF potential. 6. Never use
synthetic data as the only evidence of model performance. 7. Never put
duplicate/near-duplicate incidents across train and test. 8. Never
overwrite the original raw datasets. 9. Never silently change existing
application behavior. 10. Never let the AI become the final safety
authority. 11. Preserve original report text. 12. Track source and
provenance for every record. 13. Prefer official/publicly credible safety
sources. 14. Check dataset licenses before using external data. 15.
Benchmark before replacing the current model.
