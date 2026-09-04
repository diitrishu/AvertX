# Model Comparison (Phase 18)

Winner is selected using real, held-out test performance only (rule 15:
benchmark before replacing). No model is treated as a winner by default;
"keep current" is a valid outcome and, for the SIF classifier, is the
outcome here.

## SIF binary classifier

| Candidate | HF ID | License | Val threshold (precision-floor 0.5) | Test precision | Test recall | Test F1 | Test PR-AUC | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **MiniLM (current production)** | sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 | 0.13 | 0.352 | 1.0 | 0.5208 | 0.494 | 15.75 |
| BGE-Small | BAAI/bge-small-en-v1.5 | MIT | 0.13 | 0.347 | 1.0 | 0.5149 | 0.477 | 28.13 |
| SafetyBERT | — | — | **skipped** — no qualifying checkpoint found (see benchmark_results.json's skip_reason; both candidate HF repos checked and neither meets the bar) | | | | | |

Test-set base rate (always-predict-Yes precision) is 0.347 -- the positive
class prevalence, which is the correct random baseline for PR-AUC here, not
0.5.

**Decision: keep MiniLM in production. No swap.**

Rationale: BGE-Small does not beat MiniLM on any test metric that matters
(marginally lower precision, identical recall, lower F1, lower PR-AUC,
*and* roughly 2x the inference latency). Neither model clears a bar that
would justify calling either one a reliable binary classifier on real
narratives — both land within a few points of the base rate at the
recall=1.0 operating point, and precision only exceeds 0.50 in a narrow
threshold window on either model (see data/threshold_tradeoff_summary.md).
That weakness is real and is not solved by picking between these two models
— it's a training-data-size problem (1,325 labeled rows), not a
model-choice problem. This is exactly why Phase 12 built a three-way output
contract (Yes/No/NEEDS_REVIEW) instead of forcing a binary call: the honest
answer, most of the time, is "not confident enough to say."

## Life-Saving Rule multi-class classifier

No formal multi-candidate benchmark exists here (unlike the SIF classifier,
only one embedding model — MiniLM, matching the SIF classifier by design,
per train_rule_classifier.py's own docstring — has ever been tried for this
task). What exists instead is two *versions* of the same
MiniLM+LogisticRegression approach:

| Artifact | Status | Test accuracy | NEEDS_REVIEW rate on test | Test set size / caveat |
|---|---|---|---|---|
| `rule_classifier.joblib` (v1, currently in production) | in_production | not measured — no eval report exists for this specific file | not measured | — |
| `rule_classifier_v2.joblib` (Phase 13) | trained, evaluated, **not promoted** | 0.5417 | 91.67% | only 24 test rows; 4 of 9 rule classes have zero test examples — see data/rule_classifier_eval.md's explicit warning not to treat their per-class numbers as meaningful |

**Decision: no swap, pending a human call.** This is the one place this
document does NOT declare a winner, because rule 15's "benchmark before
replacing" can't be satisfied here — there is no apples-to-apples
comparison between v1 and v2 on the same held-out rows, because v1 was
never independently evaluated against `data/splits/test.csv` in the first
place. Declaring v2 the winner just because it has a report (however weak)
and v1 doesn't would be evaluating on vibes, not evidence. See
`models/registry.json`'s `rule_classifier_v2` entry for the concrete next
step a human should take (run v1 through the same eval harness that
produced `data/rule_classifier_eval.md`, then compare like-for-like).

## What would change this conclusion

More labeled real data for both classifiers. The SIF classifier's precision
ceiling and the rule classifier's near-total NEEDS_REVIEW rate are both
downstream of the same root cause identified in PHASE9_DECISION.md: 1,325
labeled rows is not enough for a 2-class problem, let alone a 9-class one,
and the 859,115-row real corpus stays an unlabeled reserve until a human
signs off on a labeling effort. No architecture change or model swap fixes
that; only more human-reviewed labels do.
