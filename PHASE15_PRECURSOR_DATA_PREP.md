# Phase 15 — Data Readiness for Precursor / Recurring-Pattern Analysis

## Framing, deliberately

This document is about surfacing **recurring precursor patterns** (which
site/activity/rule combinations recur, and how often) so a human reviewer
can prioritize attention. It is explicitly NOT about predicting fatalities,
predicting who will be hurt, or forecasting individual future events — none
of that is supported by this data or attempted here. "Recurrence" is a
historical count, not a forecast.

## What already exists in the codebase

`backend/model.py::build_recurrence_table` / `compute_risk_score` and
`backend/database.py::get_dashboard_stats`'s `top_precursors` query already
implement a form of this: they count how often a given
(site, activity, life_saving_rule) combination has occurred, and use that
count as 25% of the `risk_score` shown per-report, and as a ranked list on
the dashboard. So the *mechanism* for precursor surfacing already exists.

## A finding that needs a human decision (not fixed here, per rule 10)

`build_recurrence_table` filters its input to `source == "synthetic"` only:

```python
syn = df[df["source"] == "synthetic"]
for _, row in syn.iterrows():
    key = (row["site"], row["activity"], row["life_saving_rule"])
    recurrence_counts[key] = recurrence_counts.get(key, 0) + 1
```

`oil_safety_reports_merged.csv` (the file this reads) is the 1,325-row seed
set: 900 synthetic rows + 425 real `ihm_stefanini` rows. This filter means
the recurrence table — and therefore 25% of every report's `risk_score` —
is built **only from the 900 synthetic rows**, excluding all 425 real ones.
This is a real, pre-existing behavior (not introduced by this session's
work), confirmed by directly inspecting `oil_safety_reports_merged.csv`'s
`source` column. It matters because rule 6 (never use synthetic data as the
only evidence) is squarely about exactly this pattern, even though it's
recurrence-weighting rather than classifier training.

This is flagged here rather than silently changed because switching the
filter to include real rows (or to use the 859,115-row real corpus once
site/activity/rule fields are populated for it — see below) changes the
`risk_score` value shown for existing reports, which is a visible behavior
change a human should sign off on, not something an AI should decide
unilaterally under rule 10.

## What's needed to extend recurrence beyond the 1,325-row seed set

The 859,115-row real, unlabeled corpus (`data/processed/safety_master.csv`,
Phase 4/8) already carries `site`, `activity`, and (once Phase 7's
extraction finishes — see status note below) `unsafe_act`/`unsafe_condition`
fields for most rows. What it does NOT reliably carry for most sources is
`life_saving_rule` — that's UNKNOWN outside the seed set by the same Phase
9 decision that scoped the SIF classifier to the seed set (see
PHASE9_DECISION.md). So a precursor table keyed on
(site, activity, life_saving_rule) can't be built from the full corpus
today without either:

1. A human-reviewed rule-labeling pass on a sample of the real corpus
   (same caution as Phase 9's SIF labeling decision — this is a labeling
   decision, not a code change), or
2. Keying recurrence on fields that ARE populated for the full corpus
   instead — `site` + `activity` + `hazard` (from Phase 4's mapping) is a
   weaker but immediately available substitute, worth a human's
   consideration as an interim option.

## Status of Phase 7 (structured field extraction) — complete

`data/extract_structured_fields.py` finished during this session. Verified
directly against `data/processed/safety_master_extracted.csv` (not just
trusted from the run's own printed summary): 860,440 rows total, matching
`safety_master.csv` exactly; non-UNKNOWN field counts cross-checked two
ways (script's own matched/skipped/ambiguous/unknown tally vs. an
independent pandas re-count of the output file) and they agree exactly,
including the seed-vs-non-seed split for `barrier` (944 pre-filled seed
rows correctly left untouched, 0 wrongly overwritten; 5,001 new matches
elsewhere). Match rates are intentionally low for a strict keyword-only
matcher (0.06% unsafe_act, 1.32% unsafe_condition, 0.58% barrier, 23.65%
potential_consequence) — that reflects the deterministic design (rule 3:
never guess, leave UNKNOWN or log to `extraction_ambiguous.csv` — 14,642
rows — when the narrative doesn't literally match the vocabulary), not a
bug. `unsafe_act` / `unsafe_condition` / `barrier` / `potential_consequence`
are now available for the full corpus (see PHASE14_HYBRID_ARCHITECTURE.md,
stage 3) — useful context fields for a future precursor analysis, but as
noted above, not a substitute for `life_saving_rule`, which Phase 7
explicitly does not touch.

## Recommendation

Do not build a new precursor-detection feature in this session. The
mechanism already exists (`top_precursors`); what's missing is (a) a human
decision on the synthetic-only recurrence-table finding above, and (b) a
human-reviewed labeling pass if precursor analysis should ever cover the
real corpus rather than just the 1,325-row seed set. Both are labeling /
scope decisions for a person, not an AI, per rule 10.
