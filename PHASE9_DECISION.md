# Phase 9 Decision Record — Real:Synthetic Ratio & Labeling Status

## What changed since DATASET_AUDIT.md

The original audit (Phase 1) found the seed dataset at 32% real / 68% synthetic,
against a 70-90% real target. Phase 3-4 ingestion of external sources (OSHA, NIOSH,
PHMSA, IHM Stefanini) brought the total corpus to 860,440 rows, 99.9% of them real.
On its face this looks like the ratio problem is solved. It isn't — see below.

## The actual constraint: labels, not row count

`sif_potential` is `UNKNOWN` for every row outside `sif_guard_seed`, by design —
Phase 4's mapping script deliberately never inferred it (rule 3: never fabricate
labels). So the corpus is **1,325 labeled rows (382 Yes / 943 No) plus 859,115
unlabeled real narratives**. Adding real rows didn't add real training signal for
the SIF classifier; it added a large unlabeled pool.

## Severity-signal survey (Chunk 3, this session)

Checked whether any non-seed source has an existing field that could anchor a
label:

| Source | Severity field found | Usable for a label? |
|---|---|---|
| severe_injury_reports_jboysen | Hospitalized, Amputation (integer counts) | Yes-only, weak (see below) |
| hse_abstracts_2015_2017_ruqaiyaship | Fatal / Nonfatal (61.2% / 38.8%) | Yes-only, weak (already in schema) |
| phmsa_hazmat_incidents | Fatality/injury indicators (0.06% / 0.9% positive) | Yes-only, weak; not yet in master schema |
| niosh_injury_narrative_coding | none (narrative + OIICS event code only) | No — narrative-only |
| ihm_stefanini_raw | Accident Level / Potential Accident Level (already used in seed mapping) | Already exploited — this is the source of the seed's 425 real rows |

The severity fields that exist are all **one-directional**: an actual
hospitalization/amputation/fatality is real evidence of high SIF potential, but
their absence is not evidence of low potential (a real near-miss can have zero
actual harm and still be genuinely SIF-Yes — that's the entire premise of SIF
logic, and exactly what rule 5 protects against getting collapsed). So none of
these fields can ever support a "No" label, only a conservative "Yes."

Also resolved: the "425 vs 426" IHM Stefanini row-count question from
`DATA_SOURCES.md` is not a data-loss issue — `OIL-00426` in the seed dataset is
simply where the synthetic block begins in that file's indexing, not a dropped row.

## Decision (human-reviewed, per rule 10)

**Skip weak/derived labeling for now.** Rather than auto-generating
`sif_potential_weak = Yes` values from the severity fields above — even
conservatively, even clearly tagged as derived — the decision is to leave every
non-seed row's `sif_potential` as `UNKNOWN` and treat the 859,115-row real corpus
as an **unlabeled pool**, not a supervised training addition, until a real
human-reviewed labeling pass happens.

Rationale: a heuristic label, even a defensible one, is still an unaudited
judgment call flowing into a safety-relevant model. Rule 10 (never let the AI
become the final safety authority) argues for holding this until a person
explicitly signs off on turning it on — which didn't happen here, so it stays off.

## What this means going forward

- **The supervised SIF classifier (Phase 11+) trains and evaluates on the
  1,325-row seed dataset only**, for now — real:synthetic ratio there is still
  32:68, still short of the 70-90% real target, and that gap is NOT closed by
  this session's external sourcing work, despite the corpus now being 99.9% real
  by row count.
- **The 859,115-row real corpus is retained as a reserve asset**, useful for
  things that don't need `sif_potential` labels: domain-adapted embedding
  pretraining, vocabulary/terminology analysis, or as a candidate pool for a
  future proper human labeling effort — not fed into Phase 10's train/test split
  as labeled data.
- Phase 10 (splitting) should be scoped to the seed dataset's 1,325 rows unless
  and until more labeled data exists. Splitting the unlabeled corpus isn't
  meaningful for a supervised classifier's train/test split.
- If a future labeling effort does happen (crowdsourced review, in-house HSE
  review of a sample, etc.), this document's severity-signal survey is the
  starting point for which sources are worth prioritizing (severe_injury and
  hse_abstracts have the cleanest existing anchors).
