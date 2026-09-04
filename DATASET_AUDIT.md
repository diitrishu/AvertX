# DATASET_AUDIT.md — SIF-Guard Seed Dataset Audit

**Scope:** `oil_safety_reports_merged.csv` (the current training/seed dataset). This is a
**read-only audit** — the file itself was not modified, sorted, deduplicated, or
re-labeled in producing this report. All numbers below come from direct inspection of
the file as it exists today.

**Purpose:** establish ground truth on what this dataset actually contains — its
composition, its label quality, its leakage risks, and its gaps — before any new data
is sourced, any schema is redesigned, or any model is retrained. Every later phase
(external sourcing, taxonomy mapping, train/test splitting, model benchmarking) depends
on the findings here being accurate.

---

## 1. Shape and structure

- **1,325 rows × 13 columns**
- Columns: `report_id, date, site, activity, report_type, report_text, sif_potential,
  life_saving_rule, hazard, barrier, barrier_status, risk_level, source`
- All columns are stored as `object` (string) dtype — no numeric/boolean typing yet in
  the raw file (relevant when designing the normalized schema in Phase 4).
- `report_id` is 100% unique, 100% present, consistent format (`OIL-NNNNN`, zero-padded
  5 digits, sequential 00001–01325).
- **No fully-duplicated rows.** No duplicate `report_id` values.

## 2. Source composition

| Source | Rows | % of dataset |
|---|---|---|
| `synthetic` | 900 | 67.9% |
| `real_ihm_stefanini` | 425 | 32.1% |

This is **below** the 70–90% real-data target stated in the project's own rules
(Phase 9). The current dataset is roughly **32% real / 68% synthetic** — the inverse of
the target ratio. This is the single most consequential finding of this audit and
should directly drive the external-data-sourcing priority in Phase 2/3: the project
needs substantially more real incident narratives (ideally oil & gas domain) before the
synthetic share can be brought down to an acceptable level, or the "real" evaluation set
needs to be built from a source large enough to stand alone regardless of what the
training mix looks like.

`report_id` numbering confirms provenance is at least positionally traceable: real rows
occupy `OIL-00001`–`OIL-00425`, synthetic rows occupy `OIL-00426`–`OIL-01325`. There is
no separate `source_record_id` column tying a real row back to its original row index
in the source Kaggle file — see §7 (Provenance gaps).

## 3. Missing values

| Column | Missing (NaN) | Notes |
|---|---|---|
| `life_saving_rule` | 381 | 100% within the synthetic subset (see §5) |
| `barrier` | 381 | same 381 synthetic rows |
| `barrier_status` | 381 | same 381 synthetic rows |
| all other columns | 0 | no blank-but-non-null strings found anywhere |

No column has silent/placeholder missingness disguised as a valid-looking string (e.g.
no stray `"N/A"`, `"-"`, or empty-string values masquerading as data) outside of the
already-documented `"Unmapped"` / `"Unknown (not recorded in source)"` literals the
merge script itself introduced for the real subset.

## 4. Duplicate and near-duplicate narratives

### 4.1 Exact duplicates (whole dataset, case-sensitive exact text match)

- **329 duplicate `report_text` values** (996 unique / 1,325 total rows)
- **166 duplicate-text groups, covering 495 rows total**
- Duplicate-count distribution: 2×(84 groups), 3×(42), 4×(22), 5×(8), 6×(3), 7×(4),
  8×(1), 9×(1), 10×(1)
- **Zero cross-source duplicates** — no real narrative and synthetic narrative ever
  share identical text.
- **Zero label inconsistency** — within every one of the 166 duplicate-text groups,
  `sif_potential` and `life_saving_rule` are 100% consistent. Duplication is not
  masking a labeling bug.
- The duplicate rows are near-entirely **templated synthetic housekeeping/observation
  reports** that vary only site name and date — a known and expected artifact of a
  template-based synthetic generator, not a data-quality defect in the labels
  themselves.

**Implication for Phase 10 (splitting):** these 495 duplicated synthetic rows must be
split as *groups*, not as independent rows — if two copies of the same templated
narrative land in both train and test, that is direct leakage even though the exact-dup
check shows no label inconsistency. The split logic must dedupe-then-group, or drop
duplicates entirely before splitting, not just split on row index.

### 4.2 Near-duplicates (case/punctuation-normalized), real subset

Restricting to the 425 real rows and normalizing text (lowercased, punctuation
stripped, whitespace collapsed) to catch near-identical text that exact matching
would miss:

- **12 near-duplicate groups, 26 rows** (411 unique normalized texts out of 425)
- **Zero label inconsistency** in any of these 12 groups
- Spot-checking the actual rows (e.g. `OIL-00167`/`OIL-00168`, `OIL-00262`–`OIL-00264`,
  `OIL-00303`/`OIL-00304`) shows these are **genuine duplicate rows already present in
  the original Kaggle source file** (the same incident narrative appears 2–3 times
  verbatim, sometimes 3× as with the "chuteo de ore in hopper op5" incident) — this is a
  known characteristic of the IHM Stefanini Kaggle dataset, not something introduced by
  the merge script.

**Implication:** these are legitimate duplicate *incidents* in the real data (not a
scraping/merge bug), but they carry the same leakage risk as the synthetic duplicates —
if `OIL-00262` and `OIL-00263` (identical text) land on opposite sides of a train/test
split, the model has effectively seen its test answer during training. The gold test
set (Phase 10) must dedupe the real subset by normalized text before sampling, and the
12 identified groups should collapse to one representative row each for the held-out
set.

I did not run a fuzzy/edit-distance similarity pass (e.g. Levenshtein or embedding
cosine similarity) beyond exact-normalized matching — that is a reasonable follow-up
once the embedding pipeline is stood up in Phase 11, but normalized-exact matching
already surfaces the clear cases and is cheap enough to re-run at every future
dataset revision.

## 5. `life_saving_rule` gap — real subset

The 381 missing `life_saving_rule` values are entirely synthetic; the real subset has
zero NaN there — but not zero *usable* labels either:

| Source | `life_saving_rule` value | Rows |
|---|---|---|
| real | `Unmapped` | 361 |
| real | Line of Fire | 24 |
| real | Energy Isolation | 19 |
| real | Driving | 9 |
| real | Working at Height | 9 |
| real | Bypassing Safety Controls | 2 |
| real | Confined Space | 1 |
| synthetic | `NaN` | 381 |
| synthetic | (8 other rules) | 519 (evenly spread, 37–96 each) |

Cross-tabbing against `sif_potential` confirms this is a clean pattern, not a bug:

- The 381 synthetic `NaN` rows are **100% `sif_potential = No`** — the generator simply
  didn't assign a rule to non-SIF synthetic rows, which is defensible but means the
  rule classifier currently has **zero synthetic No-SIF training signal with an
  explicit rule label** (not a problem for the rule classifier, which should only ever
  be trained/evaluated on SIF-Yes rows anyway, but worth stating plainly).
- The 361 real `Unmapped` rows split **139 Yes / 222 No** — this is the real gap: **139
  real, SIF-Yes reports have no usable Life-Saving Rule label** because their original
  `Critical Risk` category (from the Kaggle source) had no clean 1:1 IOGP mapping. These
  139 rows are usable for the binary SIF classifier but must be excluded from
  Life-Saving-Rule classifier training and evaluation — training on `Unmapped` as if it
  were a real 10th class would corrupt the taxonomy.
- The 36 real non-`Unmapped`/Yes rows and 28 real non-`Unmapped`/No rows are the only
  real data currently carrying a trustworthy, mapped Life-Saving Rule label — a small
  number or a rule classifier that must lean heavily on synthetic data. This is the
  second-most important sourcing priority after the real:synthetic ratio (§2) and should
  be flagged explicitly in `DATA_SOURCES.md`.

## 6. Narrative length — real vs. synthetic (shortcut-learning risk)

| Source | Mean narrative length | Range |
|---|---|---|
| real | 65.7 words | 16–183 |
| synthetic | 18.5 words | 10–34 |
| **overall** | 33.6 words | — |

A ~3.5× gap. This is flagged as a **shortcut-learning risk**: any model trained on this
mix has a trivial, semantically-empty signal available to it — narrative length (or
correlated stylistic features like sentence count, punctuation density, vocabulary
diversity) — that separates real from synthetic almost perfectly on its own. Because
`sif_potential` rates also differ by source (real 41.2% Yes vs. synthetic 23.0% Yes,
§8), a model that partially learns "longer text → real → higher prior on Yes" would
show inflated validation performance that doesn't reflect genuine semantic
understanding of SIF risk language, and would likely under-perform on new real-world
text that happens to be short (a one-line near-miss report, for instance — exactly the
kind of report SIF-Guard most needs to catch).

**Mitigation to carry into Phase 9–11:** the gold test set should be **real-only** (already
planned), which removes the length confound from evaluation entirely. During training,
this is worth monitoring via a length-stratified error analysis once a baseline model
exists (Phase 11) rather than trying to "fix" it by truncating/padding text, which would
be a fabrication of evidence, not a fix.

## 7. Provenance gaps

Per project rule 12 ("track source and provenance for every record"):

- `source` (`real_ihm_stefanini` / `synthetic`) is present and reliable for every row.
- There is **no `source_record_id`** column linking a real row back to its original row
  index/identifier in the source Kaggle CSV. `report_id` is an internally-assigned
  sequential ID (`OIL-00001`...), not a provenance key. If the original Kaggle file (or
  a specific row within it) is ever needed again — to re-verify a label, respond to a
  license question, or re-run the mapping with corrected logic — there is currently no
  reliable way to trace an `OIL-XXXXX` row back to it except positional order (rows
  0–424 of the merged file are, by construction, rows 0–424 of the real source in the
  same order — this should be verified against `merge_real_data.py` before being relied
  upon, and is fragile to any future re-run of that script with different row
  ordering).
- **Recommendation for Phase 4 (normalized schema):** add an explicit
  `source_record_id` field now, populated from the original Kaggle row index, so this
  fragile positional link doesn't have to be preserved by convention.

## 8. Label distribution and consistency checks

**Overall `sif_potential`:** 71.2% No / 28.8% Yes (943 No / 382 Yes).

By source: real 41.2% Yes / 58.8% No; synthetic 23.0% Yes / 77.0% No (matches the
project's stated SIF-density reference figure for the synthetic generator).

**`risk_level` vs. `sif_potential`** — this is a notable internal-consistency finding:

| `risk_level` | `sif_potential = No` | `sif_potential = Yes` |
|---|---|---|
| LOW | 837 | 0 |
| MEDIUM | 106 | 47 |
| HIGH | 0 | 335 |

`risk_level = HIGH` is a **perfect predictor** of `sif_potential = Yes` (335/335), and
`risk_level = LOW` is a **perfect predictor** of `sif_potential = No` (837/837); only
the MEDIUM band (153 rows) has any actual mixing. This means `risk_level` is not an
independent signal — it was very likely derived from (or derived in lockstep with)
`sif_potential` at data-generation/mapping time, not assigned independently from the
narrative text.

**This is a direct leakage risk if `risk_level` is ever used as a model input feature.**
It is not currently listed as a classifier input in the architecture description, and it
must stay that way — `risk_level` should be treated as a downstream/derived field (fine
to display in the UI, fine to use for sorting the reports inbox) and explicitly excluded
from any feature set feeding the SIF or Life-Saving-Rule classifiers. This should be
written explicitly into the Phase 4 schema design and the Phase 12/13 classifier
contracts as a "do not use as a feature" field, alongside `sif_potential` and
`life_saving_rule` themselves (which are of course the labels, not features).

`report_type` vs. `sif_potential` shows normal, non-degenerate mixing in all four
categories (Incident 38.3% Yes, Near Miss 32.8% Yes, UA 22.5% Yes, UC 19.7% Yes) — no
leakage concern there.

## 9. `site` / `activity` / `date` sanity checks

- **17 unique sites**, cleanly source-segregated: 5 named oil & gas sites
  (`Site A–E`) appear only in synthetic rows; 12 `Real-Local_NN` placeholder sites
  appear only in real rows. No cross-contamination, no site typos/near-duplicate
  spellings found.
- **16 unique activities.** Real-subset activities (`Mining`, `Metals`,
  `Maintenance`, `Production Operations`, etc.) and synthetic-subset activities
  (`Confined Space Entry`, `Hot Work`, `Working at Height`, `Lifting Operations`,
  etc.) are visibly drawn from two different vocabularies reflecting the two source
  domains (mining/metals vs. oil & gas) — expected, consistent with the known
  domain-mismatch already documented in the README, not a data error.
- **Date range: 2016-01-01 to 2026-08-31.** Distribution is bimodal by construction:
  2016 (285 rows) and 2017 (140 rows) are the real Kaggle data's actual incident
  dates; 2025 (532 rows) and 2026 (368 rows) are synthetic-generator dates (including
  dates into "the future" relative to when the original data was collected, though
  not relative to today). This is cosmetic — dates aren't used as a model feature —
  but worth knowing before anyone builds a time-based visualization that would
  otherwise show a confusing 8-year gap.
- `hazard` has 60 unique values including a large `"None identified"` bucket (693
  rows, 52.3%) and `"Others"` (232 rows) — together accounting for ~70% of rows. The
  remaining ~30% is a long tail of specific hazard types (Pressed, Manual Tools,
  Chemical substances, Venomous Animals, etc.). This is a plausible real-world
  distribution (most UA/UC/housekeeping reports genuinely don't have a specific
  hazard to name) but the long tail is thin per-category (many hazards have single
  digits of rows) — not enough per-category volume to train a dedicated hazard
  classifier today; worth revisiting once more real data is sourced.
- `barrier` / `barrier_status`: beyond the 381 synthetic NaNs already covered in §3,
  the 425 real rows are uniformly `"Unknown (not recorded in source)"` /
  `"Unknown"` — confirming the README's statement that barrier data isn't in the
  original Kaggle source and wasn't fabricated to fill the gap. Good practice,
  consistent with rule 1.

## 10. PII scan

- A regex sweep of `report_text` for email addresses and phone-number patterns
  returned **zero matches** across all 1,325 rows.
- However, manual inspection of the near-duplicate real rows (§4.2) shows **real
  full names embedded in narrative text** — e.g. "Felipe and Divino de Morais",
  "Rafael and Danillo da Silva", "Márcio and Sérgio", "Robson and Manoel da Silva".
  This is expected for a real accident-investigation dataset (the original Kaggle
  narratives were written by real safety investigators about real workers) and is
  already present in the public source data, not introduced by this project — but it
  should be flagged now, before any of this data is published, screenshotted, or
  shared outside the immediate project team. If `DATASET_AUDIT` findings or example
  rows are ever shown externally, redact worker names first.

## 11. Summary of findings requiring action in later phases

1. **Real:synthetic ratio is 32:68, inverted from the 70–90% real target** — the
   single biggest gap; drives the urgency of Phase 2/3 external sourcing. *(→ Phase 2/3)*
2. **139 real, SIF-Yes rows have no usable Life-Saving Rule label** (`Unmapped`) —
   rule classifier is short on real positive examples. *(→ Phase 2/3, Phase 13)*
3. **495 rows (166 groups) of exact-duplicate text + 26 rows (12 groups) of real
   near-duplicate text** must be deduplicated at the group level before any
   train/test split, not just row-deduplicated. *(→ Phase 10)*
4. **Real vs. synthetic narrative length gap (65.7 vs 18.5 words)** is a
   shortcut-learning risk; mitigated by a real-only gold test set, monitored via
   length-stratified error analysis post-baseline. *(→ Phase 10, Phase 11)*
5. **No `source_record_id` for real rows** — provenance is currently positional and
   fragile. Add an explicit field in the Phase 4 schema. *(→ Phase 4)*
6. **`risk_level` is a near-perfect derived proxy for `sif_potential`** (HIGH↔Yes,
   LOW↔No, 100% consistent) — must be explicitly excluded from any classifier
   feature set to avoid leakage. *(→ Phase 4, Phase 12/13 contracts)*
7. **Real narratives contain real worker names** — no emails/phone numbers found,
   but names should be redacted before any external sharing of raw examples.
   *(→ handling discipline for all future phases, not a code change)*

No label fabrication, no forced/invented mappings, and no evidence of the merge
script inventing data were found anywhere in this audit — the existing dataset is
honest about what it does and doesn't know (`Unmapped`, `Unknown (not recorded in
source)`) rather than guessing. The problems found here are gaps and risks to
manage, not integrity violations to fix.
