# DATA_SOURCES.md — Verified External Data Sources for SIF-Guard

This is the authoritative record of which external datasets are actually being used to
train/evaluate SIF-Guard's models, as opposed to `data/README.md`, which is a broader
candidate catalog (17+ possible sources, most not yet downloaded or verified). Every
source listed here has been independently verified — downloaded file inspected
directly, license confirmed against the source's own documentation or the live
source page, not assumed from a description.

No fabricated URLs, no fabricated licenses, no fabricated row counts. Where something
is genuinely uncertain, it's marked "needs verification" rather than guessed.

Raw ingestion layout and per-file provenance detail: see `data/raw/MANIFEST.csv`.

---

## Sources in active use

### 1. SIF-Guard seed dataset (internal)
- **File:** `oil_safety_reports_merged.csv` (1,325 rows)
- **Composition:** 425 real rows (IHM Stefanini, CC0) + 900 synthetic oil & gas rows
- **License:** N/A — project-owned; underlying real rows CC0
- **Status:** Existing production dataset. See `DATASET_AUDIT.md` for full quality audit.

### 2. OSHA Severe Injury Reports (jboysen Kaggle mirror)
- **Source:** OSHA, U.S. Department of Labor
- **Official page:** https://www.osha.gov/severe-injury-reports
- **Mirror used:** https://www.kaggle.com/datasets/jboysen/injured-workers
- **Rows:** 21,730 (verified by direct count)
- **License:** Public domain (U.S. federal government work, 17 U.S.C. §105). Verified
  by confirming the column schema (`ID, UPA, EventDate, Final Narrative, Hospitalized,
  Amputation, Primary NAICS, ...`) matches OSHA's own dashboard fields exactly.
- **Narrative field:** `Final Narrative`
- **Domain:** Cross-industry U.S. severe injuries (hospitalizations, amputations) — not
  oil & gas specific.

### 3. OSHA HSE Construction Accident Abstracts, 2015-2017 (ruqaiyaship Kaggle mirror)
- **Source:** OSHA, U.S. Department of Labor
- **Mirror used:** https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517
- **Rows:** 4,847 (verified by direct count)
- **License:** Underlying OSHA data is public domain; the Kaggle uploader's own
  redistribution terms have not been separately confirmed — **needs verification**
  before any external redistribution of this specific file (fine for internal
  model training either way, since the source data itself is government/public-domain).
- **Narrative field:** `Abstract Text` (also `Event Description`)
- **Domain:** Construction.
- **Companion file:** `oiics_201_code_list.xlsx` — OIICS code reference, not incident
  data itself; public domain.

### 4. NIOSH Injury Narrative Coding Dataset (HuggingFace)
- **Source:** NIOSH (National Institute for Occupational Safety and Health) — built
  from a 2018 NASA Tournament Lab competition run in collaboration with NIOSH, using
  the official OIICS occupational-injury classification system.
- **URL:** https://huggingface.co/datasets/mayerantoine/injury-narrative-coding
- **Rows:** 153,956 train + 75,864 test = 229,820 total (exact match verified against
  the dataset card and the downloaded files)
- **License:** Apache 2.0 — confirmed directly from the dataset card.
- **Narrative field:** `text`
- **Labels:** `event` — 48 OIICS event codes across 7 categories (verified: 48 unique
  values present in the training split)
- **Note:** This is a NIOSH dataset, not an OSHA one — filed separately in
  `data/raw/other/` rather than `data/raw/osha/` to avoid conflating two distinct
  federal agencies. Only the train split has been copied into `data/raw/`; the test
  split must stay held out once Phase 10 splitting logic exists, not be used for
  training.

### 5. PHMSA Hazmat Transportation Incident Reports
- **Source:** PHMSA (Pipeline and Hazardous Materials Safety Administration), U.S. DOT
- **Mirror used:** Data Liberation Project — https://github.com/data-liberation-project/phmsa-hazmat-incident-reports
- **Size:** ~697 monthly CSVs (1971–2025), 872 MB, 200 columns
- **License:** Verified directly from the mirror repo's own README — raw files under
  `data/fetched/` (what this project uses) are public domain; other processed/derived
  files in that repo are CC BY-SA 4.0; the repo's own code is MIT.
- **Domain:** Hazmat transportation incidents, includes oil & gas-related hazmat
  shipments but is not oil & gas E&P/operations-specific.
- **Note:** Left in place at its cloned location rather than duplicated into
  `data/raw/` (see `data/raw/MANIFEST.csv` for why). Which of its 200 columns holds
  the actual free-text narrative has **not yet been confirmed** — flagged as a Phase 4
  task, not assumed here.

### 6. IHM Stefanini Industrial Safety Database (fresh raw copy)
- **Source:** IHM Stefanini (Brazil), via Kaggle
- **URL:** https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database
- **Rows:** 426 (this raw copy) — note the version already merged into the SIF-Guard
  seed dataset uses 425 of these rows; the 1-row discrepancy has not been root-caused
  and is flagged here rather than silently reconciled.
- **License:** CC0 / Public Domain.
- **Narrative field:** `Description`
- **Domain:** Manufacturing/mining, 12 plants across 3 countries — not oil & gas.
- **Note:** Kept as an untouched raw input alongside the already-merged version, per
  the project rule against overwriting original raw datasets — useful if the merge
  mapping ever needs to be redone or audited.

---

## Sources evaluated but NOT yet obtained (known gaps)

- **BSEE (Bureau of Safety and Environmental Enforcement) offshore incident data** —
  listed as a plausible candidate in `data/README.md` (Category A) but no file has
  been downloaded or license-confirmed. `data/raw/bsee/NOTE.md` documents this gap
  explicitly rather than leaving it silently empty.
- Every other entry in `data/README.md` (Categories A/B/C/D not listed above — MSHA
  narrative fatality reports, NEISS, UK HSE RIDDOR, Safe Work Australia, etc.) remains
  an unverified candidate, not an active data source. Confidence ratings there reflect
  "the source plausibly exists," not "content confirmed."

---

## Summary

| # | Dataset | Rows | Domain | License status |
|---|---|---|---|---|
| 1 | SIF-Guard seed | 1,325 | Oil & gas | Project-owned / CC0 (real subset) |
| 2 | OSHA Severe Injury (jboysen) | 21,730 | Cross-industry | Verified public domain |
| 3 | OSHA HSE Abstracts 15-17 | 4,847 | Construction | Public domain (source); Kaggle terms unverified |
| 4 | NIOSH Injury Narratives | 229,820 | Cross-industry | Verified Apache 2.0 |
| 5 | PHMSA Hazmat Incidents | ~697 files / 872MB | Hazmat transport | Verified public domain (raw) |
| 6 | IHM Stefanini (raw) | 426 | Manufacturing/mining | Verified CC0 |

None of these are oil & gas E&P/operations-specific beyond the existing seed dataset —
this remains the central limitation carried forward from `DATASET_AUDIT.md` and should
directly shape how aggressively Phase 4's schema and Phase 9's real:synthetic ratio
work try to compensate for domain mismatch rather than assuming it away.
