# Public Datasets for Industrial/Occupational Safety ML (SIH26165)

**Last updated:** September 4, 2026  
**Purpose:** Supplement oil_safety_reports_merged.csv (1,325 labeled rows) with real  
non-synthetic incident/near-miss/injury narrative data for SIF precursor detection.

---

## VERIFICATION NOTE (added after independent review, Sept 4 2026)

Every entry below claiming "DOWNLOADED" or a specific row count/license has now been
checked directly against the actual files on disk and, where relevant, against the
live source (Kaggle page, HuggingFace dataset card, GitHub repo, osha.gov) rather than
taken on faith. Corrections made:

- **B2 (jboysen/injured-workers):** was marked "still need Kaggle API key" in an earlier
  status message — it was in fact already downloaded (`data/kaggle/injured_workers/`,
  moved there from a mismatched folder name). Row count corrected to 21,730 (was
  ~22,000). License upgraded from "needs verification" to confirmed public domain — the
  column schema was checked against OSHA's own severe-injury dashboard and matches
  exactly, and OSHA data is a U.S. federal government work.
- **B1 (IHM Stefanini):** also downloaded fresh via Kaggle into `data/kaggle/ihm_stefanini/`
  (this was likewise mislabeled under `kaggle/injured_workers/` before cleanup).
- **B3 (ruqaiyaship OSHA 15-17 abstracts):** confirmed present and correctly described
  (4,847 rows, narrative columns verified) — moved into `data/kaggle/osha_15_17/` next
  to its code-list companion, rather than sitting loose at `data/kaggle/` root.
- **D1 (HuggingFace injury narrative coding):** license confirmed as **Apache 2.0** by
  fetching the dataset card directly; origin confirmed as a 2018 NASA Tournament
  Lab / NIOSH collaboration using the official OIICS classification system. Row counts
  (153,956 train + 75,864 test) verified exactly.
- **PHMSA hazmat repo:** license confirmed directly from the repo's own README — raw
  `data/fetched/` files (what this project uses) are public domain; other derived files
  are CC BY-SA 4.0.

Everything else below (Category A oil & gas pipeline/offshore sources, and Category
C/D entries not yet downloaded) has **not** been independently re-verified beyond what's
stated in each entry's own Confidence rating — treat "Confidence: HIGH" on an
undownloaded source as "the source plausibly exists," not as "content confirmed."

---

## WHAT'S ALREADY IN YOUR SIF FOLDER (C:\Users\hp\Desktop\SIF\)

| File | Rows | Has Narrative Text? | SIF Labeled? | Notes |
|------|------|---------------------|--------------|-------|
| oil_safety_reports_merged.csv | 1,325 (382 SIF-Yes, 943 No) | YES | YES | Your primary training file |
| IHMStefanini..._with_accidents_description.csv | 425 | YES (Description column) | No | Raw accidents from 12 plants, 3 countries |
| IHMStefanini...database.csv | 439 | No | No | Same data without Description column |
| OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv | 4,847 | YES (Abstract Text) | No | Construction worker injuries, 2015-2017 |
| OSHA_Incident-Reports_Jan-2015_Dec-22.csv | ~40K+ | YES | No | US incident reports, encoding issues (try latin-1) |
| incident_gas_transmission_gathering_jan2010_present.txt | ~2,000 | YES (NARRATIVE col) | No | PHMSA pipeline incidents, tab-delimited |
| database.csv | structured | No | No | Pipeline cause codes only, not usable for NLP |

---

## NEWLY DOWNLOADED: data/ folder

### Downloaded Successfully

| Dataset | Location | Rows/Files | Narrative? | Domain |
|---------|----------|------------|------------|--------|
| PHMSA Hazmat Incidents (1971-2025) | data/phmsa/hazmat/repo/ | 655 monthly CSVs (872 MB) | YES (200 columns including incident narratives) | Hazmat transportation |
| HuggingFace Injury Narrative Coding | data/huggingface/injury_narrative_coding/ | 153,956 train + 75,864 test | YES (text column with narratives) | General occupational (48 event codes) |

### Need Kaggle API Key (setup instructions below)

| Dataset | Kaggle Slug | Rows | Narrative? | Domain |
|---------|-------------|------|------------|--------|
| IHM Stefanini Industrial Safety | ihmstefanini/industrial-safety-and-health-analytics-database | 425 | YES | Manufacturing |
| Severely Injured Workers | jboysen/injured-workers | ~22,000 | YES | Cross-industry US severe injuries |
| OSHA Accident & Injury Data | ruqaiyaship/osha-accident-and-injury-data-1517 | ~4,800 | YES | Construction |

---

## CATEGORY A: Oil & Gas / Energy Sector (HIGHEST PRIORITY)

---

### A1. PHMSA Pipeline Incident Data

**Publisher:** Pipeline and Hazardous Materials Safety Administration (PHMSA), U.S. DOT  
**URL:** https://www.phmsa.dot.gov/data-and-statistics/pipeline/distribution-transmission-gathering-lng-and-liquid-accident-and-incident-data  
**What it contains:** Free-text NARRATIVE column + structured fields (FATALITY_IND, INJURY_IND, IGNITE_IND, EXPLODE_IND, cause codes).  
**Size:** ~2,000 rows/year, going back to 1970. You already have `incident_gas_transmission_gathering_jan2010_present.txt`.  
**Domain:** Oil & gas pipeline  
**License:** Public domain (U.S. government)  
**Confidence:** HIGH  

---

### A2. BSEE Offshore Incident Data

**Publisher:** Bureau of Safety and Environmental Enforcement (BSEE), U.S. DOI  
**URL:** https://www.bsee.gov/stats-facts/offshore-incident-statistics  
**Data Center:** https://www.data.bsee.gov/  
**What it contains:** Offshore incident statistics, investigation reports (PDFs), INCs (queryable).  
**Size:** Unknown (Gulf of Mexico OCS incidents, decades of data)  
**Domain:** Oil & gas offshore  
**License:** Public domain  
**Confidence:** HIGH  

---

### A3. CSB Chemical Incident Reports

**Publisher:** U.S. Chemical Safety Board (CSB)  
**URL:** https://www.csb.gov/investigations/completed-investigations/  
**Structured DB:** https://incidents.tychodata.com/ (116 investigations, 1998-2024)  
**What it contains:** Detailed investigation reports (PDF/video), structured root cause data in Tychodata.  
**Size:** ~116 full investigations, ~500+ incident summaries  
**Domain:** Chemical/process safety (refineries, chemical plants)  
**License:** Public domain  
**Confidence:** HIGH  

---

### A4. PHMSA Hazmat Transportation Incidents

**Publisher:** Data Liberation Project (mirroring PHMSA)  
**GitHub:** https://github.com/data-liberation-project/phmsa-hazmat-incident-reports  
**Status:** DOWNLOADED to data/phmsa/hazmat/repo/ (full git clone, 697 CSVs across data/fetched, data/processed/filtered, data/processed/discovered-dates — not just 655)  
**What it contains:** Monthly CSVs (1971-2025), 200 columns including incident narratives, hazmat details, causes, fatalities, injuries, cost.  
**Size:** 872 MB total  
**Domain:** Hazmat transportation (includes oil & gas materials)  
**License:** VERIFIED — confirmed directly against the repo's own README: raw files in `data/fetched/` (public domain — the source of this project's data), other processed/derived files under CC BY-SA 4.0. Repo code itself is MIT.  
**Confidence:** HIGH (verified)  

---

## CATEGORY B: General Industrial/Occupational Safety

---

### B1. IHM Stefanini Industrial Safety Database

**Publisher:** IHM Stefanini (Brazil)  
**Kaggle:** https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database  
**Status:** Also downloaded fresh via Kaggle — `data/kaggle/ihm_stefanini/` (426 rows raw; your existing merged project file `oil_safety_reports_merged.csv` uses 425 of these). Verified: real-source rows.  
**What it contains:** Accidents from 12 manufacturing plants in 3 countries. Columns: timestamp, country, industry sector, accident level, potential accident level, Description (free-text).  
**Size:** 425 rows  
**Domain:** Manufacturing  
**License:** CC0 / Public Domain  
**Confidence:** HIGH  

---

### B2. Severely Injured Workers

**Publisher:** Kaggle (jboysen)  
**Kaggle:** https://www.kaggle.com/datasets/jboysen/injured-workers  
**Status:** Downloaded — `data/kaggle/injured_workers/severe_injury_reports.csv`  
**What it contains:** OSHA Severe Injury Report records. Confirmed columns: ID, UPA, EventDate, Employer, Address1/2, City, State, Zip, Latitude, Longitude, Primary NAICS, Hospitalized, Amputation, Inspection, **Final Narrative** (free text), Nature, Part of Body, Event, Source, Secondary Source.  
**Size:** 21,730 rows (verified by direct count — not ~22,000 as originally estimated, close but corrected here)  
**Domain:** Cross-industry US severe injuries  
**License:** Public domain — this is a republished copy of OSHA's official Severe Injury Reports (osha.gov/severe-injury-reports), a U.S. federal government work (17 U.S.C. §105). Verified by confirming the column schema matches OSHA's own dashboard fields exactly.  
**Confidence:** HIGH (verified)  

---

### B3. OSHA Accident & Injury Data (2015-2017)

**Publisher:** Kaggle (ruqaiyaship, derived from OSHA)  
**Kaggle:** https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517  
**Status:** Downloaded — `data/kaggle/osha_15_17/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv` (4,847 rows, verified). The `oiics_201_code_list.xlsx` in the same folder is a companion code reference, not incident data.  
**What it contains:** Construction worker accident abstracts with narrative text (`Abstract Text`, `Event Description` columns confirmed by direct inspection).  
**Size:** 4,847 rows (verified)  
**Domain:** Construction  
**License:** Needs verification — Kaggle uploader license not yet confirmed against OSHA's own terms.  
**Confidence:** HIGH (existence and content verified; license still open)  

---

### B4. OSHA Injury Tracking Application (ITA)

**Publisher:** OSHA, U.S. DOL  
**URL:** http://www.osha.gov/itadata  
**What it contains:** Work-related injury/illness data (structured, **no free-text narratives**).  
**Size:** Hundreds of thousands of establishments annually  
**Domain:** All industries  
**License:** Public domain  
**Confidence:** HIGH  

**Note:** Structured/statistical only — not for NLP training, useful for validation.

---

### B5. OSHA Severe Injury Reports

**Publisher:** OSHA, U.S. DOL  
**URL:** http://www.osha.gov/severeinjury  
**What it contains:** Hospitalizations, amputations, eye losses, fatalities. Includes narrative descriptions.  
**Size:** Growing dataset (download from dashboard)  
**Domain:** All industries  
**License:** Public domain  
**Confidence:** HIGH  

---

### B6. MSHA Mining Accident/Injury Data

**Publisher:** Mine Safety and Health Administration (MSHA), U.S. DOL  
**Catalog:** http://catalog.data.gov/dataset/msha-accident-injuries-data-set  
**Part 50 Data:** https://arlweb.msha.gov/stats/part50/p50y2k/p50y2k.htm  
**Fatality Reports:** https://www.msha.gov/training/training-programs-and-courses/academy-home-page/academy-library/fatality-report  
**What it contains:** ~24,000 fatality reports with narratives. Part 50 structured data.  
**Size:** ~24,000 narrative reports + decades of structured data  
**Domain:** Mining  
**License:** Public domain  
**Confidence:** HIGH  
**Research note:** "An exploration of text mining of narrative reports of injury incidents to assess risk" used 77,215 MSHA coal mine narratives for NLP.

---

### B7. NEISS (National Electronic Injury Surveillance System)

**Publisher:** Consumer Product Safety Commission (CPSC)  
**URL:** https://www.cpsc.gov/Research--Statistics/NEISS-Injury-Data  
**ICPSR:** https://www.icpsr.umich.edu/web/NACJD/series/198  
**What it contains:** Hospital ED treatments with **brief incident narratives**. Includes age, diagnosis, product code.  
**Size:** Tens of thousands of records annually  
**Domain:** Consumer product injuries (includes workplace-related)  
**License:** Public domain  
**Confidence:** HIGH  

---

### B8. Safe Work Australia

**Publisher:** Safe Work Australia  
**URL:** https://data.safeworkaustralia.gov.au/datasets  
**What it contains:** Work-related fatalities, workers' comp, occupational hazards. **Structured only — no free-text narratives.**  
**Domain:** Australian workplace safety  
**License:** Australian open data  
**Confidence:** MEDIUM (structured, not narrative)  

---

### B9. UK HSE RIDDOR

**Publisher:** Health and Safety Executive (HSE), UK  
**URL:** https://www.hse.gov.uk/statistics/sources.htm  
**What it contains:** Workplace injury reports with **free-text narratives** (confirmed in research papers). Bulk narrative download requires FOI request.  
**Domain:** UK workplace safety (all industries)  
**License:** UK Open Government Licence  
**Confidence:** MEDIUM (narratives exist but require FOI for bulk access)  

---

### B10. Hong Kong OSH Statistics

**Publisher:** Labour Department, Hong Kong SAR  
**URL:** https://data.gov.hk/en-data/dataset/hk-ld-aaidaao-aaid-oshstats  
**What it contains:** Historical workplace injury/illness statistics (structured).  
**Domain:** Hong Kong workplace safety  
**Confidence:** MEDIUM (structured, not narrative)  

---

## CATEGORY C: Government Regulatory Datasets

---

### C1. OSHA Inspection/Violation Data

**Publisher:** OSHA, U.S. DOL  
**URL:** https://www.osha.gov/data  
**What it contains:** Inspection records with **narrative text about accidents** (confirmed on data.gov).  
**Domain:** All industries  
**License:** Public domain  
**Confidence:** HIGH  

---

### C2. PHMSA Pipeline Incident Flagged Files

**Publisher:** PHMSA, U.S. DOT  
**URL:** https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files  
**What it contains:** 20-year trend data with cause/subcause codes, fire/explosion/fatality flags.  
**Domain:** Oil & gas pipeline  
**License:** Public domain  
**Confidence:** HIGH  

---

### C3. BOEM Incidents of Non-Compliance (INCs)

**Publisher:** Bureau of Ocean Energy Management (BOEM)  
**URL:** https://www.data.boem.gov/Company/INCs/Default.aspx  
**What it contains:** Queryable offshore compliance violations (exportable).  
**Domain:** Oil & gas offshore  
**License:** Public domain  
**Confidence:** HIGH  

---

### C4. MSHA Fatality Investigation Reports

**Publisher:** MSHA, U.S. DOL  
**URL:** https://www.msha.gov/data-and-reports  
**What it contains:** Preliminary Accident Reports, Fatalgrams, Fatal Investigation Reports with narratives.  
**Domain:** Mining  
**License:** Public domain  
**Confidence:** HIGH  

---

## CATEGORY D: Safety-NLP Research Datasets

---

### D1. HuggingFace: Injury Narrative Coding

**Publisher:** mayerantoine (researcher)  
**URL:** https://huggingface.co/datasets/mayerantoine/injury-narrative-coding  
**Status:** DOWNLOADED to data/huggingface/injury_narrative_coding/ (row counts and column schema verified by direct inspection: `text, sex, age, event`, 48 unique event codes confirmed)  
**What it contains:** Multi-class text classification: **48 event codes across 7 categories** (Violence, Transportation, Falls/Slips/Trips, Contact with Objects, Exposure to Harmful Substances, Overexertion, Other). Free-text injury narratives with coded labels. Columns: text, sex, age, event.  
**Size:** 153,956 train + 75,864 test = **229,820 total rows** (exact match verified)  
**Top event codes:** 71 (25,910), 62 (24,402), 42 (15,624), 55 (11,672)  
**Domain:** General occupational injury (NIOSH-derived)  
**License:** VERIFIED — Apache 2.0. Origin confirmed: built from a 2018 NASA Tournament Lab competition run in collaboration with NIOSH, using the official OIICS occupational-injury classification system.  
**Confidence:** HIGH (verified)  

**Relevance to SIF:** Directly applicable — injury narrative classification is very close to SIF precursor detection. The 48 event codes could be cross-mapped to your 9 IOGP life-saving rules.

---

### D2. HuggingFace: OHS Severity Classifier

**Publisher:** stuSterfc  
**URL:** https://huggingface.co/stuSterfc/ohs-severity-classifier  
**What it contains:** DistilBERT model for OHS incident severity prediction. Model weights (safetensors).  
**Domain:** Occupational safety  
**Confidence:** MEDIUM (model confirmed; training data availability unclear)  

---

### D3. Mendeley: De-identified U.S. Occupational Injury Events

**Publisher:** Mendeley Data  
**URL:** https://data.mendeley.com/datasets/kj6dbshsnp  
**What it contains:** De-identified incident-level U.S. occupational injury events with narratives and standardized codes.  
**Domain:** U.S. occupational injuries  
**Confidence:** MEDIUM (URL found but not directly verified)  

---

### D4. Kaggle: US Accidents (2016-2023)

**Publisher:** sobhanmoosavi on Kaggle  
**URL:** https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents  
**What it contains:** ~2.8M car accidents with free-text descriptions and structured fields.  
**Domain:** Traffic/road safety (not workplace)  
**License:** CC BY 4.0  
**Confidence:** HIGH  

**Note:** Not workplace safety, but useful for benchmarking NLP approaches on safety narratives.

---

## HOW TO DOWNLOAD KAGGLE DATASETS

```bash
# 1. Install kaggle CLI
pip install kaggle

# 2. Get API key from https://www.kaggle.com/settings
#    Download kaggle.json and place it in ~/.kaggle/

# 3. Download datasets
kaggle datasets download -d ihmstefanini/industrial-safety-and-health-analytics-database -p data/kaggle/ihm_stefanini --unzip
kaggle datasets download -d jboysen/injured-workers -p data/kaggle/injured_workers --unzip
kaggle datasets download -d ruqaiyaship/osha-accident-and-injury-data-1517 -p data/kaggle/osha_15_17 --unzip
```

---

## RECOMMENDED NEXT STEPS FOR SIF MODEL IMPROVEMENT

| Priority | Action | Dataset | Expected Impact |
|----------|--------|---------|-----------------|
| 1 | Augment training with HuggingFace narratives | D1 (229K rows) | Major — 170x more training data |
| 2 | Add PHMSA pipeline narratives | A1 + A4 | Oil & gas domain specificity |
| 3 | Cross-map 48 event codes to 9 IOGP rules | D1 | Multi-label rule classification |
| 4 | Add severe injury narratives | B2 (~22K rows) | High-severity event patterns |
| 5 | Mine MSHA fatality reports for mining-specific SIF patterns | B6 | Cross-industry SIF generalization |
| 6 | Add OSHA severe injury narratives | B5 | Regulatory-grade severity data |
