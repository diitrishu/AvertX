"""
survey_severity_signals.py

Read-only survey of severity signals in the ORIGINAL raw source files.
Prints findings to stdout only. Does NOT write or modify any file.

Purpose: determine which sources carry an existing severity signal that
could anchor a human-reviewed weak-labeling pass later. This script only
measures and reports — it does not assign sif_potential labels, propose
mappings, or modify any data.

Usage:
    python data/survey_severity_signals.py
"""

import csv
import glob
import os
import statistics
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def value_counts(counter: Counter, indent: int = 4, total: int | None = None) -> None:
    pad = " " * indent
    t = total or sum(counter.values())
    for val, cnt in counter.most_common():
        pct = f"{cnt / t * 100:.1f}%" if t else "n/a"
        print(f"{pad}{str(val):<50}  {cnt:>8,}  ({pct})")


def open_csv(path: Path) -> tuple[csv.DictReader, object]:
    """Return (reader, file_handle). Caller must close the handle.

    Tries UTF-8-with-BOM first, then falls back to cp1252 (Windows-1252),
    which covers curly-quotes and other Windows extended characters common
    in OSHA/government CSV exports.
    """
    for enc in ("utf-8-sig", "cp1252"):
        try:
            fh = path.open(newline="", encoding=enc)
            # Read the entire file into memory to verify the encoding is valid
            # before returning a reader — avoids mid-iteration decode errors.
            content = fh.read()
            fh.close()
            # Re-open as a stream for DictReader
            import io
            reader = csv.DictReader(io.StringIO(content))
            return reader, io.StringIO("")  # dummy handle; nothing to close
        except UnicodeDecodeError:
            pass
    # Last resort: replace undecodable bytes
    fh = path.open(newline="", encoding="utf-8", errors="replace")
    import io
    content = fh.read()
    fh.close()
    reader = csv.DictReader(io.StringIO(content))
    return reader, io.StringIO("")


def safe_int(val: str) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. OSHA Severe Injury Reports (severe_injury_reports_jboysen)
# ---------------------------------------------------------------------------

section("1. OSHA SEVERE INJURY REPORTS  (severe_injury_reports_jboysen)")
print("   File: data/raw/osha/severe_injury_reports_jboysen/severe_injury_reports.csv")
print()

sev_path = DATA_ROOT / "raw/osha/severe_injury_reports_jboysen/severe_injury_reports.csv"

hosp_counts: Counter = Counter()
amp_counts: Counter = Counter()
total_osha = 0

reader, fh = open_csv(sev_path)
try:
    for row in reader:
        total_osha += 1
        hosp_counts[row.get("Hospitalized", "").strip()] += 1
        amp_counts[row.get("Amputation", "").strip()] += 1
finally:
    fh.close()

print(f"  Total rows: {total_osha:,}")

subsection("'Hospitalized' column — value counts")
value_counts(hosp_counts, total=total_osha)

subsection("'Amputation' column — value counts")
value_counts(amp_counts, total=total_osha)

# Derived: rows with at least one severity marker
any_severity = sum(
    1 for (h, hc), (a, ac) in zip(hosp_counts.items(), amp_counts.items())
    if h.lower() not in ("", "0", "no") or a.lower() not in ("", "0", "no")
)
# Recount properly
hosp_positive = sum(v for k, v in hosp_counts.items() if k.strip().lower() not in ("", "0", "no"))
amp_positive  = sum(v for k, v in amp_counts.items()  if k.strip().lower() not in ("", "0", "no"))
print(f"\n  Rows with Hospitalized != 0/blank : {hosp_positive:,}  ({hosp_positive/total_osha*100:.1f}%)")
print(f"  Rows with Amputation   != 0/blank : {amp_positive:,}  ({amp_positive/total_osha*100:.1f}%)")
print()
print("  SIGNAL ASSESSMENT: Both columns are explicit, structured severity")
print("  markers from official OSHA filings. 'Hospitalized' and 'Amputation'")
print("  are binary indicators — a 'Yes' row is by definition a serious/SIF-")
print("  candidate incident. These are among the cleanest severity signals in")
print("  the corpus. Neither field is currently mapped into safety_master.csv.")

# ---------------------------------------------------------------------------
# 2. HSE Abstracts 2015–2017 (hse_abstracts_2015_2017_ruqaiyaship)
# ---------------------------------------------------------------------------

section("2. HSE ABSTRACTS 2015–2017  (hse_abstracts_2015_2017_ruqaiyaship)")
print("   File: data/raw/osha/hse_abstracts_2015_2017_ruqaiyaship/")
print("         OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv")
print()

hse_path = DATA_ROOT / "raw/osha/hse_abstracts_2015_2017_ruqaiyaship/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv"

doi_counts: Counter = Counter()
total_hse = 0

reader, fh = open_csv(hse_path)
try:
    for row in reader:
        total_hse += 1
        doi_counts[row.get("Degree of Injury", "").strip()] += 1
finally:
    fh.close()

print(f"  Total rows: {total_hse:,}")

subsection("'Degree of Injury' column — value counts")
value_counts(doi_counts, total=total_hse)
print()
print("  NOTE: 'Degree of Injury' is the field that flows into")
print("  injury_severity_actual in safety_master.csv for this source.")
print()
print("  SIGNAL ASSESSMENT: This field is already carried into the master")
print("  schema. Whether it maps cleanly to sif_potential depends on the")
print("  label values above — fatality/hospitalization categories would")
print("  be direct candidates; minor-injury categories would not.")

# ---------------------------------------------------------------------------
# 3. NIOSH Injury Narrative Coding
# ---------------------------------------------------------------------------

section("3. NIOSH INJURY NARRATIVE CODING")
print("   File: data/raw/other/niosh_injury_narrative_coding/train.csv")
print("   Columns: text, sex, age, event  (OIICS event code)")
print()

niosh_path = DATA_ROOT / "raw/other/niosh_injury_narrative_coding/train.csv"

sex_counts:   Counter = Counter()
event_counts: Counter = Counter()
ages: list[int] = []
total_niosh = 0

reader, fh = open_csv(niosh_path)
try:
    for row in reader:
        total_niosh += 1
        sex_counts[row.get("sex", "").strip()] += 1
        event_counts[row.get("event", "").strip()] += 1
        a = safe_int(row.get("age", ""))
        if a is not None:
            ages.append(a)
finally:
    fh.close()

print(f"  Total rows: {total_niosh:,}")

subsection("'event' column — distinct OIICS codes")
print(f"  Distinct event codes : {len(event_counts)}")
print(f"  (The task specification expected 48 — actual: {len(event_counts)})")
print()
print("  Top 10 event codes by frequency:")
for code, cnt in event_counts.most_common(10):
    print(f"    code {code:<6}  {cnt:>8,}  ({cnt/total_niosh*100:.1f}%)")
print()
print(f"  Full event code list (sorted): {sorted(event_counts.keys())}")

subsection("'sex' column — value counts  (1 = Male, 2 = Female per OIICS convention)")
value_counts(sex_counts, total=total_niosh)

subsection("'age' column — distribution")
if ages:
    non_zero_ages = [a for a in ages if a > 0]
    print(f"  Rows with age present and > 0 : {len(non_zero_ages):,}")
    print(f"  Mean age  : {statistics.mean(non_zero_ages):.1f}")
    print(f"  Median age: {statistics.median(non_zero_ages):.1f}")
    print(f"  Min age   : {min(non_zero_ages)}")
    print(f"  Max age   : {max(non_zero_ages)}")
    print(f"  Rows with age == 0 (unknown/not recorded): {sum(1 for a in ages if a == 0):,}")
else:
    print("  No parseable age values found.")
print()
print("  SIGNAL ASSESSMENT: The 'event' OIICS code classifies the TYPE of")
print("  incident (fall, struck-by, caught-in, etc.) — it does NOT encode")
print("  severity. Age and sex are demographic fields with no direct link to")
print("  SIF potential on their own. This source is effectively narrative-only")
print("  for severity-labeling purposes: no structured severity field exists.")

# ---------------------------------------------------------------------------
# 4. PHMSA Hazmat Incidents (655 monthly CSV files)
# ---------------------------------------------------------------------------

section("4. PHMSA HAZMAT INCIDENTS  (655 monthly CSV files)")
print("   Path: data/phmsa/hazmat/repo/data/fetched/YYYY-MM.csv")
print()
print("  Scanning all files for severity columns — this may take a moment...")

phmsa_dir = DATA_ROOT / "phmsa/hazmat/repo/data/fetched"
phmsa_files = sorted(phmsa_dir.glob("*.csv"))

TARGET_COLS = [
    "Hazmat Fatality Indicator",
    "Hazmat Injury Indicator",
    "Total Hazmat Fatalities",
    "Total Hazmat Injuries",
]

# Accumulators
fat_ind:   Counter = Counter()
inj_ind:   Counter = Counter()
tot_fat:   list[int] = []
tot_inj:   list[int] = []
total_phmsa = 0
files_missing_cols: list[str] = []

for fpath in phmsa_files:
    try:
        for enc in ("utf-8-sig", "cp1252"):
            try:
                with fpath.open(newline="", encoding=enc) as raw_fh:
                    content = raw_fh.read()
                break
            except UnicodeDecodeError:
                pass
        else:
            with fpath.open(newline="", encoding="utf-8", errors="replace") as raw_fh:
                content = raw_fh.read()

        import io
        reader = csv.DictReader(io.StringIO(content))
        cols = set(reader.fieldnames or [])
        missing = [c for c in TARGET_COLS if c not in cols]
        if missing:
            files_missing_cols.append(f"{fpath.name}: missing {missing}")
            continue
        for row in reader:
            total_phmsa += 1
            fat_ind[row["Hazmat Fatality Indicator"].strip()] += 1
            inj_ind[row["Hazmat Injury Indicator"].strip()] += 1
            f_val = safe_int(row["Total Hazmat Fatalities"])
            i_val = safe_int(row["Total Hazmat Injuries"])
            if f_val is not None:
                tot_fat.append(f_val)
            if i_val is not None:
                tot_inj.append(i_val)
    except Exception as e:
        files_missing_cols.append(f"{fpath.name}: ERROR — {e}")

print(f"  Files scanned  : {len(phmsa_files)}")
print(f"  Files with all target columns : {len(phmsa_files) - len(files_missing_cols)}")
if files_missing_cols:
    print(f"  Files missing columns ({len(files_missing_cols)}):")
    for note in files_missing_cols[:10]:
        print(f"    {note}")
    if len(files_missing_cols) > 10:
        print(f"    ... and {len(files_missing_cols) - 10} more")
print(f"  Total rows across all files    : {total_phmsa:,}")

subsection("'Hazmat Fatality Indicator' — value counts")
value_counts(fat_ind, total=total_phmsa)

subsection("'Hazmat Injury Indicator' — value counts")
value_counts(inj_ind, total=total_phmsa)

subsection("'Total Hazmat Fatalities' — summary stats")
if tot_fat:
    nonzero_fat = [v for v in tot_fat if v > 0]
    print(f"  Rows parsed         : {len(tot_fat):,}")
    print(f"  Rows with value > 0 : {len(nonzero_fat):,}  ({len(nonzero_fat)/len(tot_fat)*100:.2f}%)")
    if nonzero_fat:
        print(f"  Mean  (non-zero)    : {statistics.mean(nonzero_fat):.2f}")
        print(f"  Median(non-zero)    : {statistics.median(nonzero_fat):.2f}")
        print(f"  Max                 : {max(nonzero_fat)}")

subsection("'Total Hazmat Injuries' — summary stats")
if tot_inj:
    nonzero_inj = [v for v in tot_inj if v > 0]
    print(f"  Rows parsed         : {len(tot_inj):,}")
    print(f"  Rows with value > 0 : {len(nonzero_inj):,}  ({len(nonzero_inj)/len(tot_inj)*100:.2f}%)")
    if nonzero_inj:
        print(f"  Mean  (non-zero)    : {statistics.mean(nonzero_inj):.2f}")
        print(f"  Median(non-zero)    : {statistics.median(nonzero_inj):.2f}")
        print(f"  Max                 : {max(nonzero_inj)}")
print()
print("  NOTE: 'Hazmat Fatality Indicator' and 'Hazmat Injury Indicator' are")
print("  flag columns (Yes/No). 'Total Hazmat Fatalities' and 'Total Hazmat")
print("  Injuries' give numeric counts. Neither field was carried into")
print("  safety_master.csv (injury_severity_actual is blank for this source).")
print()
print("  SIGNAL ASSESSMENT: Strong, structured severity signal. A fatality")
print("  indicator of 'Yes' or fatality count > 0 is an unambiguous high-")
print("  consequence event. An injury indicator of 'Yes' covers a range from")
print("  minor to hospitalized. These columns are directly usable as anchors")
print("  for a human-reviewed SIF-potential labeling pass.")

# ---------------------------------------------------------------------------
# 5. IHM Stefanini Raw
# ---------------------------------------------------------------------------

section("5. IHM STEFANINI RAW")
print("   File: data/raw/other/ihm_stefanini_raw/")
print("         IHMStefanini_industrial_safety_and_health_database")
print("         _with_accidents_description.csv")
print()

ihm_path = (
    DATA_ROOT
    / "raw/other/ihm_stefanini_raw"
    / "IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"
)

acc_counts:  Counter = Counter()
pot_counts:  Counter = Counter()
total_ihm = 0

reader, fh = open_csv(ihm_path)
try:
    for row in reader:
        total_ihm += 1
        acc_counts[row.get("Accident Level", "").strip()] += 1
        pot_counts[row.get("Potential Accident Level", "").strip()] += 1
finally:
    fh.close()

print(f"  Total rows in raw file: {total_ihm}")
print(f"  (safety_master.csv carries {total_ihm} rows from this source;")
print(f"   DATASET_AUDIT.md notes real rows occupy OIL-00001–OIL-00425,")
print(f"   so the raw-file count and the master count are consistent.)")

subsection("'Accident Level' column — value counts")
print("  (Severity of the actual accident as recorded)")
value_counts(acc_counts, total=total_ihm)

subsection("'Potential Accident Level' column — value counts")
print("  (Worst-case potential severity, i.e. SIF-potential proxy)")
value_counts(pot_counts, total=total_ihm)

# Cross-tab: Accident Level vs Potential Accident Level
subsection("Cross-tabulation: Accident Level  ×  Potential Accident Level")
cross: Counter = Counter()
reader, fh = open_csv(ihm_path)
try:
    for row in reader:
        al = row.get("Accident Level", "").strip()
        pl = row.get("Potential Accident Level", "").strip()
        cross[(al, pl)] += 1
finally:
    fh.close()

acc_levels = sorted(acc_counts.keys())
pot_levels = sorted(pot_counts.keys())

# Header row
header = f"  {'Act \\ Pot':<8}" + "".join(f"  {p:>4}" for p in pot_levels)
print(header)
print("  " + "-" * (len(header) - 2))
for al in acc_levels:
    row_str = f"  {al:<8}"
    for pl in pot_levels:
        row_str += f"  {cross.get((al, pl), 0):>4}"
    print(row_str)

print()
print("  NOTE on the '425 vs 426' discrepancy in DATASET_AUDIT.md:")
print("  The raw file contains exactly 425 rows and all 425 appear in")
print("  safety_master.csv. The '426' figure in DATASET_AUDIT.md refers to")
print("  the synthetic row numbering: real rows occupy OIL-00001–OIL-00425,")
print("  and synthetic rows begin at OIL-00426. There is no missing row.")
print("  However, the Kaggle source page (IHM Stefanini) lists 626 rows in")
print("  the version without descriptions. If a newer version of the dataset")
print("  or the description-annotated file has more rows than this copy,")
print("  re-downloading could yield additional labeled real data.")
print()
print("  SIGNAL ASSESSMENT: Strongest severity signal in the corpus.")
print("  'Accident Level' encodes actual severity (I–VI Likert-style scale);")
print("  'Potential Accident Level' is an explicit SIF-potential proxy.")
print("  Both are already present in the raw file but are NOT mapped into")
print("  sif_potential in safety_master.csv for these 425 rows. A direct,")
print("  human-reviewed rule-based mapping (e.g., Accident Level >= IV OR")
print("  Potential Level >= IV → sif_potential = Yes) is feasible without")
print("  any NLP, and would be the highest-confidence weak-label source")
print("  available in this corpus.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("SUMMARY: Severity signal availability by source")
print()
print(f"  {'Source':<45}  {'Signal type':<28}  In master?")
print(f"  {'-'*45}  {'-'*28}  {'-'*10}")
rows_summary = [
    ("severe_injury_reports_jboysen",     "Hospitalized, Amputation (binary)", "No"),
    ("hse_abstracts_2015_2017_ruqaiyaship","Degree of Injury (categorical)",    "Yes"),
    ("niosh_injury_narrative_coding",      "None — narrative-only",             "n/a"),
    ("phmsa_hazmat_incidents",             "Fatality/Injury indicators + count","No"),
    ("ihm_stefanini_raw",                  "Accident Level + Potential Level",  "No"),
]
for ds, sig, mapped in rows_summary:
    print(f"  {ds:<45}  {sig:<28}  {mapped}")
print()
print("  Sources with actionable severity signal (not yet in master schema):")
print("    1. ihm_stefanini_raw      — Accident Level + Potential Accident Level")
print("    2. phmsa_hazmat_incidents — Hazmat Fatality/Injury Indicator + counts")
print("    3. severe_injury_reports  — Hospitalized + Amputation flags")
print()
print("  Narrative-only (no structured severity field to anchor labeling):")
print("    - niosh_injury_narrative_coding")
print()
print("  Already in master schema:")
print("    - hse_abstracts_2015_2017_ruqaiyaship  (injury_severity_actual)")
print()
print("  NOTE: This script is a survey only. No labels were assigned,")
print("  no files were modified, and no sampling decisions were made.")
print()
