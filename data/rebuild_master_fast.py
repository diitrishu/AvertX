#!/usr/bin/env python3
"""
Fast, fully-vectorized rebuild of data/processed/master_raw.csv.

The IDE model's original build_master_schema.py used df.iterrows() per source,
which is far too slow for the 678K-row PHMSA source in a time-boxed environment.
This produces byte-for-byte the same 29-column schema and semantics (verified
against the original script's mapping logic), but uses vectorized pandas ops
throughout, and includes the unsafe_act fix for hse_abstracts_2015_2017_ruqaiyaship
(now UNKNOWN, not mapped from Human Factor).

Idempotent: re-running produces identical output (only ingestion_date changes).
"""
import csv
import glob
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = Path(".").resolve()
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "MANIFEST.csv"
OUTPUT_PATH = Path("/tmp/master_raw_new.csv")
SCHEMA_VERSION = "1.0.0"
INGESTION_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
UNKNOWN = "UNKNOWN"

MASTER_COLUMNS = [
    "record_id", "source_organization", "source_dataset", "source_url",
    "source_record_id", "source_type", "is_synthetic", "date", "site",
    "industry_domain", "report_type", "activity", "narrative_text", "hazard",
    "unsafe_act", "unsafe_condition", "barrier", "barrier_status",
    "potential_consequence", "actual_outcome", "sif_potential", "sif_reason",
    "life_saving_rule", "injury_severity_actual", "age", "sex", "naics_code",
    "ingestion_date", "schema_version",
]


def u(series):
    """Vectorized _u(): blank/NaN -> UNKNOWN string, else stringified value."""
    s = series.astype(object)
    s = s.where(s.notna(), UNKNOWN)
    s = s.astype(str)
    s = s.mask(s.str.strip().isin(["", "nan", "NaN", "None"]), UNKNOWN)
    return s


def const(val, n):
    return pd.Series([val] * n)


def base_frame(n):
    df = pd.DataFrame(index=range(n))
    for col in MASTER_COLUMNS:
        df[col] = UNKNOWN
    df["ingestion_date"] = INGESTION_DATE
    df["schema_version"] = SCHEMA_VERSION
    return df


def dedupe_ids(id_series):
    """Match original suffixing behavior for duplicate source_record_id within a source."""
    counts = {}
    out = []
    for v in id_series:
        if v in counts:
            counts[v] += 1
            out.append(f"{v}_dup{counts[v]}")
        else:
            counts[v] = 0
            out.append(v)
    return out


def read_csv_robust(path, **kwargs):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 1, f"could not decode {path} with utf-8/latin-1/cp1252")


manifest_rows = {r["dataset_name"]: r for r in csv.DictReader(open(MANIFEST_PATH, encoding="utf-8"))}
frames = []


# ---------------------------------------------------------------- seed ----
m = manifest_rows["sif_guard_seed"]
df = read_csv_robust(m["local_path"])
n = len(df)
out = base_frame(n)
out["source_organization"] = m["source_organization"]
out["source_dataset"] = "sif_guard_seed"
out["source_url"] = m["source_url"]
out["source_record_id"] = df["report_id"].astype(str)
out["source_type"] = df["source"].astype(str)
out["is_synthetic"] = df["source"].astype(str).eq("synthetic")
out["date"] = u(df["date"])
out["site"] = u(df["site"])
out["industry_domain"] = "oil_and_gas"
out["report_type"] = u(df["report_type"])
out["activity"] = u(df["activity"])
out["narrative_text"] = u(df["report_text"])
out["hazard"] = u(df["hazard"])
out["barrier"] = u(df["barrier"])
out["barrier_status"] = u(df["barrier_status"])
out["sif_potential"] = u(df["sif_potential"])
out["life_saving_rule"] = u(df["life_saving_rule"])
out["record_id"] = "sif_guard_seed__" + out["source_record_id"]
frames.append(out)
print(f"sif_guard_seed: {n} rows")

# ---------------------------------------------- osha severe injury --------
m = manifest_rows["severe_injury_reports_jboysen"]
df = read_csv_robust(m["local_path"])
n = len(df)
out = base_frame(n)
out["source_organization"] = m["source_organization"]
out["source_dataset"] = "severe_injury_reports_jboysen"
out["source_url"] = m["source_url"]
sr_id = df["ID"].astype(str) if "ID" in df.columns else pd.Series(range(n)).astype(str)
out["source_record_id"] = dedupe_ids(sr_id.tolist())
out["source_type"] = "real_osha_severe_injury"
out["is_synthetic"] = False
out["date"] = u(df.get("EventDate", pd.Series([None] * n)))
city = u(df.get("City", pd.Series([None] * n)))
state = u(df.get("State", pd.Series([None] * n)))
out["site"] = (city + ", " + state).where((city != UNKNOWN) & (state != UNKNOWN), city.where(city != UNKNOWN, state))
out["industry_domain"] = "general_industrial"
out["report_type"] = "severe_injury_report"
out["narrative_text"] = u(df.get("Final Narrative", pd.Series([None] * n)))
out["hazard"] = u(df.get("NatureTitle", pd.Series([None] * n)))
out["actual_outcome"] = u(df.get("EventTitle", pd.Series([None] * n)))
out["naics_code"] = u(df.get("Primary NAICS", pd.Series([None] * n)))
out["record_id"] = "severe_injury_reports_jboysen__" + out["source_record_id"].astype(str)
frames.append(out)
print(f"severe_injury_reports_jboysen: {n} rows")

# ---------------------------------------------- osha hse abstracts (FIXED) -
m = manifest_rows["hse_abstracts_2015_2017_ruqaiyaship"]
df = read_csv_robust(m["local_path"])
n = len(df)
out = base_frame(n)
out["source_organization"] = m["source_organization"]
out["source_dataset"] = "hse_abstracts_2015_2017_ruqaiyaship"
out["source_url"] = m["source_url"]
sr_id = df["summary_nr"].astype(str) if "summary_nr" in df.columns else pd.Series(range(n)).astype(str)
out["source_record_id"] = dedupe_ids(sr_id.tolist())
out["source_type"] = "real_osha_hse_abstracts"
out["is_synthetic"] = False
out["date"] = u(df.get("Event Date", pd.Series([None] * n)))
out["industry_domain"] = "construction"
out["report_type"] = "hse_abstract"
out["activity"] = u(df.get("Task Assigned", pd.Series([None] * n)))
out["narrative_text"] = u(df.get("Abstract Text", pd.Series([None] * n)))
out["hazard"] = u(df.get("Environmental Factor", pd.Series([None] * n)))
out["unsafe_act"] = UNKNOWN  # <-- the fix: was Human Factor before
out["actual_outcome"] = u(df.get("Nature of Injury", pd.Series([None] * n)))
out["injury_severity_actual"] = u(df.get("Degree of Injury", pd.Series([None] * n)))
out["record_id"] = "hse_abstracts_2015_2017_ruqaiyaship__" + out["source_record_id"].astype(str)
frames.append(out)
print(f"hse_abstracts_2015_2017_ruqaiyaship: {n} rows")

# ---------------------------------------------------- ihm stefanini raw ---
m = manifest_rows["ihm_stefanini_raw"]
df = read_csv_robust(m["local_path"])
n = len(df)
out = base_frame(n)
out["source_organization"] = m["source_organization"]
out["source_dataset"] = "ihm_stefanini_raw"
out["source_url"] = m["source_url"]
out["source_record_id"] = [str(i) for i in range(n)]
out["source_type"] = "real_ihm_stefanini"
out["is_synthetic"] = False
date_col = "Data" if "Data" in df.columns else None
out["date"] = u(df[date_col]) if date_col else UNKNOWN
local_col = "Local" if "Local" in df.columns else None
country_col = "Countries" if "Countries" in df.columns else None
if local_col and country_col:
    out["site"] = u(df[local_col]) + ", " + u(df[country_col])
out["industry_domain"] = u(df["Industry Sector"]) if "Industry Sector" in df.columns else "Mining"
desc_col = "Description" if "Description" in df.columns else None
out["narrative_text"] = u(df[desc_col]) if desc_col else UNKNOWN
out["record_id"] = "ihm_stefanini_raw__" + out["source_record_id"].astype(str)
frames.append(out)
print(f"ihm_stefanini_raw: {n} rows")

# ------------------------------------------------------------ niosh ------
m = manifest_rows["niosh_injury_narrative_coding"]
df = read_csv_robust(m["local_path"])
n = len(df)
out = base_frame(n)
out["source_organization"] = m["source_organization"]
out["source_dataset"] = "niosh_injury_narrative_coding"
out["source_url"] = m["source_url"]
out["source_record_id"] = [str(i) for i in range(n)]
out["source_type"] = "real_niosh_injury_narrative"
out["is_synthetic"] = False
out["industry_domain"] = "general_industrial"
out["narrative_text"] = u(df["text"]) if "text" in df.columns else UNKNOWN
out["age"] = u(df["age"]) if "age" in df.columns else UNKNOWN
out["sex"] = u(df["sex"]) if "sex" in df.columns else UNKNOWN
out["hazard"] = u(df["event"]) if "event" in df.columns else UNKNOWN  # OIICS event code, not a hazard label
out["record_id"] = "niosh_injury_narrative_coding__" + out["source_record_id"].astype(str)
frames.append(out)
print(f"niosh_injury_narrative_coding: {n} rows")

# ------------------------------------------------------------- phmsa -----
m = manifest_rows["phmsa_hazmat_incidents"]
combined_path = "/tmp/phmsa_slim.csv"
d = read_csv_robust(combined_path)
nf = len(d)
o = base_frame(nf)
o["source_organization"] = m["source_organization"]
o["source_dataset"] = "phmsa_hazmat_incidents"
o["source_url"] = m["source_url"]
rep = u(d["Report Number"]) if "Report Number" in d.columns else pd.Series([UNKNOWN] * nf)
fallback = pd.Series([f"phmsa_{i}" for i in range(nf)])
sr_id = rep.where(rep != UNKNOWN, fallback)
# dedupe duplicate Report Numbers across the combined file using a vectorized cumcount
sr_id_df = sr_id.to_frame(name="id")
sr_id_df["occurrence"] = sr_id_df.groupby("id").cumcount()
o["source_record_id"] = sr_id_df["id"].where(sr_id_df["occurrence"] == 0, sr_id_df["id"] + "_dup" + sr_id_df["occurrence"].astype(str))
o["source_type"] = "real_phmsa_hazmat"
o["is_synthetic"] = False
o["date"] = u(d["Date Of Incident"]) if "Date Of Incident" in d.columns else UNKNOWN
city = u(d["Incident City"]) if "Incident City" in d.columns else pd.Series([UNKNOWN] * nf)
state = u(d["Incident State"]) if "Incident State" in d.columns else pd.Series([UNKNOWN] * nf)
o["site"] = (city + ", " + state).where((city != UNKNOWN) & (state != UNKNOWN), city.where(city != UNKNOWN, state))
o["industry_domain"] = "hazmat_transportation"
o["report_type"] = "hazmat_incident_report"
o["activity"] = u(d["Mode Of Transportation"]) if "Mode Of Transportation" in d.columns else UNKNOWN
o["narrative_text"] = u(d["Description Of Events"]) if "Description Of Events" in d.columns else UNKNOWN
o["hazard"] = u(d["Commodity Short Name"]) if "Commodity Short Name" in d.columns else UNKNOWN
o["record_id"] = "phmsa_hazmat_incidents__" + pd.Series(range(nf)).astype(str)
phmsa_all = o
frames.append(phmsa_all)
print(f"phmsa_hazmat_incidents: {len(phmsa_all)} rows (from combined file, {combined_path})")

# ----------------------------------------------------------- write -------
final = pd.concat(frames, ignore_index=True)[MASTER_COLUMNS]

dupe_ids = final["record_id"].duplicated().sum()
print(f"total rows: {len(final)}  |  duplicate record_ids: {dupe_ids}")
assert dupe_ids == 0, "record_id collision — fix id scheme before writing"

final.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")
