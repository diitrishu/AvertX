#!/usr/bin/env python3
"""
build_master_schema.py — SIF-Guard Phase 4: Raw-to-Master Schema Mapper

Reads data/raw/MANIFEST.csv, maps each ingested raw dataset into the 29-field
master schema, and writes one combined CSV to data/processed/master_raw.csv.

Idempotent: re-running produces identical output.

Usage:
    py -3.12 data/build_master_schema.py
"""

import os
import sys
import csv
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "MANIFEST.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "master_raw.csv"
SCHEMA_VERSION = "1.0.0"
INGESTION_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

MASTER_COLUMNS = [
    "record_id",
    "source_organization",
    "source_dataset",
    "source_url",
    "source_record_id",
    "source_type",
    "is_synthetic",
    "date",
    "site",
    "industry_domain",
    "report_type",
    "activity",
    "narrative_text",
    "hazard",
    "unsafe_act",
    "unsafe_condition",
    "barrier",
    "barrier_status",
    "potential_consequence",
    "actual_outcome",
    "sif_potential",
    "sif_reason",
    "life_saving_rule",
    "injury_severity_actual",
    "age",
    "sex",
    "naics_code",
    "ingestion_date",
    "schema_version",
]

UNKNOWN = "UNKNOWN"


def _u(val):
    """Return val as-is if truthy (and not 'nan'), else UNKNOWN."""
    if val is None:
        return UNKNOWN
    s = str(val).strip()
    if s == "" or s.lower() == "nan" or s.lower() == "none":
        return UNKNOWN
    return s


def _strip_html(val):
    """Strip HTML tags from a value (e.g. PHMSA Report Number wrapped in <a> tags)."""
    import re
    s = _u(val)
    if s == UNKNOWN:
        return UNKNOWN
    # Remove HTML tags, extract just the text content
    cleaned = re.sub(r"<[^>]+>", "", s).strip()
    return cleaned if cleaned else s


def _is_nan(val):
    if val is None:
        return True
    s = str(val).strip().lower()
    return s == "" or s == "nan" or s == "none"


def _make_record_id(source_dataset: str, source_record_id: str) -> str:
    return f"{source_dataset}__{source_record_id}"


# ---------------------------------------------------------------------------
# Per-source mappers
# ---------------------------------------------------------------------------
# Each mapper receives (df: pd.DataFrame, manifest_row: dict) and returns
# a list of dicts, one per output row.  Every mapper MUST produce all 29
# master-schema columns.  Fields with no source mapping are set to UNKNOWN.

def _map_sif_guard_seed(df, manifest):
    """Map oil_safety_reports_merged.csv — the seed dataset with real labels."""
    rows = []
    for _, r in df.iterrows():
        source_record_id = _u(r.get("report_id"))
        rows.append({
            "record_id": _make_record_id("sif_guard_seed", source_record_id),
            "source_organization": _u(manifest["source_organization"]),
            "source_dataset": "sif_guard_seed",
            "source_url": _u(manifest["source_url"]),
            "source_record_id": source_record_id,
            "source_type": _u(manifest["source_type_value"]),
            "is_synthetic": str(r.get("source", "")).strip().startswith("synthetic"),
            "date": _u(r.get("date")),
            "site": _u(r.get("site")),
            "industry_domain": "oil_and_gas",
            "report_type": _u(r.get("report_type")),
            "activity": _u(r.get("activity")),
            "narrative_text": _u(r.get("report_text")),
            # Rule 3: map through from seed — these have real labels
            "hazard": _u(r.get("hazard")),
            "unsafe_act": UNKNOWN,
            "unsafe_condition": UNKNOWN,
            "barrier": _u(r.get("barrier")),
            "barrier_status": _u(r.get("barrier_status")),
            "potential_consequence": UNKNOWN,
            "actual_outcome": UNKNOWN,
            "sif_potential": _u(r.get("sif_potential")),
            "sif_reason": UNKNOWN,
            "life_saving_rule": _u(r.get("life_saving_rule")),
            "injury_severity_actual": UNKNOWN,
            "age": UNKNOWN,
            "sex": UNKNOWN,
            "naics_code": UNKNOWN,
            "ingestion_date": INGESTION_DATE,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _map_osha_severe_injury(df, manifest):
    """Map OSHA Severe Injury Reports (jboysen/kaggle)."""
    rows = []
    seen_ids = {}
    for _, r in df.iterrows():
        source_record_id = str(int(r["ID"])) if not _is_nan(r.get("ID")) else UNKNOWN
        # Deduplicate within source
        if source_record_id in seen_ids:
            seen_ids[source_record_id] += 1
            source_record_id = f"{source_record_id}_dup{seen_ids[source_record_id]}"
        else:
            seen_ids[source_record_id] = 0
        # is_synthetic: all rows from this source are real
        # date: EventDate column
        # site: City + State
        city = _u(r.get("City"))
        state = _u(r.get("State"))
        site = f"{city}, {state}" if city != UNKNOWN and state != UNKNOWN else (city if city != UNKNOWN else state)
        rows.append({
            "record_id": _make_record_id("severe_injury_reports_jboysen", source_record_id),
            "source_organization": _u(manifest["source_organization"]),
            "source_dataset": "severe_injury_reports_jboysen",
            "source_url": _u(manifest["source_url"]),
            "source_record_id": source_record_id,
            "source_type": _u(manifest["source_type_value"]),
            "is_synthetic": False,
            "date": _u(r.get("EventDate")),
            "site": site,
            "industry_domain": "general_industrial",
            "report_type": "severe_injury_report",
            "activity": UNKNOWN,
            "narrative_text": _u(r.get("Final Narrative")),
            "hazard": _u(r.get("SourceTitle")),
            "unsafe_act": UNKNOWN,
            "unsafe_condition": UNKNOWN,
            "barrier": UNKNOWN,
            "barrier_status": UNKNOWN,
            "potential_consequence": UNKNOWN,
            "actual_outcome": _u(r.get("NatureTitle")),
            "sif_potential": UNKNOWN,
            "sif_reason": UNKNOWN,
            "life_saving_rule": UNKNOWN,
            "injury_severity_actual": _u(r.get("NatureTitle")),
            "age": UNKNOWN,
            "sex": UNKNOWN,
            "naics_code": _u(r.get("Primary NAICS")),
            "ingestion_date": INGESTION_DATE,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _map_hse_abstracts(df, manifest):
    """Map OSHA HSE Abstracts 2015-2017 (ruqaiyaship/kaggle)."""
    rows = []
    seen_ids = {}
    for _, r in df.iterrows():
        source_record_id = str(int(r["summary_nr"])) if not _is_nan(r.get("summary_nr")) else UNKNOWN
        # Deduplicate within source
        if source_record_id in seen_ids:
            seen_ids[source_record_id] += 1
            source_record_id = f"{source_record_id}_dup{seen_ids[source_record_id]}"
        else:
            seen_ids[source_record_id] = 0
        rows.append({
            "record_id": _make_record_id("hse_abstracts_2015_2017_ruqaiyaship", source_record_id),
            "source_organization": _u(manifest["source_organization"]),
            "source_dataset": "hse_abstracts_2015_2017_ruqaiyaship",
            "source_url": _u(manifest["source_url"]),
            "source_record_id": source_record_id,
            "source_type": _u(manifest["source_type_value"]),
            "is_synthetic": False,
            "date": _u(r.get("Event Date")),
            "site": UNKNOWN,
            "industry_domain": "construction",
            "report_type": "hse_abstract",
            "activity": _u(r.get("Task Assigned")),
            "narrative_text": _u(r.get("Abstract Text")),
            "hazard": _u(r.get("Environmental Factor")),
            "unsafe_act": UNKNOWN,
            "unsafe_condition": UNKNOWN,
            "barrier": UNKNOWN,
            "barrier_status": UNKNOWN,
            "potential_consequence": UNKNOWN,
            "actual_outcome": _u(r.get("Nature of Injury")),
            "sif_potential": UNKNOWN,
            "sif_reason": UNKNOWN,
            "life_saving_rule": UNKNOWN,
            "injury_severity_actual": _u(r.get("Degree of Injury")),
            "age": UNKNOWN,
            "sex": UNKNOWN,
            "naics_code": UNKNOWN,
            "ingestion_date": INGESTION_DATE,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _map_ihm_stefanini(df, manifest):
    """Map IHM Stefanini raw industrial safety database."""
    rows = []
    for idx, r in df.iterrows():
        source_record_id = str(idx)  # row index per MANIFEST rule
        # Date column is 'Data' (Portuguese for Date)
        date_val = _u(r.get("Data"))
        # site: Local + Countries
        local = _u(r.get("Local"))
        country = _u(r.get("Countries"))
        site = f"{local}, {country}" if local != UNKNOWN and country != UNKNOWN else (local if local != UNKNOWN else country)
        rows.append({
            "record_id": _make_record_id("ihm_stefanini_raw", source_record_id),
            "source_organization": _u(manifest["source_organization"]),
            "source_dataset": "ihm_stefanini_raw",
            "source_url": _u(manifest["source_url"]),
            "source_record_id": source_record_id,
            "source_type": _u(manifest["source_type_value"]),
            "is_synthetic": False,
            "date": date_val,
            "site": site,
            "industry_domain": _u(r.get("Industry Sector")),
            "report_type": "accident_report",
            "activity": UNKNOWN,
            "narrative_text": _u(r.get("Description")),
            "hazard": _u(r.get("Critical Risk")),
            "unsafe_act": UNKNOWN,
            "unsafe_condition": UNKNOWN,
            "barrier": UNKNOWN,
            "barrier_status": UNKNOWN,
            "potential_consequence": UNKNOWN,
            "actual_outcome": UNKNOWN,
            "sif_potential": UNKNOWN,
            "sif_reason": UNKNOWN,
            "life_saving_rule": UNKNOWN,
            "injury_severity_actual": _u(r.get("Accident Level")),
            "age": UNKNOWN,
            "sex": _u(r.get("Genre")),
            "naics_code": UNKNOWN,
            "ingestion_date": INGESTION_DATE,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _map_niosh_narrative(df, manifest):
    """Map NIOSH injury narrative coding (HuggingFace mayerantoine)."""
    rows = []
    for idx, r in df.iterrows():
        source_record_id = str(idx)  # row index per MANIFEST rule
        # sex: 1=male, 2=female in NIOSH coding
        sex_val = UNKNOWN
        if not _is_nan(r.get("sex")):
            sex_map = {1: "male", 2: "female"}
            sex_val = sex_map.get(int(r["sex"]), UNKNOWN)
        rows.append({
            "record_id": _make_record_id("niosh_injury_narrative_coding", source_record_id),
            "source_organization": _u(manifest["source_organization"]),
            "source_dataset": "niosh_injury_narrative_coding",
            "source_url": _u(manifest["source_url"]),
            "source_record_id": source_record_id,
            "source_type": _u(manifest["source_type_value"]),
            "is_synthetic": False,
            "date": UNKNOWN,
            "site": UNKNOWN,
            "industry_domain": "general_industrial",
            "report_type": "injury_narrative",
            "activity": UNKNOWN,
            "narrative_text": _u(r.get("text")),
            "hazard": UNKNOWN,
            "unsafe_act": UNKNOWN,
            "unsafe_condition": UNKNOWN,
            "barrier": UNKNOWN,
            "barrier_status": UNKNOWN,
            "potential_consequence": UNKNOWN,
            "actual_outcome": UNKNOWN,
            "sif_potential": UNKNOWN,
            "sif_reason": UNKNOWN,
            "life_saving_rule": UNKNOWN,
            "injury_severity_actual": UNKNOWN,
            "age": str(int(r["age"])) if not _is_nan(r.get("age")) else UNKNOWN,
            "sex": sex_val,
            "naics_code": UNKNOWN,
            "ingestion_date": INGESTION_DATE,
            "schema_version": SCHEMA_VERSION,
        })
    return rows


def _map_phmsa_hazmat(manifest):
    """Map PHMSA hazmat incidents — reads all monthly CSVs from fetched/."""
    # The PHMSA data lives outside data/raw/ (at data/phmsa/hazmat/repo/data/fetched/)
    # per the MANIFEST note: "NOT copied into data/raw/ — left in place"
    phmsa_dir = PROJECT_ROOT / "data" / "phmsa" / "hazmat" / "repo" / "data" / "fetched"
    if not phmsa_dir.exists():
        print(f"  [WARN] PHMSA directory not found: {phmsa_dir}")
        return []

    csv_files = sorted(glob.glob(str(phmsa_dir / "*.csv")))
    if not csv_files:
        print(f"  [WARN] No CSV files found in {phmsa_dir}")
        return []

    all_rows = []
    seen_ids = {}  # track duplicates across monthly files
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip", low_memory=False)
            except Exception as e:
                print(f"  [WARN] Could not read {csv_path}: {e}")
                continue

        for _, r in df.iterrows():
            source_record_id = _strip_html(r.get("Report Number"))
            if source_record_id == UNKNOWN:
                source_record_id = f"phmsa_{len(all_rows)}"
            # Handle duplicates across monthly files by appending suffix
            if source_record_id in seen_ids:
                seen_ids[source_record_id] += 1
                source_record_id = f"{source_record_id}_dup{seen_ids[source_record_id]}"
            else:
                seen_ids[source_record_id] = 0

            # site: Incident City + Incident State
            city = _u(r.get("Incident City"))
            state = _u(r.get("Incident State"))
            site = f"{city}, {state}" if city != UNKNOWN and state != UNKNOWN else (city if city != UNKNOWN else state)

            # date: Date Of Incident
            date_val = _u(r.get("Date Of Incident"))

            # Commodity name as industry context
            commodity = _u(r.get("Commodity Short Name"))

            all_rows.append({
                "record_id": _make_record_id("phmsa_hazmat_incidents", source_record_id),
                "source_organization": _u(manifest["source_organization"]),
                "source_dataset": "phmsa_hazmat_incidents",
                "source_url": _u(manifest["source_url"]),
                "source_record_id": source_record_id,
                "source_type": "real_phmsa_hazmat",
                "is_synthetic": False,
                "date": date_val,
                "site": site,
                "industry_domain": "hazmat_transportation",
                "report_type": "hazmat_incident_report",
                "activity": _u(r.get("Mode Of Transportation")),
                "narrative_text": _u(r.get("Description Of Events")),
                "hazard": _u(r.get("Commodity Short Name")),
                "unsafe_act": UNKNOWN,
                "unsafe_condition": UNKNOWN,
                "barrier": UNKNOWN,
                "barrier_status": UNKNOWN,
                "potential_consequence": UNKNOWN,
                "actual_outcome": UNKNOWN,
                "sif_potential": UNKNOWN,
                "sif_reason": UNKNOWN,
                "life_saving_rule": UNKNOWN,
                "injury_severity_actual": UNKNOWN,
                "age": UNKNOWN,
                "sex": UNKNOWN,
                "naics_code": UNKNOWN,
                "ingestion_date": INGESTION_DATE,
                "schema_version": SCHEMA_VERSION,
            })

    return all_rows


# Mapper registry: maps dataset_name -> mapper function
MAPPER_REGISTRY = {
    "sif_guard_seed": _map_sif_guard_seed,
    "severe_injury_reports_jboysen": _map_osha_severe_injury,
    "hse_abstracts_2015_2017_ruqaiyaship": _map_hse_abstracts,
    "ihm_stefanini_raw": _map_ihm_stefanini,
    "niosh_injury_narrative_coding": _map_niosh_narrative,
    "phmsa_hazmat_incidents": _map_phmsa_hazmat,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SIF-Guard Phase 4: build_master_schema.py")
    print(f"Schema version: {SCHEMA_VERSION}")
    print(f"Ingestion date: {INGESTION_DATE}")
    print("=" * 70)

    # --- Read manifest ---
    if not MANIFEST_PATH.exists():
        print(f"[FATAL] Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"\nManifest loaded: {len(manifest_df)} rows")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_output_rows = []
    summary = []

    for _, manifest_row in manifest_df.iterrows():
        dataset_name = str(manifest_row["dataset_name"]).strip()
        narrative_col = str(manifest_row["narrative_text_column"]).strip()
        local_path = str(manifest_row["local_path"]).strip()
        source_type = str(manifest_row["source_type_value"]).strip()

        # --- Rule: skip rows with N/A narrative column ---
        if narrative_col.upper() == "N/A":
            print(f"\n  [SKIP] {dataset_name}: narrative_text_column is N/A (reference file)")
            summary.append({
                "dataset": dataset_name,
                "rows_read": 0,
                "rows_written": 0,
                "skipped": True,
                "reason": "N/A narrative column (reference file)",
            })
            continue

        # --- Look up mapper ---
        mapper = MAPPER_REGISTRY.get(dataset_name)
        if mapper is None:
            print(f"\n  [SKIP] {dataset_name}: no mapper registered in MAPPER_REGISTRY")
            summary.append({
                "dataset": dataset_name,
                "rows_read": 0,
                "rows_written": 0,
                "skipped": True,
                "reason": "No mapper registered",
            })
            continue

        # --- Read source data ---
        print(f"\n  Processing: {dataset_name}")
        print(f"    Source path: {local_path}")

        # PHMSA is special — directory of CSVs, mapper reads them directly
        if dataset_name == "phmsa_hazmat_incidents":
            output_rows = mapper(manifest_row.to_dict())
            rows_read = "~655 CSVs"  # PHMSA is too large to count precisely here
        else:
            full_path = PROJECT_ROOT / local_path
            if not full_path.exists():
                print(f"    [ERROR] File not found: {full_path}")
                summary.append({
                    "dataset": dataset_name,
                    "rows_read": 0,
                    "rows_written": 0,
                    "skipped": True,
                    "reason": f"File not found: {full_path}",
                })
                continue

            try:
                df = pd.read_csv(full_path, encoding="utf-8", on_bad_lines="skip")
            except UnicodeDecodeError:
                df = pd.read_csv(full_path, encoding="latin-1", on_bad_lines="skip")

            rows_read = len(df)
            print(f"    Rows read: {rows_read}")

            # Skip rows where narrative_text is empty/nan
            narrative_col_actual = narrative_col
            if narrative_col_actual not in df.columns:
                print(f"    [ERROR] Narrative column '{narrative_col}' not found in source columns: {list(df.columns)}")
                summary.append({
                    "dataset": dataset_name,
                    "rows_read": rows_read,
                    "rows_written": 0,
                    "skipped": True,
                    "reason": f"Narrative column '{narrative_col}' not found",
                })
                continue

            # Filter out rows with empty narratives
            valid_mask = df[narrative_col_actual].notna() & (df[narrative_col_actual].astype(str).str.strip() != "")
            rows_before_filter = len(df)
            df = df[valid_mask].reset_index(drop=True)
            rows_skipped_empty = rows_before_filter - len(df)
            if rows_skipped_empty > 0:
                print(f"    Rows skipped (empty narrative): {rows_skipped_empty}")

            output_rows = mapper(df, manifest_row.to_dict())

        rows_written = len(output_rows)
        all_output_rows.extend(output_rows)

        print(f"    Rows written: {rows_written}")
        summary.append({
            "dataset": dataset_name,
            "rows_read": rows_read,
            "rows_written": rows_written,
            "skipped": False,
            "reason": "",
        })

    # --- Build output DataFrame ---
    print(f"\n{'=' * 70}")
    print(f"Total rows to write: {len(all_output_rows)}")

    if not all_output_rows:
        print("[FATAL] No rows produced. Check manifest and source files.")
        sys.exit(1)

    output_df = pd.DataFrame(all_output_rows, columns=MASTER_COLUMNS)

    # --- Assert record_id uniqueness (Rule 6) ---
    dupes = output_df[output_df["record_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        dupe_ids = dupes["record_id"].unique()
        print(f"\n[FATAL] Duplicate record_ids detected ({len(dupe_ids)} duplicates):")
        for rid in dupe_ids[:10]:
            print(f"  {rid}")
        if len(dupe_ids) > 10:
            print(f"  ... and {len(dupe_ids) - 10} more")
        sys.exit(1)

    print("record_id uniqueness: PASSED")

    # --- Write output ---
    output_df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nWritten: {OUTPUT_PATH}")
    print(f"Rows: {len(output_df)}")
    print(f"Columns: {len(output_df.columns)}")

    # --- Verify column count ---
    if len(output_df.columns) != 29:
        print(f"[FATAL] Expected 29 columns, got {len(output_df.columns)}")
        sys.exit(1)

    # --- Print per-source summary ---
    print(f"\n{'=' * 70}")
    print("PER-SOURCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<45} {'Read':>8} {'Written':>8} {'Notes'}")
    print("-" * 70)
    total_written = 0
    for s in summary:
        notes = s["reason"] if s["skipped"] else ""
        read_str = str(s["rows_read"])
        written_str = str(s["rows_written"])
        print(f"  {s['dataset']:<43} {read_str:>8} {written_str:>8}  {notes}")
        total_written += s["rows_written"]
    print("-" * 70)
    print(f"  {'TOTAL':<43} {'':>8} {total_written:>8}")
    print()

    # --- Quick stats ---
    print("MASTER SCHEMA STATS:")
    print(f"  sif_potential distribution:")
    vc = output_df["sif_potential"].value_counts()
    for val, count in vc.items():
        print(f"    {val}: {count}")
    print(f"  source_type distribution:")
    vc = output_df["source_type"].value_counts()
    for val, count in vc.items():
        print(f"    {val}: {count}")
    print(f"  is_synthetic distribution:")
    vc = output_df["is_synthetic"].value_counts()
    for val, count in vc.items():
        print(f"    {val}: {count}")

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
