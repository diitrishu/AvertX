#!/usr/bin/env python3
"""
Phase 3 ingestion script for SIF-Guard.

Organizes verified external/raw safety datasets into data/raw/{oil_existing,osha,bsee,other}/
WITHOUT modifying the content of any source file (copy2 preserves bytes + timestamps).
Writes data/raw/MANIFEST.csv documenting, for every ingested file: category, source
organization, source URL, license, row count, how to derive source_record_id, what
source_type value to assign, which column holds narrative text (if any), and notes.

Re-runnable: skips a copy if the destination already exists and is the same size.
"""
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # will be set to SIF root by caller via chdir
RAW = Path("data/raw")

CATEGORIES = ["oil_existing", "osha", "bsee", "other"]

MANIFEST_ROWS = []

def ingest(category, dataset_name, src_path, source_org, source_url, license_str,
           row_count, source_record_id_field, source_type_value, narrative_col, notes,
           copy_file=True):
    dest_dir = RAW / category / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_path)
    if copy_file:
        dest = dest_dir / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        local_path = str(dest)
    else:
        # Too large / already in a stable untouched location (e.g. a git clone) —
        # reference in place rather than duplicate.
        local_path = str(src)

    MANIFEST_ROWS.append({
        "category": category,
        "dataset_name": dataset_name,
        "local_path": local_path,
        "source_organization": source_org,
        "source_url": source_url,
        "license": license_str,
        "row_count": row_count,
        "source_record_id_field": source_record_id_field,
        "source_type_value": source_type_value,
        "narrative_text_column": narrative_col,
        "notes": notes,
    })


def main():
    for c in CATEGORIES:
        (RAW / c).mkdir(parents=True, exist_ok=True)

    # --- oil_existing ---------------------------------------------------
    ingest(
        category="oil_existing",
        dataset_name="sif_guard_seed",
        src_path="oil_safety_reports_merged.csv",
        source_org="SIF-Guard project (internal)",
        source_url="N/A (internal project file)",
        license_str="N/A (project-owned; underlying real rows sourced from IHM Stefanini, CC0)",
        row_count=1325,
        source_record_id_field="report_id (already unique, e.g. OIL-00001)",
        source_type_value="real_ihm_stefanini | synthetic (already present in `source` column)",
        narrative_col="report_text",
        notes=("The current production seed dataset, already blending 425 real IHM "
               "Stefanini rows with 900 synthetic oil & gas rows. Treated as its own "
               "raw input for the Phase 4 master merge, not decomposed further here. "
               "See DATASET_AUDIT.md for full quality findings on this file."),
    )

    # --- osha -------------------------------------------------------------
    ingest(
        category="osha",
        dataset_name="severe_injury_reports_jboysen",
        src_path="data/kaggle/injured_workers/severe_injury_reports.csv",
        source_org="OSHA, U.S. Department of Labor (via Kaggle mirror: jboysen/injured-workers)",
        source_url="https://www.kaggle.com/datasets/jboysen/injured-workers ; official source: https://www.osha.gov/severe-injury-reports",
        license_str="Public domain (U.S. federal government work, 17 U.S.C. §105) — verified by matching column schema to OSHA's own SIR dashboard fields",
        row_count=21730,
        source_record_id_field="ID",
        source_type_value="real_osha_severe_injury",
        narrative_col="Final Narrative",
        notes="Hospitalizations, amputations, and related severe injuries. Cross-industry, not oil & gas specific.",
    )

    ingest(
        category="osha",
        dataset_name="hse_abstracts_2015_2017_ruqaiyaship",
        src_path="data/kaggle/osha_15_17/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv",
        source_org="OSHA, U.S. Department of Labor (via Kaggle mirror: ruqaiyaship/osha-accident-and-injury-data-1517)",
        source_url="https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517",
        license_str="Needs verification — underlying OSHA data is public domain, but the Kaggle uploader's own redistribution terms have not been separately confirmed",
        row_count=4847,
        source_record_id_field="summary_nr",
        source_type_value="real_osha_hse_abstracts",
        narrative_col="Abstract Text",
        notes="Construction-worker accident abstracts, 2015-2017.",
    )

    ingest(
        category="osha",
        dataset_name="oiics_code_list_reference",
        src_path="data/kaggle/osha_15_17/oiics_201_code_list.xlsx",
        source_org="OSHA / BLS OIICS reference taxonomy",
        source_url="https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517",
        license_str="Public domain (U.S. government classification reference)",
        row_count="N/A — reference/lookup table, not incident records",
        source_record_id_field="N/A",
        source_type_value="N/A (reference table, not ingested as incident rows)",
        narrative_col="N/A",
        notes="OIICS code definitions — useful for mapping HuggingFace 'event' codes and understanding OSHA abstract categories, not itself a source of incident narratives.",
    )

    # --- bsee ---------------------------------------------------------------
    # Nothing verified/downloaded yet. Do not fabricate a source here.

    # --- other ----------------------------------------------------------
    ingest(
        category="other",
        dataset_name="ihm_stefanini_raw",
        src_path="data/kaggle/ihm_stefanini/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv",
        source_org="IHM Stefanini (Brazil) via Kaggle (ihmstefanini/industrial-safety-and-health-analytics-database)",
        source_url="https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database",
        license_str="CC0 / Public Domain",
        row_count=426,
        source_record_id_field="row index (no native ID column in source file)",
        source_type_value="real_ihm_stefanini",
        narrative_col="Description",
        notes=("Fresh raw copy of the same source already blended into oil_existing/sif_guard_seed. "
               "Kept here as the untouched raw input for provenance/re-mapping purposes, per rule "
               "'never overwrite original raw datasets.' Mining/metals domain, not oil & gas."),
    )

    ingest(
        category="other",
        dataset_name="niosh_injury_narrative_coding",
        src_path="data/huggingface/injury_narrative_coding/train.csv",
        source_org="NIOSH (National Institute for Occupational Safety and Health), via HuggingFace (mayerantoine/injury-narrative-coding)",
        source_url="https://huggingface.co/datasets/mayerantoine/injury-narrative-coding",
        license_str="Apache 2.0 — verified directly from the dataset card",
        row_count=153956,
        source_record_id_field="row index (no native ID column in source file)",
        source_type_value="real_niosh_injury_narrative",
        narrative_col="text",
        notes=("Filed under 'other' rather than 'osha' — NIOSH and OSHA are distinct US "
               "DOL/HHS agencies and should not be conflated. train split only copied here; "
               "test split (75,864 rows) stays at data/huggingface/injury_narrative_coding/test.csv "
               "and should be held out, not used for training, once split logic is designed in Phase 10."),
        copy_file=True,
    )

    ingest(
        category="other",
        dataset_name="phmsa_hazmat_incidents",
        src_path="data/phmsa/hazmat/repo/data/fetched",
        source_org="PHMSA (Pipeline and Hazardous Materials Safety Administration), U.S. DOT, via Data Liberation Project mirror",
        source_url="https://github.com/data-liberation-project/phmsa-hazmat-incident-reports",
        license_str="Public domain (raw data/fetched/ files, confirmed from the repo's own README); repo code MIT; other derived files CC BY-SA 4.0",
        row_count="~697 monthly CSVs, 1971-2025 (row count not fully tallied — 872MB total)",
        source_record_id_field="varies by monthly file — needs a per-file incident-ID column check in Phase 4",
        source_type_value="real_phmsa_hazmat",
        narrative_col="not yet confirmed which of the 200 columns holds free narrative text — flagged for Phase 4 schema mapping",
        notes=("NOT copied into data/raw/ — left in place at data/phmsa/hazmat/repo/data/fetched/ "
               "because it is a 1.1GB git clone already in a stable, untouched location; duplicating "
               "it would waste disk with no provenance benefit. Referenced here by path only."),
        copy_file=False,
    )

    # --- write manifest ---------------------------------------------------
    manifest_path = RAW / "MANIFEST.csv"
    fieldnames = ["category", "dataset_name", "local_path", "source_organization",
                  "source_url", "license", "row_count", "source_record_id_field",
                  "source_type_value", "narrative_text_column", "notes"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in MANIFEST_ROWS:
            w.writerow({
                "category": row["category"],
                "dataset_name": row["dataset_name"],
                "local_path": row["local_path"],
                "source_organization": row["source_organization"],
                "source_url": row["source_url"],
                "license": row["license"],
                "row_count": row["row_count"],
                "source_record_id_field": row["source_record_id_field"],
                "source_type_value": row["source_type_value"],
                "narrative_text_column": row["narrative_text_column"],
                "notes": row["notes"],
            })

    print(f"Wrote {manifest_path} with {len(MANIFEST_ROWS)} entries.")

    # bsee gap note
    bsee_note = RAW / "bsee" / "NOTE.md"
    bsee_note.write_text(
        "# BSEE (Bureau of Safety and Environmental Enforcement) — not yet sourced\n\n"
        "No BSEE offshore incident dataset has been downloaded or verified as of this "
        "ingestion pass. `data/README.md` Category A lists BSEE as a candidate source "
        "(offshore Gulf of Mexico OCS incidents, public domain) but the URL/portal was "
        "not yet fetched or license-confirmed against actual downloadable content.\n\n"
        "This folder is intentionally empty. Per project rule #1 (never fabricate data) "
        "and rule #2 (never fabricate dataset URLs), nothing is placed here until a real "
        "BSEE dataset is located, downloaded, and verified the same way the OSHA/PHMSA/"
        "NIOSH sources were in this phase.\n",
        encoding="utf-8",
    )
    print(f"Wrote {bsee_note}")


if __name__ == "__main__":
    main()
