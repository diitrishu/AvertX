#!/usr/bin/env python3
"""
build_taxonomy_mapping.py — SIF-Guard Phase 4: Taxonomy & Legacy Term Mapping

1. Reads life_saving_rules_taxonomy.json and writes taxonomy/life_saving_rules.csv
2. Scans all raw sources for risk-category columns, extracts distinct values,
   and writes taxonomy/legacy_term_mapping.csv with conservative IOGP rule mappings.

This script only READS from data/raw/ and life_saving_rules_taxonomy.json.
It only WRITES to taxonomy/. It does NOT modify master_raw.csv or any raw files.

Usage:
    py -3.12 data/build_taxonomy_mapping.py
"""

import csv
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = PROJECT_ROOT / "taxonomy"
TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY_JSON = PROJECT_ROOT / "life_saving_rules_taxonomy.json"
RULES_CSV = TAXONOMY_DIR / "life_saving_rules.csv"
LEGACY_CSV = TAXONOMY_DIR / "legacy_term_mapping.csv"

# ---------------------------------------------------------------------------
# 9 IOGP Life-Saving Rule IDs (canonical, stable identifiers)
# ---------------------------------------------------------------------------

RULE_IDS = {
    "bypassing_safety_controls": "Bypassing Safety Controls",
    "confined_space": "Confined Space",
    "driving": "Driving",
    "energy_isolation": "Energy Isolation",
    "hot_work": "Hot Work",
    "line_of_fire": "Line of Fire",
    "safe_mechanical_lifting": "Safe Mechanical Lifting",
    "work_authorisation": "Work Authorisation",
    "working_at_height": "Working at Height",
}

UNMAPPED = "Unmapped"

# ---------------------------------------------------------------------------
# Step 1: Build life_saving_rules.csv from taxonomy JSON
# ---------------------------------------------------------------------------


def build_rules_csv():
    """Read taxonomy JSON and write life_saving_rules.csv."""
    print("=" * 70)
    print("Step 1: Building taxonomy/life_saving_rules.csv")
    print("=" * 70)

    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    rules = taxonomy["life_saving_rules"]
    print(f"  Rules found in JSON: {len(rules)}")

    rows = []
    for i, rule in enumerate(rules, start=1):
        rule_name = rule["rule"]
        # Generate stable rule_id from rule name
        rule_id = rule_name.lower().replace(" ", "_").replace("/", "_")
        description = f"IOGP Life-Saving Rule: {rule_name}"
        hazard_examples = "; ".join(rule.get("typical_hazards", []))
        barrier_examples = "; ".join(rule.get("barriers", []))

        rows.append({
            "rule_id": rule_id,
            "rule_name": rule_name,
            "description": description,
            "hazard_examples": hazard_examples,
            "barrier_examples": barrier_examples,
        })

    with open(RULES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rule_id", "rule_name", "description",
                                                "hazard_examples", "barrier_examples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written: {RULES_CSV} ({len(rows)} rules)")
    for r in rows:
        print(f"    {r['rule_id']}: {r['rule_name']}")


# ---------------------------------------------------------------------------
# Step 2: Build legacy_term_mapping.csv by scanning raw sources
# ---------------------------------------------------------------------------
# Mapping philosophy (from the user's brief):
#   - Only map when there is a genuinely clear, confident 1:1 match.
#   - "Unmapped" is the correct answer for vague, ambiguous, or unrelated terms.
#   - Do NOT force mappings to make categories look cleaner.
#   - This is a mapping TABLE only — it does not modify master_raw.csv.

# --- Conservative mapping tables ---
# Each maps: source_dataset -> { source_term -> rule_id }
# Only entries where the mapping is genuinely unambiguous are included.
# Anything not in the dict gets mapped to Unmapped.

IHM_STEFANINI_MAP = {
    # Confident 1:1 mappings only
    "Vehicles and Mobile Equipment": "driving",
    "Confined space": "confined_space",
    "Electrical Shock": "energy_isolation",
    "Electrical installation": "energy_isolation",
    "Power lock": "energy_isolation",
    "Blocking and isolation of energies": "energy_isolation",
    "Machine Protection": "bypassing_safety_controls",
    "Fall prevention": "working_at_height",
    "Fall": "working_at_height",
    "Suspended Loads": "safe_mechanical_lifting",
    "Traffic": "driving",
    "Individual protection equipment": "bypassing_safety_controls",
    "Pressurized Systems": "energy_isolation",
    "Liquid Metal": "hot_work",
    "Burn": "hot_work",
    # Everything else → Unmapped (including "Others", "Pressed", "Manual Tools",
    # "Chemical substances", "Venomous Animals", "Cut", "Projection", "Bees",
    # "remains of choco", "Poll", "Plates", "\nNot applicable", etc.)
    # These are legitimately unmappable to a single IOGP rule.
}

OSHA_SEVERE_INJURY_EVENT_MAP = {
    # Transportation incidents (2xx) → Driving
    20: "driving", 210: "driving", 211: "driving", 216: "driving", 219: "driving",
    227: "driving", 229: "driving", 240: "driving", 249: "driving",
    250: "driving", 253: "driving", 256: "driving", 257: "driving",
    258: "driving", 259: "driving", 260: "driving", 270: "driving",
    2312: "driving", 2313: "driving", 2321: "driving", 2322: "driving",
    2323: "driving", 2329: "driving",
    2410: "driving", 2411: "driving", 2412: "driving", 2413: "driving",
    2419: "driving", 2420: "driving", 2421: "driving", 2422: "driving",
    2423: "driving", 2429: "driving", 2430: "driving", 2431: "driving",
    2432: "driving", 2433: "driving", 2440: "driving", 2441: "driving",
    2442: "driving", 2443: "driving", 2449: "driving",
    2521: "driving", 2522: "driving",
    2610: "driving", 2611: "driving", 2612: "driving", 2613: "driving",
    2615: "driving", 2621: "driving", 2622: "driving", 2629: "driving",
    2630: "driving", 2631: "driving", 2632: "driving", 2634: "driving",
    2636: "driving", 2637: "driving", 2638: "driving", 2639: "driving",
    2710: "driving", 2711: "driving", 2712: "driving", 2713: "driving",
    2719: "driving", 2720: "driving",
    2151: "driving", 2153: "driving", 2312: "driving",
    # Fire/explosion (3xx) → Hot Work
    30: "hot_work", 310: "hot_work", 311: "hot_work", 312: "hot_work",
    313: "hot_work", 314: "hot_work", 315: "hot_work", 316: "hot_work",
    317: "hot_work", 319: "hot_work", 320: "hot_work", 321: "hot_work",
    322: "hot_work", 323: "hot_work", 329: "hot_work",
    # Falls from elevation (430-4xx) → Working at Height
    430: "working_at_height", 440: "working_at_height",
    45: "working_at_height",
    # Same-level falls (410-429) → Unmapped (not a height issue)
    40: UNMAPPED, 49: UNMAPPED, 410: UNMAPPED, 419: UNMAPPED,
    420: UNMAPPED, 422: UNMAPPED, 423: UNMAPPED, 424: UNMAPPED,
    429: UNMAPPED,
    # Exposure to electricity (510) → Energy Isolation
    510: "energy_isolation",
    # Temperature/chemical exposure → Unmapped (not cleanly one rule)
    50: UNMAPPED, 530: UNMAPPED, 531: UNMAPPED, 532: UNMAPPED,
    533: UNMAPPED, 534: UNMAPPED, 541: UNMAPPED, 542: UNMAPPED,
    550: UNMAPPED, 553: UNMAPPED, 558: UNMAPPED, 562: UNMAPPED,
    57: UNMAPPED,
    # Contact with objects/equipment (6xx) → Line of Fire
    620: "line_of_fire", 629: "line_of_fire", 630: "line_of_fire",
    639: "line_of_fire", 640: "energy_isolation", 642: "energy_isolation",
    643: "energy_isolation", 644: "energy_isolation", 649: "energy_isolation",
    651: "line_of_fire", 655: "line_of_fire", 656: "line_of_fire",
    659: "line_of_fire", 662: "line_of_fire", 664: "line_of_fire",
    669: "line_of_fire", 61: UNMAPPED, 69: UNMAPPED,
    # Overexertion (7xx) → Unmapped
    70: UNMAPPED, 710: UNMAPPED, 718: UNMAPPED, 719: UNMAPPED,
    730: UNMAPPED, 738: UNMAPPED, 74: UNMAPPED, 78: UNMAPPED,
    79: UNMAPPED,
    # Violence/animals → Unmapped
    129: UNMAPPED, 1110: UNMAPPED, 1111: UNMAPPED, 1112: UNMAPPED,
    1113: UNMAPPED, 1114: UNMAPPED, 1119: UNMAPPED, 1210: UNMAPPED,
    1211: UNMAPPED, 1212: UNMAPPED, 1213: UNMAPPED, 1214: UNMAPPED,
    1215: UNMAPPED, 1219: UNMAPPED, 1221: UNMAPPED, 1229: UNMAPPED,
    1310: UNMAPPED, 1311: UNMAPPED, 1312: UNMAPPED, 1313: UNMAPPED,
    1319: UNMAPPED, 1320: UNMAPPED, 1321: UNMAPPED, 1322: UNMAPPED,
    1323: UNMAPPED, 1324: UNMAPPED, 1329: UNMAPPED, 1381: UNMAPPED,
}

OSHA_HSE_EVENT_MAP = {
    "Fall (from elevation)": "working_at_height",
    "Struck-by": "line_of_fire",
    "Caught in or between": "energy_isolation",
    "Shock": "energy_isolation",
    "Fall (same level)": UNMAPPED,
    "Struck against": "line_of_fire",
    "Inhalation": UNMAPPED,
    "Other": UNMAPPED,
    "Bite/sting/scratch": UNMAPPED,
    "Rubbed/abraded": UNMAPPED,
    "Ingestion": UNMAPPED,
    "Rep. Motion/pressure": UNMAPPED,
    "Absorption": UNMAPPED,
    "Card-vascular/resp. fail.": UNMAPPED,
}

OSHA_HSE_HUMAN_FACTOR_MAP = {
    "Safety Devices Removed/Inoperable": "bypassing_safety_controls",
    "Lockout/Tagout Procedure Malfunction": "energy_isolation",
    "Insufficient /Lack/Engineering Controls": "bypassing_safety_controls",
    "Position Inappropriate For Task": "line_of_fire",
    "Mater-Handling Procedure Inappropriate": "safe_mechanical_lifting",
    "Equipment Inappropriate For Operation": "bypassing_safety_controls",
    "Insufficient /Lack/Written Work Practice Program": "work_authorisation",
    "Insufficient /Lack/Protective Work Clothing/Equipment": "bypassing_safety_controls",
    "Misjudgment, Hazardous Situation": "work_authorisation",
    "Other": UNMAPPED,
}

OSHA_HSE_ENV_FACTOR_MAP = {
    "Materials Handling Equip./Method": "safe_mechanical_lifting",
    "Work-Surface/Facility-Layout Condition": "line_of_fire",
    "Pinch Point Action": "line_of_fire",
    "Overhead Moving/Falling Object Action": "line_of_fire",
    "Catch Point/Puncture Action": "line_of_fire",
    "Flying Object Action": "line_of_fire",
    "Shear Point Action": "energy_isolation",
    "Weather, Earthquake, Etc.": UNMAPPED,
    "Gas/Vapor/Mist/Fume/Smoke/Dust": "confined_space",
    "Other": UNMAPPED,
}

PHMSA_FAILURE_CAUSE_MAP = {
    "Forklift Accident": "driving",
    "Human Error": "work_authorisation",
    "Improper Preparation for Transportation": "work_authorisation",
    "Inadequate Preparation for Transportation": "work_authorisation",
    "Loose Closure, Component, or Device": "bypassing_safety_controls",
    "Defective Component or Device": "bypassing_safety_controls",
    "Dropped": "line_of_fire",
    "Impact with Sharp or Protruding Object (e.g., nails)": "line_of_fire",
    "Abrasion": "line_of_fire",
    "Broken Component or Device": "bypassing_safety_controls",
    "Inadequate Blocking and Bracing": "work_authorisation",
    "Conveyer or Material Handling Equipment Mishap": "line_of_fire",
    "Too Much Weight on Package": "line_of_fire",
    "Freezing": UNMAPPED,
    "Valve Open": "energy_isolation",
}

# NIOSH event codes (48 codes mapped to 7 OIICS categories)
# These are numeric codes where the top-level code maps to a broad category.
# Only mapping codes where there's a genuinely confident IOGP match.
NIOSH_EVENT_MAP = {
    # 2xx = Transportation incidents → Driving
    20: "driving", 21: "driving", 22: "driving", 23: "driving",
    24: "driving", 25: "driving", 26: "driving",
    # 3xx = Fires and explosions → Hot Work
    30: "hot_work", 31: "hot_work", 32: "hot_work",
    # 4xx = Falls, slips, trips → Working at Height (elevation falls) or Unmapped
    40: UNMAPPED, 41: UNMAPPED, 42: UNMAPPED, 43: UNMAPPED,
    # 5xx = Exposure to harmful substances → Unmapped (not cleanly one rule)
    50: UNMAPPED, 51: UNMAPPED, 52: UNMAPPED, 53: UNMAPPED,
    54: UNMAPPED, 55: UNMAPPED,
    # 6xx = Contact with objects/equipment → Line of Fire
    60: "line_of_fire", 61: "line_of_fire", 62: "line_of_fire",
    63: "line_of_fire", 64: "energy_isolation", 65: "line_of_fire",
    66: "line_of_fire",
    # 7xx = Overexertion → Unmapped
    70: UNMAPPED, 71: UNMAPPED, 72: UNMAPPED, 73: UNMAPPED,
    # Other categories → Unmapped
    10: UNMAPPED, 11: UNMAPPED, 12: UNMAPPED, 13: UNMAPPED,
    # 99 = Other → Unmapped
    99: UNMAPPED,
}

# Master registry: dataset_name -> list of (source_column, mapping_dict, column_description)
SOURCE_CATEGORY_COLUMNS = [
    {
        "dataset": "ihm_stefanini_raw",
        "column": "Critical Risk",
        "description": "Hazard/risk category assigned by IHM Stefanini",
        "map": IHM_STEFANINI_MAP,
    },
    {
        "dataset": "severe_injury_reports_jboysen",
        "column": "Event",
        "description": "OSHA event type code (OIICS numeric)",
        "map": OSHA_SEVERE_INJURY_EVENT_MAP,
    },
    {
        "dataset": "severe_injury_reports_jboysen",
        "column": "EventTitle",
        "description": "OSHA event type narrative (OIICS-coded)",
        "map": {},  # Text descriptions — mapped via numeric Event column above
    },
    {
        "dataset": "severe_injury_reports_jboysen",
        "column": "NatureTitle",
        "description": "Nature of injury (OIICS-coded)",
        "map": {},  # Nature of injury is an outcome, not a precursor — all Unmapped
    },
    {
        "dataset": "severe_injury_reports_jboysen",
        "column": "SourceTitle",
        "description": "Source of injury (object/substance that caused harm)",
        "map": {},  # Source of injury is descriptive, not directly mappable — all Unmapped
    },
    {
        "dataset": "hse_abstracts_2015_2017_ruqaiyaship",
        "column": "Event type",
        "description": "OSHA HSE event type (e.g. Fall, Struck-by)",
        "map": OSHA_HSE_EVENT_MAP,
    },
    {
        "dataset": "hse_abstracts_2015_2017_ruqaiyaship",
        "column": "Human Factor",
        "description": "Human factor contributing to the incident",
        "map": OSHA_HSE_HUMAN_FACTOR_MAP,
    },
    {
        "dataset": "hse_abstracts_2015_2017_ruqaiyaship",
        "column": "Environmental Factor",
        "description": "Environmental factor contributing to the incident",
        "map": OSHA_HSE_ENV_FACTOR_MAP,
    },
    {
        "dataset": "phmsa_hazmat_incidents",
        "column": "Failure Cause Description",
        "description": "PHMSA hazmat incident failure cause",
        "map": PHMSA_FAILURE_CAUSE_MAP,
    },
    {
        "dataset": "niosh_injury_narrative_coding",
        "column": "event",
        "description": "NIOSH OIICS event code (numeric, top-level category)",
        "map": NIOSH_EVENT_MAP,
    },
]


def _strip_html(val):
    """Strip HTML tags from a value."""
    import re
    s = str(val).strip()
    cleaned = re.sub(r"<[^>]+>", "", s).strip()
    return cleaned if cleaned else s


def build_legacy_mapping():
    """Scan raw sources, extract distinct risk-category values, build mapping table."""
    print("\n" + "=" * 70)
    print("Step 2: Building taxonomy/legacy_term_mapping.csv")
    print("=" * 70)

    rows = []
    total_terms = 0
    mapped_count = 0
    unmapped_count = 0

    for source in SOURCE_CATEGORY_COLUMNS:
        dataset = source["dataset"]
        column = source["column"]
        mapping = source["map"]
        description = source["description"]

        print(f"\n  {dataset} -> {column}")
        print(f"    Description: {description}")

        # Get distinct values from the raw source
        distinct_values = _get_distinct_values(dataset, column)
        if distinct_values is None:
            print(f"    [SKIP] Column or dataset not found")
            continue

        print(f"    Distinct values: {len(distinct_values)}")

        for term in sorted(distinct_values, key=lambda x: str(x).lower()):
            term_str = str(term).strip()
            if term_str == "" or term_str.lower() == "nan":
                continue

            # Look up mapping — try string first, then numeric
            rule_id = mapping.get(term_str, UNMAPPED)
            if rule_id == UNMAPPED:
                try:
                    rule_id = mapping.get(int(float(term_str)), UNMAPPED)
                except (ValueError, TypeError):
                    pass

            # Determine confidence
            if rule_id == UNMAPPED:
                confidence = "n/a"
            else:
                # Check if multiple source terms map to the same rule
                # (still high confidence if it's a deliberate 1:1 in our dict)
                confidence = "high"

            notes = ""
            if rule_id == UNMAPPED:
                notes = f"No confident IOGP rule match for '{term_str}' in {column}"

            rows.append({
                "source_dataset": dataset,
                "source_term": term_str,
                "mapped_rule_id": rule_id,
                "mapping_confidence": confidence,
                "notes": notes,
            })

            total_terms += 1
            if rule_id == UNMAPPED:
                unmapped_count += 1
            else:
                mapped_count += 1

    # Write CSV
    with open(LEGACY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "source_term",
                                                "mapped_rule_id", "mapping_confidence", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written: {LEGACY_CSV}")
    print(f"  Total distinct terms mapped: {total_terms}")
    print(f"    Mapped to IOGP rule: {mapped_count}")
    print(f"    Unmapped: {unmapped_count}")
    print(f"    Mapping rate: {mapped_count/total_terms*100:.1f}% (higher is not better — "
          "many terms are legitimately unmappable)")

    # Summary by source
    print(f"\n  By source:")
    df = pd.DataFrame(rows)
    for dataset, group in df.groupby("source_dataset"):
        mapped = (group["mapped_rule_id"] != UNMAPPED).sum()
        total = len(group)
        print(f"    {dataset}: {mapped}/{total} mapped "
              f"({total - mapped} unmapped)")


def _get_distinct_values(dataset_name: str, column_name: str):
    """Read the raw source file and return sorted distinct values of the column."""
    # Path mappings (must match MANIFEST.csv local_path)
    PATH_MAP = {
        "ihm_stefanini_raw": "data/raw/other/ihm_stefanini_raw/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv",
        "severe_injury_reports_jboysen": "data/raw/osha/severe_injury_reports_jboysen/severe_injury_reports.csv",
        "hse_abstracts_2015_2017_ruqaiyaship": "data/raw/osha/hse_abstracts_2015_2017_ruqaiyaship/OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv",
        "phmsa_hazmat_incidents": None,  # special: directory of CSVs
        "niosh_injury_narrative_coding": "data/raw/other/niosh_injury_narrative_coding/train.csv",
    }

    local_path = PATH_MAP.get(dataset_name)
    full_path = PROJECT_ROOT / local_path if local_path else None

    if dataset_name == "phmsa_hazmat_incidents":
        # Sample from a few monthly files to get distinct values
        import glob
        phmsa_dir = PROJECT_ROOT / "data" / "phmsa" / "hazmat" / "repo" / "data" / "fetched"
        csv_files = sorted(glob.glob(str(phmsa_dir / "*.csv")))
        if not csv_files:
            return None
        # Read from 3 representative files (first, middle, last)
        sample_files = [csv_files[0], csv_files[len(csv_files) // 2], csv_files[-1]]
        all_values = set()
        for fp in sample_files:
            try:
                df = pd.read_csv(fp, encoding="utf-8", on_bad_lines="skip",
                                 usecols=[column_name], nrows=10000, low_memory=False)
                vals = df[column_name].dropna().unique()
                all_values.update(str(v).strip() for v in vals)
            except Exception:
                try:
                    df = pd.read_csv(fp, encoding="latin-1", on_bad_lines="skip",
                                     usecols=[column_name], nrows=10000, low_memory=False)
                    vals = df[column_name].dropna().unique()
                    all_values.update(str(v).strip() for v in vals)
                except Exception:
                    continue
        return sorted(all_values) if all_values else None

    if full_path is None or not full_path.exists():
        return None

    try:
        df = pd.read_csv(full_path, encoding="utf-8", on_bad_lines="skip",
                         usecols=[column_name])
    except UnicodeDecodeError:
        df = pd.read_csv(full_path, encoding="latin-1", on_bad_lines="skip",
                         usecols=[column_name])
    except Exception:
        return None

    if column_name not in df.columns:
        return None

    # For numeric columns (like NIOSH event codes), convert to int-like strings
    vals = df[column_name].dropna().unique()
    cleaned = set()
    for v in vals:
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            continue
        # Try to normalize numeric codes (e.g. 71.0 -> 71)
        try:
            f = float(s)
            if f == int(f):
                s = str(int(f))
        except ValueError:
            pass
        cleaned.add(s)

    return sorted(cleaned)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("SIF-Guard Phase 4: build_taxonomy_mapping.py")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Taxonomy dir: {TAXONOMY_DIR}")

    build_rules_csv()
    build_legacy_mapping()

    print("\n" + "=" * 70)
    print("Done.")
    print(f"  Rules CSV:      {RULES_CSV}")
    print(f"  Legacy mapping: {LEGACY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
