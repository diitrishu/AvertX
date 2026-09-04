"""
extract_structured_fields.py -- Phase 7 (scoped)

Fills four fields in safety_master.csv using DETERMINISTIC keyword/phrase
matching against narrative_text:

    unsafe_act           -- what the person did wrong (active error)
    unsafe_condition     -- what the environment/equipment state was
    barrier              -- what control was present (or absent)
    potential_consequence -- what COULD have happened

SCOPE CONSTRAINTS (per PHASE9_DECISION.md):
  - sif_potential and life_saving_rule are NOT touched. Those are still
    UNKNOWN outside sif_guard_seed and remain gated on human review.
  - Fields that already have a non-UNKNOWN value are NEVER overwritten.
  - Only literal case-insensitive substring / regex matching on a fixed
    vocabulary. No LLM, no semantic similarity, no fuzzy matching.
    Every extraction is auditable to an exact quoted substring.

VOCABULARY SOURCE:
  taxonomy/life_saving_rules.csv -- hazard_examples and barrier_examples
  columns are used as the starting point for barrier vocabulary and as
  context for the other three fields. Additional terms are added below
  where they are common real-world safety phrases (all justified inline).

AMBIGUITY HANDLING:
  If a narrative matches terms from MULTIPLE different categories for the
  same field, the field is left UNKNOWN and the row is logged to
  data/processed/extraction_ambiguous.csv for human review.

OUTPUTS:
  data/processed/safety_master_extracted.csv
    -- same schema as safety_master.csv plus 8 companion columns:
       unsafe_act_evidence_span, unsafe_act_evidence_start, unsafe_act_evidence_end
       unsafe_condition_evidence_span, unsafe_condition_evidence_start, ...
       barrier_evidence_span, barrier_evidence_start, barrier_evidence_end
       potential_consequence_evidence_span, ...

  data/processed/extraction_ambiguous.csv
    -- record_id, field, matched_categories (pipe-separated), matched_spans

Run with:
    python data/extract_structured_fields.py
"""

import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(__file__).parent
MASTER_CSV  = DATA_ROOT / "processed" / "safety_master.csv"
TAXONOMY_CSV = DATA_ROOT.parent / "taxonomy" / "life_saving_rules.csv"

OUT_CSV     = DATA_ROOT / "processed" / "safety_master_extracted.csv"
AMBIG_CSV   = DATA_ROOT / "processed" / "extraction_ambiguous.csv"

UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Vocabulary definition
# ---------------------------------------------------------------------------
# Structure: { field_name: { category_label: [phrase, phrase, ...] } }
#
# Phrases are plain strings; matching is case-insensitive substring search.
# Where a phrase contains special regex characters it is re.escape()'d
# automatically by the matcher.
#
# JUSTIFICATION POLICY: every term must be a real-world safety phrase that
# would plausibly appear verbatim (or near-verbatim) in an incident narrative
# written by a safety officer, worker, or investigator. Terms invented purely
# for this project are not included. Plurals and minor variants of taxonomy
# terms are explicitly noted.
#
# The four fields target different aspects of an incident:
#   unsafe_act        -- worker action / behaviour error (active failure)
#   unsafe_condition  -- physical/equipment/environmental state (latent failure)
#   barrier           -- named safety control (from taxonomy barrier_examples
#                        plus common real-world variants)
#   potential_consequence -- outcome that COULD have occurred

VOCABULARY: dict[str, dict[str, list[str]]] = {

    # =========================================================================
    # UNSAFE_ACT
    # What the person DID that was unsafe. Active, volitional acts.
    # Sourced from: taxonomy hazard_examples (act-based ones), OSHA/IOGP
    # incident vocabulary, NIOSH common unsafe act phrases.
    # =========================================================================
    "unsafe_act": {

        "bypassed_safety_control": [
            # taxonomy: "Disabled safety interlock", "Overridden alarm",
            # "Removed machine guard"
            "bypassed",
            "bypass",
            "overrode",
            "overridden",
            "disabled the",
            "disabled interlock",
            "removed the guard",
            "removed machine guard",
            "removed guard",
            "silenced the alarm",
            "silenced alarm",
            "defeated the interlock",
            "defeated interlock",
            "jumped out",         # common phrase for bypassing interlock
        ],

        "entered_without_permit": [
            # taxonomy: "Confined space entry permit" (missing/bypassed)
            "entered without",
            "entered the confined space without",
            "entered without a permit",
            "entered without permit",
            "entered without gas test",
            "entered without testing",
            "no entry permit",
            "without completing entry",
        ],

        "worked_without_isolation": [
            # taxonomy: "Lockout/Tagout (LOTO)", "Isolation verification"
            "without isolating",
            "without isolation",
            "without locking out",
            "without lockout",
            "without loto",
            "bypassed the loto",
            "re-energized",
            "re energized",
            "started the equipment while",
            "energized the equipment while",
            "worked on energized",
            "worked without de-energizing",
            "worked without de energizing",
        ],

        "worked_without_permit": [
            # taxonomy: "Permit to Work (PTW)", "Work Authorisation" hazard
            "without a permit",
            "without a hot work permit",
            "without obtaining a permit",
            "without completing a permit",
            "without ptw",
            "no permit was obtained",
            "no permit to work",
            "started work without",
            "began work without",
            "commenced work without",
            "work started without",
            "work began without a",
        ],

        "worked_at_height_without_protection": [
            # taxonomy: "Fall from height", "Fall protection/harness" absent
            "without a harness",
            "without harness",
            "without fall protection",
            "without wearing a harness",
            "not wearing a harness",
            "no harness",
            "working at height without",
            "worked at height without",
            "climbed without",
        ],

        "stood_in_line_of_fire": [
            # taxonomy: "Line of Fire", "Positioning awareness"
            "stood in the line of fire",
            "standing in the line of fire",
            "positioned in the line of fire",
            "placed themselves in",
            "walked under the",
            "walked under a suspended",
            "stood beneath",
            "standing beneath",
            "positioned beneath",
            "under the load",
            "beneath the load",
        ],

        "drove_unsafely": [
            # taxonomy: Driving rule -- speed, fatigue, seatbelt
            "without a seatbelt",
            "without seatbelt",
            "not wearing a seatbelt",
            "not wearing seatbelt",
            "exceeded the speed limit",
            "speeding",
            "driving while fatigued",
            "driving fatigued",
            "driving while distracted",
            "used a mobile phone while driving",
            "used phone while driving",
        ],

        "lifted_without_plan": [
            # taxonomy: "Lift plan" absent, "Competent rigger/banksman"
            "without a lift plan",
            "without lift plan",
            "no banksman",
            "no competent banksman",
            "without a banksman",
            "without a rigger",
            "without a competent rigger",
            "rigged the load without",
        ],

        "conducted_hot_work_without_check": [
            # taxonomy: Hot Work -- permit and gas test required
            "welding without a permit",
            "welding without permit",
            "cutting without a permit",
            "cutting without permit",
            "hot work without a permit",
            "hot work without permit",
            "welding in a",           # "welding in a confined/hazardous area"
            "grinding without",
            "burning without",
            "commenced hot work",
        ],
    },

    # =========================================================================
    # UNSAFE_CONDITION
    # State of equipment, environment, or system -- not what a person did,
    # but what was WRONG before/during the incident.
    # =========================================================================
    "unsafe_condition": {

        "defective_equipment": [
            "defective",
            "faulty equipment",
            "faulty",
            "damaged equipment",
            "worn equipment",
            "malfunctioning",
            "not functioning properly",
            "not functioning correctly",
            "failed equipment",
            "equipment failure",
            "equipment malfunction",
            "out of service",
            "unserviceable",
        ],

        "missing_or_inadequate_guarding": [
            "no guard",
            "no machine guard",
            "missing guard",
            "guard removed",
            "guard was missing",
            "guard was not in place",
            "unguarded",
            "guarding was absent",
            "guarding removed",
            "machine guard was not",
            "no guarding",
        ],

        "slippery_or_unstable_surface": [
            "slippery",
            "wet floor",
            "wet surface",
            "oily floor",
            "oily surface",
            "icy surface",
            "icy ground",
            "unstable surface",
            "unstable ground",
            "uneven ground",
            "uneven surface",
        ],

        "inadequate_lighting": [
            "poor lighting",
            "inadequate lighting",
            "insufficient lighting",
            "no lighting",
            "dim lighting",
            "low visibility",
            "reduced visibility",
        ],

        "hazardous_atmosphere": [
            # taxonomy: confined space hazards
            "toxic atmosphere",
            "oxygen deficient",
            "oxygen-deficient",
            "lack of oxygen",
            "low oxygen",
            "flammable atmosphere",
            "flammable vapour",
            "flammable vapor",
            "flammable gas",
            "toxic gas",
            "gas present",
            "gas detected",
            "gas leak",
            "gas release",
            "vapour present",
            "vapor present",
            "fumes present",
            "hazardous atmosphere",
        ],

        "unsecured_or_unstable_load": [
            # taxonomy: line of fire, safe mechanical lifting
            "unsecured load",
            "unsecured object",
            "unstable load",
            "unstable scaffold",
            "scaffold was unstable",
            "load was unsecured",
            "improperly rigged",
            "rigged incorrectly",
            "incorrectly rigged",
            "rigging failed",
            "rigging failure",
            "suspended load",
            "load suspended",
        ],

        "energized_equipment": [
            # taxonomy: energy isolation
            "still energized",
            "remained energized",
            "found energized",
            "was still live",
            "live circuit",
            "live wire",
            "live cable",
            "live electrical",
            "energized circuit",
            "energized line",
            "stored energy",
            "residual energy",
            "pressurized line",
            "pressurized pipe",
            "pressurized system",
            "pressure still present",
        ],

        "inadequate_signage_or_demarcation": [
            # taxonomy: signage, barricading
            "no signage",
            "inadequate signage",
            "faded signage",
            "missing signage",
            "no barricade",
            "no barricading",
            "inadequate barricading",
            "barricade missing",
            "barricade was not in place",
            "no exclusion zone",
            "exclusion zone not established",
            "area not barricaded",
        ],

        "poor_housekeeping": [
            "poor housekeeping",
            "clutter",
            "cluttered",
            "obstructed walkway",
            "blocked walkway",
            "blocked exit",
            "blocked aisle",
            "debris on the floor",
            "debris on floor",
        ],
    },

    # =========================================================================
    # BARRIER
    # Named safety control that was present (in any state).
    # Values taken directly from taxonomy barrier_examples plus documented
    # real-world variants and synonyms used in OSHA/IOGP literature.
    # Category labels match the taxonomy values used in sif_guard_seed so
    # any new extractions are consistent with existing data.
    # =========================================================================
    "barrier": {

        # ---- energy_isolation barriers ----
        "Lockout/Tagout (LOTO)": [
            # taxonomy exact; common variants
            "lockout/tagout",
            "lock out tag out",
            "lock-out tag-out",
            "lockout tagout",
            "loto",
            "loto tag",
            "loto lock",
            "locked out",
            "tagged out",
            "lock out",
            "tag out",
            "lockout",
            "tagout",
        ],
        "Isolation verification": [
            "isolation verification",
            "isolation confirmed",
            "verified isolation",
            "isolation check",
            "energy verified",
            "de-energized and verified",
            "deenergized and verified",
        ],
        "Zero-energy check": [
            "zero-energy check",
            "zero energy check",
            "zero energy state",
            "zero-energy state",
            "zero energy verified",
        ],
        "Isolation certificate": [
            "isolation certificate",
            "isolation cert",
        ],

        # ---- hot_work barriers ----
        "Hot work permit": [
            "hot work permit",
            "hotwork permit",
            "hot-work permit",
        ],
        "Fire watch": [
            "fire watch",
            "firewatch",
            "fire watcher",
        ],
        "Gas testing (flammable atmosphere)": [
            "gas testing",
            "gas test",
            "flammable gas test",
            "gas tested",
            "atmosphere tested",
            "atmosphere test",
            "tested for gas",
            "tested for flammable",
            "gas monitor",
            "gas monitoring",
            "gas detector",
        ],
        "Fire extinguishing equipment on site": [
            "fire extinguisher",
            "fire extinguishers",
            "fire suppression",
            "fire fighting equipment",
            "firefighting equipment",
        ],

        # ---- confined_space barriers ----
        "Confined space entry permit": [
            "confined space permit",
            "confined space entry permit",
            "entry permit",
            "cs permit",
        ],
        "Continuous atmosphere monitoring": [
            "continuous atmosphere monitoring",
            "continuous gas monitoring",
            "continuous monitoring",
            "atmospheric monitoring",
        ],
        "Rescue plan/standby": [
            "rescue plan",
            "rescue standby",
            "standby rescue",
            "attendant standing by",
            "rescue team on standby",
            "rescue team standing by",
        ],

        # ---- working_at_height barriers ----
        "Fall protection/harness": [
            "fall protection",
            "full body harness",
            "safety harness",
            "harness",
            "fall arrest",
            "personal fall arrest",
            "lanyard",
        ],
        "Guardrails": [
            "guardrail",
            "guardrails",
            "guard rail",
            "guard rails",
            "handrail",
            "handrails",
        ],
        "Scaffold inspection tag": [
            "scaffold inspection tag",
            "scaffold tag",
            "scaffolding tag",
            "scaffold inspection",
        ],
        "Anchor point verification": [
            "anchor point",
            "anchor points",
            "tie-off point",
            "tie off point",
        ],

        # ---- line_of_fire barriers ----
        "Barricading/exclusion zone": [
            "barricade",
            "barricades",
            "barricading",
            "exclusion zone",
            "exclusion zones",
            "restricted area",
            "keep-out zone",
            "keep out zone",
            "exclusion area",
        ],
        "Suspended load control": [
            "suspended load control",
            "tag line",
            "tagline",
            "tag lines",
            "controlled load",
        ],
        "Positioning awareness": [
            "positioning awareness",
            "awareness of position",
            "line of fire awareness",
        ],
        "Signage": [
            "safety sign",
            "safety signs",
            "warning sign",
            "warning signs",
            "hazard sign",
            "hazard signs",
            "signage",
            "caution tape",
            "caution sign",
        ],

        # ---- safe_mechanical_lifting barriers ----
        "Lift plan": [
            "lift plan",
            "lifting plan",
            "crane lift plan",
            "rigging plan",
        ],
        "Rigging inspection": [
            "rigging inspection",
            "rigging inspected",
            "sling inspection",
            "slings inspected",
        ],
        "Competent rigger/banksman": [
            "competent rigger",
            "qualified rigger",
            "certified rigger",
            "banksman",
            "banks man",
            "signal person",
            "competent banksman",
        ],
        "Load chart compliance": [
            "load chart",
            "load charts",
            "crane capacity",
            "rated capacity",
            "safe working load",
        ],

        # ---- work_authorisation barriers ----
        "Permit to Work (PTW)": [
            "permit to work",
            "work permit",
            "working permit",
        ],
        "Risk assessment / JSA": [
            "risk assessment",
            "job safety analysis",
            "job hazard analysis",
            "toolbox talk",
            "task hazard analysis",
            "risk assess",
        ],
        "Supervisory authorization": [
            "supervisory authorization",
            "supervisor authorization",
            "supervisor approval",
            "supervisor authoriz",
            "authorised by supervisor",
            "authorized by supervisor",
            "management authoriz",
            "management approval",
        ],

        # ---- bypassing_safety_controls barriers ----
        "Interlock system": [
            "interlock",
            "interlocks",
            "interlocking",
            "safety interlock",
            "safety interlocks",
        ],
        "Safety guard": [
            "machine guard",
            "safety guard",
            "safety guards",
            "machine guards",
            "point of operation guard",
        ],
        "Alarm system": [
            "alarm system",
            "alarm systems",
            "safety alarm",
            "process alarm",
            "gas alarm",
            "fire alarm",
            "emergency alarm",
            "alarm activated",
            "alarm triggered",
        ],
        "Permit conditions": [
            "permit conditions",
            "conditions of the permit",
            "conditions of permit",
        ],

        # ---- driving barriers ----
        "Seatbelt use": [
            "seatbelt",
            "seat belt",
            "safety belt",
        ],
        "Speed compliance": [
            "speed limit",
            "speed compliance",
            "speed restriction",
        ],
        "Journey management plan": [
            "journey management",
            "journey plan",
            "travel plan",
            "trip plan",
        ],
        "Fatigue management": [
            "fatigue management",
            "fatigue risk",
            "fatigue assessment",
            "hours of service",
            "driving hours",
            "rest requirement",
        ],
    },

    # =========================================================================
    # POTENTIAL_CONSEQUENCE
    # What COULD have happened (counterfactual harm), regardless of actual
    # outcome. Taken from taxonomy hazard_examples; these are the named
    # potential outcomes used across IOGP / OSHA literature.
    # =========================================================================
    "potential_consequence": {

        "fatality_or_life_threatening_injury": [
            "fatal",
            "fatality",
            "killed",
            "death",
            "life-threatening",
            "life threatening",
            "could have been killed",
            "could have been fatal",
            "could have resulted in death",
            "could have resulted in fatality",
            "near fatal",
        ],

        "electrocution_or_electrical_shock": [
            # taxonomy: energy_isolation hazard
            "electrocution",
            "electrocuted",
            "electrical shock",
            "electric shock",
            "shock",           # kept specific: only when near electrical context
            "contacted energized",
            "contact with energized",
            "contacted live",
            "contact with live",
        ],

        "fire_or_explosion": [
            # taxonomy: hot_work hazard
            "fire",
            "explosion",
            "ignition",
            "ignited",
            "caught fire",
            "deflagration",
            "detonation",
            "flash fire",
            "flashback",
        ],

        "fall_from_height": [
            # taxonomy: working_at_height
            "fall from height",
            "fall from elevation",
            "fell from height",
            "fell from elevation",
            "falling from height",
            "could have fallen",
            "could have resulted in a fall",
        ],

        "struck_by_object": [
            # taxonomy: line_of_fire, safe_mechanical_lifting
            "struck by",
            "struck by a",
            "struck by falling",
            "struck by moving",
            "hit by",
            "hit by a falling",
            "hit by falling",
            "dropped object",
            "falling object",
            "load drop",
            "load dropped",
        ],

        "engulfment_or_asphyxiation": [
            # taxonomy: confined_space
            "engulfment",
            "engulfed",
            "asphyxiation",
            "asphyxiated",
            "suffocation",
            "suffocated",
            "oxygen depletion",
            "overcome by gas",
            "overcome by fumes",
            "inhaled toxic",
        ],

        "chemical_exposure_or_burn": [
            "chemical burn",
            "chemical exposure",
            "chemical splash",
            "acid burn",
            "caustic burn",
            "corrosive",
            "toxic exposure",
            "toxic chemical",
            "hazardous substance",
            "hazmat release",
            "hazardous material release",
            "chemical release",
            "spill",
        ],

        "crush_or_entrapment": [
            "crushed",
            "crush injury",
            "entrapment",
            "trapped",
            "caught between",
            "caught in",
            "pinched",
            "run over",
        ],

        "vehicle_collision": [
            # taxonomy: driving
            "vehicle collision",
            "collision",
            "rollover",
            "roll over",
            "vehicle rollover",
            "pedestrian strike",
            "struck a pedestrian",
            "struck pedestrian",
            "vehicle struck",
        ],
    },
}


# ---------------------------------------------------------------------------
# Evidence-span companion column names
# ---------------------------------------------------------------------------

EVIDENCE_COLS = {}
for field in ("unsafe_act", "unsafe_condition", "barrier", "potential_consequence"):
    EVIDENCE_COLS[field] = {
        "span":  f"{field}_evidence_span",
        "start": f"{field}_evidence_start",
        "end":   f"{field}_evidence_end",
    }


# ---------------------------------------------------------------------------
# Pre-compile all patterns
# ---------------------------------------------------------------------------
# compiled_vocab: { field: { category: [ compiled_regex, ... ] } }

def _compile_vocab(vocab: dict) -> dict:
    compiled = {}
    for field, categories in vocab.items():
        compiled[field] = {}
        for cat, phrases in categories.items():
            compiled[field][cat] = [
                re.compile(re.escape(p), re.IGNORECASE)
                for p in phrases
            ]
    return compiled


COMPILED = _compile_vocab(VOCABULARY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_vocabulary() -> None:
    """Print the full vocabulary to stdout so it is reviewable."""
    section("VOCABULARY IN USE (full list)")
    for field, categories in VOCABULARY.items():
        print(f"\n  [{field}]")
        for cat, phrases in categories.items():
            print(f"    {cat}:")
            for p in phrases:
                print(f"      - {repr(p)}")


def match_field(
    text: str,
    field: str,
) -> list[tuple[str, re.Match]]:
    """
    Return all (category, first_match) pairs where any phrase for that
    category matches in text.  At most one entry per category (the first
    match found for that category is returned).
    """
    hits: list[tuple[str, re.Match]] = []
    for cat, patterns in COMPILED[field].items():
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                hits.append((cat, m))
                break   # first matching phrase for this category is enough
    return hits


def should_skip(row: dict, field: str) -> bool:
    """Return True if the field already has a real (non-UNKNOWN) value."""
    val = row.get(field, "").strip()
    return val != UNKNOWN and val != ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # Print vocabulary first so a reviewer can check it before the run
    print_vocabulary()

    section("STARTING EXTRACTION")
    print(f"  Input : {MASTER_CSV}")
    print(f"  Output: {OUT_CSV}")
    print(f"  Ambiguous log: {AMBIG_CSV}")
    print()
    print("  SCOPE: fills unsafe_act, unsafe_condition, barrier,")
    print("  potential_consequence ONLY.  sif_potential and life_saving_rule")
    print("  are never touched.  Fields with existing non-UNKNOWN values are")
    print("  never overwritten.  All extractions are keyword-match only.")

    # ------------------------------------------------------------------
    # Stat counters
    # ------------------------------------------------------------------
    TARGET_FIELDS = ("unsafe_act", "unsafe_condition", "barrier",
                     "potential_consequence")

    # counts[field][source_dataset] = {"matched", "ambiguous", "unknown", "skipped"}
    counts: dict[str, dict[str, dict[str, int]]] = {
        f: defaultdict(lambda: {"matched": 0, "ambiguous": 0,
                                "unknown": 0, "skipped": 0})
        for f in TARGET_FIELDS
    }
    ambiguous_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Read input columns
    # ------------------------------------------------------------------
    with MASTER_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames_in = list(reader.fieldnames or [])

    # Build output fieldnames: original + 12 evidence-span columns
    span_cols = []
    for f in TARGET_FIELDS:
        ev = EVIDENCE_COLS[f]
        span_cols += [ev["span"], ev["start"], ev["end"]]
    fieldnames_out = fieldnames_in + span_cols

    # ------------------------------------------------------------------
    # Streaming pass: read, extract, write
    # ------------------------------------------------------------------
    section("PROCESSING")
    print("  Reading and writing row by row (streaming) ...")

    rows_processed = 0
    with (
        MASTER_CSV.open(newline="", encoding="utf-8") as fh_in,
        OUT_CSV.open("w", newline="", encoding="utf-8") as fh_out,
    ):
        reader = csv.DictReader(fh_in)
        writer = csv.DictWriter(fh_out, fieldnames=fieldnames_out)
        writer.writeheader()

        for row in reader:
            rows_processed += 1
            if rows_processed % 100_000 == 0:
                print(f"    ... {rows_processed:,} rows processed")

            ds   = row["source_dataset"]
            text = row["narrative_text"]

            # Skip entirely-unknown narratives (PHMSA blanks, etc.)
            text_usable = text.strip().upper() != UNKNOWN and text.strip() != ""

            # Add blank evidence-span columns for every row first
            for f in TARGET_FIELDS:
                ev = EVIDENCE_COLS[f]
                row[ev["span"]]  = ""
                row[ev["start"]] = ""
                row[ev["end"]]   = ""

            if text_usable:
                for field in TARGET_FIELDS:
                    ev = EVIDENCE_COLS[field]

                    # Never overwrite existing real values
                    if should_skip(row, field):
                        counts[field][ds]["skipped"] += 1
                        continue  # leave evidence cols blank for this field

                    hits = match_field(text, field)

                    if not hits:
                        counts[field][ds]["unknown"] += 1
                        # leave field as UNKNOWN, evidence cols blank

                    elif len(hits) == 1:
                        # Single category match -- confident extraction
                        cat, m = hits[0]
                        counts[field][ds]["matched"] += 1
                        row[field]         = cat
                        row[ev["span"]]    = m.group(0)
                        row[ev["start"]]   = str(m.start())
                        row[ev["end"]]     = str(m.end())

                    else:
                        # Multiple categories matched -- ambiguous, do not fill
                        # de-duplicate: a category that matches in multiple
                        # patterns still counts as one category
                        unique_cats = list(dict.fromkeys(c for c, _ in hits))
                        if len(unique_cats) == 1:
                            # Same category matched via multiple phrases -- fine
                            cat, m = hits[0]
                            counts[field][ds]["matched"] += 1
                            row[field]         = cat
                            row[ev["span"]]    = m.group(0)
                            row[ev["start"]]   = str(m.start())
                            row[ev["end"]]     = str(m.end())
                        else:
                            counts[field][ds]["ambiguous"] += 1
                            ambiguous_rows.append({
                                "record_id":         row["record_id"],
                                "source_dataset":    ds,
                                "field":             field,
                                "matched_categories": "|".join(unique_cats),
                                "matched_spans": "|".join(
                                    repr(m.group(0)) for _, m in hits[:len(unique_cats)]
                                ),
                            })
            else:
                # Unusable narrative -- count as unknown for all fields
                for field in TARGET_FIELDS:
                    if not should_skip(row, field):
                        counts[field][ds]["unknown"] += 1

            writer.writerow(row)

    print(f"    ... {rows_processed:,} rows processed (done)")

    # ------------------------------------------------------------------
    # Write ambiguous log
    # ------------------------------------------------------------------
    section("WRITING AMBIGUOUS LOG")
    ambig_fields = ["record_id", "source_dataset", "field",
                    "matched_categories", "matched_spans"]
    with AMBIG_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ambig_fields)
        writer.writeheader()
        writer.writerows(ambiguous_rows)
    print(f"  Wrote {len(ambiguous_rows):,} ambiguous rows to {AMBIG_CSV}")

    # ------------------------------------------------------------------
    # Match-rate report
    # ------------------------------------------------------------------
    section("MATCH-RATE REPORT")

    ALL_DS_KEY = "_ALL_SOURCES"
    for field in TARGET_FIELDS:
        print(f"\n  [{field}]")
        print(f"  {'Source dataset':<45}  {'Matched':>8}  {'Ambiguous':>10}  "
              f"{'Skipped':>8}  {'Unknown':>8}  {'Total':>8}")
        print(f"  {'-'*45}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

        totals = {"matched": 0, "ambiguous": 0, "skipped": 0, "unknown": 0}
        for ds in sorted(counts[field].keys()):
            c = counts[field][ds]
            total_ds = c["matched"] + c["ambiguous"] + c["skipped"] + c["unknown"]
            print(f"  {ds:<45}  {c['matched']:>8,}  {c['ambiguous']:>10,}  "
                  f"{c['skipped']:>8,}  {c['unknown']:>8,}  {total_ds:>8,}")
            for k in totals:
                totals[k] += c[k]

        grand_total = sum(totals.values())
        print(f"  {'TOTAL':<45}  {totals['matched']:>8,}  "
              f"{totals['ambiguous']:>10,}  {totals['skipped']:>8,}  "
              f"{totals['unknown']:>8,}  {grand_total:>8,}")
        if grand_total > 0:
            match_pct  = totals["matched"]   / grand_total * 100
            ambig_pct  = totals["ambiguous"] / grand_total * 100
            print(f"  Match rate: {match_pct:.2f}%   Ambiguous rate: {ambig_pct:.2f}%")

    section("DONE")
    print(f"  Output CSV : {OUT_CSV}")
    print(f"  Ambiguous  : {AMBIG_CSV}  ({len(ambiguous_rows):,} rows)")
    print()
    print("  Reminder: sif_potential and life_saving_rule were NOT touched.")
    print("  All extractions are keyword-match only and are auditable via")
    print("  the *_evidence_span / *_evidence_start / *_evidence_end columns.")
    print()


if __name__ == "__main__":
    main()
