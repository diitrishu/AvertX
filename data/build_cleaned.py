#!/usr/bin/env python3
"""
build_cleaned.py — SIF-Guard Phase 4: Diagnostic Column Injection

Reads data/processed/master_raw.csv (860,440 rows) and writes
data/processed/safety_master.csv with three new boolean/id columns added:

  1. is_duplicate_group_id  — int group id for rows sharing normalized narrative
     text WITHIN the same source_dataset. 0 = unique (no dup within source).
  2. is_empty_or_malformed   — True if narrative_text is UNKNOWN, empty, or <5 chars.
  3. has_possible_pii        — True if narrative_text contains a simple regex match
     for email addresses or phone numbers.

STRICT RULES:
  - No rows are dropped.
  - No text is modified, normalized, rewritten, or redacted.
  - No other columns are touched.
  - Idempotent: re-running produces identical output.

Usage:
    py -3.12 data/build_cleaned.py
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "master_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "safety_master.csv"

# ---------------------------------------------------------------------------
# Normalisation helpers (for duplicate detection only — NOT applied to output)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.  Used only to
    compute is_duplicate_group_id — the original text is never altered."""
    t = str(text).lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# PII detection (very simple — email + phone patterns only)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\d)"                         # not preceded by digit
    r"(?:\+?\d{1,3}[-.\s]?)?"          # optional country code
    r"(?:\(?\d{2,4}\)?[-.\s]?)"        # area / region code
    r"\d{3,4}[-.\s]?\d{3,4}"           # local number
    r"(?!\d)"                           # not followed by digit
)


def _has_pii(text: str) -> bool:
    """Return True if text looks like it contains an email or phone number."""
    s = str(text)
    return bool(_EMAIL_RE.search(s) or _PHONE_RE.search(s))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("SIF-Guard Phase 4: build_cleaned.py")
    print("=" * 70)

    # --- Read ---
    if not INPUT_PATH.exists():
        print(f"[FATAL] Input not found: {INPUT_PATH}")
        sys.exit(1)

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    total = len(df)
    print(f"  Rows: {total:,}")
    print(f"  Columns: {len(df.columns)} (before adding diagnostic columns)")

    # --- 1. Duplicate group ids (within same source_dataset) ---
    print("\nComputing duplicate groups (within same source_dataset)...")
    df["_norm_text"] = df["narrative_text"].apply(_normalize_for_dedup)
    df["_group_key"] = df["source_dataset"] + "||" + df["_norm_text"]

    # Assign group ids: each unique (source, normalized_text) pair gets an int.
    # Only groups with >1 member get a nonzero id; singletons stay 0.
    group_codes = pd.Categorical(df["_group_key"]).codes  # fast hash-based
    # Count members per group
    group_sizes = pd.Series(group_codes).value_counts()
    # Singleton groups (size == 1) get id 0; real dup groups get 1..N
    singleton_groups = set(group_sizes[group_sizes == 1].index)
    # Build mapping: original code -> output group id
    code_to_group = {}
    next_id = 1
    for code in sorted(group_sizes.index):
        if code in singleton_groups:
            code_to_group[code] = 0
        else:
            code_to_group[code] = next_id
            next_id += 1

    df["is_duplicate_group_id"] = pd.Series(group_codes).map(code_to_group).values

    # Clean up temp columns
    df.drop(columns=["_norm_text", "_group_key"], inplace=True)

    dup_groups = (df["is_duplicate_group_id"] > 0).sum()
    n_groups = df["is_duplicate_group_id"].nunique() - (1 if 0 in df["is_duplicate_group_id"].values else 0)
    print(f"  Duplicate group rows: {dup_groups:,}")
    print(f"  Distinct duplicate groups: {n_groups:,}")

    # Per-source breakdown
    dup_by_source = (
        df[df["is_duplicate_group_id"] > 0]
        .groupby("source_dataset")["is_duplicate_group_id"]
        .nunique()
    )
    for src, ngrp in dup_by_source.items():
        print(f"    {src}: {ngrp} dup groups")

    # --- 2. Empty / malformed narrative ---
    print("\nFlagging empty/malformed narratives...")
    narrative = df["narrative_text"].fillna("")
    narrative_stripped = narrative.str.strip()
    df["is_empty_or_malformed"] = (
        (narrative_stripped == "")
        | (narrative_stripped.str.upper() == "UNKNOWN")
        | (narrative_stripped.str.len() < 5)
    )
    n_empty = df["is_empty_or_malformed"].sum()
    print(f"  Flagged: {n_empty:,} / {total:,} ({n_empty/total*100:.2f}%)")

    # --- 3. Possible PII ---
    print("\nScanning for possible PII (email/phone patterns)...")
    df["has_possible_pii"] = df["narrative_text"].apply(_has_pii)
    n_pii = df["has_possible_pii"].sum()
    print(f"  Flagged: {n_pii:,} / {total:,} ({n_pii/total*100:.2f}%)")

    # --- Write ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nWritten: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)} ({len(df.columns) - 29} new diagnostic columns)")

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total rows:             {total:>10,}")
    print(f"  Duplicate groups:       {n_groups:>10,}  ({dup_groups:,} rows tagged)")
    print(f"  Empty / malformed:      {n_empty:>10,}")
    print(f"  Possible PII:           {n_pii:>10,}")
    print(f"  New columns added:      is_duplicate_group_id, is_empty_or_malformed, has_possible_pii")
    print(f"\nDone. No rows dropped, no text modified.")


if __name__ == "__main__":
    main()
