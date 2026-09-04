"""
analyze_real_synthetic_ratio.py

Read-only analysis of data/processed/safety_master.csv.
Prints a composition report to stdout. Does NOT write or modify any file.

Usage:
    python data/analyze_real_synthetic_ratio.py
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MASTER_CSV = Path(__file__).parent / "processed" / "safety_master.csv"

# Keywords that signal "plausibly oil & gas / hazmat / pipeline" relevance
OG_KEYWORDS = {"oil", "gas", "hazmat", "pipeline"}


def word_count(text: str) -> int:
    return len(text.split())


def is_og_relevant(domain: str) -> bool:
    domain_lower = domain.lower()
    return any(kw in domain_lower for kw in OG_KEYWORDS)


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{n / total * 100:.1f}%"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print()
    print(f"  --- {title} ---")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

rows_by_ds: dict[str, list[dict]] = defaultdict(list)

with MASTER_CSV.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rows_by_ds[row["source_dataset"]].append(row)

all_rows: list[dict] = [r for rows in rows_by_ds.values() for r in rows]
grand_total = len(all_rows)

dataset_order = sorted(rows_by_ds.keys(), key=lambda ds: -len(rows_by_ds[ds]))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("  SAFETY MASTER CSV — COMPOSITION REPORT")
print(f"  Source: {MASTER_CSV}")
print(f"  Total rows: {grand_total:,}")
print("=" * 70)

# ---------------------------------------------------------------------------
# Per-dataset sections
# ---------------------------------------------------------------------------

for ds in dataset_order:
    rows = rows_by_ds[ds]
    n = len(rows)

    section(f"Dataset: {ds}  ({n:,} rows, {fmt_pct(n, grand_total)} of total)")

    # 1. Row count already in section header — confirm here for clarity
    print(f"  Row count : {n:,}")
    print(f"  % of total: {fmt_pct(n, grand_total)}")

    # 2. industry_domain value counts
    subsection("industry_domain value counts")
    domain_counts = Counter(r["industry_domain"] for r in rows)
    for domain, cnt in domain_counts.most_common():
        print(f"    {domain:<40}  {cnt:>8,}  ({fmt_pct(cnt, n)})")

    # 3. is_synthetic breakdown
    subsection("is_synthetic breakdown")
    synth_counts = Counter(r["is_synthetic"] for r in rows)
    for val, cnt in sorted(synth_counts.items()):
        print(f"    {val:<10}  {cnt:>8,}  ({fmt_pct(cnt, n)})")
    if synth_counts.get("True", 0) == 0:
        print("    (entirely non-synthetic, as expected for this dataset)")
    elif synth_counts.get("False", 0) == 0:
        print("    (entirely synthetic)")
    else:
        print("    (mixed real + synthetic)")

    # 4. narrative_text word-count distribution (exclude is_empty_or_malformed rows)
    subsection("narrative_text word-count distribution  (is_empty_or_malformed == False)")
    good_rows = [r for r in rows if r["is_empty_or_malformed"] == "False"]
    excluded = n - len(good_rows)
    if good_rows:
        wc = [word_count(r["narrative_text"]) for r in good_rows]
        print(f"    Rows used  : {len(good_rows):,}  (excluded {excluded:,} malformed/empty)")
        print(f"    Mean       : {statistics.mean(wc):.1f} words")
        print(f"    Median     : {statistics.median(wc):.1f} words")
        print(f"    Min        : {min(wc):,} words")
        print(f"    Max        : {max(wc):,} words")
    else:
        print(f"    No usable rows (all {excluded:,} excluded as malformed/empty).")

    # 5. Oil & gas relevance proxy via industry_domain keywords
    subsection(
        'industry_domain relevance proxy  (contains "oil", "gas", "hazmat", or "pipeline")'
    )
    og_rows = [r for r in rows if is_og_relevant(r["industry_domain"])]
    other_rows = n - len(og_rows)
    print(f"    Plausibly oil & gas / hazmat / pipeline : {len(og_rows):>8,}  ({fmt_pct(len(og_rows), n)})")
    print(f"    Other / general industrial              : {other_rows:>8,}  ({fmt_pct(other_rows, n)})")
    if og_rows:
        og_domain_counts = Counter(r["industry_domain"] for r in og_rows)
        print("    Matching domain values:")
        for domain, cnt in og_domain_counts.most_common():
            print(f"      {domain:<40}  {cnt:>8,}")

# ---------------------------------------------------------------------------
# 6. Overall real vs synthetic summary
# ---------------------------------------------------------------------------

section("OVERALL: real vs synthetic")

total_real = sum(1 for r in all_rows if r["is_synthetic"] == "False")
total_synth = sum(1 for r in all_rows if r["is_synthetic"] == "True")

print(f"  Total real rows      (is_synthetic == False): {total_real:,}  ({fmt_pct(total_real, grand_total)})")
print(f"  Total synthetic rows (is_synthetic == True) : {total_synth:,}  ({fmt_pct(total_synth, grand_total)})")

subsection("Each source's contribution to the real total")
print(f"  {'Dataset':<45}  {'Real rows':>10}  {'% of real total':>16}")
print(f"  {'-'*45}  {'-'*10}  {'-'*16}")
for ds in dataset_order:
    real_in_ds = sum(1 for r in rows_by_ds[ds] if r["is_synthetic"] == "False")
    print(f"  {ds:<45}  {real_in_ds:>10,}  {fmt_pct(real_in_ds, total_real):>16}")

print()
print("  NOTE: Sampling decisions (what to keep / drop / downsample) are")
print("  intentionally excluded from this script — review numbers first.")
print()
