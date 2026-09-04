"""
build_splits.py — Phase 10: train / validation / test split for the SIF classifier.

Reads  : data/processed/safety_master.csv
Writes : data/splits/train.csv
         data/splits/validation.csv
         data/splits/test.csv

Only rows from source_dataset == "sif_guard_seed" are touched.
All other rows in safety_master.csv are ignored completely.

Algorithm (follows the spec exactly):
  1. Filter to seed rows; drop is_empty_or_malformed == True.
  2. Assign a split_unit per row (duplicate-group id if non-zero, else record_id).
  3. Build test set from real, non-duplicated rows; sample to ~15% target.
  4. Assign remaining split_units to val (greedy, ~15% target) then train.
  5. Write the three CSVs with the full column set.
  6. Print verification report.

Run with:
    python data/build_splits.py
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT  = Path(__file__).parent
MASTER_CSV = DATA_ROOT / "processed" / "safety_master.csv"
SPLITS_DIR = DATA_ROOT / "splits"

TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV   = SPLITS_DIR / "validation.csv"
TEST_CSV  = SPLITS_DIR / "test.csv"

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(n: int, total: int) -> str:
    """'n (pct%)' string."""
    pct = n / total * 100 if total else 0.0
    return f"{n:>5,}  ({pct:5.1f}%)"


def bool_col(val: str) -> bool:
    """Parse 'True'/'False' string columns as used in safety_master.csv."""
    return val.strip().lower() == "true"


def write_split(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def section(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def print_split_stats(name: str, rows: list[dict], total: int) -> None:
    n = len(rows)
    yes  = sum(1 for r in rows if r["sif_potential"] == "Yes")
    no   = sum(1 for r in rows if r["sif_potential"] == "No")
    syn  = sum(1 for r in rows if bool_col(r["is_synthetic"]))
    real = n - syn
    print(f"  {name:<14}  rows: {fmt(n, total)}")
    print(f"               sif_potential  Yes: {fmt(yes, n)}   No: {fmt(no, n)}")
    print(f"               is_synthetic   True (synth): {fmt(syn, n)}   "
          f"False (real): {fmt(real, n)}")


# ---------------------------------------------------------------------------
# Step 1 — Load and filter
# ---------------------------------------------------------------------------

section("STEP 1 — Load sif_guard_seed rows from safety_master.csv")

fieldnames: list[str] = []
seed_rows:  list[dict] = []

with MASTER_CSV.open(newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    fieldnames = list(reader.fieldnames or [])
    for row in reader:
        if row["source_dataset"] == "sif_guard_seed":
            seed_rows.append(row)

print(f"  Rows with source_dataset == 'sif_guard_seed': {len(seed_rows):,}")

malformed = [r for r in seed_rows if bool_col(r["is_empty_or_malformed"])]
print(f"  Rows dropped (is_empty_or_malformed == True): {len(malformed):,}")

working: list[dict] = [r for r in seed_rows if not bool_col(r["is_empty_or_malformed"])]
N = len(working)
print(f"  Rows remaining after filter:                  {N:,}")

# ---------------------------------------------------------------------------
# Step 2 — Assign split_unit per row
# ---------------------------------------------------------------------------

section("STEP 2 — Assign split_units")

# split_unit = duplicate-group id (int) if non-zero, else record_id (str).
# We keep a mapping from unit → list of rows.
unit_to_rows: dict = defaultdict(list)

for row in working:
    dup_id = int(row["is_duplicate_group_id"])
    unit   = dup_id if dup_id != 0 else row["record_id"]
    unit_to_rows[unit].append(row)

singleton_units = [u for u in unit_to_rows if isinstance(u, str)]
group_units     = [u for u in unit_to_rows if isinstance(u, int)]

print(f"  Singleton split_units (own record_id): {len(singleton_units):,}")
print(f"  Duplicate-group split_units:           {len(group_units):,}")
print(f"  Total split_units:                     {len(unit_to_rows):,}")

# ---------------------------------------------------------------------------
# Step 3 — Build TEST set
# ---------------------------------------------------------------------------

section("STEP 3 — Build TEST set")

target_test = round(N * 0.15)
print(f"  Total working rows (N):  {N:,}")
print(f"  Target test size (~15%): {target_test:,}")

# Candidate pool: real AND not in any duplicate group at all.
test_candidate_units = [
    u for u in singleton_units
    if not bool_col(unit_to_rows[u][0]["is_synthetic"])
       and int(unit_to_rows[u][0]["is_duplicate_group_id"]) == 0
]
n_candidates = sum(len(unit_to_rows[u]) for u in test_candidate_units)

print(f"  Test candidate pool (real, no dup group): {n_candidates:,} rows "
      f"({len(test_candidate_units):,} units)")

if n_candidates < target_test:
    shortfall = target_test - n_candidates
    print()
    print("  !! WARNING: test candidate pool is SMALLER than the 15% target.")
    print(f"  !!   Target  : {target_test:,} rows")
    print(f"  !!   Pool    : {n_candidates:,} rows")
    print(f"  !!   Shortfall: {shortfall:,} rows")
    print("  !!   Using entire pool for test — NOT padding with synthetic or")
    print("  !!   duplicated rows. Test set will be smaller than targeted.")
    test_units_selected = test_candidate_units[:]
else:
    rng = random.Random(SEED)
    shuffled_candidates = test_candidate_units[:]
    rng.shuffle(shuffled_candidates)
    # Greedily pick units until we reach target_test rows.
    # Since all candidates are singletons (1 row per unit), this is a
    # straight random.sample — but we do it as a greedy walk for
    # consistency with the val-assignment step below.
    test_units_selected = []
    test_row_count = 0
    for u in shuffled_candidates:
        rows_in_unit = unit_to_rows[u]
        if test_row_count + len(rows_in_unit) <= target_test:
            test_units_selected.append(u)
            test_row_count += len(rows_in_unit)
        if test_row_count >= target_test:
            break

test_unit_set: set = set(test_units_selected)
test_rows: list[dict] = [r for u in test_units_selected for r in unit_to_rows[u]]

print(f"  Test rows selected: {len(test_rows):,}")

# ---------------------------------------------------------------------------
# Step 4 — Assign remaining split_units to val / train
# ---------------------------------------------------------------------------

section("STEP 4 — Assign remaining units to VALIDATION and TRAIN")

target_val = round(N * 0.15)
print(f"  Target validation size (~15% of N={N}): {target_val:,}")

remaining_units = [u for u in unit_to_rows if u not in test_unit_set]

rng2 = random.Random(SEED)
rng2.shuffle(remaining_units)

val_units:   list = []
train_units: list = []
val_row_count = 0

for u in remaining_units:
    rows_in_unit = unit_to_rows[u]
    if val_row_count < target_val:
        val_units.append(u)
        val_row_count += len(rows_in_unit)
    else:
        train_units.append(u)

val_unit_set:   set = set(val_units)
train_unit_set: set = set(train_units)

val_rows:   list[dict] = [r for u in val_units   for r in unit_to_rows[u]]
train_rows: list[dict] = [r for u in train_units for r in unit_to_rows[u]]

print(f"  Validation rows: {len(val_rows):,}")
print(f"  Train rows:      {len(train_rows):,}")

# ---------------------------------------------------------------------------
# Step 5 — Write CSVs
# ---------------------------------------------------------------------------

section("STEP 5 — Write output files")

write_split(TRAIN_CSV, fieldnames, train_rows)
write_split(VAL_CSV,   fieldnames, val_rows)
write_split(TEST_CSV,  fieldnames, test_rows)

print(f"  Wrote: {TRAIN_CSV}")
print(f"  Wrote: {VAL_CSV}")
print(f"  Wrote: {TEST_CSV}")

# ---------------------------------------------------------------------------
# Step 6 — Verification report
# ---------------------------------------------------------------------------

section("STEP 6 — VERIFICATION REPORT")

total_out = len(train_rows) + len(val_rows) + len(test_rows)

print(f"\n  Total rows in three splits: {total_out:,}  "
      f"(working set was {N:,})")
if total_out != N:
    print(f"  !! MISMATCH: {total_out} != {N}  — investigate before using these splits.")
else:
    print("  Row-count check: OK (all working rows accounted for)")

print()
print_split_stats("train",      train_rows, N)
print()
print_split_stats("validation", val_rows,   N)
print()
print_split_stats("test",       test_rows,  N)

# --- Overlap checks ---
print()
print("  Split-unit overlap checks (must all be 0):")

tv_overlap = train_unit_set & val_unit_set
te_overlap = train_unit_set & test_unit_set
ve_overlap = val_unit_set   & test_unit_set

def overlap_line(label: str, overlap: set) -> None:
    status = "OK" if len(overlap) == 0 else f"!! VIOLATION — {len(overlap)} shared units"
    print(f"    train ∩ {label:<12}: {len(overlap):>4} units  →  {status}")

overlap_line("validation", tv_overlap)
overlap_line("test       ", te_overlap)
overlap_line("val ∩ test ", ve_overlap)

# --- Test-set integrity ---
print()
test_synthetic_count = sum(1 for r in test_rows if bool_col(r["is_synthetic"]))
test_dup_count       = sum(1 for r in test_rows if int(r["is_duplicate_group_id"]) != 0)

def integrity_line(label: str, count: int, must_be_zero: bool = True) -> None:
    if must_be_zero:
        status = "OK" if count == 0 else f"!! VIOLATION — expected 0"
    else:
        status = "OK"
    print(f"    {label:<48}: {count:>5}  →  {status}")

print("  Test-set integrity checks (must all be 0):")
integrity_line("is_synthetic == True rows in test",      test_synthetic_count)
integrity_line("is_duplicate_group_id != 0 rows in test", test_dup_count)

# --- Summary line ---
all_clear = (
    total_out == N
    and len(tv_overlap) == 0
    and len(te_overlap) == 0
    and len(ve_overlap) == 0
    and test_synthetic_count == 0
    and test_dup_count == 0
)

print()
if all_clear:
    print("  ALL CHECKS PASSED — splits are ready for use.")
else:
    print("  !! ONE OR MORE CHECKS FAILED — do not use these splits.")

print()
