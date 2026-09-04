"""
seed_supabase.py -- bulk-insert reports from oil_safety_reports_merged.csv
Uses executemany for a single round-trip per batch instead of 1325 inserts.
Run from project root: python scripts/seed_supabase.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import psycopg2
import psycopg2.extras

url = os.environ["SUPABASE_DB_URL"]
conn = psycopg2.connect(url, connect_timeout=30)
conn.autocommit = False
cur = conn.cursor()

df = pd.read_csv("oil_safety_reports_merged.csv")
print(f"CSV: {len(df)} rows -- building batch ...")

rows = []
for _, row in df.iterrows():
    rl   = str(row.get("risk_level", "") or "")
    rl   = rl if rl in ("High", "Medium", "Low") else "Low"
    sif  = str(row.get("sif_potential", "") or "")
    crit = 1 if (sif == "Yes" and rl == "High") else 0
    rows.append((
        str(row.get("report_id", "") or ""),
        str(row.get("report_text", "") or ""),
        str(row.get("date", "") or ""),
        str(row.get("site", "") or ""),
        str(row.get("activity", "") or ""),
        str(row.get("report_type", "") or ""),
        str(row.get("source", "") or ""),
        sif,
        str(row.get("life_saving_rule", "") or ""),
        rl,
        crit,
    ))

print(f"Inserting {len(rows)} rows in one batch ...")
psycopg2.extras.execute_values(
    cur,
    """
    INSERT INTO reports
      (report_id, report_text, date, site, activity, report_type, source,
       sif_potential, life_saving_rule, risk_level, status, critical)
    VALUES %s
    ON CONFLICT (report_id) DO NOTHING
    """,
    [(
        r[0], r[1], r[2], r[3], r[4], r[5], r[6],
        r[7], r[8], r[9], "Closed", r[10],
    ) for r in rows],
    page_size=500,
)

conn.commit()
inserted = cur.rowcount
conn.close()
print(f"Done -- {inserted} rows inserted (duplicates skipped)")

# Verify
conn2 = psycopg2.connect(url, connect_timeout=15)
cur2  = conn2.cursor()
cur2.execute("SELECT COUNT(*) FROM reports")
total = cur2.fetchone()[0]
cur2.execute("SELECT COUNT(*) FROM reports WHERE sif_potential = 'Yes'")
sif_yes = cur2.fetchone()[0]
cur2.execute("SELECT COUNT(*) FROM reports WHERE critical = 1")
critical_count = cur2.fetchone()[0]
conn2.close()
print(f"Supabase reports: {total} total | {sif_yes} SIF-Yes | {critical_count} critical")
