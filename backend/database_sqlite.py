"""
SQLite database for persisting analyzed reports, users, and audit history.
Seeds reports from oil_safety_reports_merged.csv on first run and seeds
two bootstrap Admin accounts (see DEFAULT_ADMINS below).
"""

import os, sqlite3, json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sif_database.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "oil_safety_reports_merged.csv")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE,
            report_text TEXT,
            date TEXT,
            site TEXT,
            activity TEXT,
            report_type TEXT,
            source TEXT,
            sif_potential TEXT,
            life_saving_rule TEXT,
            confidence REAL,
            risk_score INTEGER,
            rule_confidence REAL,
            risk_level TEXT,
            top_phrases TEXT,
            analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Users ────────────────────────────────────────────────────────
    # Roles: Reporter (default for every signup), Supervisor, HSE, Admin.
    # Admin is never self-registerable -- see _seed_admins below.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Reporter',
            site TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Audit log — every safety-critical change is recorded here ─────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id INTEGER,
            actor_name TEXT,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            field TEXT,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # ── Migrate older report DBs that predate the lifecycle columns ───
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(reports)").fetchall()}
    migrations = {
        "reporter_id": "INTEGER",
        "status": "TEXT DEFAULT 'Submitted'",
        "critical": "INTEGER DEFAULT 0",
        # Phase 16 -- explainability metadata (Phase 12/13 output contracts).
        # Additive only: existing rows get NULL/default here, nothing above
        # is touched, and report_text (the original submitted narrative) is
        # never rewritten by this migration.
        "sif_reasons": "TEXT",
        "sif_model_name": "TEXT",
        "sif_model_version": "TEXT",
        "sif_decision_thresholds": "TEXT",
        "sif_prediction_timestamp": "TEXT",
        "rule_needs_review": "INTEGER DEFAULT 0",
        "rule_review_reason": "TEXT",
        "rule_runner_up": "TEXT",
        "rule_runner_up_confidence": "REAL",
        "rule_model_name": "TEXT",
        "rule_model_version": "TEXT",
        "rule_prediction_timestamp": "TEXT",
    }
    for col, coltype in migrations.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE reports ADD COLUMN {col} {coltype}")
    conn.commit()

    # Seed reports if empty
    count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    if count == 0:
        _seed_from_csv(conn)

    _seed_admins(conn)
    conn.close()

def _seed_from_csv(conn):
    """Seed database from oil_safety_reports_merged.csv."""
    import pandas as pd
    if not os.path.exists(CSV_PATH):
        print("[db] CSV not found, skipping seed.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"[db] Seeding {len(df)} reports from CSV...")

    for _, row in df.iterrows():
        risk_level = "High" if row.get("risk_level", "") == "High" else (
            "Medium" if row.get("risk_level", "") == "Medium" else "Low"
        )
        sif_potential = row.get("sif_potential", "")
        # Historical/seed reports are treated as already closed out --
        # they represent the organization's existing record, not new
        # live submissions awaiting review.
        status = "Closed"
        critical = 1 if (sif_potential == "Yes" and risk_level == "High") else 0
        try:
            conn.execute("""
                INSERT OR IGNORE INTO reports
                (report_id, report_text, date, site, activity, report_type, source,
                 sif_potential, life_saving_rule, risk_level, status, critical)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("report_id", ""),
                row.get("report_text", ""),
                str(row.get("date", "")),
                row.get("site", ""),
                row.get("activity", ""),
                row.get("report_type", ""),
                row.get("source", ""),
                sif_potential,
                row.get("life_saving_rule", ""),
                risk_level,
                status,
                critical,
            ))
        except Exception:
            pass
    conn.commit()
    print(f"[db] Seeded {len(df)} reports.")

# ── Bootstrap admin accounts ────────────────────────────────────────
# Real-world rule: Admin is the role that can grant safety-critical
# authority (Supervisor/HSE), so it is never a self-service signup
# choice. At least two named accounts exist so the org is never locked
# out if one admin is unavailable. These are bootstrap credentials --
# change them after first login in a real deployment.
DEFAULT_ADMINS = [
    {"name": "System Admin 1", "email": "admin1@email.com", "password": "admin123",
     "legacy_emails": ["admin1@oil-hsse.local"]},
    {"name": "System Admin 2", "email": "admin2@oil-hsse.local", "password": "ChangeMe@Admin2",
     "legacy_emails": []},
]

def _seed_admins(conn):
    from backend.auth import hash_password
    for a in DEFAULT_ADMINS:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (a["email"],)).fetchone()
        if existing:
            continue

        # If this bootstrap admin previously existed under an older
        # placeholder email (e.g. we renamed admin1@oil-hsse.local to
        # something easier to type), rename that row in place instead of
        # creating a second Admin account -- keeps the same user id, role
        # history and audit trail rather than orphaning the old one.
        legacy_row = None
        for legacy_email in a.get("legacy_emails", []):
            legacy_row = conn.execute("SELECT id FROM users WHERE email=?", (legacy_email,)).fetchone()
            if legacy_row:
                break
        if legacy_row:
            conn.execute(
                "UPDATE users SET email=?, password_hash=?, name=? WHERE id=?",
                (a["email"], hash_password(a["password"]), a["name"], legacy_row["id"]),
            )
            log_audit(conn, None, "System", "credentials_changed", "user", legacy_row["id"],
                      field="email", old_value=legacy_email, new_value=a["email"],
                      reason="Bootstrap admin renamed to a simpler dev login")
            print(f"[db] Renamed bootstrap admin -> {a['email']}")
            continue

        conn.execute("""
            INSERT INTO users (name, email, password_hash, role, status)
            VALUES (?, ?, ?, 'Admin', 'active')
        """, (a["name"], a["email"], hash_password(a["password"])))
        print(f"[db] Seeded admin account: {a['email']}")
    conn.commit()

# ── Users ────────────────────────────────────────────────────────────
def create_user(conn, name, email, password_hash, role="Reporter", site=""):
    cur = conn.execute("""
        INSERT INTO users (name, email, password_hash, role, site, status)
        VALUES (?, ?, ?, ?, ?, 'active')
    """, (name, email, password_hash, role, site))
    conn.commit()
    return cur.lastrowid

def get_user_by_email(conn, email):
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None

def get_user_by_id(conn, user_id):
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None

def list_users(conn):
    rows = conn.execute("SELECT id, name, email, role, site, status, created_at FROM users ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def update_user_role(conn, user_id, new_role):
    conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()

def update_user_status(conn, user_id, new_status):
    conn.execute("UPDATE users SET status=? WHERE id=?", (new_status, user_id))
    conn.commit()

# ── Audit log ────────────────────────────────────────────────────────
def log_audit(conn, actor_user_id, actor_name, action, target_type, target_id,
              field=None, old_value=None, new_value=None, reason=None):
    conn.execute("""
        INSERT INTO audit_log
        (actor_user_id, actor_name, action, target_type, target_id, field, old_value, new_value, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (actor_user_id, actor_name, action, target_type, str(target_id), field, old_value, new_value, reason))
    conn.commit()

def get_audit_log(conn, target_type=None, target_id=None, limit=100):
    query = "SELECT * FROM audit_log WHERE 1=1"
    params = []
    if target_type:
        query += " AND target_type=?"; params.append(target_type)
    if target_id:
        query += " AND target_id=?"; params.append(str(target_id))
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def insert_report(conn, data: dict):
    report_id = data.get("report_id") or f"DRAFT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    conn.execute("""
        INSERT OR REPLACE INTO reports
        (report_id, report_text, date, site, activity, report_type, source,
         sif_potential, life_saving_rule, confidence, risk_score, rule_confidence,
         risk_level, top_phrases, analyzed_at, reporter_id, status, critical,
         sif_reasons, sif_model_name, sif_model_version, sif_decision_thresholds,
         sif_prediction_timestamp, rule_needs_review, rule_review_reason,
         rule_runner_up, rule_runner_up_confidence, rule_model_name,
         rule_model_version, rule_prediction_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id,
        data.get("report_text", ""),
        data.get("date", str(datetime.now().date())),
        data.get("site", "Unknown"),
        data.get("activity", "Unknown"),
        data.get("report_type", "Manual"),
        data.get("source", "api"),
        data.get("sif_potential", "No"),
        data.get("life_saving_rule", "N/A"),
        data.get("confidence", 0),
        data.get("risk_score", 0),
        data.get("rule_confidence", 0),
        data.get("risk_level", "Low"),
        json.dumps(data.get("top_contributing_phrases", [])),
        datetime.now().isoformat(),
        data.get("reporter_id"),
        data.get("status", "Submitted"),
        1 if data.get("critical") else 0,
        # ── Phase 16 explainability metadata (additive; None if caller
        # didn't provide it -- e.g. CSV-seeded legacy rows) ──────────────
        json.dumps(data.get("sif_reasons")) if data.get("sif_reasons") is not None else None,
        data.get("sif_model_name"),
        data.get("sif_model_version"),
        json.dumps(data.get("sif_decision_thresholds")) if data.get("sif_decision_thresholds") is not None else None,
        data.get("sif_prediction_timestamp"),
        1 if data.get("rule_needs_review") else 0,
        data.get("rule_review_reason"),
        data.get("rule_runner_up"),
        data.get("rule_runner_up_confidence"),
        data.get("rule_model_name"),
        data.get("rule_model_version"),
        data.get("rule_prediction_timestamp"),
    ))
    conn.commit()
    row = conn.execute("SELECT id FROM reports WHERE report_id=?", (report_id,)).fetchone()
    return row["id"] if row else None

def update_report_status(conn, report_pk_id, new_status):
    conn.execute("UPDATE reports SET status=? WHERE id=?", (new_status, report_pk_id))
    conn.commit()

def get_report_by_pk(conn, report_pk_id):
    row = conn.execute("SELECT * FROM reports WHERE id=?", (report_pk_id,)).fetchone()
    return dict(row) if row else None

def get_dashboard_stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    sif_yes = conn.execute("SELECT COUNT(*) FROM reports WHERE sif_potential='Yes'").fetchone()[0]
    sif_no = total - sif_yes
    density = round(100 * sif_yes / total, 1) if total > 0 else 0
    critical_open = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE critical=1 AND status != 'Closed'"
    ).fetchone()[0]

    # By lifecycle status -- powers the Supervisor/HSE "New" and "Awaiting
    # Review" KPI tiles without them having to re-derive it from the raw
    # report list.
    status_rows = conn.execute("""
        SELECT COALESCE(NULLIF(status, ''), 'Submitted') as status, COUNT(*) as cnt
        FROM reports GROUP BY status
    """).fetchall()
    by_status = {r["status"]: r["cnt"] for r in status_rows}

    # By life-saving rule
    rule_counts = conn.execute("""
        SELECT life_saving_rule, COUNT(*) as cnt
        FROM reports WHERE sif_potential='Yes'
        GROUP BY life_saving_rule ORDER BY cnt DESC
    """).fetchall()

    # Site-wise
    site_rows = conn.execute("""
        SELECT site,
               COUNT(*) as total,
               SUM(CASE WHEN sif_potential='Yes' THEN 1 ELSE 0 END) as sif_count
        FROM reports GROUP BY site ORDER BY sif_count DESC
    """).fetchall()

    # Activity-wise
    activity_rows = conn.execute("""
        SELECT activity,
               COUNT(*) as total,
               SUM(CASE WHEN sif_potential='Yes' THEN 1 ELSE 0 END) as sif_count
        FROM reports GROUP BY activity ORDER BY sif_count DESC
    """).fetchall()

    # Top recurring precursors
    precursors = conn.execute("""
        SELECT site, activity, life_saving_rule, COUNT(*) as cnt
        FROM reports
        WHERE sif_potential='Yes' AND life_saving_rule NOT IN ('Unmapped','None','N/A','')
        GROUP BY site, activity, life_saving_rule
        HAVING cnt >= 2
        ORDER BY cnt DESC LIMIT 5
    """).fetchall()

    return {
        "total_reports": total,
        "sif_yes": sif_yes,
        "sif_no": sif_no,
        "sif_density": density,
        "critical_open": critical_open,
        "by_status": by_status,
        "by_rule": [{"rule": r["life_saving_rule"], "count": r["cnt"]} for r in rule_counts],
        "by_site": [{"site": r["site"], "total": r["total"],
                      "sif_count": r["sif_count"],
                      "density": round(100*r["sif_count"]/r["total"], 1) if r["total"] else 0}
                     for r in site_rows],
        "by_activity": [{"activity": r["activity"], "total": r["total"],
                          "sif_count": r["sif_count"],
                          "density": round(100*r["sif_count"]/r["total"], 1) if r["total"] else 0}
                         for r in activity_rows],
        "top_precursors": [
            {"site": r["site"], "activity": r["activity"],
             "rule": r["life_saving_rule"], "count": r["cnt"]}
            for r in precursors
        ],
    }

# Statuses that read as "still with HSE/Supervisor, not yet actioned" from
# a Reporter's point of view -- collapses the finer internal lifecycle
# into the three buckets a field user actually cares about.
_UNDER_REVIEW_STATUSES = ("Submitted", "Reviewed", "Investigation")

def get_my_stats(conn, reporter_id):
    """Scoped to one Reporter's own submissions -- never touches
    organization-wide counts. See the Reporter-vs-HSE dashboard split:
    a field user should see what happened to *their* reports, not the
    site's SIF density or precursor analytics."""
    rows = conn.execute(
        "SELECT status FROM reports WHERE reporter_id=?", (reporter_id,)
    ).fetchall()
    total = len(rows)
    under_review = sum(1 for r in rows if (r["status"] or "Submitted") in _UNDER_REVIEW_STATUSES)
    action_in_progress = sum(1 for r in rows if r["status"] == "Action Assigned")
    closed = sum(1 for r in rows if r["status"] == "Closed")

    recent = conn.execute("""
        SELECT report_id, report_type, status, date, id
        FROM reports WHERE reporter_id=?
        ORDER BY id DESC LIMIT 5
    """, (reporter_id,)).fetchall()

    return {
        "total": total,
        "under_review": under_review,
        "action_in_progress": action_in_progress,
        "closed": closed,
        "recent": [dict(r) for r in recent],
    }

def get_reports(conn, site=None, activity=None, sif_potential=None,
                life_saving_rule=None, reporter_id=None, critical_first=False,
                page=1, per_page=25):
    query = "SELECT * FROM reports WHERE 1=1"
    params = []
    if site:
        query += " AND site=?"; params.append(site)
    if activity:
        query += " AND activity=?"; params.append(activity)
    if sif_potential:
        query += " AND sif_potential=?"; params.append(sif_potential)
    if life_saving_rule:
        query += " AND life_saving_rule=?"; params.append(life_saving_rule)
    if reporter_id is not None:
        query += " AND reporter_id=?"; params.append(reporter_id)

    count_q = query.replace("SELECT *", "SELECT COUNT(*)")
    total = conn.execute(count_q, params).fetchone()[0]

    if critical_first:
        query += " ORDER BY critical DESC, (status != 'Closed') DESC, id DESC"
    else:
        query += " ORDER BY id DESC"
    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])

    rows = conn.execute(query, params).fetchall()
    return {
        "reports": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }

print("[db] Module loaded.")
