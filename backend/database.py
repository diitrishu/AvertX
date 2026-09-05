"""
database_supabase.py

Drop-in replacement for database.py that targets Supabase (PostgreSQL)
instead of local SQLite.  All public function signatures are identical
to database.py so main.py requires zero changes.

Configuration (set these environment variables before starting the backend):
    SUPABASE_DB_URL   -- PostgreSQL connection string, e.g.:
                         postgresql://postgres.[ref]:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
                         Get it from: Supabase dashboard -> Project Settings -> Database -> Connection string (Transaction mode)

Usage: rename this file to database.py (or swap the import in main.py).
       The SQLite database.py is kept as database_sqlite.py for rollback.
"""

import os
import json
from datetime import datetime
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get("SUPABASE_DB_URL", "")

if not _DB_URL:
    raise RuntimeError(
        "SUPABASE_DB_URL environment variable is not set.\n"
        "Set it to the Supabase Transaction Mode connection string:\n"
        "  postgresql://postgres.[ref]:[password]@[host]:6543/postgres\n"
        "Find it at: Supabase dashboard -> Project Settings -> Database -> "
        "Connection string (Transaction mode)"
    )


def get_db():
    """
    Return a new psycopg2 connection.
    Caller is responsible for closing it (same contract as SQLite version).
    RealDictCursor makes rows behave like dicts, matching sqlite3.Row.
    """
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# Schema bootstrap
# (called once at startup — idempotent; schema already created by migration
#  001_initial_schema.sql, this is just a safety net)
# ---------------------------------------------------------------------------

def init_db():
    """
    Verify the three core tables exist.  Schema creation is handled by
    supabase/migrations/001_initial_schema.sql — run that in the Supabase
    SQL editor before first startup.  This function only checks connectivity
    and seeds the admin accounts if they are somehow missing.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        for table in ("users", "reports", "audit_log"):
            cur.execute(
                "select exists(select 1 from information_schema.tables "
                "where table_schema='public' and table_name=%s)",
                (table,),
            )
            exists = cur.fetchone()["exists"]
            if not exists:
                raise RuntimeError(
                    f"Table '{table}' not found in Supabase. "
                    f"Run supabase/migrations/001_initial_schema.sql first."
                )
        _seed_admins(conn)
        conn.commit()
        print("[db] Supabase connection OK — all tables present.")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Admin seeding (mirrors database.py _seed_admins exactly)
# ---------------------------------------------------------------------------

DEFAULT_ADMINS = [
    {"name": "System Admin 1",  "email": "admin1@email.com",       "password": "admin123"},
    {"name": "System Admin 2",  "email": "admin2@oil-hsse.local",  "password": "ChangeMe@Admin2"},
]


def _seed_admins(conn):
    from backend.auth import hash_password
    cur = conn.cursor()
    for a in DEFAULT_ADMINS:
        cur.execute("select id from users where email=%s", (a["email"],))
        if cur.fetchone():
            continue
        cur.execute(
            "insert into users (name, email, password_hash, role, status) "
            "values (%s, %s, %s, 'Admin', 'active') on conflict (email) do nothing",
            (a["name"], a["email"], hash_password(a["password"])),
        )
        print(f"[db] Seeded admin account: {a['email']}")


# ---------------------------------------------------------------------------
# CSV seed (called once if reports table is empty)
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "oil_safety_reports_merged.csv")


def _seed_from_csv(conn):
    import pandas as pd
    if not os.path.exists(CSV_PATH):
        print("[db] CSV not found, skipping seed.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"[db] Seeding {len(df)} reports from CSV...")
    cur = conn.cursor()
    for _, row in df.iterrows():
        risk_level = row.get("risk_level", "")
        risk_level = risk_level if risk_level in ("High", "Medium", "Low") else "Low"
        sif_potential = row.get("sif_potential", "")
        critical = 1 if (sif_potential == "Yes" and risk_level == "High") else 0
        try:
            cur.execute(
                """
                insert into reports
                  (report_id, report_text, date, site, activity, report_type, source,
                   sif_potential, life_saving_rule, risk_level, status, critical)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'Closed',%s)
                on conflict (report_id) do nothing
                """,
                (
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
                    critical,
                ),
            )
        except Exception:
            pass
    conn.commit()
    print(f"[db] Seeded {len(df)} reports.")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(conn, name, email, password_hash, role="Reporter", site=""):
    cur = conn.cursor()
    cur.execute(
        "insert into users (name, email, password_hash, role, site, status) "
        "values (%s,%s,%s,%s,%s,'active') returning id",
        (name, email, password_hash, role, site),
    )
    row = cur.fetchone()
    conn.commit()
    return row["id"]


def get_user_by_email(conn, email):
    cur = conn.cursor()
    cur.execute("select * from users where email=%s", (email,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_user_by_id(conn, user_id):
    cur = conn.cursor()
    cur.execute("select * from users where id=%s", (user_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def list_users(conn):
    cur = conn.cursor()
    cur.execute(
        "select id, name, email, role, site, status, created_at "
        "from users order by created_at desc"
    )
    return [dict(r) for r in cur.fetchall()]


def update_user_role(conn, user_id, new_role):
    cur = conn.cursor()
    cur.execute("update users set role=%s where id=%s", (new_role, user_id))
    conn.commit()


def update_user_status(conn, user_id, new_status):
    cur = conn.cursor()
    cur.execute("update users set status=%s where id=%s", (new_status, user_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_audit(conn, actor_user_id, actor_name, action, target_type, target_id,
              field=None, old_value=None, new_value=None, reason=None):
    cur = conn.cursor()
    cur.execute(
        """
        insert into audit_log
          (actor_user_id, actor_name, action, target_type, target_id,
           field, old_value, new_value, reason)
        values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (actor_user_id, actor_name, action, target_type, str(target_id),
         field, old_value, new_value, reason),
    )
    conn.commit()


def get_audit_log(conn, target_type=None, target_id=None, limit=100):
    cur = conn.cursor()
    query = "select * from audit_log where 1=1"
    params: list = []
    if target_type:
        query += " and target_type=%s"; params.append(target_type)
    if target_id:
        query += " and target_id=%s"; params.append(str(target_id))
    query += " order by id desc limit %s"
    params.append(limit)
    cur.execute(query, params)
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Reports — insert
# ---------------------------------------------------------------------------

def insert_report(conn, data: dict):
    report_id = data.get("report_id") or f"DRAFT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    cur = conn.cursor()
    cur.execute(
        """
        insert into reports
          (report_id, report_text, date, site, activity, report_type, source,
           sif_potential, life_saving_rule, confidence, risk_score, rule_confidence,
           risk_level, top_phrases, analyzed_at, reporter_id, status, critical,
           sif_reasons, sif_model_name, sif_model_version, sif_decision_thresholds,
           sif_prediction_timestamp, rule_needs_review, rule_review_reason,
           rule_runner_up, rule_runner_up_confidence, rule_model_name,
           rule_model_version, rule_prediction_timestamp)
        values
          (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
           %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (report_id) do update set
          report_text=excluded.report_text,
          site=excluded.site, activity=excluded.activity,
          sif_potential=excluded.sif_potential,
          life_saving_rule=excluded.life_saving_rule,
          confidence=excluded.confidence,
          risk_score=excluded.risk_score,
          rule_confidence=excluded.rule_confidence,
          risk_level=excluded.risk_level,
          top_phrases=excluded.top_phrases,
          analyzed_at=excluded.analyzed_at,
          status=excluded.status,
          critical=excluded.critical
        returning id
        """,
        (
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
        ),
    )
    row = cur.fetchone()
    conn.commit()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# Reports — read / update
# ---------------------------------------------------------------------------

def update_report_status(conn, report_pk_id, new_status):
    cur = conn.cursor()
    cur.execute("update reports set status=%s where id=%s", (new_status, report_pk_id))
    conn.commit()


def get_report_by_pk(conn, report_pk_id):
    cur = conn.cursor()
    cur.execute("select * from reports where id=%s", (report_pk_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_reports(conn, site=None, activity=None, sif_potential=None,
                life_saving_rule=None, reporter_id=None, critical_first=False,
                page=1, per_page=25):
    # Exclude seed/training rows — only show real submitted reports.
    query = "select * from reports where source not in ('real_ihm_stefanini', 'synthetic')"
    params: list = []
    if site:
        query += " and site=%s"; params.append(site)
    if activity:
        query += " and activity=%s"; params.append(activity)
    if sif_potential:
        query += " and sif_potential=%s"; params.append(sif_potential)
    if life_saving_rule:
        query += " and life_saving_rule=%s"; params.append(life_saving_rule)
    if reporter_id is not None:
        query += " and reporter_id=%s"; params.append(reporter_id)

    count_q = query.replace("select *", "select count(*)")
    cur = conn.cursor()
    cur.execute(count_q, params)
    total = cur.fetchone()["count"]

    if critical_first:
        query += " order by critical desc, (status != 'Closed') desc, id desc"
    else:
        query += " order by id desc"
    query += " limit %s offset %s"
    params.extend([per_page, (page - 1) * per_page])

    cur.execute(query, params)
    return {
        "reports": [dict(r) for r in cur.fetchall()],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def get_dashboard_stats(conn):
    cur = conn.cursor()

    # Filter out seed/training rows — only count reports submitted through
    # the real application. Seed rows have source = 'real_ihm_stefanini'
    # or 'synthetic'; real submissions come from 'api', 'pdf_upload',
    # 'bulk_upload', or any other source set by the application.
    SEED_SOURCES = ('real_ihm_stefanini', 'synthetic')
    seed_filter = "source NOT IN %s"
    seed_param = (SEED_SOURCES,)

    cur.execute(f"select count(*) from reports where {seed_filter}", seed_param)
    total = cur.fetchone()["count"]

    cur.execute(f"select count(*) from reports where sif_potential='Yes' and {seed_filter}", seed_param)
    sif_yes = cur.fetchone()["count"]
    sif_no = total - sif_yes
    density = round(100 * sif_yes / total, 1) if total > 0 else 0

    cur.execute(
        f"select count(*) from reports where critical=1 and status != 'Closed' and {seed_filter}",
        seed_param,
    )
    critical_open = cur.fetchone()["count"]

    cur.execute(
        f"select coalesce(nullif(status,''),'Submitted') as status, count(*) as cnt "
        f"from reports where {seed_filter} group by status",
        seed_param,
    )
    by_status = {r["status"]: r["cnt"] for r in cur.fetchall()}

    cur.execute(
        f"select life_saving_rule, count(*) as cnt from reports "
        f"where sif_potential='Yes' and {seed_filter} group by life_saving_rule order by cnt desc",
        seed_param,
    )
    rule_counts = cur.fetchall()

    cur.execute(
        f"select site, count(*) as total, "
        f"sum(case when sif_potential='Yes' then 1 else 0 end) as sif_count "
        f"from reports where {seed_filter} group by site order by sif_count desc",
        seed_param,
    )
    site_rows = cur.fetchall()

    cur.execute(
        f"select activity, count(*) as total, "
        f"sum(case when sif_potential='Yes' then 1 else 0 end) as sif_count "
        f"from reports where {seed_filter} group by activity order by sif_count desc",
        seed_param,
    )
    activity_rows = cur.fetchall()

    cur.execute(
        f"select site, activity, life_saving_rule, count(*) as cnt "
        f"from reports "
        f"where sif_potential='Yes' "
        f"  and life_saving_rule not in ('Unmapped','None','N/A','') "
        f"  and {seed_filter} "
        f"group by site, activity, life_saving_rule "
        f"having count(*) >= 2 "
        f"order by cnt desc limit 5",
        seed_param,
    )
    precursors = cur.fetchall()

    return {
        "total_reports": total,
        "sif_yes": sif_yes,
        "sif_no": sif_no,
        "sif_density": density,
        "critical_open": critical_open,
        "by_status": by_status,
        "by_rule": [{"rule": r["life_saving_rule"], "count": r["cnt"]} for r in rule_counts],
        "by_site": [
            {"site": r["site"], "total": r["total"], "sif_count": r["sif_count"],
             "density": round(100 * r["sif_count"] / r["total"], 1) if r["total"] else 0}
            for r in site_rows
        ],
        "by_activity": [
            {"activity": r["activity"], "total": r["total"], "sif_count": r["sif_count"],
             "density": round(100 * r["sif_count"] / r["total"], 1) if r["total"] else 0}
            for r in activity_rows
        ],
        "top_precursors": [
            {"site": r["site"], "activity": r["activity"],
             "rule": r["life_saving_rule"], "count": r["cnt"]}
            for r in precursors
        ],
    }


_UNDER_REVIEW_STATUSES = ("Submitted", "Reviewed", "Investigation")


def get_my_stats(conn, reporter_id):
    cur = conn.cursor()
    cur.execute("select status from reports where reporter_id=%s", (reporter_id,))
    rows = cur.fetchall()
    total = len(rows)
    under_review = sum(1 for r in rows if (r["status"] or "Submitted") in _UNDER_REVIEW_STATUSES)
    action_in_progress = sum(1 for r in rows if r["status"] == "Action Assigned")
    closed = sum(1 for r in rows if r["status"] == "Closed")

    cur.execute(
        "select report_id, report_type, status, date, id from reports "
        "where reporter_id=%s order by id desc limit 5",
        (reporter_id,),
    )
    return {
        "total": total,
        "under_review": under_review,
        "action_in_progress": action_in_progress,
        "closed": closed,
        "recent": [dict(r) for r in cur.fetchall()],
    }


print("[db] Supabase database module loaded.")
