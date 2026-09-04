"""
FastAPI backend for SIH26165 AI Safety Analyst.

Endpoints:
  Auth:        /auth/signup, /auth/login, /auth/me
  Reports:     /analyze/text, /analyze/pdf, /reports/bulk-upload,
               /reports, /reports/{id}/status, /reports/filters
  Dashboard:   /dashboard/stats (Supervisor/HSE/Admin), /dashboard/my-stats (own reports)
  Admin:       /admin/users, /admin/users/{id}/role, /admin/users/{id}/status
  Audit:       /audit-log
"""

import os, io, json
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # loads .env before any module reads SUPABASE_DB_URL

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.model import predict, explain_prediction, build_recurrence_table, RULE_SEVERITY
from backend.database import (
    init_db, get_db, insert_report, get_dashboard_stats, get_my_stats, get_reports,
    create_user, get_user_by_email, get_user_by_id, list_users,
    update_user_role, update_user_status, log_audit, get_audit_log,
    update_report_status, get_report_by_pk,
)
from backend.auth import (
    hash_password, verify_password, create_token, get_current_user,
    require_roles, ROLES,
)

app = FastAPI(title="AI Safety Analyst", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Taxonomy lookup ──────────────────────────────────────────────────
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "..", "life_saving_rules_taxonomy.json")
taxonomy = {}
if os.path.exists(TAXONOMY_PATH):
    with open(TAXONOMY_PATH) as f:
        raw = json.load(f)
    for rule_obj in raw.get("life_saving_rules", []):
        taxonomy[rule_obj["rule"]] = {
            "typical_hazards": rule_obj.get("typical_hazards", []),
            "typical_barriers": rule_obj.get("barriers", []),
        }

# A report is flagged CRITICAL -- and jumps to the top of the HSE/Reviewer
# inbox -- the instant the AI returns a SIF-potential call with a high
# risk score. This is what makes the "immediate alert to reviewer" real
# rather than a UI decoration: the flag is set at analysis time, in the
# same request that produced the classification.
CRITICAL_RISK_THRESHOLD = 70

def _is_critical(result: dict) -> bool:
    return result.get("sif_potential") == "Yes" and result.get("risk_score", 0) >= CRITICAL_RISK_THRESHOLD

# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    conn = get_db()
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "oil_safety_reports_merged.csv"))
    build_recurrence_table(df)
    conn.close()
    print("[api] Startup complete.")

# ═══════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════

@app.post("/auth/signup")
def signup(body: dict):
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    site = (body.get("site") or "").strip()

    if not name or not email or not password:
        raise HTTPException(400, "name, email and password are required")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    conn = get_db()
    if get_user_by_email(conn, email):
        conn.close()
        raise HTTPException(409, "An account with this email already exists")

    # Every new signup starts as Reporter. Elevation to Supervisor/HSE/Admin
    # is a deliberate act by an existing Admin (see /admin/users/{id}/role),
    # never a self-service choice -- see the earlier design discussion.
    user_id = create_user(conn, name, email, hash_password(password), role="Reporter", site=site)
    user = get_user_by_id(conn, user_id)
    log_audit(conn, user_id, name, "signup", "user", user_id, new_value="role=Reporter")
    conn.close()

    token = create_token(user)
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"],
                                       "role": user["role"], "site": user["site"]}}

@app.post("/auth/login")
def login(body: dict):
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    conn = get_db()
    user = get_user_by_email(conn, email)
    conn.close()

    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    if user["status"] != "active":
        raise HTTPException(403, "This account has been deactivated")

    token = create_token(user)
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"],
                                       "role": user["role"], "site": user["site"]}}

@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user

# ═══════════════════════════════════════════════════════════════════
#  ADMIN — user management & role elevation
# ═══════════════════════════════════════════════════════════════════

@app.get("/admin/users")
def admin_list_users(admin: dict = Depends(require_roles("Admin"))):
    conn = get_db()
    users = list_users(conn)
    conn.close()
    return {"users": users, "roles": ROLES}

@app.post("/admin/users/{user_id}/role")
def admin_set_role(user_id: int, body: dict, admin: dict = Depends(require_roles("Admin"))):
    new_role = body.get("role")
    reason = body.get("reason", "")
    if new_role not in ROLES:
        raise HTTPException(400, f"role must be one of {ROLES}")

    conn = get_db()
    target = get_user_by_id(conn, user_id)
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")

    old_role = target["role"]
    update_user_role(conn, user_id, new_role)
    log_audit(conn, admin["id"], admin["name"], "role_change", "user", user_id,
              field="role", old_value=old_role, new_value=new_role, reason=reason)
    conn.close()
    return {"ok": True, "user_id": user_id, "old_role": old_role, "new_role": new_role}

@app.post("/admin/users/{user_id}/status")
def admin_set_status(user_id: int, body: dict, admin: dict = Depends(require_roles("Admin"))):
    new_status = body.get("status")
    reason = body.get("reason", "")
    if new_status not in ("active", "disabled"):
        raise HTTPException(400, "status must be 'active' or 'disabled'")

    conn = get_db()
    target = get_user_by_id(conn, user_id)
    if not target:
        conn.close()
        raise HTTPException(404, "User not found")

    old_status = target["status"]
    update_user_status(conn, user_id, new_status)
    log_audit(conn, admin["id"], admin["name"], "status_change", "user", user_id,
              field="status", old_value=old_status, new_value=new_status, reason=reason)
    conn.close()
    return {"ok": True, "user_id": user_id, "old_status": old_status, "new_status": new_status}

@app.get("/audit-log")
def audit_log(target_type: Optional[str] = None, target_id: Optional[str] = None,
              user: dict = Depends(require_roles("Admin", "HSE"))):
    conn = get_db()
    entries = get_audit_log(conn, target_type=target_type, target_id=target_id)
    conn.close()
    return {"entries": entries}

# ═══════════════════════════════════════════════════════════════════
#  REPORTS — create, analyze, list, advance lifecycle status
# ═══════════════════════════════════════════════════════════════════

@app.post("/analyze/text")
def analyze_text(body: dict, user: dict = Depends(get_current_user)):
    report_text = body.get("report_text", "")
    site = body.get("site", "Unknown")
    activity = body.get("activity", "Unknown")
    if not report_text.strip():
        raise HTTPException(400, "report_text is required")

    result = predict(report_text, site=site, activity=activity)

    rule = result["life_saving_rule"]
    tax = taxonomy.get(rule, {})
    typical_hazards = tax.get("typical_hazards", [])
    typical_barriers = tax.get("typical_barriers", [])

    rs = result["risk_score"]
    risk_level = "High" if rs >= 70 else ("Medium" if rs >= 40 else "Low")
    critical = _is_critical(result)

    response = {
        **result,
        "typical_hazard": ", ".join(typical_hazards) if typical_hazards else "N/A",
        "typical_barrier": ", ".join(typical_barriers) if typical_barriers else "N/A",
        "risk_level": risk_level,
        "critical": critical,
        "status": "Submitted",
    }

    conn = get_db()
    report_pk = insert_report(conn, {
        "report_text": report_text,
        "site": site,
        "activity": activity,
        "report_type": "Manual",
        "source": "api",
        "reporter_id": user["id"],
        **response,
    })
    if critical:
        log_audit(conn, None, "AI Engine", "critical_flag_raised", "report", report_pk,
                  new_value=f"sif_potential=Yes risk_score={rs}",
                  reason="Auto-flagged for immediate Reviewer alert")
    conn.close()

    return {**response, "id": report_pk}

@app.post("/analyze/pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    site: str = Form("Unknown"),
    activity: str = Form("Unknown"),
    user: dict = Depends(get_current_user),
):
    contents = await file.read()
    text = ""

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[pdf] pdfplumber failed: {e}")

    if not text.strip():
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(contents, dpi=200)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            print(f"[pdf] OCR failed: {e}")
            text = "(Could not extract text from PDF)"

    result = predict(text, site=site, activity=activity)
    rule = result["life_saving_rule"]
    tax = taxonomy.get(rule, {})
    rs = result["risk_score"]
    risk_level = "High" if rs >= 70 else ("Medium" if rs >= 40 else "Low")
    critical = _is_critical(result)

    response = {
        **result,
        "extracted_text": text,
        "typical_hazard": ", ".join(tax.get("typical_hazards", [])) or "N/A",
        "typical_barrier": ", ".join(tax.get("typical_barriers", [])) or "N/A",
        "risk_level": risk_level,
        "critical": critical,
        "status": "Submitted",
    }

    conn = get_db()
    report_pk = insert_report(conn, {
        "report_text": text,
        "site": site,
        "activity": activity,
        "report_type": "PDF Upload",
        "source": "pdf_upload",
        "reporter_id": user["id"],
        **{k: v for k, v in response.items() if k != "extracted_text"},
    })
    if critical:
        log_audit(conn, None, "AI Engine", "critical_flag_raised", "report", report_pk,
                  new_value=f"sif_potential=Yes risk_score={rs}",
                  reason="Auto-flagged for immediate Reviewer alert")
    conn.close()

    return {**response, "id": report_pk}

@app.post("/reports/bulk-upload")
async def bulk_upload(file: UploadFile = File(...), user: dict = Depends(require_roles("HSE", "Admin"))):
    contents = await file.read()
    filename = file.filename or "upload.csv"

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        df = pd.read_csv(io.BytesIO(contents))

    if "report_text" not in df.columns:
        raise HTTPException(400, "CSV must have a 'report_text' column")

    results = []
    conn = get_db()
    for _, row in df.iterrows():
        text = str(row.get("report_text", ""))
        site = str(row.get("site", "Unknown"))
        activity = str(row.get("activity", "Unknown"))
        if not text.strip():
            continue
        result = predict(text, site=site, activity=activity)
        rs = result["risk_score"]
        risk_level = "High" if rs >= 70 else ("Medium" if rs >= 40 else "Low")
        critical = _is_critical(result)

        insert_report(conn, {
            "report_id": str(row.get("report_id", "")),
            "report_text": text,
            "date": str(row.get("date", "")),
            "site": site,
            "activity": activity,
            "report_type": str(row.get("report_type", "Bulk")),
            "source": "bulk_upload",
            "reporter_id": user["id"],
            "status": "Submitted",
            "critical": critical,
            **result,
            "risk_level": risk_level,
        })
        results.append(result)
    conn.close()

    sif_count = sum(1 for r in results if r["sif_potential"] == "Yes")
    return {
        "total_processed": len(results),
        "sif_flagged": sif_count,
        "non_sif": len(results) - sif_count,
    }

@app.get("/reports")
def list_reports(
    site: Optional[str] = Query(None),
    activity: Optional[str] = Query(None),
    sif_potential: Optional[str] = Query(None),
    life_saving_rule: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    user: dict = Depends(get_current_user),
):
    # Reporters only ever see their own submissions. Every other role
    # (Supervisor/HSE/Admin) sees the shared inbox, sorted with critical
    # and still-open reports first -- this is the "immediate alert" view.
    reporter_id = user["id"] if user["role"] == "Reporter" else None
    critical_first = user["role"] != "Reporter"

    conn = get_db()
    result = get_reports(conn, site=site, activity=activity,
                         sif_potential=sif_potential,
                         life_saving_rule=life_saving_rule,
                         reporter_id=reporter_id,
                         critical_first=critical_first,
                         page=page, per_page=per_page)
    conn.close()
    return result

@app.post("/reports/{report_pk}/status")
def set_report_status(report_pk: int, body: dict, user: dict = Depends(require_roles("Supervisor", "HSE", "Admin"))):
    new_status = body.get("status")
    reason = body.get("reason", "")
    valid = ("Submitted", "Reviewed", "Investigation", "Action Assigned", "Closed")
    if new_status not in valid:
        raise HTTPException(400, f"status must be one of {valid}")

    conn = get_db()
    report = get_report_by_pk(conn, report_pk)
    if not report:
        conn.close()
        raise HTTPException(404, "Report not found")

    old_status = report["status"]
    update_report_status(conn, report_pk, new_status)
    log_audit(conn, user["id"], user["name"], "status_change", "report", report_pk,
              field="status", old_value=old_status, new_value=new_status, reason=reason)
    conn.close()
    return {"ok": True, "report_id": report_pk, "old_status": old_status, "new_status": new_status}

@app.get("/reports/filters")
def report_filters(user: dict = Depends(get_current_user)):
    conn = get_db()
    sites = [r[0] for r in conn.execute(
        "SELECT DISTINCT site FROM reports WHERE site != '' ORDER BY site"
    ).fetchall()]
    activities = [r[0] for r in conn.execute(
        "SELECT DISTINCT activity FROM reports WHERE activity != '' ORDER BY activity"
    ).fetchall()]
    rules = [r[0] for r in conn.execute(
        "SELECT DISTINCT life_saving_rule FROM reports WHERE life_saving_rule != '' ORDER BY life_saving_rule"
    ).fetchall()]
    conn.close()
    return {"sites": sites, "activities": activities, "rules": rules}

# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════

@app.get("/dashboard/stats")
def dashboard_stats(user: dict = Depends(require_roles("Supervisor", "HSE", "Admin"))):
    # Organization-wide SIF density, site risk, precursor analytics --
    # this is HSE-level intelligence. Reporter is deliberately excluded:
    # a field user should not see the org's overall SIF rate or other
    # sites' incidents. See /dashboard/my-stats for their view.
    conn = get_db()
    stats = get_dashboard_stats(conn)
    conn.close()
    return stats

@app.get("/dashboard/my-stats")
def my_dashboard_stats(user: dict = Depends(get_current_user)):
    conn = get_db()
    stats = get_my_stats(conn, user["id"])
    conn.close()
    return stats

# ── Health check ─────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2", "threshold": 0.393}
