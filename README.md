# AI Safety Analyst — SIH26165 (Oil India Limited)

An HSSE safety-intelligence platform: field reports go in as free text, an
NLP model classifies them for **SIF potential** (Serious Injury or
Fatality potential — not the same as actual outcome), maps them to IOGP
Life-Saving Rules, and the report then moves through a real lifecycle
(submitted → reviewed → investigated where it matters → actioned →
closed) with role-based access at every step.

This is the single source of truth for the project — it replaces the
older `RUN_COMMANDS.md` and `README_dataset.md`, folded in below.

---

## 1. Quick start

### First-time setup (once per clone)

`model_artifacts/embedding_model_ref.joblib` (~87MB) is not committed to
this repo -- it's a re-downloadable public model, not project-specific
data. Regenerate it once, after installing dependencies (section 2):

```bash
python scripts/setup_embedding_model.py
```

### Backend (FastAPI + ML models)

```bash
cd C:\Users\hp\Desktop\SIF
py -3.12 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Do not run `python -m backend.main`** — that only imports the module
(loads the model, prints a deprecation warning) and exits; it does not
start a server. You must invoke `uvicorn` directly, as above, or via
`run_app.bat`.

Wait for these lines before using the app — they confirm the database
schema, model artifacts, and bootstrap accounts all loaded correctly:

```
[model] Ready.
[db] Seeded admin account: admin1@email.com   (or "[db] Renamed bootstrap admin -> admin1@email.com" on a DB that already existed)
[db] Seeded admin account: admin2@oil-hsse.local
[api] Startup complete.
```

### Frontend (React + Vite)

```bash
cd C:\Users\hp\Desktop\SIF\frontend
npm run dev
```

### Or both at once

Double-click `run_app.bat` from the project root.

### Access

- **App:** http://localhost:5173
- **API:** http://localhost:8000
- **Interactive API docs (Swagger):** http://localhost:8000/docs

Whenever backend code changes, **restart the uvicorn process** — it does
not hot-reload. A stale process is the single most common cause of
"it worked yesterday" bugs here (new routes 404, new DB columns silently
missing).

---

## 2. Prerequisites

- Python 3.12 with: `fastapi`, `uvicorn`, `pandas`, `scikit-learn`,
  `sentence-transformers`, `torch`, `joblib`, `lime`, `pdfplumber`,
  `pytesseract`, `pdf2image`, `python-multipart`, `pyjwt`, `bcrypt`
  (see `backend/requirements.txt`)
- Node.js with npm

```bash
pip install -r backend/requirements.txt --break-system-packages
python scripts/setup_embedding_model.py
cd frontend && npm install
```

---

## 3. Accounts & roles

Every new signup (`Create account` on the login screen) is created as a
**Reporter** — this is enforced server-side, not just hidden in the UI.
Elevating someone to Supervisor, HSE, or Admin is a deliberate action
taken by an Admin from the **Users & Roles** screen, and every role
change is written to the audit log (who changed what, when, and why).

### Bootstrap Admin accounts

Two Admin accounts are seeded automatically on first backend startup so
the system is never locked out of user management. **Change these
passwords after first login in any real deployment** — they're
plain-text in `backend/database.py` for development convenience only.

| Email | Password |
|---|---|
| `admin1@email.com` | `admin123` |
| `admin2@oil-hsse.local` | `ChangeMe@Admin2` |

### The four roles

| Role | Can do | Cannot do |
|---|---|---|
| **Reporter** | Submit reports, see only their own reports and status | See other reports, org-wide stats, change any status |
| **Supervisor** | See the review queue (critical-first), advance report status | Manage users, see org-wide SIF/site analytics dashboard |
| **HSE** | Everything Supervisor can, plus the full analytics dashboard, bulk CSV upload, audit log | Manage users/roles |
| **Admin** | Manage users and roles, view audit log | Nothing safety-related is Admin-exclusive by design — an IT admin having database access doesn't mean they get to call something "Non-SIF" |

Each role sees a different **Dashboard**: Reporter gets "My Reports" (own
counts + recent list, nothing org-wide), Supervisor gets a lighter KPI
row plus a review-queue preview, HSE/Admin get the full SIF
density/site-risk/precursor-analytics dashboard. This is enforced by the
backend (`GET /dashboard/stats` requires Supervisor/HSE/Admin — a
Reporter gets a 403, not just a hidden button), not only by what the
frontend chooses to render.

---

## 4. Report lifecycle

```
Submitted → Reviewed → Investigation → Action Assigned → Closed
```

A report is auto-flagged **critical** at the moment the AI returns
`sif_potential = Yes` with a risk score ≥ 70 — in the same request that
produced the classification, not as a separate step. Critical reports
sort to the top of the Supervisor/HSE reports inbox automatically.

Supervisor, HSE, and Admin can advance a report's status from the
Reports Log; every status change is recorded in the audit log with the
old value, new value, and actor.

---

## 5. Architecture

- **Backend:** FastAPI, JWT-based auth (stateless, `Authorization: Bearer`
  header), SQLite for persistence (`sif_database.db`, seeded from
  `oil_safety_reports_merged.csv` on first run)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` embeddings +
  Logistic Regression classifiers (SIF potential, Life-Saving Rule), LIME
  for explainability. Both classifiers return one of **three** outcomes
  (Yes / No / `NEEDS_REVIEW`), not a forced binary call — see
  `backend/sif_output_contract.py` and `backend/rule_output_contract.py`.
  This replaced an earlier single-threshold design after
  `data/benchmark_results.json` (a real, duplicate-free, real-narrative-only
  gold test set — not the training set) showed the SIF classifier's
  precision barely clears its 34.7% base rate at full recall (test
  precision 0.352, recall 1.0, PR-AUC 0.494 at the validation-selected
  threshold 0.13). The classifier is not scrapped over this — see
  `MODEL_COMPARISON.md` for why no candidate model beat it — but its
  honest performance is why "not confident enough, ask a human" is a first
  -class output instead of a forced guess. Full benchmark:
  `data/benchmark_results.md`.
- **Frontend:** React + TypeScript + Vite + Tailwind CSS + Recharts,
  single-file `App.tsx`
- **Auth:** bcrypt password hashing, JWT sessions (12h TTL), role stored
  in the token and re-checked server-side per endpoint via
  `require_roles(...)` dependencies — the frontend hiding a button is a
  UX nicety, the backend check is the actual security boundary

---

## 6. API reference

Auth (no token required):
- `POST /auth/signup` — creates a Reporter account, returns `{token, user}`
- `POST /auth/login` — returns `{token, user}`

Authenticated (any role):
- `GET /auth/me` — current user info
- `POST /analyze/text` — analyze a text report, persists it, tags reporter + critical flag
- `POST /analyze/pdf` — same, from an uploaded PDF (OCR fallback if no text layer)
- `GET /reports` — paginated, filtered; Reporters see only their own, everyone else sees critical-first
- `GET /reports/filters` — distinct site/activity/rule values for filter dropdowns
- `GET /dashboard/my-stats` — the calling user's own report counts + recent list

Supervisor / HSE / Admin only:
- `POST /reports/{id}/status` — advance a report's lifecycle status
- `GET /dashboard/stats` — org-wide SIF density, site risk, precursor analytics

HSE / Admin only:
- `POST /reports/bulk-upload` — CSV/Excel batch analysis
- `GET /audit-log` — recent audit entries (optionally filtered by target)

Admin only:
- `GET /admin/users` — list all users
- `POST /admin/users/{id}/role` — change a user's role (audit-logged)
- `POST /admin/users/{id}/status` — activate/disable a user (audit-logged)

Unauthenticated:
- `GET /health` — liveness check, model name, decision threshold

---

## 7. Dataset

`oil_safety_reports_merged.csv` — **1,325 rows**, blending two sources so
the model isn't trained purely on synthetic text, and it's honest about
which is which via a `source` column (`real_ihm_stefanini` / `synthetic`).

**425 real rows** — Kaggle's "Industrial Safety and Health Analytics
Database" (IHM Stefanini), genuine accident narratives from a
mining/metals operation, each carrying a real `Potential Accident Level`
field. **900 synthetic rows** — OIL-flavored, grounded in the IOGP
Life-Saving Rules taxonomy and the SIH26165 problem statement's own
reference figures (SIF density ~20–25%).

**Why blend instead of using only the real data:** the real dataset gives
a genuine potential-severity ground truth and real incident language, but
it's mining/metals (not oil & gas vocabulary — no LOTO, gas testing,
confined space entry, hot work permit language), ~85% of its risk
categories have no clean IOGP rule equivalent, and every row is a
recorded accident with no near-miss/unsafe-act/unsafe-condition rows —
which is most of what a real UA/UC stream looks like, and exactly what
SIF logic exists to catch ("nothing happened, but it very easily could
have"). The synthetic data fills those three gaps; the real data proves
the classifier isn't purely trained on invented text.

**Mapping decisions applied to the real rows** (see `merge_real_data.py`):
`sif_potential` comes directly from the real `Potential Accident Level`
field (IV/V/VI → Yes, I/II/III → No — no heuristic guessing).
`life_saving_rule` is mapped from `Critical Risk` only where the category
has an unambiguous IOGP match (Pressurized Systems → Energy Isolation,
Suspended Loads/Projection/Vehicles → Line of Fire, Fall → Working at
Height, Confined space → Confined Space, Machine Protection → Bypassing
Safety Controls); everything else is labeled `"Unmapped"` (361 of 425
rows) rather than force-fit, though those rows still count for the binary
SIF classifier. `barrier`/`barrier_status` aren't in the source data and
are marked `"Unknown"` rather than fabricated.

**Class balance (merged):** 382 SIF-Yes / 943 SIF-No overall (~29%) — real
subset alone runs ~41% (accident logs skew higher than a raw UA/UC
stream since every row is already an accident), synthetic subset alone
~23% (calibrated to the problem statement's own reference figure).

**Files:** `oil_safety_reports_merged.csv` is the one the classifier
trains on. `merge_real_data.py` re-runs the mapping/merge logic.
`life_saving_rules_taxonomy.json` is the fixed 9-rule reference.
`generate_dataset.py` / `taxonomy.py` / `report_texts.py` are the
synthetic generator. Still on disk but not yet used: the OSHA construction
abstracts and incident reports, and DOT pipeline-accident data (structured
codes only, no free-text narrative) — a reasonable stretch-goal source of
more real English incident text if time allows.

---

## 8. Project structure

```
SIF/
├── backend/
│   ├── main.py                    FastAPI app, all endpoints
│   ├── auth.py                    password hashing, JWT, role-check dependencies
│   ├── database.py                SQLite schema, migrations, seeding, queries
│   ├── model.py                   model loading, prediction, LIME, risk score
│   ├── sif_output_contract.py     3-way (Yes/No/NEEDS_REVIEW) SIF decision wrapper
│   ├── rule_output_contract.py    3-way Life-Saving Rule decision wrapper
│   └── requirements.txt
├── frontend/
│   └── src/App.tsx        entire UI — auth screens, role-scoped dashboards,
│                          reports log, analyze, bulk upload, admin
├── scripts/
│   └── setup_embedding_model.py   regenerates the ~87MB embedding model
│                                  reference (not committed -- see .gitignore)
├── model_artifacts/       trained classifier + metadata (embedding reference
│                          excluded from git, see scripts/ above)
├── models/                 registry.json + README.md -- catalogs every
│                          trained artifact and which is actually in production
├── taxonomy/               life_saving_rules.csv, legacy_term_mapping.csv
├── data/                   pipeline scripts + small evidence artifacts only
│                          (splits/, MANIFEST.csv, benchmark/eval reports) --
│                          multi-GB intermediate outputs are NOT committed,
│                          see data/README.md to regenerate them
├── oil_safety_reports_merged.csv   seed dataset (synthetic + real IHM Stefanini data)
├── life_saving_rules_taxonomy.json  IOGP rule → hazard/barrier lookup
├── merge_real_data.py     real-dataset mapping/merge logic
├── train_model.py          trains sif_classifier.joblib + rule_classifier.joblib
├── train_rule_classifier.py  trains rule_classifier_v2.joblib (candidate, see MODEL_COMPARISON.md)
├── DATASET_AUDIT.md, DATA_SOURCES.md, PHASE9_DECISION.md,
│   PHASE14_HYBRID_ARCHITECTURE.md, PHASE15_PRECURSOR_DATA_PREP.md,
│   MODEL_COMPARISON.md, README_AI.md   design/decision record -- read
│                          README_AI.md first, it indexes the rest
├── sif_database.db        SQLite database (created on first run, gitignored)
└── run_app.bat            starts both servers
```

---

## 9. Known limitations (by design, not oversight)

These were deliberately scoped out of the hackathon build:

- No SSO/MFA — JWT + bcrypt only
- No org/site-scoped permissions yet (Supervisor sees all open reports,
  not just "their area" — there's no site-assignment field on users yet)
- No Investigation workspace, formal risk-matrix UI, or corrective-action
  tracking beyond the status field — the lifecycle *states* exist, the
  dedicated workspaces for Investigator/HSE Manager/Corporate roles don't
- Admin credentials are seeded in plain text in source for development
  convenience — must move to environment/secret-manager config before any
  real deployment

---

## 10. Troubleshooting

**"Not Found" on signup/login, or a dashboard shows an error banner
about the backend needing a restart:** the uvicorn process is running
older code. Stop it and restart with the command in section 1 — it does
not hot-reload on file changes.

**Sign-in form seems unresponsive after typing:** hard-refresh the
browser tab (Ctrl+Shift+R) to clear any leftover state from a prior
error, then retry.

**Report filters/pagination don't update the table:** confirmed fixed —
the Reports Log re-fetches on every filter/page change via a `useEffect`.
