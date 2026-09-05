# AvertX — AI Safety Analyst (SIH26165 / Oil India Limited)

An HSSE safety-intelligence platform for oil & gas operations. Field
incident reports go in as free text; an NLP model classifies them for
**SIF potential** (Serious Injury or Fatality potential — not the same
as actual outcome), maps them to IOGP Life-Saving Rules, and the report
moves through a structured lifecycle (Submitted → Reviewed → Investigation
→ Action Assigned → Closed) with role-based access at every step.

**Live deployment:**

| Service | URL |
|---|---|
| Frontend | https://avertx-frontend.onrender.com |
| Backend API | https://avertx.onrender.com |
| ML Service | https://avertx-ml.onrender.com |
| Database | Supabase PostgreSQL (ap-northeast-1) |

> **Cold-start notice:** free-tier services sleep after 15 minutes of
> inactivity. The first request after idle takes 30–60 seconds. Hit
> `https://avertx.onrender.com/health` and
> `https://avertx-ml.onrender.com/health` in your browser to wake both
> services before using the app.

---

## 1. Quick start (local development)

### Prerequisites

- Python 3.12
- Node.js with npm
- A `.env` file at the project root (copy `.env.example` and fill in values)

### Backend

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Wait for these lines before using the app:

```
[model] Using ONNX encoder (low-memory mode)   # or SentenceTransformer fallback
[db] Supabase connection OK — all tables present.
[api] Startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Or both at once

```bash
run_app.bat
```

### Access

| | URL |
|---|---|
| App | http://localhost:5173 |
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

---

## 2. Environment variables

Copy `.env.example` to `.env` and fill in:

```
SUPABASE_DB_URL=postgresql://postgres.[ref]:[password]@[host]:6543/postgres
JWT_SECRET=your-secret-here
ML_SERVICE_URL=https://avertx-ml.onrender.com   # omit for local dev
```

`ML_SERVICE_URL`: when set, the backend forwards all prediction calls to
the ML microservice instead of loading the model in-process. Leave unset
for local development to use the local model directly.

---

## 3. Architecture

```
Browser (React + Vite)
        |
        | HTTPS
        v
Backend API  (FastAPI — auth, reports, DB, lifecycle)
        |
        |-- Supabase PostgreSQL (users, reports, audit_log)
        |
        | HTTP (when ML_SERVICE_URL is set)
        v
ML Service  (FastAPI — ONNX encoder, LR classifiers)
```

### Backend (`backend/`)

- **FastAPI** — all endpoints, JWT auth, CORS
- **PostgreSQL via psycopg2** — Supabase hosted database
- **`database.py`** — all DB queries (drop-in replacement for the
  original SQLite version, which is preserved as `database_sqlite.py`)
- **`model.py`** — model loading with ONNX-first, SentenceTransformer
  fallback; lazy LIME loading
- **`onnx_encoder.py`** — pure ONNX + HuggingFace tokenizer encoder,
  no PyTorch dependency at runtime
- **`sif_output_contract.py`** — 3-way SIF decision (Yes / No /
  NEEDS_REVIEW) replacing the original single-threshold binary call
- **`rule_output_contract.py`** — 3-way Life-Saving Rule prediction

### ML Service (`ml_service/`)

Separate lightweight microservice carrying only the inference stack.
Deployed independently on Render free tier (~200 MB RAM).

- `POST /predict` — takes `{report_text, site, activity}`, returns
  `{sif_potential, confidence, life_saving_rule, rule_confidence, risk_score}`
- `GET /health` — liveness check

### Frontend (`frontend/src/App.tsx`)

React + TypeScript + Vite + Tailwind CSS + Recharts. Single-file
component. API base is `VITE_API_URL` (production) or `/api` proxy
(local dev via `vite.config.ts`).

### Database (Supabase)

Three tables: `users`, `reports`, `audit_log`. All with Row Level
Security enabled. Schema: `supabase/migrations/001_initial_schema.sql`.
Seeded with 1,325 reports from `oil_safety_reports_merged.csv`.

---

## 4. Model

**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim),
exported to ONNX (`model_artifacts/onnx_encoder/`) for production
deployment without PyTorch.

**Classifiers:** Logistic Regression (scikit-learn, `class_weight=
"balanced"`) trained on 927 rows from the Phase 10 split.

**Performance (Phase 11 benchmark on real-only gold test set, 199 rows):**

| Metric | all-MiniLM-L6-v2 | BAAI/bge-small-en-v1.5 |
|---|---|---|
| Threshold (val-selected) | 0.13 | 0.13 |
| Precision | 0.352 | 0.347 |
| Recall | 1.000 | 1.000 |
| F1 | 0.521 | 0.515 |
| PR-AUC | 0.494 | 0.477 |
| FN rate | 0.000 | 0.000 |
| Latency (CPU) | 17.9 ms | 29.8 ms |

SafetyBERT was evaluated but skipped: both candidates on HuggingFace
(`adanish91/safetybert`, `invalidexception/safetybert`) lack a declared
license and neither is suitable for this task — see
`data/benchmark_results.md` for the full finding.

**Why precision is low:** the model learned topic patterns from synthetic
data, not severity signals from real incidents. This is a data problem,
not a model capacity problem — see `PHASE9_DECISION.md` and
`data/threshold_tradeoff_summary.md`. The next investment should be
better labels and more real safety narratives before testing stronger
models.

**Decision output:** both classifiers return one of three values
(Yes / No / NEEDS_REVIEW) via output contracts, not a forced binary. A
NEEDS_REVIEW result surfaces to the reviewer queue rather than silently
becoming a Yes or No.

---

## 5. Data pipeline

The corpus behind this project was built across Phases 3–7:

| Source | Rows | Notes |
|---|---|---|
| `sif_guard_seed` (IHM Stefanini real + synthetic) | 1,325 | Labeled — used for training |
| OSHA severe injury reports | 21,578 | Unlabeled reserve |
| HSE construction abstracts 2015–17 | 4,847 | Unlabeled reserve |
| NIOSH injury narrative coding | 153,956 | Unlabeled reserve |
| PHMSA hazmat incidents | 678,309 | Unlabeled reserve |
| **Total** | **860,440** | 1,325 labeled / 859,115 unlabeled |

Key pipeline scripts (all read-only analysis unless noted):

| Script | Purpose |
|---|---|
| `data/build_splits.py` | Phase 10: train/val/test splits (leakage-checked) |
| `data/benchmark_models.py` | Phase 11: embedding model benchmark |
| `data/analyze_threshold_tradeoff.py` | Phase 11 post-hoc diagnostic |
| `data/analyze_real_synthetic_ratio.py` | Corpus composition report |
| `data/survey_severity_signals.py` | Severity field survey across raw sources |
| `data/extract_structured_fields.py` | Phase 7 (scoped): keyword extraction |
| `scripts/seed_supabase.py` | One-shot DB seed from CSV |
| `scripts/export_encoder_onnx.py` | Export MiniLM to ONNX format |

Splits: `data/splits/train.csv` (927 rows), `validation.csv` (199),
`test.csv` (199 — real-only, no duplicates, no synthetic).

---

## 6. Accounts & roles

Bootstrap admin accounts (seeded automatically on first startup):

| Email | Password |
|---|---|
| `admin1@email.com` | `admin123` |
| `admin2@oil-hsse.local` | `ChangeMe@Admin2` |

**Change these passwords after first login in any real deployment.**

| Role | Can do |
|---|---|
| **Reporter** | Submit reports, view own submissions only |
| **Supervisor** | Review queue (critical-first), advance report status |
| **HSE** | Everything Supervisor + analytics dashboard, bulk upload, audit log |
| **Admin** | User management, role changes (all audit-logged) |

Role enforcement is server-side, not just UI. Every role change is
written to `audit_log`.

---

## 7. Report lifecycle

```
Submitted → Reviewed → Investigation → Action Assigned → Closed
```

A report is auto-flagged **critical** the moment `sif_potential = Yes`
with risk score ≥ 70. Critical reports sort to the top of the
Supervisor/HSE inbox. Every status advance is audit-logged.

---

## 8. API reference

**Auth (no token):**
- `POST /auth/signup` — creates Reporter account, returns `{token, user}`
- `POST /auth/login` — returns `{token, user}`
- `GET /health` — liveness check

**Any authenticated role:**
- `GET /auth/me`
- `POST /analyze/text` — classify + persist a text report
- `POST /analyze/pdf` — classify + persist a PDF (OCR fallback)
- `GET /reports` — paginated; Reporters see own only, others see critical-first
- `GET /reports/filters` — distinct site/activity/rule for dropdowns
- `GET /dashboard/my-stats` — own report counts

**Supervisor / HSE / Admin:**
- `POST /reports/{id}/status` — advance lifecycle status
- `GET /dashboard/stats` — org-wide SIF density, site risk, precursors

**HSE / Admin:**
- `POST /reports/bulk-upload` — CSV/Excel batch analysis
- `GET /audit-log`

**Admin:**
- `GET /admin/users`
- `POST /admin/users/{id}/role`
- `POST /admin/users/{id}/status`

---

## 9. Project structure

```
AvertX/
├── backend/
│   ├── main.py                   FastAPI app, all endpoints
│   ├── auth.py                   bcrypt, JWT, role-check dependencies
│   ├── database.py               PostgreSQL/Supabase queries (psycopg2)
│   ├── database_sqlite.py        Original SQLite version (kept for local fallback)
│   ├── model.py                  Model loading (ONNX-first), LIME, risk score
│   ├── onnx_encoder.py           ONNX inference wrapper (no PyTorch)
│   ├── sif_output_contract.py    3-way SIF decision contract
│   ├── rule_output_contract.py   3-way rule prediction contract
│   ├── requirements.txt          Full deps (local dev, includes torch)
│   └── requirements-render.txt  Stripped deps for Render free tier (~103MB RAM)
│
├── ml_service/
│   ├── main.py                   Inference microservice (POST /predict)
│   └── requirements.txt          onnxruntime, transformers, sklearn only
│
├── frontend/
│   └── src/App.tsx               Entire UI (React + TS + Vite + Tailwind)
│
├── model_artifacts/
│   ├── onnx_encoder/             MiniLM exported to ONNX (committed, ~87MB)
│   ├── sif_classifier.joblib     Trained LR classifier
│   ├── rule_classifier.joblib    Trained rule LR classifier
│   └── model_metadata.json       Threshold, class lists, rule severities
│
├── data/
│   ├── splits/                   train/val/test CSVs (Phase 10)
│   ├── processed/                safety_master.csv, extracted fields
│   ├── benchmark_results.json    Phase 11 model benchmark (full metrics)
│   ├── benchmark_results.md      Human-readable benchmark table
│   ├── threshold_tradeoff.csv    Full precision/recall curve data
│   ├── threshold_tradeoff_summary.md  Post-hoc diagnostic
│   └── [pipeline scripts]        see section 5
│
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql  Full DB schema + RLS + seed
│
├── scripts/
│   ├── seed_supabase.py          One-shot report seeding
│   └── export_encoder_onnx.py    MiniLM → ONNX export
│
├── taxonomy/
│   ├── life_saving_rules.csv     IOGP rules + hazard/barrier examples
│   └── legacy_term_mapping.csv   Source vocabulary → rule mapping
│
├── oil_safety_reports_merged.csv  Training dataset (1,325 rows)
├── life_saving_rules_taxonomy.json IOGP rule → hazard/barrier lookup
├── train_model.py                 Trains sif_classifier.joblib
├── train_rule_classifier.py       Trains rule_classifier.joblib
├── .env.example                   Environment variable template
├── DATASET_AUDIT.md               Phase 1 data quality audit
├── DATA_SOURCES.md                All dataset provenance + licenses
├── PHASE9_DECISION.md             Labels/ratio decision record
├── MODEL_COMPARISON.md            Phase 18 model selection record
├── README_AI.md                   AI collaboration notes (indexes decisions)
└── run_app.bat                    Starts both servers (local dev)
```

---

## 10. Deployment (Render free tier)

Three separate services, all free:

### ML Service (`avertx-ml`)

| Field | Value |
|---|---|
| Repo | `diitrishu/AvertX` |
| Root Directory | *(blank)* |
| Build Command | `pip install -r ml_service/requirements.txt` |
| Start Command | `uvicorn ml_service.main:app --host 0.0.0.0 --port $PORT` |
| Environment vars | none |

### Backend API (`avertx`)

| Field | Value |
|---|---|
| Build Command | `pip install -r backend/requirements-render.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Environment vars | `SUPABASE_DB_URL`, `JWT_SECRET`, `ML_SERVICE_URL` |

### Frontend (`avertx-frontend`)

| Field | Value |
|---|---|
| Root Directory | `frontend` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |
| Environment vars | `VITE_API_URL=https://avertx.onrender.com` |
| Redirects/Rewrites | `/* → /index.html` (Rewrite) |

Deploy ML service first. Get its URL. Set `ML_SERVICE_URL` on the
backend before deploying it.

**RAM budget (free tier = 512 MB):**

| Service | Actual RAM |
|---|---|
| Backend (no ML) | ~150 MB |
| ML service (ONNX, no torch) | ~200 MB |
| Frontend (static) | 0 MB |

---

## 11. Known limitations

- **Model precision is low** (0.352 on real test set at full recall) — this
  is a label quality and training data problem, not a model architecture
  problem. Short or ambiguous inputs may get spurious SIF-Yes predictions.
  The NEEDS_REVIEW output class exists specifically to handle low-confidence
  cases rather than forcing a wrong answer.
- Free-tier cold starts: both backend and ML service sleep after 15 min
  idle. Wake them via `/health` before use.
- No SSO/MFA — JWT + bcrypt only.
- No org/site-scoped permissions — Supervisor sees all open reports.
- Admin credentials are seeded in source — change after first login.
- LIME explanations are disabled on the Render deployment (not available
  via remote ML service call). Available in local dev only.
- `pytesseract` PDF OCR requires Tesseract to be installed separately
  on the host OS. It is not available on Render's free tier.

---

## 12. Troubleshooting

**Analyze returns an error in the UI:** both services may be cold. Visit
`https://avertx.onrender.com/health` and
`https://avertx-ml.onrender.com/health` to wake them, then retry.

**Backend starts but predictions fail:** check `ML_SERVICE_URL` is set in
Render's environment variables for the `avertx` service. If not set, the
backend tries to load the model locally and will OOM on free tier.

**"Not Found" on the root URL:** expected — the API has no root endpoint.
Use `/health` to confirm the service is alive.

**Frontend routes return 404 on refresh:** the Redirects/Rewrites rule
(`/* → /index.html, Rewrite`) is not set on the static site. Add it in
Render → avertx-frontend → Redirects/Rewrites.

**Local backend: model loads but DB connection fails:** `SUPABASE_DB_URL`
is not in your `.env` file. Copy `.env.example` and fill in the value
from Supabase dashboard → Project Settings → Database → Connection string
(Transaction mode, port 6543).

**Re-running splits/training locally:**
```bash
python data/build_splits.py        # Phase 10 splits
python train_model.py              # SIF classifier
python train_rule_classifier.py    # rule classifier
python scripts/export_encoder_onnx.py  # ONNX export after retraining
```
