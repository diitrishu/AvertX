-- ============================================================
-- SIF-Guard: Initial Supabase schema migration
-- Translated from backend/database.py (SQLite)
-- Project ref: nupcindayjsyesaflvmp
-- ============================================================
-- Run order: this file only. No prior migrations exist.
-- Safe to re-run (all statements use IF NOT EXISTS / OR IGNORE).
-- ============================================================

-- ------------------------------------------------------------
-- EXTENSIONS
-- ------------------------------------------------------------
-- pgcrypto is used for password hashing on the server side
-- (bcrypt stays in the Python backend; this is just a safety net)
create extension if not exists pgcrypto;


-- ------------------------------------------------------------
-- TABLE: users
-- Matches database.py: id, name, email, password_hash, role,
-- site, status, created_at
-- Roles: Reporter (default), Supervisor, HSE, Admin
-- ------------------------------------------------------------
create table if not exists users (
    id          bigserial primary key,
    name        text      not null,
    email       text      not null unique,
    password_hash text    not null,
    role        text      not null default 'Reporter'
                check (role in ('Reporter', 'Supervisor', 'HSE', 'Admin')),
    site        text      not null default '',
    status      text      not null default 'active'
                check (status in ('active', 'disabled')),
    created_at  timestamptz not null default now()
);

-- Index for login lookups by email
create index if not exists idx_users_email on users (email);


-- ------------------------------------------------------------
-- TABLE: reports
-- Matches database.py insert_report() including all Phase 16
-- explainability columns added via ALTER TABLE migrations.
-- ------------------------------------------------------------
create table if not exists reports (
    id                          bigserial primary key,
    report_id                   text      unique,
    report_text                 text,
    date                        text,
    site                        text,
    activity                    text,
    report_type                 text,
    source                      text,

    -- AI classification outputs
    sif_potential               text,
    life_saving_rule            text,
    confidence                  double precision,
    risk_score                  integer,
    rule_confidence             double precision,
    risk_level                  text,
    top_phrases                 text,   -- JSON array stored as text
    analyzed_at                 timestamptz,

    -- Lifecycle
    reporter_id                 bigint references users(id) on delete set null,
    status                      text    not null default 'Submitted'
                                check (status in (
                                    'Submitted', 'Reviewed',
                                    'Investigation', 'Action Assigned', 'Closed'
                                )),
    critical                    integer not null default 0, -- 0/1 boolean

    -- Phase 16: explainability metadata (nullable; NULL for legacy/seeded rows)
    sif_reasons                 text,   -- JSON
    sif_model_name              text,
    sif_model_version           text,
    sif_decision_thresholds     text,   -- JSON
    sif_prediction_timestamp    text,
    rule_needs_review           integer not null default 0,
    rule_review_reason          text,
    rule_runner_up              text,
    rule_runner_up_confidence   double precision,
    rule_model_name             text,
    rule_model_version          text,
    rule_prediction_timestamp   text,

    -- Timestamps
    created_at  timestamptz not null default now()
);

-- Indexes matching the query patterns in database.py
create index if not exists idx_reports_reporter_id  on reports (reporter_id);
create index if not exists idx_reports_status        on reports (status);
create index if not exists idx_reports_sif_potential on reports (sif_potential);
create index if not exists idx_reports_site          on reports (site);
create index if not exists idx_reports_critical      on reports (critical);
-- Critical-first inbox sort used in get_reports()
create index if not exists idx_reports_inbox_sort
    on reports (critical desc, status, id desc);


-- ------------------------------------------------------------
-- TABLE: audit_log
-- Matches database.py log_audit() exactly.
-- Every safety-critical change (role change, status change,
-- critical flag raised, etc.) is appended here — never updated.
-- ------------------------------------------------------------
create table if not exists audit_log (
    id              bigserial primary key,
    actor_user_id   bigint references users(id) on delete set null,
    actor_name      text,
    action          text      not null,
    target_type     text      not null,
    target_id       text      not null,
    field           text,
    old_value       text,
    new_value       text,
    reason          text,
    created_at      timestamptz not null default now()
);

create index if not exists idx_audit_log_target
    on audit_log (target_type, target_id);
create index if not exists idx_audit_log_actor
    on audit_log (actor_user_id);


-- ------------------------------------------------------------
-- ROW LEVEL SECURITY
-- Enable RLS on all three tables.
-- The FastAPI backend connects via the service role key
-- (bypasses RLS) so these policies are a defence-in-depth
-- layer against direct PostgREST/client access.
-- ------------------------------------------------------------

alter table users      enable row level security;
alter table reports    enable row level security;
alter table audit_log  enable row level security;

-- Service role bypasses RLS automatically in Supabase.
-- These policies cover the anon/authenticated roles that
-- the frontend might use if it ever queries Supabase directly.
-- Note: CREATE POLICY IF NOT EXISTS requires PG16; we use DO blocks for PG15.

-- users: no direct client access (all user ops go through backend)
do $$ begin
  if not exists (select 1 from pg_policies where tablename='users' and policyname='users_no_anon_access') then
    execute 'create policy "users_no_anon_access" on users for all to anon using (false)';
  end if;
end $$;

do $$ begin
  if not exists (select 1 from pg_policies where tablename='users' and policyname='users_no_authenticated_direct_access') then
    execute 'create policy "users_no_authenticated_direct_access" on users for all to authenticated using (false)';
  end if;
end $$;

-- reports: same — backend-only via service role
do $$ begin
  if not exists (select 1 from pg_policies where tablename='reports' and policyname='reports_no_anon_access') then
    execute 'create policy "reports_no_anon_access" on reports for all to anon using (false)';
  end if;
end $$;

do $$ begin
  if not exists (select 1 from pg_policies where tablename='reports' and policyname='reports_no_authenticated_direct_access') then
    execute 'create policy "reports_no_authenticated_direct_access" on reports for all to authenticated using (false)';
  end if;
end $$;

-- audit_log: append-only via backend; no direct client reads
do $$ begin
  if not exists (select 1 from pg_policies where tablename='audit_log' and policyname='audit_log_no_anon_access') then
    execute 'create policy "audit_log_no_anon_access" on audit_log for all to anon using (false)';
  end if;
end $$;

do $$ begin
  if not exists (select 1 from pg_policies where tablename='audit_log' and policyname='audit_log_no_authenticated_direct_access') then
    execute 'create policy "audit_log_no_authenticated_direct_access" on audit_log for all to authenticated using (false)';
  end if;
end $$;


-- ------------------------------------------------------------
-- SEED: Bootstrap admin accounts
-- Mirrors database.py DEFAULT_ADMINS.
-- Passwords use pgcrypto bcrypt (cost 12) so they are
-- compatible with the Python bcrypt library used in auth.py.
-- IMPORTANT: change these passwords immediately after first login
-- in any non-development deployment.
-- ------------------------------------------------------------
insert into users (name, email, password_hash, role, status)
values
    (
        'System Admin 1',
        'admin1@email.com',
        crypt('admin123', gen_salt('bf', 12)),
        'Admin',
        'active'
    ),
    (
        'System Admin 2',
        'admin2@oil-hsse.local',
        crypt('ChangeMe@Admin2', gen_salt('bf', 12)),
        'Admin',
        'active'
    )
on conflict (email) do nothing;
