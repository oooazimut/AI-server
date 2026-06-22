-- Sprint 11: move agent data from public schema to per-agent schemas.
-- Run ONCE on the production DB before deploying the Sprint 11 build.
-- Idempotent: ensure_schema() uses CREATE IF NOT EXISTS, so re-runs are safe.

-- Create target schemas (ensure_schema() also does this, but this is explicit)
CREATE SCHEMA IF NOT EXISTS bitrix24;
CREATE SCHEMA IF NOT EXISTS logistics;
CREATE SCHEMA IF NOT EXISTS pto;

-- ── bitrix24 ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bitrix24.incomplete_proposals (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    responsible_id INTEGER,
    deadline TEXT,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending_review',
    created_at TEXT NOT NULL,
    manager_notified_at TEXT,
    responsible_response TEXT,
    responded_at TEXT
);

INSERT INTO bitrix24.incomplete_proposals
SELECT * FROM public.incomplete_proposals
ON CONFLICT DO NOTHING;

DROP TABLE IF EXISTS public.incomplete_proposals;

-- ── logistics ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS logistics.employees (
    id INTEGER PRIMARY KEY,
    bitrix_user_id INTEGER,
    full_name TEXT NOT NULL,
    position TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS logistics.vehicles (
    id INTEGER PRIMARY KEY,
    brand_model TEXT NOT NULL,
    registration_number TEXT NOT NULL,
    debit_card_number TEXT NOT NULL DEFAULT '',
    ppr_card_number TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS logistics.vehicle_usage_requests (
    id SERIAL PRIMARY KEY,
    request_date TEXT NOT NULL,
    user_id INTEGER,
    dialog_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    sent_at TEXT,
    response_text TEXT,
    responded_at TEXT,
    parsed_json TEXT,
    reminder_count INTEGER NOT NULL DEFAULT 0,
    last_reminder_at TEXT,
    escalated_at TEXT,
    UNIQUE (request_date, user_id)
);

CREATE TABLE IF NOT EXISTS logistics.employee_daily_statuses (
    id SERIAL PRIMARY KEY,
    status_date TEXT NOT NULL,
    employee_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE (status_date, employee_id)
);

CREATE TABLE IF NOT EXISTS logistics.vehicle_daily_assignments (
    id SERIAL PRIMARY KEY,
    assignment_date TEXT NOT NULL,
    vehicle_id INTEGER NOT NULL,
    employee_id INTEGER,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE (assignment_date, vehicle_id)
);

INSERT INTO logistics.employees SELECT * FROM public.employees ON CONFLICT DO NOTHING;
INSERT INTO logistics.vehicles SELECT * FROM public.vehicles ON CONFLICT DO NOTHING;
INSERT INTO logistics.vehicle_usage_requests SELECT * FROM public.vehicle_usage_requests ON CONFLICT DO NOTHING;
INSERT INTO logistics.employee_daily_statuses SELECT * FROM public.employee_daily_statuses ON CONFLICT DO NOTHING;
INSERT INTO logistics.vehicle_daily_assignments SELECT * FROM public.vehicle_daily_assignments ON CONFLICT DO NOTHING;

DROP TABLE IF EXISTS public.employees;
DROP TABLE IF EXISTS public.vehicles;
DROP TABLE IF EXISTS public.vehicle_usage_requests;
DROP TABLE IF EXISTS public.employee_daily_statuses;
DROP TABLE IF EXISTS public.vehicle_daily_assignments;
