-- Schema for PLAN.md section 2.
--
-- Timestamps are UTC ISO-8601 strings. The one deliberate exception is a
-- class's weekly start_time/end_time, which are local wall-clock "HH:MM":
-- a class that meets at 09:00 meets at 09:00 on both sides of a daylight
-- saving change, and storing those as UTC would silently move them an hour.
-- Dates on course_exception are local "YYYY-MM-DD" for the same reason.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS course (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT    NOT NULL,
    title       TEXT,
    instructor  TEXT,
    location    TEXT,
    weekday     INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    start_time  TEXT    NOT NULL,
    end_time    TEXT    NOT NULL,
    term_start  TEXT,
    term_end    TEXT,
    notes       TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    deleted_at  TEXT
);

CREATE TABLE IF NOT EXISTS course_exception (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id    INTEGER NOT NULL REFERENCES course(id),
    date         TEXT    NOT NULL,
    kind         TEXT    NOT NULL CHECK (kind IN ('cancelled', 'moved', 'room_change')),
    new_start    TEXT,
    new_end      TEXT,
    new_location TEXT,
    note         TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    deleted_at   TEXT
);

CREATE TABLE IF NOT EXISTS assignment (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id    INTEGER REFERENCES course(id),
    title        TEXT    NOT NULL,
    due_at       TEXT,
    est_hours    REAL,
    progress_pct INTEGER NOT NULL DEFAULT 0 CHECK (progress_pct BETWEEN 0 AND 100),
    status       TEXT    NOT NULL DEFAULT 'todo'
                 CHECK (status IN ('todo', 'in_progress', 'done', 'dropped')),
    notes        TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    deleted_at   TEXT
);

CREATE TABLE IF NOT EXISTS reminder (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    remind_at    TEXT    NOT NULL,
    related_type TEXT CHECK (related_type IN ('course', 'assignment') OR related_type IS NULL),
    related_id   INTEGER,
    notes        TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    deleted_at   TEXT
);

-- Invariant #3: every turn writes a row here, including errors, empty
-- transcripts, and turns that ended in a clarification. This is the probe's
-- training data and the only debugging surface for a system whose input is
-- sound. Append-only, so it carries no soft-delete columns.
CREATE TABLE IF NOT EXISTS turn_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    ts                TEXT    NOT NULL,
    transcript        TEXT,
    asr_conf          REAL,
    probe_score       REAL,
    probe_label       TEXT,
    tool_name         TEXT,
    tool_args_json    TEXT,
    tool_result_json  TEXT,
    reply_text        TEXT,
    ms_asr            INTEGER,
    ms_prefill        INTEGER,
    ms_gen            INTEGER,
    ms_tts            INTEGER,
    hidden_state_path TEXT
);

-- Invariant #4: writes go through here and deletes are soft, so
-- undo_last_write can reverse any mutation. undone_at is an addition to the
-- column list in PLAN.md: without it a second undo would redo the first.
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    table_name  TEXT    NOT NULL,
    row_id      INTEGER NOT NULL,
    op          TEXT    NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
    before_json TEXT,
    after_json  TEXT,
    turn_id     INTEGER REFERENCES turn_log(id),
    undone_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_course_live      ON course(deleted_at, weekday);
CREATE INDEX IF NOT EXISTS idx_exception_course ON course_exception(course_id, date);
CREATE INDEX IF NOT EXISTS idx_assignment_due   ON assignment(deleted_at, due_at);
CREATE INDEX IF NOT EXISTS idx_reminder_at      ON reminder(deleted_at, remind_at);
CREATE INDEX IF NOT EXISTS idx_audit_open       ON audit_log(undone_at, id);
CREATE INDEX IF NOT EXISTS idx_turn_session     ON turn_log(session_id, ts);
