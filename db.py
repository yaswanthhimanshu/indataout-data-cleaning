# db.py — MySQL persistence layer for inDataOut
# Tables: projects, cleaning_history, dataset_versions
# No auth — project_id tracked via session cookie

"""MySQL metadata and audit-history layer for inDataOut.

This module owns project rows, ordered cleaning-history rows, per-action
manifests, and immutable dataset-version metadata. It does not read or write
DataFrame files directly; callers in ``app.py`` coordinate filesystem snapshots
and decide when a database transaction is safe to commit.

Connection helpers commit on successful context exit and roll back on raised
exceptions. Multi-resource operations that span MySQL plus the filesystem are
mitigated by app-level temp-file promotion; they are not true distributed
transactions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

# python-dotenv's default load_dotenv() searches upward from the current
# working directory, which is fine for `python app.py` but not reliable for a
# packaged .exe launched via double-click (cwd there can be anything,
# depending on how Windows/pywebview invokes it). Point it explicitly at the
# folder the running program actually lives in instead.
if getattr(sys, "frozen", False):
    _env_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    _env_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_env_dir, ".env"))

log = logging.getLogger(__name__)


class StaleDatasetVersionError(RuntimeError):
    """Raised before persistence when a mutation targets an older state."""

    def __init__(self, expected: int, current: int):
        super().__init__(f"stale dataset version: expected {expected}, current {current}")
        self.expected = expected
        self.current = current

# --- DB config -----------------------------------------------------------
# Pulled from environment / .env — see .env.example. DB_PASSWORD has no
# default on purpose: fail loudly at startup rather than silently trying
# an empty password against a real MySQL server.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "indataout"),
}

# --- Connection pool ---------------------------------------------------------

_pool: Optional[pooling.MySQLConnectionPool] = None


def _get_pool() -> pooling.MySQLConnectionPool:
    """Return the process-wide connection pool, creating it lazily."""
    # Lazy init; reuses pool across requests
    global _pool
    if _pool is None:
        if not DB_CONFIG["password"]:
            raise RuntimeError(
                "DB_PASSWORD is not set. Copy .env.example to .env and fill in "
                "your MySQL credentials before starting the app."
            )
        _pool = pooling.MySQLConnectionPool(
            pool_name="indataout",
            pool_size=5,                                          # max concurrent connections
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            autocommit=False,
            charset="utf8mb4",
            use_unicode=True,
        )
        log.info("MySQL pool ready")
    return _pool


@contextmanager
def _conn():
    """Yield a MySQL connection and own commit/rollback for this DB unit."""
    # Commit on success, rollback + re-raise on error, always release to pool
    cnx = _get_pool().get_connection()
    try:
        yield cnx
        cnx.commit()
    except Exception:
        cnx.rollback()
        raise
    finally:
        cnx.close()


# --- Schema ------------------------------------------------------------------

_CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS projects (
        id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(255)  NOT NULL,
        notes         TEXT,
        original_file VARCHAR(255)  NOT NULL,
        upload_sid    VARCHAR(64)   NOT NULL,       -- links to session upload folder
        file_path     VARCHAR(512),                 -- latest cleaned file path
        row_count     INT UNSIGNED  DEFAULT 0,
        col_count     INT UNSIGNED  DEFAULT 0,
        state_version BIGINT UNSIGNED NOT NULL DEFAULT 0,
        status        ENUM('active','cleaned','archived') DEFAULT 'active',
        created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                                             ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_upload_sid (upload_sid),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS cleaning_history (
        id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        project_id    INT UNSIGNED  NOT NULL,
        action_name   VARCHAR(128)  NOT NULL,
        action_config JSON,                         -- serialised options dict
        rows_before   INT UNSIGNED  DEFAULT 0,
        rows_after    INT UNSIGNED  DEFAULT 0,
        cells_changed INT UNSIGNED  DEFAULT 0,
        status        ENUM('applied','undone') NOT NULL DEFAULT 'applied',
        dq_after      DECIMAL(5,2),                 -- cached data_quality_score(cleaned_df) at apply time
        version_before BIGINT UNSIGNED NOT NULL DEFAULT 0,
        version_after  BIGINT UNSIGNED NOT NULL DEFAULT 0,
        applied_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
        INDEX idx_project_id (project_id),
        -- get_cleaning_history() filters WHERE project_id = %s and orders BY
        -- applied_at ASC — the single-column index above covers the filter
        -- but not the ordering, so MySQL still has to filesort the whole
        -- per-project result set (the "Out of sort memory" error on large
        -- histories). This composite index covers filter + order together,
        -- so no sort step is needed at all. (A standalone index on
        -- applied_at alone would NOT fix this — it isn't scoped per-project,
        -- so MySQL can't use it to avoid the sort for a filtered query.)
        INDEX idx_project_applied (project_id, applied_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_versions (
        id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        project_id    INT UNSIGNED  NOT NULL,
        version_no    SMALLINT UNSIGNED NOT NULL DEFAULT 1,  -- scoped per project
        file_path     VARCHAR(512)  NOT NULL,
        label         VARCHAR(128),
        row_count     INT UNSIGNED  DEFAULT 0,
        col_count     INT UNSIGNED  DEFAULT 0,
        created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
        UNIQUE KEY uq_project_version (project_id, version_no),
        INDEX idx_project_id (project_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    -- Per-action manifests (the precise cell/row/column diff that powers
    -- independent Undo/Reapply), keyed 1:1 to cleaning_history via history_id.
    -- Deliberately a SEPARATE table, not a column on cleaning_history —
    -- that's what keeps this from reintroducing the JSON-column bloat/
    -- "Out of sort memory" problem manifests were moved out of the DB for
    -- in the first place: get_cleaning_history()'s query never touches this
    -- table at all, so a manifest's size has zero effect on it.
    --
    -- This replaces file-based manifest storage (uploads/<sid>/manifests/),
    -- which turned out to have a real failure mode in practice: a file can
    -- go missing (moved app folder, temp-dir cleanup, UPLOAD_FOLDER change)
    -- while the DB row referencing it lives on — silently breaking Undo/
    -- Reapply with no way to tell why. A LONGTEXT column in the same
    -- database as everything else doesn't have that failure mode.
    CREATE TABLE IF NOT EXISTS manifests (
        id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        history_id    INT UNSIGNED NOT NULL,
        manifest      LONGTEXT NOT NULL,
        created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (history_id) REFERENCES cleaning_history(id) ON DELETE CASCADE,
        UNIQUE KEY uq_history_id (history_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def _add_column_if_missing(cur, table: str, column: str, ddl_clause: str) -> None:
    """MySQL 5.7 doesn't support `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`
    (that syntax is MySQL 8.0.29+ only) — it would raise a syntax error there,
    which the old code silently swallowed via a bare `except: pass`, so
    migrations quietly never ran on 5.7 at all. Checking information_schema
    first (same pattern setup_database.py already uses) works identically on
    5.7 and 8.0, and lets a genuine failure surface instead of being masked.

    `ddl_clause` is the column definition + placement, e.g.
    "TEXT AFTER name" or "DECIMAL(5,2) AFTER status".
    """
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (DB_CONFIG["database"], table, column),
    )
    exists = cur.fetchone()[0]
    if exists:
        return
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_clause}")
    log.info(f"Migration: added {table}.{column}")


def _add_index_if_missing(cur, table: str, index_name: str, index_columns: str) -> None:
    """Same rationale as _add_column_if_missing but for indexes — checking
    information_schema works identically on MySQL 5.7 and 8.0. index_columns
    is the parenthesised column list, e.g. "(project_id, applied_at)".
    """
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s
        """,
        (DB_CONFIG["database"], table, index_name),
    )
    exists = cur.fetchone()[0]
    if exists:
        return
    cur.execute(f"ALTER TABLE {table} ADD INDEX {index_name} {index_columns}")
    log.info(f"Migration: added index {table}.{index_name}")


def init_db() -> None:
    """Create or migrate schema safely without deleting existing user data."""
    # Called once at Flask startup; idempotent (IF NOT EXISTS)
    with _conn() as cnx:
        cur = cnx.cursor()
        for sql in _CREATE_TABLES:
            cur.execute(sql)
        # Migration: add notes column if it doesn't exist (existing DBs)
        try:
            _add_column_if_missing(cur, "projects", "notes", "TEXT AFTER name")
        except Exception:
            log.exception("Migration failed: projects.notes")
        try:
            _add_column_if_missing(cur, "projects", "state_version", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER col_count")
        except Exception:
            log.exception("Migration failed: projects.state_version")
        # Migration: add status column to cleaning_history if it doesn't exist
        # (existing DBs created before Undo/Reapply synced to the server)
        try:
            _add_column_if_missing(
                cur, "cleaning_history", "status",
                "ENUM('applied','undone') NOT NULL DEFAULT 'applied' AFTER cells_changed",
            )
        except Exception:
            log.exception("Migration failed: cleaning_history.status")
        # Migration: add dq_after column to cleaning_history if it doesn't exist
        # (existing DBs created before per-action DQ-score caching)
        try:
            _add_column_if_missing(cur, "cleaning_history", "dq_after", "DECIMAL(5,2) AFTER status")
        except Exception:
            log.exception("Migration failed: cleaning_history.dq_after")
        try:
            _add_column_if_missing(cur, "cleaning_history", "version_before", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER dq_after")
            _add_column_if_missing(cur, "cleaning_history", "version_after", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER version_before")
        except Exception:
            log.exception("Migration failed: cleaning_history dataset versions")
        # Migration: add the composite (project_id, applied_at) index if it
        # doesn't exist (existing DBs only had the single-column project_id
        # index, which doesn't cover get_cleaning_history's ORDER BY —
        # causing "Out of sort memory" on large per-project histories)
        try:
            _add_index_if_missing(cur, "cleaning_history", "idx_project_applied", "(project_id, applied_at)")
        except Exception:
            log.exception("Migration failed: cleaning_history.idx_project_applied")
        cur.close()
    log.info("DB tables verified")


# --- Projects ----------------------------------------------------------------

def create_project(
    name: str,
    original_file: str,
    upload_sid: str,
    file_path: str,
    row_count: int = 0,
    col_count: int = 0,
) -> int:
    """Create a project metadata row for one immutable uploaded source."""
    # Returns auto-incremented project id
    sql = """
        INSERT INTO projects (name, original_file, upload_sid, file_path, row_count, col_count)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute(sql, (name, original_file, upload_sid, file_path, row_count, col_count))
        pid = cur.lastrowid
        cur.close()
    log.info(f"project id={pid} created")
    return pid


def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    """Return one project row or ``None``; callers enforce session ownership."""
    # Returns row as dict or None
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
        row = cur.fetchone()
        cur.close()
    return row


def list_projects(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent projects with lightweight history summary fields."""
    # Use a subquery to pre-aggregate cleaning_history so MySQL doesn't have to
    # sort a huge cross-product. This avoids the "Out of sort memory" error.
    sql = """
        SELECT
            p.*,
            COALESCE(h.action_count, 0)  AS action_count,
            h.last_action_at,
            h.last_action_name
        FROM projects p
        LEFT JOIN (
            SELECT
                project_id,
                COUNT(*)       AS action_count,
                MAX(applied_at) AS last_action_at,
                SUBSTRING_INDEX(GROUP_CONCAT(action_name ORDER BY applied_at DESC SEPARATOR '|||'), '|||', 1) AS last_action_name
            FROM cleaning_history
            GROUP BY project_id
        ) h ON h.project_id = p.id
        ORDER BY p.updated_at DESC
        LIMIT %s
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
        cur.close()
    return rows


def update_project_status(project_id: int, status: str, file_path: str = None) -> None:
    """Update status and, when supplied, the latest cleaned-file pointer."""
    # file_path optional — only update when cleaned file path changes
    if file_path:
        sql, args = "UPDATE projects SET status=%s, file_path=%s WHERE id=%s", (status, file_path, project_id)
    else:
        sql, args = "UPDATE projects SET status=%s WHERE id=%s", (status, project_id)
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute(sql, args)
        cur.close()


def delete_project(project_id: int) -> None:
    """Delete project metadata; related DB rows cascade in MySQL."""
    # CASCADE removes cleaning_history + dataset_versions rows automatically
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
        cur.close()
    log.info(f"project id={project_id} deleted")


def update_project_name(project_id: int, name: str) -> None:
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute("UPDATE projects SET name=%s WHERE id=%s", (name, project_id))
        cur.close()
    log.info(f"project id={project_id} name updated")


def update_project_notes(project_id: int, notes: str) -> None:
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute("UPDATE projects SET notes=%s WHERE id=%s", (notes, project_id))
        cur.close()
    log.info(f"project id={project_id} notes updated")


# --- Cleaning history --------------------------------------------------------

def log_cleaning_action(
    project_id: int,
    action_name: str,
    action_config: Dict[str, Any],
    rows_before: int = 0,
    rows_after: int = 0,
    cells_changed: int = 0,
    dq_after: Optional[float] = None,
) -> int:
    """Insert one ordered cleaning-history row and return its action ID."""
    # action_config serialised to JSON; returns new history row id.
    # dq_after caches data_quality_score(cleaned_df) at the moment this
    # action was applied — resume_replay can read it straight off the last
    # row instead of recomputing the score from scratch (see app.py's
    # get_cached_dq_after / resume_replay).
    sql = """
        INSERT INTO cleaning_history
            (project_id, action_name, action_config, rows_before, rows_after, cells_changed, dq_after)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute(sql, (
            project_id, action_name,
            json.dumps(action_config, default=str),
            rows_before, rows_after, cells_changed, dq_after,
        ))
        hid = cur.lastrowid
        cur.close()
    return hid


def log_cleaning_action_atomic(
    project_id: int,
    action_name: str,
    action_config: Dict[str, Any],
    rows_before: int = 0,
    rows_after: int = 0,
    cells_changed: int = 0,
    dq_after: Optional[float] = None,
    manifest: Optional[Dict[str, Any]] = None,
    before_commit=None,
    expected_version: Optional[int] = None,
) -> int:
    """Insert one history row and its manifest in a single DB transaction.

    ``before_commit`` is called after the DB rows have been written but before
    commit. The app uses it to atomically promote the already-written snapshot
    file; if promotion fails, the DB transaction rolls back and no successful
    history row is visible.
    """
    history_sql = """
        INSERT INTO cleaning_history
            (project_id, action_name, action_config, rows_before, rows_after, cells_changed,
             dq_after, version_before, version_after)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    manifest_sql = """
        INSERT INTO manifests (history_id, manifest)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE manifest = VALUES(manifest)
    """
    cnx = _get_pool().get_connection()
    try:
        cur = cnx.cursor()
        cur.execute("SELECT state_version FROM projects WHERE id=%s FOR UPDATE", (project_id,))
        project_row = cur.fetchone()
        if not project_row:
            raise ValueError("project not found")
        current_version = int(project_row[0] or 0)
        submitted_version = current_version if expected_version is None else int(expected_version)
        if submitted_version != current_version:
            raise StaleDatasetVersionError(submitted_version, current_version)
        next_version = current_version + 1
        stored_config = dict(action_config or {})
        stored_config["dataset_version_before"] = current_version
        stored_config["dataset_version_after"] = next_version
        cur.execute(history_sql, (
            project_id,
            action_name,
            json.dumps(stored_config, default=str),
            rows_before,
            rows_after,
            cells_changed,
            dq_after,
            current_version,
            next_version,
        ))
        hid = cur.lastrowid
        if manifest:
            cur.execute(manifest_sql, (hid, json.dumps(manifest, default=str)))
        cur.execute("UPDATE projects SET state_version=%s WHERE id=%s", (next_version, project_id))
        if before_commit:
            before_commit(hid)
        cnx.commit()
        cur.close()
        return hid
    except Exception:
        cnx.rollback()
        raise
    finally:
        cnx.close()


def log_cleaning_actions_atomic(
    project_id: int,
    actions: List[Dict[str, Any]],
    expected_version: int,
    before_commit=None,
) -> List[Dict[str, int]]:
    """Insert sequential operation actions and promote one final snapshot atomically."""
    if not actions:
        return []
    cnx = _get_pool().get_connection()
    try:
        cur = cnx.cursor()
        cur.execute("SELECT state_version FROM projects WHERE id=%s FOR UPDATE", (project_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("project not found")
        current = int(row[0] or 0)
        if int(expected_version) != current:
            raise StaleDatasetVersionError(int(expected_version), current)
        inserted = []
        for item in actions:
            before_version = current
            current += 1
            config = dict(item.get("action_config") or {})
            config["dataset_version_before"] = before_version
            config["dataset_version_after"] = current
            cur.execute(
                """INSERT INTO cleaning_history
                   (project_id, action_name, action_config, rows_before, rows_after,
                    cells_changed, dq_after, version_before, version_after)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    project_id, item["action_name"], json.dumps(config, default=str),
                    item.get("rows_before", 0), item.get("rows_after", 0),
                    item.get("cells_changed", 0), item.get("dq_after"),
                    before_version, current,
                ),
            )
            history_id = int(cur.lastrowid)
            manifest = item.get("manifest")
            if manifest:
                cur.execute(
                    """INSERT INTO manifests (history_id, manifest) VALUES (%s,%s)
                       ON DUPLICATE KEY UPDATE manifest=VALUES(manifest)""",
                    (history_id, json.dumps(manifest, default=str)),
                )
            inserted.append({
                "id": history_id,
                "version_before": before_version,
                "version_after": current,
            })
        cur.execute("UPDATE projects SET state_version=%s WHERE id=%s", (current, project_id))
        if before_commit:
            before_commit([item["id"] for item in inserted])
        cnx.commit()
        cur.close()
        return inserted
    except Exception:
        cnx.rollback()
        raise
    finally:
        cnx.close()


def set_cleaning_action_status_atomic(
    history_id: int,
    project_id: int,
    status: str,
    expected_version: int,
    before_commit=None,
) -> int:
    """Change one action status and advance project state in one transaction."""
    if status not in ("applied", "undone"):
        raise ValueError(f"invalid status: {status!r}")
    cnx = _get_pool().get_connection()
    try:
        cur = cnx.cursor()
        cur.execute("SELECT state_version FROM projects WHERE id=%s FOR UPDATE", (project_id,))
        project_row = cur.fetchone()
        if not project_row:
            raise ValueError("project not found")
        current_version = int(project_row[0] or 0)
        if int(expected_version) != current_version:
            raise StaleDatasetVersionError(int(expected_version), current_version)
        cur.execute(
            "UPDATE cleaning_history SET status=%s WHERE id=%s AND project_id=%s",
            (status, history_id, project_id),
        )
        if cur.rowcount != 1:
            raise ValueError("history action not found")
        next_version = current_version + 1
        cur.execute("UPDATE projects SET state_version=%s WHERE id=%s", (next_version, project_id))
        if before_commit:
            before_commit()
        cnx.commit()
        cur.close()
        return next_version
    except Exception:
        cnx.rollback()
        raise
    finally:
        cnx.close()


def set_cleaning_action_status(history_id: int, status: str) -> None:
    """Persist Undo/Reapply status for a single history row."""
    # status is 'applied' or 'undone' — called when the user clicks Undo/Reapply
    # on a Change Log card, so the persisted history (and Resume) reflects it
    # instead of always showing "Applied".
    if status not in ("applied", "undone"):
        raise ValueError(f"invalid status: {status!r}")
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute("UPDATE cleaning_history SET status=%s WHERE id=%s", (status, history_id))
        cur.close()


def get_cleaning_history(project_id: int) -> List[Dict[str, Any]]:
    """Return oldest-first action history, including manifest availability."""
    # Oldest-first; parses action_config JSON back to dict.
    # has_manifest is a LEFT JOIN existence check against the manifests
    # table (indexed on history_id via its UNIQUE KEY) — cheap, and doesn't
    # touch the (project_id, applied_at) index this query relies on to avoid
    # a filesort, so it doesn't reintroduce the sort-memory problem.
    sql = """
        SELECT h.id, h.action_name, h.action_config, h.rows_before, h.rows_after,
               h.cells_changed, h.status, h.dq_after, h.version_before, h.version_after, h.applied_at,
               (m.id IS NOT NULL) AS has_manifest
        FROM cleaning_history h
        LEFT JOIN manifests m ON m.history_id = h.id
        WHERE h.project_id = %s
        ORDER BY h.applied_at ASC
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (project_id,))
        rows = cur.fetchall()
        cur.close()
    for row in rows:
        row["has_manifest"] = bool(row.get("has_manifest"))
        if isinstance(row.get("action_config"), str):
            try:
                row["action_config"] = json.loads(row["action_config"])
            except Exception:
                pass
    return rows


def get_actions_by_operation_token(project_id: int, operation_token: str) -> List[Dict[str, Any]]:
    """Return actions already committed for one durable idempotency token."""
    if not operation_token:
        return []
    sql = """
        SELECT h.id, h.action_name, h.action_config, h.rows_before, h.rows_after,
               h.cells_changed, h.status, h.version_before, h.version_after, h.applied_at,
               (m.id IS NOT NULL) AS has_manifest
        FROM cleaning_history h
        LEFT JOIN manifests m ON m.history_id=h.id
        WHERE h.project_id=%s
          AND JSON_UNQUOTE(JSON_EXTRACT(h.action_config, '$.operation_token'))=%s
        ORDER BY h.id ASC
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (project_id, operation_token))
        rows = cur.fetchall()
        cur.close()
    for row in rows:
        row["has_manifest"] = bool(row.get("has_manifest"))
        if isinstance(row.get("action_config"), str):
            row["action_config"] = json.loads(row["action_config"])
    return rows


def get_cleaning_action(history_id: int, project_id: int) -> Optional[Dict[str, Any]]:
    """Fetch one action only when it belongs to the supplied project."""
    # Single-row fetch for Undo/Reapply-via-manifest. project_id is required
    # and checked in the WHERE clause (not just as a courtesy filter) — the
    # history_id alone comes from the client, and a manifest apply mutates
    # the on-disk snapshot, so this is the guard against one project's Undo
    # click reaching into another project's row.
    sql = """
        SELECT id, project_id, action_name, action_config, status
        FROM cleaning_history
        WHERE id = %s AND project_id = %s
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (history_id, project_id))
        row = cur.fetchone()
        cur.close()
    if row and isinstance(row.get("action_config"), str):
        try:
            row["action_config"] = json.loads(row["action_config"])
        except Exception:
            pass
    return row


# --- Dataset versions --------------------------------------------------------

def save_dataset_version(
    project_id: int,
    file_path: str,
    label: str = "",
    row_count: int = 0,
    col_count: int = 0,
) -> int:
    """Register one immutable exported dataset version scoped to a project."""
    # version_no auto-increments within project scope (not global)
    sql_next = "SELECT COALESCE(MAX(version_no), 0) + 1 AS nv FROM dataset_versions WHERE project_id = %s"
    sql_ins  = """
        INSERT INTO dataset_versions
            (project_id, version_no, file_path, label, row_count, col_count)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql_next, (project_id,))
        nv = cur.fetchone()["nv"]
        cur.execute(sql_ins, (project_id, nv, file_path, label, row_count, col_count))
        vid = cur.lastrowid
        cur.close()
    log.info(f"project {project_id} → version {nv} saved")
    return vid


def get_dataset_versions(project_id: int) -> List[Dict[str, Any]]:
    """Return immutable exported versions for a project, newest first."""
    # Newest-first
    sql = """
        SELECT id, version_no, file_path, label, row_count, col_count, created_at
        FROM dataset_versions
        WHERE project_id = %s
        ORDER BY version_no DESC
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (project_id,))
        rows = cur.fetchall()
        cur.close()
    return rows


def get_dataset_version(project_id: int, version_id: int) -> Optional[Dict[str, Any]]:
    """Return a version only when it belongs to the supplied project."""
    sql = """
        SELECT id, version_no, file_path, label, row_count, col_count, created_at
        FROM dataset_versions
        WHERE project_id = %s AND id = %s
    """
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (project_id, version_id))
        row = cur.fetchone()
        cur.close()
    return row


# --- Manifests (per-action undo/reapply diffs) -------------------------------
# See the manifests table's comment in _CREATE_TABLES for why this is a
# separate table rather than a column on cleaning_history, and why it
# replaces the old file-based storage in uploads/<sid>/manifests/.

def save_manifest(history_id: int, manifest: Dict[str, Any]) -> None:
    """Persist the authoritative per-action diff used by Undo/Reapply."""
    # ON DUPLICATE KEY UPDATE covers the (rare) case of re-saving a manifest
    # for a history row that already has one, without needing a separate
    # exists-check first.
    sql = """
        INSERT INTO manifests (history_id, manifest)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE manifest = VALUES(manifest)
    """
    with _conn() as cnx:
        cur = cnx.cursor()
        cur.execute(sql, (history_id, json.dumps(manifest, default=str)))
        cur.close()


def get_manifest(history_id: int) -> Optional[Dict[str, Any]]:
    """Load a stored manifest, returning ``None`` when absent or corrupt."""
    sql = "SELECT manifest FROM manifests WHERE history_id = %s"
    with _conn() as cnx:
        cur = cnx.cursor(dictionary=True)
        cur.execute(sql, (history_id,))
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    try:
        return json.loads(row["manifest"])
    except Exception:
        log.exception(f"Failed to parse stored manifest JSON for history_id={history_id}")
        return None
