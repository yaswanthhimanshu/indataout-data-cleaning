# setup_database.py — run once before starting Flask
# Creates DB + all tables; safe to re-run (IF NOT EXISTS)
# Usage: python setup_database.py

"""Initialize the local MySQL schema used by inDataOut.

The script creates the configured database and required tables if they do not
already exist, then applies additive migrations. It owns schema setup only; it
does not delete projects, uploads, snapshots, manifests, reports, or versions.
Configuration is read from environment variables or ``.env``.
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# --- Config ------------------------------------------------------------------
# Pulled from environment / .env — see .env.example.
HOST     = os.getenv("DB_HOST", "localhost")
PORT     = int(os.getenv("DB_PORT", "3306"))
USER     = os.getenv("DB_USER", "root")
PASSWORD = os.getenv("DB_PASSWORD")
DATABASE = os.getenv("DB_NAME", "indataout")
# -----------------------------------------------------------------------------


def run(cur, label, sql):
    """Execute one setup statement and stop immediately on failure."""
    # Execute SQL; print tick/cross and exit on failure
    try:
        cur.execute(sql)
        print(f"  ✓  {label}")
    except Error as e:
        print(f"  ✗  {label} — {e}")
        sys.exit(1)


def main():
    """Create database objects and additive columns/indexes."""
    print("\n── inDataOut DB Setup ───────────────────────────────────────")

    if not PASSWORD:
        print("  ✗  DB_PASSWORD is not set.")
        print("     Copy .env.example to .env and fill in your MySQL credentials.\n")
        sys.exit(1)

    # Connect without specifying DB (may not exist yet)
    try:
        cnx = mysql.connector.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, charset="utf8mb4")
        print("  ✓  Connected to MySQL")
    except Error as e:
        print(f"  ✗  Connection failed — {e}")
        sys.exit(1)

    cur = cnx.cursor()

    # DB
    run(cur, f"CREATE DATABASE {DATABASE}",
        f"CREATE DATABASE IF NOT EXISTS `{DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    run(cur, f"USE {DATABASE}", f"USE `{DATABASE}`")

    # Tables
    print("\n── Tables ───────────────────────────────────────────────────")

    run(cur, "projects", """
        CREATE TABLE IF NOT EXISTS projects (
            id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            name          VARCHAR(255)  NOT NULL,
            notes         TEXT,
            original_file VARCHAR(255)  NOT NULL,
            upload_sid    VARCHAR(64)   NOT NULL,
            file_path     VARCHAR(512),
            row_count     INT UNSIGNED  DEFAULT 0,
            col_count     INT UNSIGNED  DEFAULT 0,
            state_version BIGINT UNSIGNED NOT NULL DEFAULT 0,
            status        ENUM('active','cleaned','archived') DEFAULT 'active',
            created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_upload_sid (upload_sid),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

    run(cur, "cleaning_history", """
        CREATE TABLE IF NOT EXISTS cleaning_history (
            id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            project_id    INT UNSIGNED  NOT NULL,
            action_name   VARCHAR(128)  NOT NULL,
            action_config JSON,
            rows_before   INT UNSIGNED  DEFAULT 0,
            rows_after    INT UNSIGNED  DEFAULT 0,
            cells_changed INT UNSIGNED  DEFAULT 0,
            status        ENUM('applied','undone') NOT NULL DEFAULT 'applied',
            dq_after      DECIMAL(5,2),
            version_before BIGINT UNSIGNED NOT NULL DEFAULT 0,
            version_after  BIGINT UNSIGNED NOT NULL DEFAULT 0,
            applied_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            INDEX idx_project_id (project_id),
            INDEX idx_project_applied (project_id, applied_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

    run(cur, "dataset_versions", """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            project_id    INT UNSIGNED  NOT NULL,
            version_no    SMALLINT UNSIGNED NOT NULL DEFAULT 1,
            file_path     VARCHAR(512)  NOT NULL,
            label         VARCHAR(128),
            row_count     INT UNSIGNED  DEFAULT 0,
            col_count     INT UNSIGNED  DEFAULT 0,
            created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            UNIQUE KEY uq_project_version (project_id, version_no),
            INDEX idx_project_id (project_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

    run(cur, "manifests", """
        CREATE TABLE IF NOT EXISTS manifests (
            id            INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            history_id    INT UNSIGNED NOT NULL,
            manifest      LONGTEXT NOT NULL,
            created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (history_id) REFERENCES cleaning_history(id) ON DELETE CASCADE,
            UNIQUE KEY uq_history_id (history_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

    cnx.commit()

    # Migrations
    print("\n── Migrations ───────────────────────────────────────────────")

    for table, column, definition in (
        ("projects", "state_version", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER col_count"),
        ("cleaning_history", "version_before", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER dq_after"),
        ("cleaning_history", "version_after", "BIGINT UNSIGNED NOT NULL DEFAULT 0 AFTER version_before"),
    ):
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME=%s
        """, (DATABASE, table, column))
        if cur.fetchone()[0]:
            print(f"  ✓  {table}.{column} already exists")
        else:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            print(f"  ✓  {table}.{column} added")

    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = 'projects'
            AND COLUMN_NAME = 'notes'
        """, (DATABASE,))

        exists = cur.fetchone()[0]

        if exists:
            print("  ✓  projects.notes column already exists")
        else:
            cur.execute("""
                ALTER TABLE projects
                ADD COLUMN notes TEXT AFTER name
            """)
            cnx.commit()
            print("  ✓  projects.notes column added")

    except Error as e:
        print(f"  ✗  notes migration — {e}")

    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = 'cleaning_history'
            AND COLUMN_NAME = 'status'
        """, (DATABASE,))

        exists = cur.fetchone()[0]

        if exists:
            print("  ✓  cleaning_history.status column already exists")
        else:
            cur.execute("""
                ALTER TABLE cleaning_history
                ADD COLUMN status ENUM('applied','undone') NOT NULL DEFAULT 'applied' AFTER cells_changed
            """)
            cnx.commit()
            print("  ✓  cleaning_history.status column added")

    except Error as e:
        print(f"  ✗  status migration — {e}")

    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = 'cleaning_history'
            AND COLUMN_NAME = 'dq_after'
        """, (DATABASE,))

        exists = cur.fetchone()[0]

        if exists:
            print("  ✓  cleaning_history.dq_after column already exists")
        else:
            cur.execute("""
                ALTER TABLE cleaning_history
                ADD COLUMN dq_after DECIMAL(5,2) AFTER status
            """)
            cnx.commit()
            print("  ✓  cleaning_history.dq_after column added")

    except Error as e:
        print(f"  ✗  dq_after migration — {e}")

    try:
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = %s
            AND TABLE_NAME = 'cleaning_history'
            AND INDEX_NAME = 'idx_project_applied'
        """, (DATABASE,))

        exists = cur.fetchone()[0]

        if exists:
            print("  ✓  cleaning_history.idx_project_applied index already exists")
        else:
            cur.execute("""
                ALTER TABLE cleaning_history
                ADD INDEX idx_project_applied (project_id, applied_at)
            """)
            cnx.commit()
            print("  ✓  cleaning_history.idx_project_applied index added")

    except Error as e:
        print(f"  ✗  idx_project_applied migration — {e}")

    # Verify
    print("\n── Verify ───────────────────────────────────────────────────")
    cur.execute("SHOW TABLES")
    found = {r[0] for r in cur.fetchall()}
    expected = {"projects", "cleaning_history", "dataset_versions", "manifests"}
    for t in sorted(expected):
        print(f"  {'✓' if t in found else '✗ MISSING'}  {t}")

    cur.close()
    cnx.close()

    if not expected.issubset(found):
        print("\n  ✗  Setup incomplete — check errors above.\n")
        sys.exit(1)

    print(f"\n── Done — start your Flask app. ─────────────────────────────")
    print(f"   DB     : {DATABASE}")
    print(f"   Tables : {', '.join(sorted(found))}\n")


if __name__ == "__main__":
    main()
