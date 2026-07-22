# app.py — inDataOut Data Cleaning
# Run: python app.py

"""Flask application boundary for the inDataOut cleaning workflow.

This module owns request/session state, immutable upload storage, current
snapshot promotion, committed action history, project/version routing, and
final export orchestration. DataFrame transformations live in ``cleaner.py``;
manifest diff/replay mechanics live in ``snapshot_diff.py``; durable metadata
and history rows live in ``db.py``.

Important invariants:
* Preview routes are read-only and must not promote authoritative state.
* A committed Apply creates one persisted history/manifest result and one
  current snapshot, or neither is considered successful.
* Undo/Reapply reconstructs state from the immutable source plus ordered active
  actions; client-rendered rows are never authoritative.
* Downloads resolve through project/version identity rather than the active
  browser session alone.
* Custom Pandas code is available only as a trusted local/desktop feature.
"""

import os
import sys
import json
import logging
from logging.handlers import RotatingFileHandler
import shutil
import secrets
import re
import math
import tempfile
import threading
import warnings
import time
from io import StringIO
from datetime import datetime, timezone

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, send_from_directory, jsonify, session as flask_session,
    g, has_request_context,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
import pandas as pd
import numpy as np

from cleaner import (
    analyze_dataframe, clean_dataframe, save_report,
    data_quality_score, data_quality_breakdown, cleaning_completeness_score, detect_outliers,
)
import db  # MySQL persistence layer

# Optional encoding detector
try:
    import chardet
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

# --- Application initialization, request IDs, CSRF, and diagnostics -----------

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB — matches the UI's stated upload limit


def _logs_dir() -> str:
    """Return the persistent diagnostic-log directory without using temp space."""
    root = os.environ.get("INDATAOUT_LOG_DIR")
    if root:
        return root
    base = os.environ.get("LOCALAPPDATA") or _persistent_dir()
    return os.path.join(base, "inDataOut", "logs")


class RequestContextFilter(logging.Filter):
    """Attach request/project correlation fields to every diagnostic log record."""

    def filter(self, record):
        has_request = has_request_context()
        record.request_id = getattr(g, "request_id", "-") if has_request else "-"
        record.project_id = flask_session.get("project_id", "-") if has_request else "-"
        record.route = request.path if has_request else "-"
        return True


def configure_logging() -> None:
    """Configure rotating diagnostic logs without logging dataset contents."""
    log_dir = _logs_dir()
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)sZ level=%(levelname)s request_id=%(request_id)s "
        "project_id=%(project_id)s route=%(route)s logger=%(name)s msg=%(message)s"
    )
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    for handler in (
        RotatingFileHandler(os.path.join(log_dir, "indataout.log"), maxBytes=1_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        root.addHandler(handler)


def csrf_token() -> str:
    """Return the session CSRF token, creating one for rendered forms if needed."""
    token = flask_session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        flask_session["_csrf_token"] = token
    return token


def _wants_json_response() -> bool:
    operation_paths = {
        "/preview_clean", "/preview_custom_code", "/get_editor_data",
        "/preview_outliers", "/apply_manifest", "/resume_replay",
    }
    return (
        request.path.startswith("/api/")
        or request.path in operation_paths
        or request.is_json
        or "application/json" in request.headers.get("Accept", "")
    )


def api_error(code: str, message: str, status: int = 400, user_action: str = "", retryable: bool = False, details: dict | None = None):
    """Return the common structured JSON error envelope used by API routes."""
    request_id = getattr(g, "request_id", secrets.token_hex(8))
    payload = {
        "ok": False,
        "success": False,
        "reason": code,
        "error": {
            "code": code,
            "message": message,
            "user_action": user_action,
            "retryable": bool(retryable),
            "request_id": request_id,
            "project_id": flask_session.get("project_id"),
            "dataset_version": int(flask_session.get("dataset_version", 0) or 0),
            "details": details or {},
        },
    }
    logging.warning("api_error code=%s status=%s retryable=%s", code, status, retryable)
    return jsonify(payload), status


def _operation_failure_explanation(exc: Exception, phase: str) -> tuple[str, str]:
    """Translate common internal failures into useful, non-sensitive UI guidance."""
    exc_name = type(exc).__name__.lower()
    exc_text = str(exc).lower()
    combined = f"{exc_name} {exc_text}"

    if isinstance(exc, MemoryError) or "out of memory" in combined or "cannot allocate memory" in combined:
        return (
            "The dataset is too large for the memory currently available, so the action was stopped without changing it.",
            "Close other memory-heavy programs, use fewer columns, or split the dataset and retry.",
        )
    if isinstance(exc, PermissionError) or "permission denied" in combined or "being used by another process" in combined:
        return (
            "Windows could not update the project snapshot because the file is locked or not writable. The dataset was left unchanged.",
            "Close programs using the project files, confirm the folder is writable, and retry.",
        )
    if "categorical" in combined and ("new category" in combined or "setitem" in combined):
        return (
            "A selected fill value was not accepted by the column's category definition, so the action was not saved.",
            "Refresh the project and retry. If it repeats, provide the diagnostic ID shown below.",
        )
    if "arrow" in combined or "feather" in combined or "serialis" in combined or "serializ" in combined:
        return (
            "The cleaned values could not be stored in the project snapshot because one column contains incompatible value types.",
            "Review recent type conversions or cell edits, then retry and provide the diagnostic ID if it repeats.",
        )
    if any(token in combined for token in (
        "mysql", "databaseerror", "operationalerror", "interfaceerror", "integrityerror",
        "connectionerror", "db down", "connection refused", "connection reset",
    )):
        return (
            "The cleaning result was calculated, but the project history database did not accept the save. The dataset was left unchanged.",
            "Check the database connection and retry the action.",
        )
    if isinstance(exc, UnicodeError) or "parsererror" in combined or "decode" in combined:
        return (
            "The dataset contains text or file formatting that could not be read consistently for this action.",
            "Check the file encoding and column formatting, then retry.",
        )
    if isinstance(exc, (TypeError, ValueError)):
        return (
            "One or more selected values are incompatible with the requested cleaning operation, so no changes were committed.",
            "Review the selected columns, values, and data types, then retry.",
        )
    if isinstance(exc, OSError):
        return (
            "The project file could not be read or updated by Windows, so the dataset was left unchanged.",
            "Check available disk space and folder access, then retry.",
        )
    if phase == "save":
        return (
            "The cleaning result was calculated, but its Change Log entry and project snapshot could not be saved together. The dataset was left unchanged.",
            "Retry once. If it repeats, provide the diagnostic ID shown below.",
        )
    return (
        "The cleaning operation stopped before any changes were committed because an unexpected processing error occurred.",
        "Retry once. If it repeats, provide the diagnostic ID shown below.",
    )


def _submitted_dataset_version(values) -> int:
    """Return the client's base version or the current session version."""
    raw = values.get("dataset_version") if values is not None else None
    return int(flask_session.get("dataset_version", 0) or 0) if raw in (None, "") else int(raw)


def _require_session_version(values) -> int:
    """Reject a mutation based on a stale dataset version."""
    submitted = _submitted_dataset_version(values)
    current = int(flask_session.get("dataset_version", 0) or 0)
    if submitted != current:
        raise db.StaleDatasetVersionError(submitted, current)
    return submitted


@app.before_request
def _request_start_and_csrf():
    """Assign request correlation and reject unverifiable mutations."""
    g.request_id = request.headers.get("X-Request-ID") or secrets.token_hex(8)
    g.started_at = datetime.now(timezone.utc)
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        expected = flask_session.get("_csrf_token")
        supplied = (
            request.headers.get("X-CSRF-Token")
            or request.form.get("csrf_token")
            or ((request.get_json(silent=True) or {}).get("csrf_token") if request.is_json else None)
        )
        if app.config.get("TESTING") and not expected:
            return None
        if not expected or supplied != expected:
            if _wants_json_response():
                return api_error(
                    "CSRF_TOKEN_INVALID",
                    "This request could not be verified.",
                    status=403,
                    user_action="Refresh the page and try again.",
                    retryable=True,
                )
            flash("This request could not be verified. Refresh the page and try again.")
            return redirect(request.referrer or url_for("index")), 403


@app.after_request
def _request_finish(response):
    """Expose request IDs and log the status/duration of every request."""
    response.headers["X-Request-ID"] = getattr(g, "request_id", "-")
    try:
        elapsed = int((datetime.now(timezone.utc) - g.started_at).total_seconds() * 1000)
        logging.info("request_complete status=%s duration_ms=%s", response.status_code, elapsed)
    except Exception:
        pass
    return response


@app.errorhandler(413)
def _file_too_large(_e):
    if _wants_json_response():
        return api_error(
            "PAYLOAD_TOO_LARGE",
            "That file is too large.",
            status=413,
            user_action="Upload a smaller file or split the dataset.",
            retryable=False,
        )
    flash("That file is too large. Max upload size is 100 MB.")
    return redirect(request.referrer or url_for('index')), 413


@app.errorhandler(Exception)
def _unhandled_request_error(exc):
    """Last-resort boundary: keep API failures structured and traceable."""
    if isinstance(exc, HTTPException):
        return exc
    logging.exception("Unhandled request failure")
    message, user_action = _operation_failure_explanation(exc, "process")
    if _wants_json_response():
        return api_error(
            "UNEXPECTED_REQUEST_FAILURE", message, status=500,
            user_action=user_action, retryable=True,
        )
    flash(f"{message} {user_action}")
    return redirect(request.referrer or url_for("index")), 500

def _persistent_dir() -> str:
    """Directory for files that must survive between runs (.secret_key, .env).

    Under a normal `python app.py` run this is just the folder app.py lives in.
    Under a PyInstaller --onefile .exe, app.py's own __file__ resolves inside a
    fresh, randomly-named _MEIxxxxx temp extraction folder every single launch
    — anything written there (like the secret key) would vanish immediately,
    silently invalidating every session on each restart. sys.executable, by
    contrast, always points at the actual .exe on disk, so we anchor there
    instead whenever we're running frozen.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _get_or_create_secret_key() -> str:
    """Persist the secret key to disk so sessions survive Flask restarts.
    Without this, every restart generates a new key → all cookies invalid →
    project_id lost from session → DB history silently skipped."""
    key_file = os.path.join(_persistent_dir(), ".secret_key")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            k = f.read().strip()
        if k:
            return k
    k = secrets.token_hex(32)
    try:
        with open(key_file, "w") as f:
            f.write(k)
        os.chmod(key_file, 0o600)
    except Exception:
        pass
    return k

app.secret_key = os.environ.get("SECRET_KEY") or _get_or_create_secret_key()

# basename filter — used in projects.html and project_detail.html
app.jinja_env.filters["basename"] = lambda p: os.path.basename(p or "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.context_processor
def inject_static_version():
    def static_version(filename: str) -> int:
        try:
            return int(os.path.getmtime(os.path.join(BASE_DIR, "static", filename)))
        except OSError:
            return 1

    return {"static_version": static_version, "csrf_token": csrf_token}


# Anchored next to the app itself (the exe's own folder when frozen, this
# file's folder in dev) — NOT the OS temp directory. Temp folders aren't
# persistent storage: Windows (and most disk-cleanup tools) can and do clear
# them, especially across a reboot, which would silently delete every
# uploaded dataset and its entire cleaning history's underlying files even
# though the DB rows describing them are still there ("Source file not found
# on disk" is exactly that scenario). Mirrors the same sys.frozen check
# db.py already uses for locating .env.
if getattr(sys, "frozen", False):
    _app_base_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    _app_base_dir = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join(_app_base_dir, "uploads"),
)
# Trusted-local boundary: unrestricted Pandas code is meant for the desktop/local
# app. Set INDATAOUT_TRUSTED_LOCAL=0 before any network-facing deployment.
ENABLE_CUSTOM_CODE = os.environ.get("INDATAOUT_TRUSTED_LOCAL", "1").lower() in ("1", "true", "yes")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "tsv", "txt"}

configure_logging()

# --- DB init — non-fatal if MySQL is down ------------------------------------

try:
    db.init_db()
    logging.info("MySQL ready")
except Exception as _db_err:
    logging.warning(f"MySQL unavailable at startup: {_db_err}")

# --- DataFrame cache ---------------------------------------------------------
# { sid: (file_mtime, dataframe) }

_df_cache: dict = {}
_dq_cache: dict = {}
_DF_CACHE_MAX = 10

# Flask + waitress (threads=8) means these module-level dicts are shared
# across concurrent request threads — e.g. two browser tabs on the same
# session, or a section Apply landing while a resume is still in flight.
# Without a lock, two threads racing a read-modify-write (the "evict oldest,
# then insert" eviction step in particular) can corrupt the dict or raise a
# rare KeyError. The lock is only held around the dict access itself, never
# across the expensive load_dataset()/data_quality_score() calls — holding it
# that long would serialize otherwise-independent requests for no reason.
_cache_lock = threading.Lock()


def get_cached_df(sid: str, file_path: str):
    """Load a source DataFrame copy, re-reading only when the file changed."""
    # Return copy from cache; re-read only when file mtime changes
    try:
        mtime = os.path.getmtime(file_path)
    except FileNotFoundError:
        with _cache_lock:
            _df_cache.pop(sid, None)
        raise
    with _cache_lock:
        cached = _df_cache.get(sid)
    if cached and cached[0] == mtime:
        return cached[1].copy()
    df = load_dataset(file_path)
    with _cache_lock:
        if len(_df_cache) >= _DF_CACHE_MAX and sid not in _df_cache:
            del _df_cache[next(iter(_df_cache))]
        _df_cache[sid] = (mtime, df)
    return df.copy()


def get_cached_dq(sid: str, file_path: str, df):
    """Cache the immutable-source quality score by file mtime."""
    # Cache DQ score for original file — recompute only when file changes
    try:
        mtime = os.path.getmtime(file_path)
    except FileNotFoundError:
        with _cache_lock:
            _dq_cache.pop(sid, None)
        raise
    with _cache_lock:
        cached = _dq_cache.get(sid)
    if cached and cached[0] == mtime:
        return cached[1]
    score = data_quality_score(df)
    with _cache_lock:
        if len(_dq_cache) >= _DF_CACHE_MAX and sid not in _dq_cache:
            del _dq_cache[next(iter(_dq_cache))]
        _dq_cache[sid] = (mtime, score)
    return score


def get_cached_dq_if_available(sid: str, file_path: str):
    """Return the immutable-source DQ score only if it is already cached."""
    try:
        mtime = os.path.getmtime(file_path)
    except FileNotFoundError:
        return None
    with _cache_lock:
        cached = _dq_cache.get(sid)
    if cached and cached[0] == mtime:
        return cached[1]
    return None


# --- Cleaned-dataset DQ score cache ------------------------------------------
# data_quality_score() isn't a cheap lookup like get_cached_dq above — it runs
# several regex/string passes per column (placeholder detection, numeric-like
# coercion, outlier IQR, ...), so on a wide/long dataset it's genuinely heavy.
# preview_clean, apply_manifest_route, and resume_replay each used to call it
# fresh on cleaned_df every single time, even when the underlying data hadn't
# actually changed since the last call (e.g. re-rendering the same state, or
# a preview that turned out to be a no-op). This caches by a cheap content
# fingerprint instead of unconditionally recomputing.

_dq_after_cache: dict = {}
_DQ_AFTER_CACHE_MAX = 10

# --- Cleaned-dataset analysis cache -------------------------------------------
# analyze_dataframe() (dtype detection, per-column stats/histograms) only
# actually needs to change when the SET OF COLUMNS changes — a pure cell-value
# edit (the overwhelming majority of undo/reapply clicks) can't affect it at
# all. Keyed by the exact column tuple so it's never served stale: any
# manifest that adds/drops/renames a column changes the tuple, which is a
# guaranteed cache miss, so this can never accidentally reuse the wrong shape.
_analysis_cache: dict = {}
_ANALYSIS_CACHE_MAX = 10
_operation_cache: dict = {}
_operation_cache_lock = threading.Lock()


def _ui_analysis(df: pd.DataFrame) -> dict:
    """Exact live-panel analysis without the duplicate 100-row preview payload."""
    rows, cols = df.shape
    total_cells = rows * cols if cols > 0 else 1
    missing_by_column = []
    missing_count = 0
    for c in df.columns:
        col_missing = int(df[c].isna().sum())
        missing_count += col_missing
        if col_missing > 0:
            missing_by_column.append({
                "column": c,
                "missing": col_missing,
                "percent": round((col_missing / rows) * 100, 2) if rows else 0,
            })

    def display_value(value):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value

    missing_rows = []
    missing_rows_total = 0
    missing_editor_limit = min(len(df), 2000)
    if missing_count > 0:
        missing_mask = df.isna().any(axis=1)
        missing_rows_total = int(missing_mask.sum())
        missing_df = df.loc[missing_mask]
        for position, idx in enumerate(missing_df.index, start=1):
            if len(missing_rows) >= missing_editor_limit:
                break
            row = missing_df.loc[idx]
            row_position = int(df.index.get_loc(idx)) if idx in df.index else position - 1
            missing_rows.append({
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "row_number": row_position + 1,
                "data_row_number": row_position + 1,
                "missing_columns": [c for c in df.columns if pd.isna(row[c])],
                "row_data": {c: display_value(row[c]) for c in df.columns},
            })

    empty_columns = [c for c in df.columns if df[c].isna().all()]
    unnamed_columns = []
    for c in df.columns:
        if str(c).strip().lower().startswith("unnamed"):
            non_missing = int(df[c].notna().sum())
            missing = int(df[c].isna().sum())
            unnamed_columns.append({
                "column": c,
                "non_missing": non_missing,
                "missing": missing,
                "percent_missing": round((missing / rows) * 100, 2) if rows else 0,
                "safe_to_drop": non_missing == 0,
            })
    constant_columns = [
        c for c in df.columns
        if df[c].notna().any() and df[c].nunique(dropna=True) <= 1
    ]
    whitespace_cells = 0
    for c in df.select_dtypes(include=["object"]).columns:
        whitespace_cells += int(df[c].map(
            lambda value: isinstance(value, str) and value != value.strip()
        ).sum())

    return {
        "shape": [rows, cols],
        "missing_count": missing_count,
        "missing_percent": round(0 if total_cells == 0 else (missing_count / total_cells) * 100, 2),
        "duplicate_count": int(df.duplicated().sum()) if rows else 0,
        "empty_columns": empty_columns,
        "unnamed_columns": unnamed_columns,
        "constant_columns": constant_columns,
        "whitespace_cells": whitespace_cells,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "missing_by_column": missing_by_column,
        "missing_rows": missing_rows,
        "missing_rows_display_limit": missing_editor_limit,
        "missing_rows_total": missing_rows_total,
    }


def get_cached_analysis(sid: str, df: pd.DataFrame, fp=None) -> dict:
    # IMPORTANT: this is intentionally NOT analyze_dataframe(). Traced every
    # consumer of the "analysis" field client-side (rebuildDtypeTable,
    # rebuildOutlierColumnList, called from updateUI) — the only two fields
    # ever read after page load are `dtypes` and `numeric_columns`. Both are
    # pure dtype metadata (no per-cell scan): `dtypes` is just each column's
    # pandas dtype name, `numeric_columns` is select_dtypes() filtering by
    # that same metadata. analyze_dataframe() additionally computes
    # missing_count, missing_by_column, duplicate_count, a up-to-2000-row
    # missing-rows list, and a preview sample — all real, full-dataset-scan
    # work that NOTHING downstream of this endpoint reads, so computing it
    # here was pure waste, not caution.
    #
    # Still fingerprint-keyed (not column-tuple-keyed): a value-only change
    # can flip a column's dtype (e.g. undoing a fill that let pandas
    # downcast int64->float64 back to float64 with NaN) without touching the
    # column set, so this must invalidate on content change, not just
    # add/drop — that was the actual bug in the column-tuple-keyed version.
    if fp is None:
        fp = _df_fingerprint(df)
    if fp is not None:
        with _cache_lock:
            cached = _analysis_cache.get(sid)
        if cached and cached[0] == fp:
            return cached[1]
    analysis = _ui_analysis(df)
    if fp is not None:
        with _cache_lock:
            if len(_analysis_cache) >= _ANALYSIS_CACHE_MAX and sid not in _analysis_cache:
                del _analysis_cache[next(iter(_analysis_cache))]
            _analysis_cache[sid] = (fp, analysis)
    return analysis


def _df_fingerprint(df: pd.DataFrame):
    # A single vectorized hash pass is far cheaper than data_quality_score's
    # several per-column regex/string passes, so this is a real time save
    # even though it still touches every cell.
    try:
        return int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        return None  # unhashable dtype edge case — treat as "always changed"


def _dataframes_match_for_history(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Return True only when values, row/column labels, and dtypes match."""
    if left.shape != right.shape:
        return False
    if list(left.index) != list(right.index) or list(left.columns) != list(right.columns):
        return False
    if {c: str(left[c].dtype) for c in left.columns} != {c: str(right[c].dtype) for c in right.columns}:
        return False
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=True, check_like=False)
        return True
    except AssertionError:
        return False


def get_cached_dq_after(sid: str, df, fp=None) -> float:
    if fp is None:
        fp = _df_fingerprint(df)
    if fp is not None:
        with _cache_lock:
            cached = _dq_after_cache.get(sid)
        if cached and cached[0] == fp:
            return cached[1]
    score = data_quality_score(df)
    if fp is not None:
        with _cache_lock:
            if len(_dq_after_cache) >= _DQ_AFTER_CACHE_MAX and sid not in _dq_after_cache:
                del _dq_after_cache[next(iter(_dq_after_cache))]
            _dq_after_cache[sid] = (fp, score)
    return score


# --- Helpers -----------------------------------------------------------------

class _FormProxy:
    # Thread-safe snapshot of request.form for ThreadPoolExecutor workers
    # Flask's request proxy dies outside its request context
    def __init__(self, data: dict):
        self._data = data  # {key: [val, ...]}

    def get(self, key, default=None):
        vals = self._data.get(key)
        return vals[-1] if vals else default

    def getlist(self, key):
        return self._data.get(key, [])

    def keys(self):
        return self._data.keys()

    def items(self):
        # Yield (key, last_value) — matches ImmutableMultiDict.items()
        for k, vals in self._data.items():
            if vals:
                yield k, vals[-1]

    def __contains__(self, key):
        return key in self._data


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalized_column_name(col_name):
    # Match cleaner.py column normalization for form field keys
    nc = str(col_name).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    nc = nc.strip().lower().replace(" ", "_")
    return "".join(ch if ord(ch) < 128 else "_" for ch in nc)


def make_json_safe(value):
    # Recursively convert pandas/numpy types and non-finite floats to JSON-safe primitives
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        lst = list(value)
        if all(isinstance(v, (str, int, bool)) or v is None for v in lst):
            return lst  # fast path: already safe
        return [make_json_safe(v) for v in lst]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


FEATURE_HISTORY_META = {
    "dataIntegrityGroup": {"label": "Data Integrity", "order": 10},
    "missingValuesGroup": {"label": "Missing Values", "order": 20},
    "dataTypesGroup": {"label": "Data Types", "order": 30},
    "outlierGroup": {"label": "Outliers", "order": 40},
    "datasetEditorSlide": {"label": "Dataset Cell Edits", "order": 50},
    "dropSelectedColumnsGroup": {"label": "Drop Selected Columns", "order": 55},
    "customCodeGroup": {"label": "Custom Pandas Code", "order": 60},
}

FEATURE_HISTORY_SUBFEATURES = {
    "dataIntegrityGroup": [
        {"key": "removeDuplicateRows", "label": "Remove Duplicate Rows", "operations": {"drop_duplicates"}},
        {"key": "dropEmptyColumns", "label": "Drop Fully Empty Columns", "operations": {"drop_empty_columns"}},
        {"key": "dropConstantColumns", "label": "Drop Constant Columns", "operations": {"drop_constant_columns"}},
        {"key": "trimWhitespace", "label": "Trim Whitespace", "operations": {"trim_whitespace"}},
    ],
}


OPERATION_FEATURE_SCOPE = {
    "fill_numeric": "missingValuesGroup",
    "fill_categorical": "missingValuesGroup",
    "manual_missing_cell_updates": "missingValuesGroup",
    "drop_missing_rows": "missingValuesGroup",
    "manual_outlier_removal": "outlierGroup",
    "outlier_removal": "outlierGroup",
    "outlier_clip": "outlierGroup",
    "drop_duplicates": "dataIntegrityGroup",
    "drop_empty_columns": "dataIntegrityGroup",
    "drop_constant_columns": "dataIntegrityGroup",
    "trim_whitespace": "dataIntegrityGroup",
    "drop_selected_columns": "dropSelectedColumnsGroup",
    "manual_column_renames": "datasetEditorSlide",
    "manual_cell_updates": "datasetEditorSlide",
    "convert_dtypes": "dataTypesGroup",
    "manual_convert_dtypes": "dataTypesGroup",
    "custom_pandas_code": "customCodeGroup",
}


def _history_scope_for(action_name: str, config: dict, operations: list | None = None) -> str:
    if operations:
        for op in operations:
            scope = OPERATION_FEATURE_SCOPE.get(str(op.get("action", "")))
            if scope:
                return scope
    active_panel = (config or {}).get("activePanelId")
    if active_panel == "datasetEditorSlide":
        return "datasetEditorSlide"
    if active_panel in FEATURE_HISTORY_META:
        return active_panel
    name = str(action_name or (config or {}).get("label", "")).lower()
    if "outlier" in name:
        return "outlierGroup"
    if "missing" in name or "fill" in name or "null" in name:
        return "missingValuesGroup"
    if "type" in name or "dtype" in name:
        return "dataTypesGroup"
    if "drop selected" in name or "drop column" in name:
        return "dropSelectedColumnsGroup"
    if "rename" in name or "edit" in name:
        return "datasetEditorSlide"
    if "custom" in name or "pandas" in name:
        return "customCodeGroup"
    return "dataIntegrityGroup"


def _titleize_action(name: str) -> str:
    text = str(name or "Cleaning action").replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Cleaning action"


def _op_label(op: dict, fallback: str) -> str:
    action = str((op or {}).get("action", ""))
    if action == "fill_numeric":
        return f"Filled {op.get('filled_cells', 0)} numeric missing cells"
    if action == "fill_categorical":
        return f"Filled {op.get('filled_cells', 0)} categorical missing cells"
    if action == "manual_missing_cell_updates":
        return f"Filled {op.get('count', 0)} missing cells manually"
    if action == "drop_missing_rows":
        return f"Dropped {op.get('dropped_row_count', 0)} rows with missing values"
    if action == "manual_outlier_removal":
        return f"Removed {op.get('rows_removed', 0)} outlier rows"
    if action == "drop_duplicates":
        return f"Removed {op.get('removed_rows', 0)} duplicate rows"
    if action == "trim_whitespace":
        return f"Trimmed whitespace in {len(op.get('columns_processed') or [])} columns"
    if action == "convert_dtypes":
        return f"Converted {len(op.get('changes') or [])} data types"
    if action == "manual_convert_dtypes":
        return f"Converted {len(op.get('changes') or {})} data types"
    if action == "manual_column_renames":
        return f"Renamed {op.get('count', 0)} columns"
    if action == "manual_cell_updates":
        return f"Edited {op.get('count', 0)} cells"
    if action == "drop_selected_columns":
        return f"Dropped {op.get('count', 0)} selected columns"
    if action in ("drop_empty_columns", "drop_constant_columns"):
        return f"Dropped {op.get('count', 0)} columns"
    if action == "custom_pandas_code":
        return "Applied custom Pandas code"
    if action:
        return _titleize_action(action)
    return fallback


def _operation_notices(operations: list[dict] | None) -> list[str]:
    """Return concise explanations for requested work that was skipped."""
    notices = []
    for op in operations or []:
        if not isinstance(op, dict):
            continue
        action = str(op.get("action") or "")
        if action in ("fill_numeric", "fill_categorical"):
            for item in op.get("skipped") or []:
                if not isinstance(item, dict):
                    continue
                column = str(item.get("column") or "Selected column")
                reason = str(item.get("reason") or "The selected strategy was not applicable.")
                notices.append(f"{column}: {reason}")
        elif action == "manual_convert_dtypes_failed":
            reason = op.get("error") or "the selected conversion was not compatible with the column values"
            notices.append(f"Data type conversion was not applied: {reason}.")
        elif action == "manual_convert_dtypes":
            for column, detail in (op.get("changes") or {}).items():
                if isinstance(detail, dict) and detail.get("success") is False:
                    reason = detail.get("note") or "the selected conversion was not compatible with the column values"
                    notices.append(f"{column}: {reason}.")
    return list(dict.fromkeys(notices))


def _no_op_message(operations: list[dict] | None, action_name: str = "") -> str:
    """Explain a harmless no-change result in terms of the selected operation."""
    notices = _operation_notices(operations)
    if notices:
        return "No changes were applied. " + " ".join(notices)

    actions = {str(op.get("action") or "") for op in operations or [] if isinstance(op, dict)}
    label = str(action_name or "").lower()
    if "drop_duplicates" in actions or "duplicate" in label:
        return "No duplicate rows were found, so the dataset was already clean for this action."
    if "trim_whitespace" in actions or "whitespace" in label:
        return "No leading or trailing whitespace was found."
    if "drop_empty_columns" in actions or "empty column" in label:
        return "No fully empty columns were found."
    if "drop_constant_columns" in actions or "constant column" in label:
        return "No constant columns were found."
    if actions.intersection({"fill_numeric", "fill_categorical", "drop_missing_rows"}) or "missing" in label:
        return "No missing cells matched the selected strategy."
    if actions.intersection({"convert_dtypes", "manual_convert_dtypes", "manual_convert_dtypes_failed"}) or "type" in label:
        return "The selected columns already use those data types, so no conversion was needed."
    if actions.intersection({"manual_outlier_removal", "outlier_removal", "outlier_clip"}) or "outlier" in label:
        return "No rows matched the selected outlier action."
    if "drop_selected_columns" in actions or "selected column" in label:
        return "No selected columns needed to be removed."
    return "No changes were needed; the dataset already satisfies this action."


def _apply_result_message(history_actions: list[dict], notices: list[str] | None = None) -> str:
    labels = [
        str((item.get("action_config") or {}).get("label") or item.get("action_name") or "").strip()
        for item in history_actions or []
    ]
    labels = [label for label in labels if label]
    if not labels:
        message = "Applied successfully."
    elif len(labels) == 1:
        message = labels[0] + "."
    else:
        message = f"Applied {len(labels)} actions: " + "; ".join(labels) + "."
    if notices:
        message += " Skipped: " + " ".join(notices)
    return message


def _history_pairs_for_operation(op: dict) -> list[dict]:
    op = op or {}
    action = str(op.get("action", ""))
    pairs: list[dict] = []

    if action == "fill_numeric":
        for detail in (op.get("details") or [])[:6]:
            pairs.append({
                "before": "NaN",
                "after": detail.get("value", "-"),
                "context": f"{detail.get('column', '')} - {len(detail.get('rows') or [])} cells",
            })
    elif action == "fill_categorical":
        for col, detail in list((op.get("details") or {}).items())[:6]:
            pairs.append({
                "before": "NaN",
                "after": detail.get("value", "-") if isinstance(detail, dict) else "-",
                "context": f"{col} - {len((detail or {}).get('rows') or []) if isinstance(detail, dict) else 0} cells",
            })
    elif action == "manual_missing_cell_updates":
        changes = op.get("changes") or op.get("updates") or []
        for change in changes[:6]:
            pairs.append({
                "before": change.get("old", "NaN"),
                "after": change.get("value", "-"),
                "context": change.get("column", ""),
            })
    elif action in ("drop_missing_rows", "manual_outlier_removal"):
        rows = op.get("row_numbers") or op.get("selected_rows") or op.get("valid_indices_dropped") or []
        for row_no in rows[:6]:
            pairs.append({"before": f"Row {row_no}", "after": "Removed", "context": ""})
        if not pairs:
            count = op.get("dropped_row_count") or op.get("rows_removed") or 0
            pairs.append({"before": f"{count} rows", "after": "Removed", "context": ""})
    elif action == "drop_duplicates":
        count = op.get("removed_rows") or op.get("duplicate_count") or 0
        pairs.append({"before": f"{count} duplicate rows", "after": "0 duplicates", "context": ""})
    elif action in ("convert_dtypes",):
        for change in (op.get("changes") or [])[:6]:
            pairs.append({
                "before": change.get("original_dtype") or change.get("from") or "-",
                "after": change.get("new_dtype") or change.get("to") or "-",
                "context": change.get("column", ""),
            })
    elif action == "manual_convert_dtypes":
        for col, info in list((op.get("changes") or {}).items())[:6]:
            if isinstance(info, dict):
                before, after = info.get("from") or info.get("original_dtype") or "-", info.get("to") or info.get("new_dtype") or "-"
            else:
                before, after = "-", info
            pairs.append({"before": before, "after": after, "context": col})
    elif action in ("drop_empty_columns", "drop_constant_columns", "drop_selected_columns"):
        for col in (op.get("columns_dropped") or op.get("columns") or [])[:6]:
            pairs.append({"before": col, "after": "Dropped", "context": ""})
    elif action == "trim_whitespace":
        for col in (op.get("columns_processed") or [])[:6]:
            pairs.append({"before": '"value "', "after": '"value"', "context": col})
    elif action == "manual_column_renames":
        changes = op.get("changes") or op.get("renames") or {}
        items = changes.items() if isinstance(changes, dict) else []
        for before, after in list(items)[:6]:
            pairs.append({"before": before, "after": after, "context": ""})
    elif action == "manual_cell_updates":
        for change in (op.get("changes") or op.get("updates") or [])[:6]:
            pairs.append({
                "before": change.get("old", "Previous value"),
                "after": change.get("value", change.get("after", "Updated")),
                "context": change.get("column", ""),
            })
    elif action == "custom_pandas_code":
        code = str(op.get("code") or op.get("source") or "").strip().splitlines()
        pairs.append({"before": "DataFrame", "after": "Custom code applied", "context": code[0] if code else ""})

    return make_json_safe(pairs)


def _compact_history_row(data: dict | None, limit: int = 4) -> str:
    """Format a persisted row as a short, human-readable value summary."""
    values = []
    for column, value in (data or {}).items():
        if value is None or value == "":
            continue
        text = str(value)
        if len(text) > 36:
            text = f"{text[:33]}..."
        values.append(f"{column}: {text}")
        if len(values) == limit:
            break
    return " | ".join(values) or "Row present"


def _operation_columns(op: dict) -> set[str]:
    """Return columns explicitly owned by one recorded operation."""
    columns: set[str] = set()
    for key in ("columns", "columns_processed", "columns_dropped", "columns_encoded"):
        value = op.get(key) or []
        if isinstance(value, list):
            columns.update(str(item) for item in value)
    details = op.get("details") or op.get("changes") or {}
    if isinstance(details, dict):
        columns.update(str(item) for item in details)
    elif isinstance(details, list):
        columns.update(str(item.get("column")) for item in details if isinstance(item, dict) and item.get("column"))
    for item in op.get("updates") or []:
        if isinstance(item, dict) and item.get("column"):
            columns.add(str(item["column"]))
    return columns


def _manifest_for_operation(manifest: dict, op: dict, operation_count: int) -> dict:
    """Return the exact manifest portion owned by one reported operation."""
    if operation_count == 1:
        return {key: value for key, value in manifest.items() if key != "row_evidence"}
    local = {
        "changed_cells": [], "dropped_rows": [], "added_rows": [],
        "dropped_columns": [], "added_columns": [], "truncated": False,
        "dtype_changes": [],
    }
    action = str(op.get("action") or "")
    row_actions = {"drop_duplicates", "drop_missing_rows", "manual_outlier_removal", "outlier_removal"}
    column_actions = {"drop_empty_columns", "drop_constant_columns", "drop_selected_columns"}
    if action == "custom_pandas_code":
        return {key: value for key, value in manifest.items() if key != "row_evidence"}
    if action in row_actions:
        requested = op.get("valid_indices_dropped") or op.get("selected_rows") or []
        requested_ids = {str(value) for value in requested}
        displayed = op.get("rows_dropped") or op.get("row_numbers") or []
        displayed_ids = {str(value) for value in displayed}
        def owns_row(item):
            row_id = str(item.get("row"))
            if not requested_ids and not displayed_ids:
                return True
            if row_id in requested_ids:
                return True
            return row_id.lstrip("-").isdigit() and str(int(row_id) + 1) in displayed_ids
        local["dropped_rows"] = [
            item for item in manifest.get("dropped_rows", []) or [] if owns_row(item)
        ]
    elif action in column_actions:
        columns = _operation_columns(op)
        local["dropped_columns"] = [
            item for item in manifest.get("dropped_columns", []) or []
            if str(item.get("col")) in columns
        ]
    elif action == "manual_column_renames":
        local["dropped_columns"] = list(manifest.get("dropped_columns", []) or [])
        local["added_columns"] = list(manifest.get("added_columns", []) or [])
    else:
        columns = _operation_columns(op)
        local["changed_cells"] = [
            item for item in manifest.get("changed_cells", []) or []
            if str(item.get("col")) in columns
        ]
        local["dtype_changes"] = [
            item for item in manifest.get("dtype_changes", []) or []
            if str(item.get("col")) in columns
        ]
        if action in ("fill_numeric", "fill_categorical"):
            details = op.get("details") or {}
            detail_items = details.values() if isinstance(details, dict) else details
            dropped_ids = {
                str(row_id) for detail in detail_items if isinstance(detail, dict)
                for row_id in (detail.get("row_ids") or [])
                if detail.get("strategy") == "drop" or detail.get("method") == "drop"
            }
            local["dropped_rows"] = [
                item for item in manifest.get("dropped_rows", []) or []
                if str(item.get("row")) in dropped_ids
            ]
    return local


def _sequential_history_actions(before_df: pd.DataFrame, final_df: pd.DataFrame,
                                operations: list[dict], manifest: dict,
                                action_name: str, action_scope: str,
                                dq_after: float | None,
                                applied_at_ms: int | None = None) -> list[dict]:
    """Split a combined Apply into independently replayable operation actions."""
    cursor = before_df.copy(deep=True)
    actions = []
    non_history_actions = {"replace_placeholders", "rename_columns", "skip_missing_values"}
    candidates = [
        op for op in operations
        if isinstance(op, dict) and op.get("action") and op.get("action") not in non_history_actions
    ]
    if len(candidates) == 1:
        op = candidates[0]
        local_manifest = _manifest_for_operation(manifest, op, 1)
        if snapshot_diff.manifest_is_empty(local_manifest):
            return []
        local_manifest["row_evidence"] = _full_row_evidence(before_df, final_df, local_manifest)
        config = {
            "label": _op_label(op, action_name),
            "operations": [op],
            "activePanelId": action_scope,
            "server_authoritative": True,
            "preview": _manifest_action_preview(before_df, final_df, local_manifest),
        }
        if applied_at_ms:
            config["applied_at_ms"] = applied_at_ms
        return [{
            "action_name": str(op.get("action"))[:128],
            "action_config": config,
            "rows_before": int(before_df.shape[0]),
            "rows_after": int(final_df.shape[0]),
            "cells_changed": _operation_cells_changed([op], local_manifest),
            "dq_after": dq_after,
            "manifest": local_manifest,
        }]
    for op in candidates:
        local_manifest = _manifest_for_operation(manifest, op, len(candidates))
        if snapshot_diff.manifest_is_empty(local_manifest):
            continue
        result = snapshot_diff.apply_manifest(cursor, local_manifest, "forward")
        if result.get("conflicts"):
            raise ValueError(f"Cannot split combined action safely: {result['conflicts'][0]}")
        next_df = result["df"]
        local_manifest["row_evidence"] = _full_row_evidence(cursor, next_df, local_manifest)
        local_preview = _manifest_action_preview(cursor, next_df, local_manifest)
        config = {
            "label": _op_label(op, action_name),
            "operations": [op],
            "activePanelId": action_scope,
            "server_authoritative": True,
            "preview": local_preview,
        }
        if applied_at_ms:
            config["applied_at_ms"] = applied_at_ms
        actions.append({
            "action_name": str(op.get("action"))[:128],
            "action_config": config,
            "rows_before": int(cursor.shape[0]),
            "rows_after": int(next_df.shape[0]),
            "cells_changed": _operation_cells_changed([op], local_manifest),
            "dq_after": dq_after,
            "manifest": local_manifest,
        })
        cursor = next_df
    if not actions or not _dataframes_match_for_history(cursor, final_df):
        raise ValueError("Combined operations could not be separated into exact reversible actions.")
    return actions


def _preview_for_history_operation(preview: dict | None, op: dict) -> dict:
    """Select the persisted manifest evidence that belongs to one visual card."""
    source = preview if isinstance(preview, dict) else {}
    action = str(op.get("action", ""))
    result = {
        key: source.get(key)
        for key in ("before_shape", "after_shape", "before_columns", "after_columns", "limit")
        if key in source
    }
    result.update({
        "changed_cells": [], "removed_rows": [], "added_rows": [],
        "removed_columns": [], "added_columns": [],
        "dtype_changes": [],
    })

    row_removals = {"drop_duplicates", "drop_missing_rows", "manual_outlier_removal", "outlier_removal"}
    column_removals = {"drop_empty_columns", "drop_constant_columns", "drop_selected_columns"}
    if action in row_removals:
        requested_rows = op.get("rows_dropped") or op.get("row_numbers") or op.get("selected_rows") or []
        requested = {str(item) for item in requested_rows}
        result["removed_rows"] = [
            item for item in (source.get("removed_rows") or [])
            if not requested or str(item.get("row_number")) in requested or str(item.get("row_id")) in requested
        ]
    elif action in column_removals:
        columns = _operation_columns(op)
        result["removed_columns"] = [
            item for item in (source.get("removed_columns") or [])
            if not columns or str(item.get("column")) in columns
        ]
    elif action == "custom_pandas_code":
        for key in ("changed_cells", "removed_rows", "added_rows", "removed_columns", "added_columns"):
            result[key] = list(source.get(key) or [])
    else:
        columns = _operation_columns(op)
        result["changed_cells"] = [
            item for item in (source.get("changed_cells") or [])
            if not columns or str(item.get("column")) in columns
        ]
        result["dtype_changes"] = [
            item for item in (source.get("dtype_changes") or [])
            if not columns or str(item.get("column")) in columns
        ]
        if action == "manual_column_renames":
            result["removed_columns"] = list(source.get("removed_columns") or [])
            result["added_columns"] = list(source.get("added_columns") or [])

    counts = {
        "total_changed_cells": len(result["changed_cells"]),
        "total_removed_rows": len(result["removed_rows"]),
        "total_added_rows": len(result["added_rows"]),
        "total_removed_columns": len(result["removed_columns"]),
        "total_added_columns": len(result["added_columns"]),
        "total_dtype_changes": len(result["dtype_changes"]),
    }
    if action == "drop_duplicates":
        counts["total_removed_rows"] = int(op.get("removed_rows") or len(result["removed_rows"]))
        result["duplicate_matches"] = list(op.get("duplicate_matches") or [])
        result["comparison_basis"] = op.get("comparison_basis") or "exact"
        result["normalization_used"] = bool(op.get("normalization_used"))
        result["comparison_columns"] = list(op.get("comparison_columns") or op.get("subset") or [])
    elif action == "drop_missing_rows":
        counts["total_removed_rows"] = int(op.get("dropped_row_count") or len(result["removed_rows"]))
    elif action in row_removals:
        counts["total_removed_rows"] = int(op.get("rows_removed") or len(result["removed_rows"]))
    elif action in column_removals:
        counts["total_removed_columns"] = int(op.get("count") or len(result["removed_columns"]))
    result.update(counts)
    result["displayed_change_count"] = sum(len(result[key]) for key in (
        "changed_cells", "removed_rows", "added_rows", "removed_columns", "added_columns", "dtype_changes"
    ))
    result["total_change_count"] = sum(counts.values())
    result["is_sample"] = result["displayed_change_count"] < result["total_change_count"]
    return make_json_safe(result)


def _history_pairs_from_preview(preview: dict | None) -> list[dict]:
    """Return a compact exact comparison from persisted manifest evidence."""
    preview = preview if isinstance(preview, dict) else {}
    pairs = []
    for item in (preview.get("changed_cells") or [])[:6]:
        pairs.append({
            "before": item.get("before"),
            "after": item.get("after"),
            "context": f"Row {item.get('row_number')} - {item.get('column')}",
        })
    for item in (preview.get("removed_rows") or [])[:6 - len(pairs)]:
        pairs.append({
            "before": _compact_history_row(item.get("data")),
            "after": "Removed",
            "context": f"Row {item.get('row_number')}",
        })
    for item in (preview.get("added_rows") or [])[:6 - len(pairs)]:
        pairs.append({
            "before": "Not present",
            "after": _compact_history_row(item.get("data")),
            "context": f"Row {item.get('row_number')}",
        })
    for item in (preview.get("removed_columns") or [])[:6 - len(pairs)]:
        pairs.append({"before": "Column existed", "after": "Dropped", "context": item.get("column")})
    for item in (preview.get("added_columns") or [])[:6 - len(pairs)]:
        pairs.append({"before": "Not present", "after": item.get("column"), "context": "Column"})
    for item in (preview.get("dtype_changes") or [])[:6 - len(pairs)]:
        pairs.append({
            "before": item.get("before"),
            "after": item.get("after"),
            "context": f"{item.get('column')} data type",
        })
    return make_json_safe(pairs)


def _history_subfeature(scope: str, operations: list[dict]) -> str | None:
    names = {str(op.get("action", "")) for op in operations if op.get("action")}
    if not names:
        return None
    matches = [
        item["key"] for item in FEATURE_HISTORY_SUBFEATURES.get(scope, [])
        if names.issubset(item["operations"])
    ]
    return matches[0] if len(matches) == 1 else None


def build_grouped_cleaning_history(history: list[dict]) -> list[dict]:
    """Build feature-local visual cards from persisted server history."""
    hidden_actions = {"label_encode", "manual_column_renames", "rename_columns"}
    groups = []
    grouped = {}
    for scope, meta in sorted(FEATURE_HISTORY_META.items(), key=lambda item: item[1]["order"]):
        group = {
            "scope": scope,
            "label": meta["label"],
            "order": meta["order"],
            "actions": [],
            "subfeatures": [
                {"key": item["key"], "label": item["label"], "actions": []}
                for item in FEATURE_HISTORY_SUBFEATURES.get(scope, [])
            ],
        }
        groups.append(group)
        grouped[scope] = group

    for sequence, entry in enumerate(history or [], start=1):
        config = entry.get("action_config") if isinstance(entry.get("action_config"), dict) else {}
        stored_ops = config.get("operations")
        operations = stored_ops if isinstance(stored_ops, list) else []
        fallback = _titleize_action(config.get("label") or entry.get("action_name"))
        source_preview = config.get("preview") if isinstance(config.get("preview"), dict) else {}
        card_operations = operations or [{}]
        for operation_index, op in enumerate(card_operations):
            if op and str(op.get("action", "")) in hidden_actions:
                continue
            scope = _history_scope_for(entry.get("action_name"), config, [op] if op else None)
            if scope not in grouped:
                scope = "dataIntegrityGroup"
            group = grouped[scope]
            subfeature_key = _history_subfeature(scope, [op]) if op else None
            target = group["actions"]
            if subfeature_key:
                target = next(item["actions"] for item in group["subfeatures"] if item["key"] == subfeature_key)

            preview = _preview_for_history_operation(source_preview, op) if op else source_preview
            pairs = _history_pairs_from_preview(preview)
            if not pairs and op:
                pairs = _history_pairs_for_operation(op)
            target.append(make_json_safe({
                "id": entry.get("id"),
                "sequence": sequence,
                "operation_index": operation_index,
                "local_number": len(target) + 1,
                "label": _op_label(op, fallback) if op else fallback,
                "applied_at": entry.get("applied_at"),
                "rows_before": entry.get("rows_before") or 0,
                "rows_after": entry.get("rows_after") or 0,
                "cells_changed": preview.get("total_change_count") or 0,
                "status": (entry.get("status") or "applied").capitalize(),
                "operation": op.get("action") if op else entry.get("action_name"),
                "operations": [op] if op else [],
                "pairs": pairs,
                "preview": preview,
                "config": config,
            }))

    for group in groups:
        group["action_count"] = len(group["actions"]) + sum(
            len(item["actions"]) for item in group["subfeatures"]
        )
    return groups


import snapshot_diff
import pyarrow.feather as feather

# --- Current-state snapshots (persistent undo/redo) --------------------------
# One Feather file per session, holding the FULL dataset exactly as it stands
# after every currently-applied (non-undone) action — kept in sync on every
# committed apply/undo/reapply. This is what makes Undo/Reapply survive an
# app restart: instead of replaying the whole cleaning pipeline from the raw
# file, we just load this file directly.
#
# Deliberately one-per-session (not one-per-action): storing a Feather file
# for every single action would multiply disk usage by the number of actions
# ever taken. The per-action *manifests* (in action_config, via
# snapshot_diff.compute_full_diff) are what make undo/redo of an individual
# action possible against this single current file.

def _current_snapshot_path(sid: str) -> str:
    return os.path.join(UPLOAD_FOLDER, sid, "_current_snapshot.feather")


def _replace_snapshot_file(tmp_path: str, final_path: str) -> None:
    """Replace the current snapshot, tolerating short Windows file locks."""
    last_error = None
    for attempt in range(6):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
    os.replace(tmp_path, final_path)


def save_current_snapshot(sid: str, df: pd.DataFrame) -> None:
    """Persist the authoritative current dataset snapshot for a session."""
    # Feather can't round-trip an arbitrary pandas Index directly (and
    # reset_index(drop=True) would destroy the original row labels every
    # other part of this app relies on for row identity) — so the index is
    # carried across as an explicit first column instead, and restored on read.
    path = _current_snapshot_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_write = df.reset_index()
    to_write.columns = ["__orig_index__"] + list(df.columns)

    # Defensive backstop: a column can end up as `object` dtype containing a
    # MIX of real pd.Timestamp values and plain date strings — most commonly
    # when apply_manifest() reconstructs a previously-dropped row or undoes a
    # cell edit on a date column (manifests store JSON-safe values, so a
    # Timestamp becomes a string there — see snapshot_diff._coerce_to_column_dtype,
    # which fixes this at the source for known cases). pyarrow/Feather can't
    # serialise that mix at all (ArrowTypeError: "Expected bytes, got a
    # 'Timestamp' object"). This catches ANY such column regardless of cause,
    # not just the two known paths, so a snapshot write can't crash the app.
    for col in to_write.columns:
        if to_write[col].dtype != "object":
            continue
        non_null = to_write[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(lambda v: isinstance(v, pd.Timestamp)).any():
            try:
                to_write[col] = pd.to_datetime(to_write[col])
            except Exception:
                logging.exception(
                    f"Could not normalise column '{col}' to datetime while saving snapshot "
                    "— saving as strings instead so the snapshot write itself doesn't fail"
                )
                to_write[col] = to_write[col].astype(str)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        # Write to a temp file first, then atomically replace — a crash or
        # kill mid-write (e.g. desktop app closed while a click is in
        # flight) must never leave a half-written, corrupt snapshot behind
        # for the next resume/undo to trip over.
        tmp_path = path + ".tmp"
        feather.write_feather(to_write, tmp_path)
        _replace_snapshot_file(tmp_path, path)


def prepare_current_snapshot_temp(sid: str, df: pd.DataFrame) -> tuple[str, str]:
    """Write a temporary current snapshot and return (tmp_path, final_path).

    The caller must promote it with ``os.replace`` only after any required
    audit/history writes have succeeded. This is used by committed Apply so a
    dataset mutation and its authoritative history entry cannot diverge.
    """
    path = _current_snapshot_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_write = df.reset_index()
    to_write.columns = ["__orig_index__"] + list(df.columns)
    for col in to_write.columns:
        if to_write[col].dtype != "object":
            continue
        non_null = to_write[col].dropna()
        if non_null.empty:
            continue
        if non_null.map(lambda v: isinstance(v, pd.Timestamp)).any():
            try:
                to_write[col] = pd.to_datetime(to_write[col])
            except Exception:
                logging.exception(
                    f"Could not normalise column '{col}' to datetime while preparing snapshot "
                    "- saving as strings instead so the snapshot write itself doesn't fail"
                )
                to_write[col] = to_write[col].astype(str)

    tmp_path = path + f".{secrets.token_hex(8)}.tmp"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        feather.write_feather(to_write, tmp_path)
    return tmp_path, path


def promote_current_snapshot_temp(tmp_path: str, final_path: str) -> None:
    """Atomically promote a prepared snapshot after history persistence succeeds."""
    _replace_snapshot_file(tmp_path, final_path)


def load_current_snapshot(sid: str):
    """Returns the persisted current DataFrame, or None if it doesn't exist
    yet — either the first action ever for this session, or a legacy session
    that predates this feature. Both are valid, non-error cases the caller
    must handle (fall back to the pre-existing behaviour)."""
    path = _current_snapshot_path(sid)
    if not os.path.exists(path):
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = feather.read_feather(path)
        return raw.set_index("__orig_index__").rename_axis(None)
    except Exception:
        logging.exception(f"Failed to read current snapshot for sid={sid} — treating as absent")
        return None


# --- Dirty-row tracking -------------------------------------------------------
# "Which rows currently differ from the original raw file" (used for the
# Cleaning Work Progress highlighting) used to be answered by
# compute_changed_rows() diffing the ENTIRE current dataframe against the
# ENTIRE raw dataframe, cell by cell, on every apply/undo/reapply/resume —
# O(dataset size) work to answer a question that only changes by a handful
# of rows per click.
#
# Instead this is maintained as a small persisted set of row labels (just
# labels — no cell data) that gets updated by re-checking ONLY the rows a
# single manifest actually touches (a handful of labels) against the raw
# file, each time that manifest is applied or undone. That's exact — not an
# approximation — because it's re-verified directly against the raw file for
# every row it could possibly affect; it just never re-checks rows nothing
# touched.

def _dirty_rows_path(sid: str) -> str:
    return os.path.join(UPLOAD_FOLDER, sid, "_dirty_rows.json")


def load_dirty_rows(sid: str) -> set:
    path = _dirty_rows_path(sid)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        logging.exception(f"Failed to read dirty-rows store for sid={sid} — treating as empty")
        return set()


def save_dirty_rows(sid: str, dirty: set) -> None:
    path = _dirty_rows_path(sid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sorted(dirty), f)
    os.replace(tmp_path, path)


def update_dirty_rows_for_manifest(sid: str, raw_df: pd.DataFrame, new_df: pd.DataFrame,
                                    manifest: dict) -> set:
    """Re-verify just the rows this one manifest could have touched against
    the raw file, and update the persisted dirty-row set accordingly. Returns
    the updated set. Rows nothing in this manifest touched are left alone —
    their dirty/clean status can't have changed by this action."""
    dirty = load_dirty_rows(sid)

    candidate_labels = set()
    for cell in manifest.get("changed_cells", []):
        candidate_labels.add(cell["row"])
    # Added/dropped rows are tracked separately as added/removed, not "changed"
    # — they're intentionally excluded from this cell-level dirty check.

    common_cols = [c for c in raw_df.columns if c in new_df.columns]
    for label in candidate_labels:
        row_key = str(label)
        row = _restore_index_dtype_for_lookup(label, raw_df.index)
        if row not in raw_df.index or row not in new_df.index or not common_cols:
            dirty.discard(row_key)
            continue
        b = raw_df.loc[row, common_cols]
        c = new_df.loc[row, common_cols]
        still_differs = False
        for col in common_cols:
            bv, cv = b[col], c[col]
            b_na = pd.isna(bv)
            c_na = pd.isna(cv)
            if b_na and c_na:
                continue
            if b_na != c_na or bv != cv:
                still_differs = True
                break
        if still_differs:
            dirty.add(row_key)
        else:
            dirty.discard(row_key)

    save_dirty_rows(sid, dirty)
    return dirty


# --- Canonical data-quality scoring -----------------------------------------
# Runtime scoring deliberately uses cleaner.data_quality_metrics via the
# fingerprint cache. Legacy incremental scoring code remains isolated in
# dq_incremental.py and is not imported by production routes.

def get_dq_after_canonical(sid: str, raw_df: pd.DataFrame, new_df: pd.DataFrame,
                            manifest: dict = None, direction: str = "", fp=None) -> float:
    """Return the full canonical quality score for the proposed current state.

    Legacy parameters are kept for older call sites. Production quality scoring
    intentionally reads ``new_df`` as a whole so Undo/Reapply, final reports and
    previews share the same formula rather than an unsafe incremental shortcut.
    """
    return get_cached_dq_after(sid, new_df, fp=fp)


def _restore_index_dtype_for_lookup(row_key, sample_index: pd.Index):
    if len(sample_index) and pd.api.types.is_integer_dtype(sample_index.dtype):
        try:
            return int(row_key)
        except (TypeError, ValueError):
            return row_key
    return row_key


def df_to_rows_safe(df, max_rows=None):
    """Convert a DataFrame to a JSON-safe list of row dicts, formatting any
    datetime columns as strings first (raw Timestamps aren't JSON-serialisable
    and make_json_safe doesn't special-case them column-wide).

    This used to be orphaned dead code — its body was sitting unreachable
    after load_current_snapshot's return statements, missing its own `def`
    line entirely, so every call site (upload, preview, resume, editor data)
    was hitting a NameError as soon as it ran.
    """
    if df is None or df.empty:
        return []
    target = df if max_rows is None else df.head(max_rows)
    dt_cols = [c for c in target.columns if pd.api.types.is_datetime64_any_dtype(target[c])]
    if dt_cols:
        target = target.copy()
        for c in dt_cols:
            target[c] = target[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    return make_json_safe(target.to_dict(orient="records"))


def row_numbers_for(df):
    return [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in df.index]


def compute_changed_rows(original_df, cleaned_df):
    # Return only rows that differ — used by Cleaning Work Progress view
    orig_idx  = set(original_df.index.tolist())
    clean_idx = set(cleaned_df.index.tolist())
    removed   = sorted([int(i) + 1 if isinstance(i, (int, np.integer)) else str(i)
                        for i in orig_idx - clean_idx])
    added_idx = clean_idx - orig_idx
    common    = list(orig_idx & clean_idx)
    common_cols = list(set(original_df.columns) & set(cleaned_df.columns))
    changed_index = []
    if common and common_cols:
        # Compare column-by-column with isna()-aware equality instead of the old
        # .astype(object).fillna("__NaN__") sentinel trick. That worked but
        # triggers pandas' "Downcasting object dtype arrays on .fillna" /
        # comparison FutureWarning on newer pandas and spams the console on
        # every single preview/apply. This gets the same result — NaN==NaN
        # counts as "unchanged", one-sided NaN or an unequal value counts as
        # "changed" — without calling fillna.
        #
        # Note: b.isna() / c.isna() always return plain bool Series (never
        # pd.NA), so gating the raw (b != c) comparison behind
        # "~b_na & ~c_na" is safe even for nullable extension dtypes
        # (Categorical, nullable Int64/boolean) whose equality comparisons can
        # themselves produce pd.NA — Kleene AND with a plain False dominates,
        # so no pd.NA ever reaches diff_mask or the final .any(axis=1).
        common_index = pd.Index(common)
        diff_mask = pd.DataFrame(False, index=common_index, columns=common_cols)
        for col in common_cols:
            b = original_df.loc[common_index, col]
            c = cleaned_df.loc[common_index, col]
            b_na = b.isna()
            c_na = c.isna()
            both_present = ~b_na & ~c_na
            only_one_na = b_na ^ c_na
            diff_mask[col] = ((both_present & (b != c)) | only_one_na).astype(bool)
        changed_index = common_index[diff_mask.any(axis=1)].tolist()
    new_cols = [c for c in cleaned_df.columns if c not in original_df.columns]
    if new_cols and not changed_index:
        changed_index = list(cleaned_df.index[:min(len(cleaned_df), 200)])
    show_idx = list(set(changed_index) | added_idx)
    if not show_idx:
        return [], [], removed, []
    show_df    = cleaned_df.loc[[i for i in show_idx if i in cleaned_df.index]]
    added_nums = [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in added_idx]
    return df_to_rows_safe(show_df), row_numbers_for(show_df), removed, added_nums


def _display_row_number_for_label(label):
    try:
        return int(label) + 1
    except (TypeError, ValueError):
        return str(label)


def _row_labels_from_operation_evidence(operations: list[dict] | None) -> tuple[set[str], set[str], bool]:
    """Return changed and removed row labels from recorded operation evidence.

    The final bool is True only when the operation evidence was sufficient and
    a full dataframe comparison can be skipped safely.
    """
    changed: set[str] = set()
    removed: set[str] = set()
    unsupported = False
    for op in operations or []:
        action = str(op.get("action") or "")
        if action in ("skip_missing_values",):
            continue
        if action in ("drop_empty_columns", "drop_constant_columns", "drop_selected_columns", "convert_dtypes", "manual_convert_dtypes"):
            continue
        if action in ("manual_cell_updates", "manual_missing_cell_updates"):
            for update in op.get("updates") or []:
                if update.get("row_index") is not None:
                    changed.add(str(update.get("row_index")))
            continue
        if action == "replace_placeholders":
            for item in op.get("replacements") or []:
                if item.get("row_index") is not None:
                    changed.add(str(item.get("row_index")))
            continue
        if action == "trim_whitespace":
            for item in op.get("details") or []:
                changed.update(str(row_id) for row_id in (item.get("row_ids") or []))
            continue
        if action == "fill_numeric":
            for item in op.get("details") or []:
                if item.get("strategy") == "drop":
                    removed.update(str(row_id) for row_id in (item.get("row_ids") or []))
                else:
                    changed.update(str(row_id) for row_id in (item.get("row_ids") or []))
            continue
        if action == "fill_categorical":
            for item in (op.get("details") or {}).values():
                if not isinstance(item, dict):
                    continue
                if item.get("strategy") == "drop" or item.get("method") == "drop":
                    removed.update(str(row_id) for row_id in (item.get("row_ids") or []))
                else:
                    changed.update(str(row_id) for row_id in (item.get("row_ids") or []))
            continue
        if action in ("manual_outlier_removal", "outlier_removal"):
            removed.update(str(row_id) for row_id in (op.get("valid_indices_dropped") or op.get("selected_rows") or []))
            continue
        if action == "drop_duplicates":
            for item in op.get("duplicate_matches") or []:
                if item.get("removed_row_id") is not None:
                    removed.add(str(item.get("removed_row_id")))
            continue
        if action == "drop_missing_rows":
            unsupported = True
            continue
        if action in ("outlier_clip",):
            for item in op.get("changes") or []:
                if item.get("row_id") is not None:
                    changed.add(str(item.get("row_id")))
            continue
        unsupported = True
    return changed, removed, not unsupported


def _changed_rows_from_evidence(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                                operations: list[dict] | None, manifest: dict | None = None):
    changed_labels: set[str] = set()
    removed_labels: set[str] = set()
    supported = True
    if manifest:
        changed_labels.update(str(item.get("row")) for item in manifest.get("changed_cells", []) if item.get("row") is not None)
        changed_labels.update(str(item.get("row")) for item in manifest.get("added_rows", []) if item.get("row") is not None)
        removed_labels.update(str(item.get("row")) for item in manifest.get("dropped_rows", []) if item.get("row") is not None)
    else:
        changed_labels, removed_labels, supported = _row_labels_from_operation_evidence(operations)
    if not supported:
        return None

    changed_index = []
    for label in changed_labels:
        row = _restore_index_dtype_for_lookup(label, cleaned_df.index)
        if row in cleaned_df.index:
            changed_index.append(row)
    changed_index = list(dict.fromkeys(changed_index))
    show_index = changed_index[:200]
    changed_recs = df_to_rows_safe(cleaned_df.loc[show_index]) if show_index else []
    changed_rnums = [_display_row_number_for_label(label) for label in list(changed_labels)[:200]]
    removed_rnums = [_display_row_number_for_label(label) for label in list(removed_labels)]
    added_rnums = [
        _display_row_number_for_label(idx)
        for idx in cleaned_df.index.difference(raw_df.index).tolist()
    ]
    return changed_recs, changed_rnums, removed_rnums, added_rnums, len(changed_labels)


def build_manifest_patch(manifest: dict, direction: str) -> dict:
    """Turn a manifest directly into a frontend patch — no dataframe access
    at all. The manifest already carries the before/after value for every
    changed cell and the full row data for every added/dropped row, so a
    patch is just picking the right side of that pair for this direction.
    Cost is O(this manifest's own size), never O(dataset size)."""
    value_key = "after" if direction == "forward" else "before"
    changed_cells = [
        {"row": c["row"], "col": c["col"], "value": c[value_key]}
        for c in manifest.get("changed_cells", [])
    ]

    rows_to_remove = manifest.get("dropped_rows", []) if direction == "forward" else manifest.get("added_rows", [])
    rows_to_add    = manifest.get("added_rows", [])    if direction == "forward" else manifest.get("dropped_rows", [])
    removed_rows = [r["row"] for r in rows_to_remove]
    added_rows   = [{"row": r["row"], "data": r["data"]} for r in rows_to_add]

    cols_to_remove = manifest.get("dropped_columns", []) if direction == "forward" else manifest.get("added_columns", [])
    cols_to_add    = manifest.get("added_columns", [])    if direction == "forward" else manifest.get("dropped_columns", [])
    removed_columns = [c["col"] for c in cols_to_remove]
    added_columns   = [{"col": c["col"], "data": c["data"]} for c in cols_to_add]

    return {
        "changed_cells":   changed_cells,
        "removed_rows":    removed_rows,
        "added_rows":      added_rows,
        "removed_columns": removed_columns,
        "added_columns":   added_columns,
    }


def _manual_update_manifest(before_df: pd.DataFrame, after_df: pd.DataFrame, operations: list | None) -> dict | None:
    """Build an exact manifest for manual cell edits without scanning the whole DataFrame."""
    allowed_actions = {"manual_cell_updates", "manual_missing_cell_updates"}
    ignored_actions = {"skip_missing_values"}
    manual_ops = []
    for op in operations or []:
        action = str(op.get("action") or "")
        if action in allowed_actions:
            manual_ops.append(op)
        elif action not in ignored_actions:
            return None
    if not manual_ops:
        return None

    manifest = {
        "changed_cells": [],
        "dropped_rows": [],
        "added_rows": [],
        "dropped_columns": [],
        "added_columns": [],
        "dtype_changes": [],
        "truncated": False,
    }
    seen = set()
    for op in manual_ops:
        for update in op.get("updates") or []:
            row = update.get("row_index")
            col = update.get("column")
            if col not in before_df.columns or col not in after_df.columns:
                return None
            row_label = _restore_index_dtype_for_lookup(row, before_df.index)
            if row_label not in before_df.index or row_label not in after_df.index:
                return None
            key = (str(row_label), str(col))
            if key in seen:
                continue
            before_value = before_df.at[row_label, col]
            after_value = after_df.at[row_label, col]
            try:
                before_missing = bool(pd.isna(before_value))
                after_missing = bool(pd.isna(after_value))
                if before_missing and after_missing:
                    same_value = True
                elif before_missing != after_missing:
                    same_value = False
                else:
                    same_value = bool(before_value == after_value)
            except (TypeError, ValueError):
                same_value = False
            if same_value:
                continue
            manifest["changed_cells"].append({
                "row": str(row_label),
                "col": str(col),
                "before": make_json_safe(before_value),
                "after": make_json_safe(after_value),
            })
            seen.add(key)

    for col in before_df.columns.intersection(after_df.columns):
        before_dtype = str(before_df[col].dtype)
        after_dtype = str(after_df[col].dtype)
        if before_dtype != after_dtype:
            manifest["dtype_changes"].append({
                "col": str(col),
                "before": before_dtype,
                "after": after_dtype,
            })
    return manifest


def _operation_manifest(before_df: pd.DataFrame, after_df: pd.DataFrame, operations: list | None) -> dict | None:
    """Build an exact per-action manifest from operation metadata.

    This avoids the expensive all-cells diff for common actions where the
    cleaner already reports the affected rows/columns. Unsupported operations
    return None so callers fall back to snapshot_diff.compute_full_diff.
    """
    ignored_actions = {"skip_missing_values"}
    supported_actions = {
        "manual_cell_updates", "manual_missing_cell_updates",
        "drop_selected_columns", "drop_empty_columns", "drop_constant_columns",
        "drop_duplicates", "manual_outlier_removal", "outlier_removal",
        "fill_numeric", "fill_categorical", "trim_whitespace",
        "manual_convert_dtypes", "convert_dtypes",
    }
    ops = operations or []
    if not ops:
        return None
    for op in ops:
        action = str(op.get("action") or "")
        if action not in supported_actions and action not in ignored_actions:
            return None

    manifest = {
        "changed_cells": [],
        "dropped_rows": [],
        "added_rows": [],
        "dropped_columns": [],
        "added_columns": [],
        "dtype_changes": [],
        "truncated": False,
    }
    seen_cells: set[tuple[str, str]] = set()
    seen_rows: set[str] = set()
    seen_cols: set[str] = set()

    def value_changed(before_value, after_value) -> bool:
        try:
            before_missing = bool(pd.isna(before_value))
            after_missing = bool(pd.isna(after_value))
            if before_missing and after_missing:
                return False
            if before_missing != after_missing:
                return True
            return not bool(before_value == after_value)
        except (TypeError, ValueError):
            return True

    def resolve_row(row_key, frame_index):
        row = _restore_index_dtype_for_lookup(row_key, frame_index)
        return row if row in frame_index else None

    def add_changed_cell(row_key, col):
        if col not in before_df.columns or col not in after_df.columns:
            return False
        before_row = resolve_row(row_key, before_df.index)
        after_row = resolve_row(row_key, after_df.index)
        if before_row is None or after_row is None:
            return False
        key = (str(before_row), str(col))
        if key in seen_cells:
            return True
        before_value = before_df.at[before_row, col]
        after_value = after_df.at[after_row, col]
        if value_changed(before_value, after_value):
            manifest["changed_cells"].append({
                "row": str(before_row),
                "col": str(col),
                "before": make_json_safe(before_value),
                "after": make_json_safe(after_value),
            })
        seen_cells.add(key)
        return True

    def add_dropped_row(row_key):
        row = resolve_row(row_key, before_df.index)
        if row is None:
            return False
        key = str(row)
        if key in seen_rows:
            return True
        series = before_df.loc[row]
        manifest["dropped_rows"].append({
            "row": key,
            "data": {str(c): make_json_safe(series[c]) for c in before_df.columns},
        })
        seen_rows.add(key)
        return True

    def add_dropped_column(col):
        if col not in before_df.columns:
            return False
        if col in after_df.columns:
            return True
        key = str(col)
        if key in seen_cols:
            return True
        series = before_df[col]
        manifest["dropped_columns"].append({
            "col": key,
            "data": {str(idx): make_json_safe(series.loc[idx]) for idx in before_df.index},
        })
        seen_cols.add(key)
        return True

    def add_dtype_changes(columns, requested_by_col=None):
        requested_by_col = requested_by_col or {}
        for col in columns:
            if col not in before_df.columns or col not in after_df.columns:
                continue
            before_dtype = str(before_df[col].dtype)
            after_dtype = str(after_df[col].dtype)
            if before_dtype != after_dtype and not any(item.get("col") == str(col) for item in manifest["dtype_changes"]):
                entry = {
                    "col": str(col),
                    "before": before_dtype,
                    "after": after_dtype,
                }
                requested = requested_by_col.get(str(col))
                if requested:
                    entry["requested"] = str(requested)
                manifest["dtype_changes"].append(entry)

    for op in ops:
        action = str(op.get("action") or "")
        if action in ignored_actions:
            continue

        if action in {"manual_cell_updates", "manual_missing_cell_updates"}:
            for update in op.get("updates") or []:
                if not add_changed_cell(update.get("row_index"), update.get("column")):
                    return None

        elif action in {"drop_selected_columns", "drop_empty_columns", "drop_constant_columns"}:
            for col in op.get("columns_dropped") or []:
                if not add_dropped_column(col):
                    return None

        elif action == "drop_duplicates":
            row_ids = [
                match.get("removed_row_id")
                for match in (op.get("duplicate_matches") or [])
                if match.get("removed_row_id") is not None
            ]
            if not row_ids and int(op.get("removed_rows") or 0):
                return None
            for row_id in row_ids:
                if not add_dropped_row(row_id):
                    return None

        elif action in {"manual_outlier_removal", "outlier_removal"}:
            row_ids = op.get("valid_indices_dropped") or op.get("selected_positions") or []
            if not row_ids and int(op.get("rows_removed") or op.get("removed_rows") or 0):
                return None
            for row_id in row_ids:
                if not add_dropped_row(row_id):
                    return None

        elif action == "fill_numeric":
            for detail in op.get("details") or []:
                col = detail.get("column")
                row_ids = detail.get("row_ids") or []
                if detail.get("strategy") == "drop":
                    for row_id in row_ids:
                        if not add_dropped_row(row_id):
                            return None
                else:
                    for row_id in row_ids:
                        if not add_changed_cell(row_id, col):
                            return None

        elif action == "fill_categorical":
            details = op.get("details") or {}
            iterable = details.items() if isinstance(details, dict) else []
            for col, detail in iterable:
                row_ids = detail.get("row_ids") or []
                if detail.get("strategy") == "drop" or detail.get("method") == "drop":
                    for row_id in row_ids:
                        if not add_dropped_row(row_id):
                            return None
                else:
                    for row_id in row_ids:
                        if not add_changed_cell(row_id, col):
                            return None

        elif action == "trim_whitespace":
            for detail in op.get("details") or []:
                col = detail.get("column")
                for row_id in detail.get("row_ids") or []:
                    if not add_changed_cell(row_id, col):
                        return None

        elif action in {"manual_convert_dtypes", "convert_dtypes"}:
            changes = op.get("changes") or {}
            columns = list(changes.keys()) if isinstance(changes, dict) else [
                item.get("column") for item in changes if item.get("column")
            ]
            requested_by_col = {}
            if isinstance(changes, dict):
                requested_by_col = {
                    str(col): info.get("requested")
                    for col, info in changes.items()
                    if isinstance(info, dict) and info.get("requested")
                }
            else:
                requested_by_col = {
                    str(item.get("column")): item.get("requested")
                    for item in changes
                    if isinstance(item, dict) and item.get("column") and item.get("requested")
                }
            common_rows = before_df.index.intersection(after_df.index)
            for col in columns:
                if col not in before_df.columns or col not in after_df.columns:
                    return None
                before_col = before_df.loc[common_rows, col]
                after_col = after_df.loc[common_rows, col]
                before_na = before_col.isna()
                after_na = after_col.isna()
                # Dtype replay alone is sufficient only when values compare
                # equal (for example 1.0 -> 1). Lossy conversions such as
                # 137.5 -> 138, text-to-number conversions, and coercions to
                # missing must retain their exact before/after cell values.
                # Those entries also drive the live Dataset Editor patch and
                # let Undo restore the original decimals/text precisely.
                only_one_na = before_na ^ after_na
                different_present = pd.Series(False, index=common_rows, dtype=bool)
                both_present = ~before_na & ~after_na
                if bool(both_present.any()):
                    present_rows = common_rows[both_present.to_numpy()]
                    left = before_col.loc[present_rows].astype(object)
                    right = after_col.loc[present_rows].astype(object)
                    different_present.loc[present_rows] = left.ne(right).fillna(False).to_numpy(dtype=bool)
                diff_mask = (only_one_na | different_present).astype(bool)
                for row_id in common_rows[diff_mask.to_numpy()]:
                    if not add_changed_cell(row_id, col):
                        return None
            add_dtype_changes(columns, requested_by_col)

    actual_dropped_rows = {str(idx) for idx in before_df.index.difference(after_df.index)}
    actual_added_rows = {str(idx) for idx in after_df.index.difference(before_df.index)}
    actual_dropped_cols = {str(col) for col in before_df.columns.difference(after_df.columns)}
    actual_added_cols = {str(col) for col in after_df.columns.difference(before_df.columns)}
    manifest_dropped_rows = {str(item.get("row")) for item in manifest["dropped_rows"]}
    manifest_dropped_cols = {str(item.get("col")) for item in manifest["dropped_columns"]}
    if actual_added_rows or actual_added_cols:
        return None
    if manifest_dropped_rows != actual_dropped_rows or manifest_dropped_cols != actual_dropped_cols:
        return None

    add_dtype_changes(before_df.columns.intersection(after_df.columns))
    return manifest


def compute_added_removed_rows(raw_df: pd.DataFrame, new_df: pd.DataFrame):
    """Index-label set difference only — O(rows), never touches cell values —
    used to report which rows exist now that didn't exist in the raw file
    (or vice versa). This is cheap enough to run in full every time; it's
    only the cell-level 'changed' diff that needs incremental tracking
    (see update_dirty_rows_for_manifest)."""
    orig_idx  = set(raw_df.index.tolist())
    new_idx   = set(new_df.index.tolist())
    removed   = sorted(int(i) + 1 if isinstance(i, (int, np.integer)) else str(i)
                        for i in orig_idx - new_idx)
    added     = sorted(int(i) + 1 if isinstance(i, (int, np.integer)) else str(i)
                        for i in new_idx - orig_idx)
    return removed, added


def refresh_report_preview(cleaned_df, report):
    # Refresh report fields that can change after post-cleaning custom code
    report["cleaned_shape"]  = list(cleaned_df.shape)
    report["rows_after"]     = int(cleaned_df.shape[0])
    report["cleaned_columns"] = list(cleaned_df.columns)
    preview_rows = min(len(cleaned_df), 100)
    dt_cols = [c for c in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[c])]
    preview = cleaned_df.head(preview_rows).copy() if dt_cols else cleaned_df.head(preview_rows)
    if dt_cols:
        for c in dt_cols:
            preview[c] = preview[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    def _row_num(label):
        if isinstance(label, int):
            return int(label) + 1
        try:
            return int(cleaned_df.index.get_loc(label)) + 1
        except Exception:
            return str(label)

    report["preview_row_numbers"]  = [_row_num(l) for l in list(preview.index)]
    report["preview_display_limit"] = preview_rows
    report["preview"]              = make_json_safe(preview.to_dict(orient="records"))
    return report


def apply_custom_code_blocks(cleaned_df, report, form):
    if not ENABLE_CUSTOM_CODE:
        return cleaned_df, report
    # Each custom-code action now owns a unique ledger key
    # (custom_code_block_<timestamp>_<random>, set by applyCell() in
    # script.js) instead of a single shared 'custom_code_block' slot — that's
    # what lets more than one custom-code action coexist and be undone/
    # reapplied independently instead of the newest one silently overwriting
    # the last. Collect every matching key and run them in the order they
    # were created (the timestamp embedded in the key, not dict/form
    # iteration order, which isn't guaranteed to be preserved everywhere
    # `form` comes from — e.g. the resume-replay _FormProxy path).
    #
    # NOTE: deliberately '_block_' in the prefix (not just 'custom_code_') so
    # this never accidentally matches the separate 'custom_code_preview' key
    # used for a preview-only run that was never committed.
    code_keys = sorted(
        (k for k in form.keys() if k.startswith("custom_code_block_")),
        key=lambda k: k.split("_")[3] if len(k.split("_")) > 3 and k.split("_")[3].isdigit() else k,
    )
    blocks = [form.get(k, "") for k in code_keys]
    blocks = [b for b in blocks if str(b).strip()]
    preview_block = form.get("custom_code_preview", "")
    if preview_block and preview_block.strip():
        blocks.append(preview_block)
    if not blocks:
        return cleaned_df, report
    from custom_code_runner import run_custom_code
    working = cleaned_df
    for block in blocks:
        result  = run_custom_code(working, block)
        working = result["df"]
        report.setdefault("operations", []).append(result["operation"])
    report = refresh_report_preview(working, report)
    return working, report


def delete_original_upload(sid, filename):
    # Delete source file after cleaned output is saved
    if not sid or not filename:
        return
    path = os.path.join(UPLOAD_FOLDER, sid, secure_filename(filename))
    try:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Deleted original upload: {filename}")
    except Exception as e:
        logging.warning(f"Could not delete upload {filename}: {e}")


def delete_session_folder(sid, reason="session cleanup"):
    # Safety check: only delete paths inside UPLOAD_FOLDER
    if not sid:
        return
    session_root = os.path.abspath(os.path.join(UPLOAD_FOLDER, str(sid)))
    upload_root  = os.path.abspath(UPLOAD_FOLDER)
    if not session_root.startswith(upload_root + os.sep):
        logging.warning(f"Refused to delete path outside upload root: {session_root}")
        return
    if os.path.isdir(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
        logging.info(f"Deleted session folder ({reason}): {sid}")
    with _cache_lock:
        _df_cache.pop(sid, None)
        _dq_cache.pop(sid, None)
        _dq_after_cache.pop(sid, None)
        _analysis_cache.pop(sid, None)


# --- Manifest files (unlimited undo/redo, no MySQL JSON-column blowup) -------
# Manifests can legitimately cover hundreds of thousands of changed cells
# now that snapshot_diff.compute_full_diff has no cap (Phase 2). Storing
# that inline in cleaning_history.action_config (a MySQL JSON column) would
# slow every query touching that table and risk hitting MySQL's max_allowed_packet
# on genuinely huge diffs. Instead each manifest is written to its own file in
# the session's manifests/ folder, and action_config stores only the filename.
# delete_session_folder() already removes this directory (and everything else
# under UPLOAD_FOLDER/<sid>) when a project is deleted — nothing extra needed there.

def _manifests_dir(sid: str) -> str:
    return os.path.join(UPLOAD_FOLDER, sid, "manifests")


def save_manifest_to_file(sid: str, manifest: dict) -> str:
    """Writes `manifest` to a new file under this session's manifests/ folder
    and returns just the filename (not the full path) — that's what gets
    stored in action_config so a later UPLOAD_FOLDER move/rename doesn't
    break every existing reference."""
    d = _manifests_dir(sid)
    os.makedirs(d, exist_ok=True)
    filename = f"manifest_{secrets.token_hex(8)}.json"
    path = os.path.join(d, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return filename


def load_manifest_from_file(sid: str, filename: str):
    """Returns the manifest dict, or None if the file is missing/unreadable —
    callers treat that the same as 'no manifest' (falls back to the
    frontend's in-memory snapBefore/snapAfter behaviour) rather than erroring."""
    if not filename:
        return None
    # secure_filename strips any path separators — filename always comes from
    # our own save_manifest_to_file, but treat it as untrusted anyway since it
    # round-trips through the DB and the request body on every Undo/Reapply.
    safe_name = secure_filename(filename)
    path = os.path.join(_manifests_dir(sid), safe_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.exception(f"Failed to read manifest file {safe_name} for sid={sid}")
        return None


def _source_path_for_project(project: dict) -> str:
    return os.path.join(UPLOAD_FOLDER, project["upload_sid"], project["original_file"])


def _find_repairable_source_path(project: dict, expected_path: str) -> str | None:
    """Find the exact original upload when a packaged app moved upload roots."""
    original_name = secure_filename(os.path.basename(project.get("original_file") or ""))
    if not original_name:
        return None

    candidates = []
    project_file = project.get("file_path")
    if project_file:
        candidates.append(project_file)
        project_file_dir = os.path.dirname(project_file)
        candidates.append(os.path.join(project_file_dir, original_name))
        candidates.append(os.path.join(os.path.dirname(project_file_dir), original_name))

    for base in (os.getcwd(), _persistent_dir(), os.path.dirname(os.path.abspath(__file__))):
        candidates.append(os.path.join(base, "uploads", project.get("upload_sid") or "", original_name))

    seen = {os.path.abspath(expected_path)}
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.basename(candidate) != original_name or not os.path.exists(candidate):
            continue
        try:
            probe = load_dataset(candidate)
        except Exception:
            logging.warning("resume_source_repair unreadable candidate=%s", candidate)
            continue
        expected_rows = int(project.get("row_count") or 0)
        expected_cols = int(project.get("col_count") or 0)
        if expected_rows and len(probe) != expected_rows:
            logging.warning("resume_source_repair row mismatch candidate=%s", candidate)
            continue
        if expected_cols and len(probe.columns) != expected_cols:
            logging.warning("resume_source_repair column mismatch candidate=%s", candidate)
            continue
        return candidate
    return None


def _ensure_project_source_available(project: dict) -> tuple[str | None, str | None]:
    """Return the canonical immutable source path, repairing moved exe uploads."""
    expected_path = _source_path_for_project(project)
    if os.path.exists(expected_path):
        return expected_path, None

    candidate = _find_repairable_source_path(project, expected_path)
    if not candidate:
        return None, None

    os.makedirs(os.path.dirname(expected_path), exist_ok=True)
    try:
        shutil.copy2(candidate, expected_path)
        logging.info(
            "resume_source_repaired project_id=%s from=%s to=%s",
            project.get("id"), candidate, expected_path,
        )
        return expected_path, candidate
    except Exception:
        logging.exception("resume_source_repair failed project_id=%s", project.get("id"))
        return None, candidate


def replay_project_actions(project_id: int, target_status: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Rebuild project state by replaying active actions in original order.

    The immutable upload is the starting point. ``target_status`` lets Undo or
    Reapply preview a status change for one history row without relying on a
    reverse patch that could overwrite later overlapping actions.
    """
    """Rebuild current state from immutable source plus ordered active actions.

    target_status can override one action status for the in-flight Undo/Reapply
    decision before the DB row is committed.
    """
    project = db.get_project(project_id)
    if not project:
        raise FileNotFoundError("Project not found.")
    source_path, _repaired_from = _ensure_project_source_available(project)
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError("Immutable source file is missing.")

    df = load_dataset(source_path)
    history = db.get_cleaning_history(project_id)
    for entry in history:
        history_id = int(entry["id"])
        status = (target_status or {}).get(history_id, entry.get("status", "applied"))
        if status != "applied":
            continue
        manifest = db.get_manifest(history_id)
        config = entry.get("action_config") or {}
        if not manifest and isinstance(config, dict):
            if config.get("manifest_file"):
                manifest = load_manifest_from_file(project["upload_sid"], config["manifest_file"])
            elif config.get("manifest"):
                manifest = config.get("manifest")
        if not manifest:
            raise RuntimeError(f"Action {history_id} cannot be replayed because its manifest is missing.")
        result = snapshot_diff.apply_manifest(df, manifest, "forward")
        if result.get("conflicts"):
            raise RuntimeError(f"Action {history_id} replay conflict: {result['conflicts'][0]}")
        df = result["df"]
    return df, history


def _aggregate_active_history_metrics(history: list[dict], target_status: dict | None = None) -> dict:
    """Summarize active action history for current-state UI metrics."""
    metrics = {
        "operations": [],
        "filled_cells": 0,
        "duplicates_removed": 0,
        "outliers_removed": 0,
        "empty_columns_removed": 0,
    }
    for entry in history or []:
        history_id = int(entry.get("id") or 0)
        status = (target_status or {}).get(history_id, entry.get("status", "applied"))
        if status != "applied":
            continue
        config = entry.get("action_config") or {}
        operations = config.get("operations") if isinstance(config, dict) else []
        if not isinstance(operations, list):
            operations = []
        metrics["operations"].extend(operations)
        for op in operations:
            action = op.get("action")
            if action in ("fill_numeric", "fill_categorical"):
                metrics["filled_cells"] += int(op.get("filled_cells") or 0)
            elif action == "manual_missing_cell_updates":
                metrics["filled_cells"] += int(op.get("count") or 0)
            elif action == "drop_duplicates":
                metrics["duplicates_removed"] += int(op.get("removed_rows") or 0)
            elif action in ("manual_outlier_removal", "outlier_removal"):
                metrics["outliers_removed"] += int(op.get("rows_removed") or op.get("removed_rows") or 0)
            elif action == "drop_empty_columns":
                metrics["empty_columns_removed"] += int(op.get("count") or 0)
    return metrics


def _patch_cell_update_old_values(operations, previous_state) -> None:
    """manual_cell_updates / manual_missing_cell_updates operations record
    their "old" value from clean_dataframe()'s `working` frame — which
    always starts from the RAW uploaded file, since clean_dataframe has no
    notion of "the current snapshot" (it's a pure function reused in several
    places that don't all have one). That's the correct "before" value the
    FIRST time a given cell is ever edited, but wrong on every subsequent
    edit of the same cell: the Change Log would show the ORIGINAL raw value
    as "before" instead of the value the cell actually held just before this
    edit. This patches "old" in place using previous_state (the last
    committed snapshot), which is the true "before" state for this action.
    Only touches entries where the row/column exist in previous_state;
    anything else is left as clean_dataframe reported it.
    """
    if previous_state is None or not operations:
        return
    for op in operations:
        if op.get("action") not in ("manual_cell_updates", "manual_missing_cell_updates"):
            continue
        for update in (op.get("updates") or []):
            col = update.get("column")
            if col not in previous_state.columns:
                continue
            label = update.get("row_index")
            if label not in previous_state.index:
                try:
                    int_label = int(label)
                except (TypeError, ValueError):
                    continue
                if int_label not in previous_state.index:
                    continue
                label = int_label
            try:
                val = previous_state.at[label, col]
            except Exception:
                continue
            try:
                is_na = bool(pd.isna(val))
            except (TypeError, ValueError):
                is_na = False
            update["old"] = "" if is_na else str(val)


def parse_clean_options(form, default_enabled=True, default_missing_strategy="fill"):
    def _bool(name, default=None):
        if default is None:
            default = default_enabled
        vals = form.getlist(name)
        return str(vals[-1]).lower() in ("1", "true", "on", "yes") if vals else default

    opts = {
        "remove_duplicates":     _bool("remove_duplicates"),
        "drop_empty_columns":    _bool("drop_empty_columns"),
        "drop_constant_columns": _bool("drop_constant_columns"),
        "trim_whitespace":       _bool("trim_whitespace"),
        "rename_columns":        _bool("rename_columns", False),
        "convert_dtypes":        _bool("convert_dtypes"),
        "handle_outliers":       None,
        "encode_categoricals":   "label" if _bool("encode_categoricals", False) else False,
        "missing_strategy":      form.get("missing_strategy", default_missing_strategy),
        "fill_numeric":          form.get("fill_numeric", "mean"),
        "fill_categorical":      form.get("fill_categorical", "mode"),
    }

    selected_outlier_rows_str = form.get("selected_outlier_rows")
    if selected_outlier_rows_str:
        try:
            opts["selected_outlier_rows"] = json.loads(selected_outlier_rows_str)
        except Exception as e:
            logging.warning(f"Could not parse selected outlier rows: {e}")
            opts["selected_outlier_rows"] = []
    else:
        opts["selected_outlier_rows"] = []

    # When outlier_scope_apply is NOT set, the outlier rows must still be
    # removed (cumulative state) but the backend must NOT emit a new
    # manual_outlier_removal operation — doing so would push a spurious
    # outlier action into whichever section (Missing Values, Data Types etc.)
    # the user was actually applying.
    opts["outlier_scope_apply"] = form.get("outlier_scope_apply") == "1"

    selected_outlier_records_str = form.get("selected_outlier_records")
    if selected_outlier_records_str:
        try:
            records = json.loads(selected_outlier_records_str)
            opts["selected_outlier_records"] = records if isinstance(records, list) else []
        except Exception as e:
            logging.warning(f"Could not parse selected outlier records: {e}")
            opts["selected_outlier_records"] = []
    else:
        opts["selected_outlier_records"] = []

    dtype_map, column_renames, cell_updates = {}, {}, []
    missing_cell_updates, drop_selected_columns = [], []
    # Per-column fill strategies: fill_numeric_col[colname] = mean|median|zero|constant
    #                             fill_categorical_col[colname] = mode|constant|drop
    numeric_col_strategies    = {}  # {col: strategy}
    categorical_col_strategies = {}  # {col: strategy}

    for key, val in form.items():
        if key.startswith("dtype_map[") and key.endswith("]"):
            col = key[len("dtype_map["):-1]
            if val and val != "auto":
                dtype_map[col] = val
        elif key.startswith("column_rename[") and key.endswith("]") and val:
            orig = key[len("column_rename["):-1]
            new = str(val).strip()
            if new and new != orig:
                column_renames[orig] = new
        elif key.startswith("cell_update["):
            m = re.match(r"^cell_update\[([^\]]+)\]\[(.+)\]$", key)
            if m:
                cell_updates.append({"row_index": m.group(1),
                                     "column": m.group(2),
                                     "value": val})
        elif key.startswith("missing_cell[") and val:
            m = re.match(r"^missing_cell\[([^\]]+)\]\[(.+)\]$", key)
            if m:
                missing_cell_updates.append({"row_index": m.group(1),
                                              "column": m.group(2),
                                              "value": val})
        elif key.startswith("drop_column[") and key.endswith("]") and val:
            # Column selectors carry exact DataFrame identifiers. Normalizing
            # here changes names such as "Customer Name" and "Unnamed: 15"
            # into identifiers that do not exist in the authoritative frame.
            drop_selected_columns.append(key[len("drop_column["):-1])
        elif key.startswith("fill_numeric_col[") and key.endswith("]") and val:
            # Per-column numeric fill strategy override
            col = key[len("fill_numeric_col["):-1]
            if col and val not in ("", "default"):
                numeric_col_strategies[col] = val
        elif key.startswith("fill_categorical_col[") and key.endswith("]") and val:
            # Per-column categorical fill strategy override
            col = key[len("fill_categorical_col["):-1]
            if col and val not in ("", "default"):
                categorical_col_strategies[col] = val

    if dtype_map:                    opts["dtype_map"]                   = dtype_map
    if column_renames:               opts["column_renames"]              = column_renames
    if cell_updates:                 opts["cell_updates"]                = cell_updates
    if missing_cell_updates:         opts["missing_cell_updates"]        = missing_cell_updates
    if drop_selected_columns:        opts["drop_selected_columns"]       = drop_selected_columns
    if numeric_col_strategies:       opts["numeric_col_strategies"]      = numeric_col_strategies
    if categorical_col_strategies:   opts["categorical_col_strategies"]  = categorical_col_strategies

    # CRITICAL: if per-column fill strategies exist but missing_strategy is still
    # "ignore", upgrade it to "fill" so the filling actually runs.
    has_per_col = bool(numeric_col_strategies or categorical_col_strategies)
    if has_per_col and opts["missing_strategy"] == "ignore":
        opts["missing_strategy"] = "fill"
    # Also ensure global strategies default to "skip" when only per-column is set,
    # so columns NOT in the per-col map are left alone.
    if has_per_col:
        if not form.get("fill_numeric"):
            opts["fill_numeric"] = "skip"
        if not form.get("fill_categorical"):
            opts["fill_categorical"] = "skip"

    return opts


def detect_encoding_with_chardet(filepath: str, n_bytes: int = 200000):
    if not HAS_CHARDET:
        return None
    try:
        with open(filepath, "rb") as f:
            raw = f.read(n_bytes)
        result = chardet.detect(raw)
        enc = result.get("encoding")
        logging.info(f"chardet result for {os.path.basename(filepath)}: {result}")
        return enc
    except Exception as e:
        logging.debug(f"chardet failed: {e}")
        return None


def load_dataset(filepath: str):
    # Robustly load CSV/TSV/Excel/JSON with encoding fallbacks
    ext = os.path.splitext(filepath)[1].lower()
    tried = []

    if ext in (".xls", ".xlsx"):
        try:
            return pd.read_excel(filepath)
        except Exception as e:
            raise ValueError("Failed to read Excel file: " + str(e)) from e

    if ext == ".json":
        try:
            return pd.read_json(filepath)
        except Exception:
            pass

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    chardet_enc = detect_encoding_with_chardet(filepath)
    if chardet_enc and chardet_enc not in encodings:
        encodings.insert(0, chardet_enc)

    sep = "\t" if filepath.lower().endswith(".tsv") else None

    for enc in encodings:
        if enc in tried:
            continue
        tried.append(enc)
        try:
            if ext == ".json":
                return pd.read_json(filepath, encoding=enc)
            elif sep is None:
                return pd.read_csv(filepath, encoding=enc, sep=None, engine="python")
            else:
                return pd.read_csv(filepath, encoding=enc, sep=sep)
        except Exception as e:
            logging.debug(f"read failed enc={enc}: {e}")

    # Final fallback — decode with replacement chars
    try:
        if ext == ".json":
            with open(filepath, "rb") as f:
                raw = f.read()
            return pd.read_json(StringIO(raw.decode("utf-8", errors="replace")))
        else:
            return pd.read_csv(filepath, encoding="utf-8", engine="python",
                               sep=sep, on_bad_lines="warn")
    except Exception:
        pass

    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        sio  = StringIO(text)
        return pd.read_json(sio) if ext == ".json" else pd.read_csv(sio, sep=sep)
    except Exception as e:
        raise ValueError(
            "Unable to parse the uploaded file as CSV/TSV/Excel/JSON. "
            "Consider re-saving as UTF-8 or uploading an Excel (.xlsx) file."
        ) from e


# --- Routes ------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html", uploaded_filename=None,
                               analysis=None, cleaned=None,
                               custom_code_enabled=ENABLE_CUSTOM_CODE)
    except Exception:
        return """<!doctype html><html><body>
            <h3>inDataOut — Data Cleaning</h3>
            <form method="post" action="/upload" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Upload</button>
            </form></body></html>"""


# --- Upload ------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    """Create a new project upload while leaving previous project folders intact."""
    if "file" not in request.files:
        flash("No file part in request.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Allowed: CSV, Excel, JSON, TSV, TXT.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)

    # Clear previous session — wipe all keys including project_id
    previous_sid = flask_session.get("upload_sid")
    if previous_sid:
        for key in ("upload_sid", "upload_filename", "cleaned_sid",
                    "cleaned_filename", "report_filename", "project_id"):
            flask_session.pop(key, None)

    session_id  = secrets.token_hex(16)
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, filename)
    flask_session["upload_sid"]      = session_id
    flask_session["upload_filename"] = filename
    flask_session["dataset_version"] = 0

    try:
        file.save(file_path)
    except Exception as e:
        logging.exception("save upload failed")
        message, user_action = _operation_failure_explanation(e, "save")
        flash(f"{message} {user_action}")
        return redirect(url_for("index"))

    try:
        df = load_dataset(file_path)
        with _cache_lock:
            _df_cache[session_id] = (os.path.getmtime(file_path), df)  # seed cache
    except Exception as e:
        logging.exception("load_dataset failed")
        flash(str(e) if isinstance(e, ValueError) else "The uploaded dataset could not be read. Check its file type, encoding, and structure.")
        return redirect(url_for("index"))

    # Create DB project row; non-fatal if MySQL is down
    try:
        project_id = db.create_project(
            name=filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title(),
            original_file=filename,
            upload_sid=session_id,
            file_path=file_path,
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
        )
        flask_session["project_id"] = project_id
        logging.info(f"project id={project_id} created")
    except Exception as e:
        logging.warning(f"DB create_project failed (non-fatal): {e}")
        flask_session["project_id"] = None

    try:
        analysis = get_cached_analysis(session_id, df, fp=_df_fingerprint(df))
        ORIGINAL_PREVIEW_ROWS = 100
        original_preview = df_to_rows_safe(df, ORIGINAL_PREVIEW_ROWS)
        analysis["preview"]                = original_preview
        analysis["original_preview"]       = original_preview
        analysis["original_total_rows"]    = len(df)
        analysis["original_preview_count"] = min(len(df), ORIGINAL_PREVIEW_ROWS)
        analysis["row_index_map"] = {
            str(i): int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            for i, idx in enumerate(df.index)
        }
        analysis["editor_columns"]   = list(df.columns)
        analysis["editor_row_count"] = len(df)
        analysis["project_id"]       = flask_session.get("project_id")  # for nav link
    except Exception as e:
        logging.exception("analyze_dataframe failed")
        message, user_action = _operation_failure_explanation(e, "process")
        flash(f"{message} {user_action}")
        return redirect(url_for("index"))

    return render_template("index.html", uploaded_filename=filename,
                           analysis=analysis, cleaned=None,
                           custom_code_enabled=ENABLE_CUSTOM_CODE)


# --- Clean -------------------------------------------------------------------

@app.route("/clean", methods=["POST"])
def clean():
    """Finalize the authoritative current dataset into a versioned CSV/report."""
    filename = request.form.get("filename")
    if not filename:
        flash("No file specified.")
        return redirect(url_for("index"))

    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        flash("Session mismatch — please re-upload your file.")
        return redirect(url_for("index"))

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        flash("File not found on server.")
        return redirect(url_for("index"))

    try:
        df = get_cached_df(sid, file_path)
    except Exception as e:
        logging.exception("get_cached_df failed")
        message, user_action = _operation_failure_explanation(e, "process")
        flash(f"The dataset could not be prepared for export: {message} {user_action}")
        return redirect(url_for("index"))

    dq_before = get_cached_dq(sid, file_path, df)
    try:
        _require_session_version(request.form)
    except (db.StaleDatasetVersionError, ValueError):
        flash("The dataset changed. Refresh the project before finalizing.")
        return redirect(url_for("index"))

    # The applied-history ledger (window._ido.appliedControlEntries on the
    # client) is the authoritative record of what's actually been applied —
    # sent as a single JSON field so it can't be corrupted by disabled/
    # re-enabled DOM elements or fields that simply aren't rendered right
    # now (e.g. outlier checkboxes on a resumed project). This mirrors the
    # replay logic /resume_replay already uses. Fall back to raw
    # request.form only for old cached pages that predate this field.
    committed_df = load_current_snapshot(sid)
    if committed_df is not None:
        cleaned_df = committed_df
        report = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "original_shape": list(df.shape),
            "cleaned_shape": list(cleaned_df.shape),
            "rows_processed": int(df.shape[0]),
            "rows_after": int(cleaned_df.shape[0]),
            "duplicates_removed": max(0, int(df.shape[0] - cleaned_df.shape[0])),
            "outliers_removed": 0,
            "filled_cells": 0,
            "columns_renamed": 0,
            "cleaned_columns": list(cleaned_df.columns),
            "empty_columns_removed": max(0, int(df.shape[1] - cleaned_df.shape[1])),
            "empty_columns": [],
            "operations": [],
            "finalized_from": "authoritative_current_snapshot",
        }
    else:
        # With no committed actions, the immutable upload is the current
        # state. Final Clean never executes browser controls or ledger data.
        cleaned_df = df.copy()
        report = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "original_shape": list(df.shape),
            "cleaned_shape": list(df.shape),
            "rows_processed": int(df.shape[0]),
            "rows_after": int(df.shape[0]),
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "filled_cells": 0,
            "columns_renamed": 0,
            "cleaned_columns": list(df.columns),
            "empty_columns_removed": 0,
            "empty_columns": [],
            "operations": [],
            "finalized_from": "immutable_source_no_committed_actions",
        }

    quality_breakdown = data_quality_breakdown(cleaned_df, df)
    dq_after = float(quality_breakdown["score"])
    report["data_quality_before"]         = dq_before
    report["data_quality_after"]          = dq_after
    report["dq_improvement"]              = round(dq_after - dq_before, 2)
    report["quality_breakdown"]           = quality_breakdown
    report["cleaning_completeness_score"] = cleaning_completeness_score(report)
    project_id = flask_session.get("project_id")
    history = db.get_cleaning_history(project_id) if project_id else []
    applied_history = [entry for entry in history if entry.get("status") == "applied"]
    report["applied_action_count"] = len(applied_history)
    report["undone_action_count"] = len(history) - len(applied_history)
    report["total_rows_removed"] = max(0, int(df.shape[0] - cleaned_df.shape[0]))
    report["columns_added"] = max(0, int(cleaned_df.shape[1] - df.shape[1]))
    report["columns_removed"] = max(0, int(df.shape[1] - cleaned_df.shape[1]))
    report["remaining_missing_cells"] = int(cleaned_df.isna().sum().sum())
    report["remaining_missing_rows"] = int(cleaned_df.isna().any(axis=1).sum())
    report["operations"] = [
        operation for entry in applied_history
        for operation in ((entry.get("action_config") or {}).get("operations") or [])
    ]
    report["filled_cells"] = sum(int(op.get("filled_cells") or (op.get("count") if op.get("action") == "manual_missing_cell_updates" else 0) or 0) for op in report["operations"])
    report["cells_manually_edited"] = sum(int(op.get("count") or 0) for op in report["operations"] if op.get("action") == "manual_cell_updates")
    report["duplicates_removed"] = sum(int(op.get("removed_rows") or 0) for op in report["operations"] if op.get("action") == "drop_duplicates")
    report["outlier_rows_removed"] = sum(int(op.get("rows_removed") or op.get("removed_rows") or 0) for op in report["operations"] if op.get("action") in ("manual_outlier_removal", "outlier_removal"))
    report["outlier_cells_clipped"] = sum(int(op.get("count") or 0) for op in report["operations"] if op.get("action") == "outlier_clip")
    report["label_encoding_mappings"] = [item for op in report["operations"] if op.get("action") == "label_encode" for item in (op.get("columns_encoded") or [])]
    report = make_json_safe(report)

    next_version = 1
    if project_id:
        try:
            versions = db.get_dataset_versions(project_id)
            next_version = (max([int(v.get("version_no", 0)) for v in versions] or [0]) + 1)
        except Exception:
            next_version = 1
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cleaned_basename = f"cleaned_{secure_filename(filename.rsplit('.', 1)[0])}_v{next_version:04d}_{timestamp}.csv"
    user_cleaned_dir = os.path.join(UPLOAD_FOLDER, sid, "cleaned")
    os.makedirs(user_cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(user_cleaned_dir, cleaned_basename)
    flask_session["cleaned_sid"]      = sid
    flask_session["cleaned_filename"] = cleaned_basename

    try:
        fd, tmp_path = tempfile.mkstemp(prefix="cleaned_", suffix=".csv", dir=user_cleaned_dir)
        os.close(fd)
        cleaned_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        with open(tmp_path, "r", encoding="utf-8-sig", newline="") as written:
            written_text = written.read()
        expected_text = cleaned_df.to_csv(index=False, lineterminator="\n")
        if written_text.replace("\r\n", "\n") != expected_text.replace("\r\n", "\n"):
            raise ValueError("Output validation mismatch after writing CSV content.")
        check_df = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
        if list(check_df.columns) != list(cleaned_df.columns) or int(check_df.shape[0]) != int(cleaned_df.shape[0]):
            raise ValueError("Output validation mismatch after writing CSV.")
        os.replace(tmp_path, cleaned_path)
    except Exception as e:
        logging.exception("to_csv failed")
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        message, user_action = _operation_failure_explanation(e, "save")
        flash(f"{message} {user_action}")
        return redirect(url_for("index"))

    report_basename  = f"report_{secure_filename(filename.rsplit('.', 1)[0])}_v{next_version:04d}_{timestamp}.json"
    user_reports_dir = os.path.join(UPLOAD_FOLDER, sid, "reports")
    os.makedirs(user_reports_dir, exist_ok=True)
    report_path = os.path.join(user_reports_dir, report_basename)
    flask_session["report_filename"] = report_basename

    try:
        report["preview"] = make_json_safe(check_df.head(100).to_dict(orient="records"))
        fd, tmp_report_path = tempfile.mkstemp(prefix="report_", suffix=".json", dir=user_reports_dir)
        os.close(fd)
        save_report(report, tmp_report_path)
        os.replace(tmp_report_path, report_path)
    except Exception as e:
        logging.exception("save_report failed")
        for path in (cleaned_path, locals().get("tmp_report_path")):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                logging.exception("Could not remove incomplete final output")
        message, user_action = _operation_failure_explanation(e, "save")
        flash(f"{message} {user_action}")
        return redirect(url_for("index"))

    # Log cleaning run to DB — non-fatal (file already saved before this block)
    if project_id:
        try:
            db.save_dataset_version(
                project_id=project_id,
                file_path=cleaned_path,
                label="After clean",
                row_count=int(cleaned_df.shape[0]),
                col_count=int(cleaned_df.shape[1]),
            )
            db.update_project_status(project_id, "cleaned", file_path=cleaned_path)
            logging.info(f"DB logged: project {project_id}")
        except Exception as e:
            logging.exception("Final version persistence failed")
            for path in (cleaned_path, report_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    logging.exception("Could not remove uncommitted final output")
            flash("Final output could not be persisted. No download version was committed.")
            return redirect(url_for("index"))

    return render_template("index.html", uploaded_filename=filename,
                           analysis=None,
                           cleaned={
                               "cleaned_filename": cleaned_basename,
                               "report_filename":  report_basename,
                               "report":           report,
                               "project_id":       project_id,  # for "View history" link
                           },
                           custom_code_enabled=ENABLE_CUSTOM_CODE)


# --- Preview clean (live) ----------------------------------------------------

def build_clean_payload(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, report: dict,
                        sid: str = None, file_path: str = None,
                        include_editor_data: bool = True,
                        include_row_index_map: bool = True,
                        manifest: dict | None = None,
                        include_quality: bool = True,
                        include_changed_rows: bool = True) -> dict:
    """Shared response-shape builder for /preview_clean and /apply_manifest.

    Given a raw/cleaned DataFrame pair and a report dict that already has its
    shape-specific keys filled in (original_shape, cleaned_shape, rows_after,
    filled_cells, duplicates_removed, outliers_removed,
    empty_columns_removed), computes everything else the frontend expects —
    dq scores, preview rows, changed-rows, missing-value breakdown, the
    editor grid payload, and the summary block — so both routes return
    identical JSON shapes and the same frontend render functions work for
    both an actual Apply and an Undo/Reapply-via-manifest.
    """
    cleaned_fp = _df_fingerprint(cleaned_df) if sid else None
    if include_quality:
        dq_before = get_cached_dq(sid, file_path, raw_df) if (sid and file_path) else data_quality_score(raw_df)
    else:
        dq_before = get_cached_dq_if_available(sid, file_path) if (sid and file_path) else None
    if include_quality:
        quality_breakdown = data_quality_breakdown(cleaned_df, raw_df)
        dq_after = float(quality_breakdown["score"])
        quality_delta = round(dq_after - dq_before, 2)
    else:
        quality_breakdown = None
        dq_after = dq_before
        quality_delta = 0.0 if dq_before is not None else None
    report["data_quality_before"]         = dq_before
    report["data_quality_after"]          = dq_after
    report["dq_improvement"]              = quality_delta
    if quality_breakdown is not None:
        report["quality_breakdown"]       = quality_breakdown
    else:
        report.pop("quality_breakdown", None)
        report["quality_pending"]         = True
        report["quality_note"]            = "Full quality score recalculates outside this fast response."
    report["cleaning_completeness_score"] = cleaning_completeness_score(report)

    preview_rows = min(len(cleaned_df), 100)
    dt_cols = [c for c in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[c])]
    preview_df = cleaned_df.head(preview_rows).copy() if dt_cols else cleaned_df.head(preview_rows)
    if dt_cols:
        for c in dt_cols:
            preview_df[c] = preview_df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    report["preview"] = make_json_safe(preview_df.to_dict(orient="records"))
    report["preview_row_numbers"] = [
        int(i) + 1 if isinstance(i, (int, np.integer)) else str(i)
        for i in preview_df.index
    ]
    report["preview_display_limit"] = preview_rows
    report["cleaned_columns"]       = list(cleaned_df.columns)

    row_index_map = None
    if include_row_index_map:
        row_index_map = {
            str(i): int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            for i, idx in enumerate(cleaned_df.index)
        }

    missing_by_column_remaining = []
    for col, count in cleaned_df.isna().sum().items():
        count = int(count)
        if count > 0:
            missing_by_column_remaining.append({
                "column": col,
                "missing": count,
                "percent": round((count / len(cleaned_df)) * 100, 2) if len(cleaned_df) else 0,
            })

    if include_changed_rows:
        evidence_rows = _changed_rows_from_evidence(raw_df, cleaned_df, report.get("operations"), manifest)
        if evidence_rows is None:
            changed_recs, changed_rnums, removed_rnums, added_rnums = compute_changed_rows(raw_df, cleaned_df)
            total_changed_rows = len(changed_recs)
        else:
            changed_recs, changed_rnums, removed_rnums, added_rnums, total_changed_rows = evidence_rows
    else:
        changed_recs, changed_rnums, removed_rnums, added_rnums, total_changed_rows = [], [], [], [], 0
    report["changed_rows_preview"] = changed_recs
    report["changed_row_numbers"]  = changed_rnums
    report["removed_row_numbers"]  = removed_rnums
    report["added_row_numbers"]    = added_rnums
    report["total_changed_rows"]   = total_changed_rows
    report["has_changes"]          = total_changed_rows > 0 or len(removed_rnums) > 0

    editor_data = df_to_rows_safe(cleaned_df) if include_editor_data else None

    cleaned_analysis = get_cached_analysis(sid, cleaned_df, fp=cleaned_fp) if sid else analyze_dataframe(cleaned_df)
    cleaned_analysis.pop("preview", None)  # duplicates report["preview"] above

    payload = {
        "success":        True,
        "preview":        report.get("preview", []),
        "changed_preview": changed_recs,
        "editor_columns": list(cleaned_df.columns),
        "report":         report,
        "analysis":       cleaned_analysis,
        "summary": {
            "rows_before":           int(report["original_shape"][0]),
            "rows_after":            int(report["rows_after"]),
            "cols_after":            int(report["cleaned_shape"][1]),
            "filled_cells":          int(report["filled_cells"]),
            "duplicates_removed":    int(report["duplicates_removed"]),
            "outliers_removed":      int(report["outliers_removed"]),
            "empty_columns_removed": int(report["empty_columns_removed"]),
            "quality_before":        dq_before,
            "quality_after":         dq_after,
            "quality_delta":         quality_delta,
            "quality_pending":       not include_quality,
            "cleaning_completeness": report["cleaning_completeness_score"],
            "missing_remaining":     int(cleaned_df.isna().sum().sum()),
            "missing_rows_remaining":int(cleaned_df.isna().any(axis=1).sum()),
            "missing_by_column_remaining": missing_by_column_remaining,
            "changed_rows":  total_changed_rows,
            "removed_rows":  len(removed_rnums),
        }
    }
    if include_editor_data:
        payload["editor_data"] = editor_data
    if include_row_index_map:
        payload["row_index_map"] = row_index_map
    return payload


def _operation_cells_changed(operations: list[dict] | None, manifest: dict | None) -> int:
    if manifest:
        return int(
            len(manifest.get("changed_cells", []) or [])
            + len(manifest.get("dropped_rows", []) or [])
            + len(manifest.get("added_rows", []) or [])
            + len(manifest.get("dropped_columns", []) or [])
            + len(manifest.get("added_columns", []) or [])
            + len(manifest.get("dtype_changes", []) or [])
        )
    total = 0
    for op in operations or []:
        total += int(op.get("count") or op.get("filled_cells") or op.get("removed_rows") or op.get("rows_removed") or 0)
    return total


def _full_row_evidence(before_df: pd.DataFrame, after_df: pd.DataFrame,
                       manifest: dict | None, limit: int = 500) -> list[dict]:
    """Capture complete immediate Before/After rows for every affected row."""
    manifest = manifest or {}

    def frame_uses_uploaded_row_numbers(frame):
        if not len(frame.index):
            return False
        if not pd.api.types.is_integer_dtype(frame.index.dtype):
            return False
        return int(frame.index.min()) == 0 and int(frame.index.max()) >= len(frame.index) - 1

    before_uploaded_rows = frame_uses_uploaded_row_numbers(before_df)
    after_uploaded_rows = frame_uses_uploaded_row_numbers(after_df)

    def display_row_number(row_id):
        try:
            numeric_id = int(row_id)
        except (TypeError, ValueError):
            return str(row_id)
        for frame, uses_uploaded_rows in ((before_df, before_uploaded_rows), (after_df, after_uploaded_rows)):
            if numeric_id in frame.index:
                return numeric_id + 1 if uses_uploaded_rows else str(row_id)
        return numeric_id + 1

    affected = []
    for item in manifest.get("changed_cells", []) or []:
        affected.append(item.get("row"))
    for key in ("dropped_rows", "added_rows"):
        for item in manifest.get(key, []) or []:
            affected.append(item.get("row"))
    all_row_ids = list(dict.fromkeys(row_id for row_id in affected if row_id is not None))
    row_ids = all_row_ids[:limit]
    columns = list(dict.fromkeys([*before_df.columns.tolist(), *after_df.columns.tolist()]))
    changed_by_row: dict[str, list[str]] = {}
    for item in manifest.get("changed_cells", []) or []:
        changed_by_row.setdefault(str(item.get("row")), []).append(str(item.get("col")))

    evidence = []
    for row_id in row_ids:
        def resolve(frame):
            if row_id in frame.index:
                return row_id
            try:
                numeric_id = int(row_id)
            except (TypeError, ValueError):
                return None
            return numeric_id if numeric_id in frame.index else None

        before_id = resolve(before_df)
        after_id = resolve(after_df)
        before = before_df.loc[before_id].to_dict() if before_id is not None else None
        after = after_df.loc[after_id].to_dict() if after_id is not None else None
        changed_columns = changed_by_row.get(str(row_id), [])
        if before is None or after is None:
            changed_columns = columns
        record_id = None
        for row in (before, after):
            if isinstance(row, dict) and "record_id" in row:
                record_id = row.get("record_id")
                break
        evidence.append({
            "row_id": row_id,
            "original_row_number": display_row_number(row_id),
            "record_id": record_id,
            "columns": columns,
            "before": before,
            "after": after,
            "changed_columns": list(dict.fromkeys(changed_columns)),
            "status": "removed" if after is None else ("added" if before is None else "changed"),
        })
    evidence = make_json_safe(evidence)
    if len(all_row_ids) > limit:
        evidence.append({
            "truncated": True,
            "shown_rows": limit,
            "total_rows": len(all_row_ids),
        })
    return evidence


def _manifest_action_preview(before_df: pd.DataFrame, after_df: pd.DataFrame,
                             manifest: dict | None, limit: int = 25) -> dict:
    """Build bounded, persisted Before/After evidence from an exact manifest."""
    manifest = manifest or {}

    def frame_uses_uploaded_row_numbers(frame):
        if not len(frame.index):
            return False
        if not pd.api.types.is_integer_dtype(frame.index.dtype):
            return False
        return int(frame.index.min()) == 0 and int(frame.index.max()) >= len(frame.index) - 1

    before_uploaded_rows = frame_uses_uploaded_row_numbers(before_df)
    after_uploaded_rows = frame_uses_uploaded_row_numbers(after_df)

    def display_row_number(row_id):
        try:
            numeric_id = int(row_id)
        except (TypeError, ValueError):
            return str(row_id)
        for frame, uses_uploaded_rows in ((before_df, before_uploaded_rows), (after_df, after_uploaded_rows)):
            if numeric_id in frame.index:
                return numeric_id + 1 if uses_uploaded_rows else str(row_id)
        return numeric_id + 1

    changed_all = manifest.get("changed_cells") or []
    removed_all = manifest.get("dropped_rows") or []
    added_all = manifest.get("added_rows") or []
    removed_cols_all = manifest.get("dropped_columns") or []
    added_cols_all = manifest.get("added_columns") or []
    dtype_all = manifest.get("dtype_changes") or []

    changed_cells = [{
        "row_id": item.get("row"),
        "row_number": display_row_number(item.get("row")),
        "column": item.get("col"),
        "before": item.get("before"),
        "after": item.get("after"),
    } for item in changed_all[:limit]]

    def preview_rows(items):
        return [{
            "row_id": item.get("row"),
            "row_number": display_row_number(item.get("row")),
            "data": item.get("data") or {},
        } for item in items[:limit]]

    def preview_columns(items):
        columns = []
        for item in items[:limit]:
            values = item.get("data") or {}
            columns.append({
                "column": item.get("col"),
                "sample": [
                    {"row_id": row_id, "value": value}
                    for row_id, value in list(values.items())[:10]
                ],
            })
        return columns

    dtype_changes = [{
        "column": item.get("col"),
        "before": item.get("before"),
        "after": item.get("after"),
    } for item in dtype_all[:limit]]

    total = sum(map(len, (changed_all, removed_all, added_all, removed_cols_all, added_cols_all, dtype_all)))
    displayed = sum(map(len, (
        changed_cells,
        removed_all[:limit],
        added_all[:limit],
        removed_cols_all[:limit],
        added_cols_all[:limit],
        dtype_changes,
    )))
    return make_json_safe({
        "before_shape": [int(before_df.shape[0]), int(before_df.shape[1])],
        "after_shape": [int(after_df.shape[0]), int(after_df.shape[1])],
        "before_columns": list(before_df.columns),
        "after_columns": list(after_df.columns),
        "changed_cells": changed_cells,
        "removed_rows": preview_rows(removed_all),
        "added_rows": preview_rows(added_all),
        "removed_columns": preview_columns(removed_cols_all),
        "added_columns": preview_columns(added_cols_all),
        "dtype_changes": dtype_changes,
        "total_changed_cells": len(changed_all),
        "total_removed_rows": len(removed_all),
        "total_added_rows": len(added_all),
        "total_removed_columns": len(removed_cols_all),
        "total_added_columns": len(added_cols_all),
        "total_dtype_changes": len(dtype_all),
        "displayed_change_count": displayed,
        "total_change_count": total,
        "is_sample": displayed < total,
        "limit": limit,
        "row_evidence": list(manifest.get("row_evidence") or [])[:limit],
        "displayed_row_count": min(len(manifest.get("row_evidence") or []), limit),
        "total_affected_rows": len(manifest.get("row_evidence") or []),
    })


def _enrich_operations_with_manifest(operations: list[dict] | None, manifest: dict | None) -> list[dict]:
    enriched = make_json_safe(operations or [])
    if not manifest:
        return enriched
    dropped_rows = manifest.get("dropped_rows") or []
    added_rows = manifest.get("added_rows") or []
    dropped_columns = manifest.get("dropped_columns") or []
    for op in enriched:
        action = op.get("action")
        if action in ("drop_duplicates", "drop_missing_rows", "manual_outlier_removal") and dropped_rows:
            op["dropped_rows_preview"] = dropped_rows[:10]
            op["dropped_rows_total"] = len(dropped_rows)
        if action == "custom_pandas_code":
            if dropped_rows:
                op["dropped_rows_preview"] = dropped_rows[:10]
                op["dropped_rows_total"] = len(dropped_rows)
            if added_rows:
                op["added_rows_preview"] = added_rows[:10]
                op["added_rows_total"] = len(added_rows)
        if action in ("drop_empty_columns", "drop_constant_columns", "drop_selected_columns") and dropped_columns:
            op["dropped_columns_preview"] = dropped_columns[:10]
            op["dropped_columns_total"] = len(dropped_columns)
    return enriched


def _persist_apply_history_and_snapshot(
    sid: str,
    cleaned_df: pd.DataFrame,
    action_name: str,
    action_config: dict,
    rows_before: int,
    rows_after: int,
    cells_changed: int,
    dq_after: float | None,
    manifest: dict | None,
    expected_version: int,
) -> int | None:
    """Persist one committed action and promote its current snapshot.

    The caller has already built ``cleaned_df`` and its manifest. When a
    project row exists, DB history/manifest insertion and snapshot promotion
    share one commit callback; if either fails, the temp snapshot is discarded
    and no successful Change Log card should be returned.
    """
    project_id = flask_session.get("project_id")
    if not project_id:
        save_current_snapshot(sid, cleaned_df)
        flask_session["dataset_version"] = expected_version + 1
        return None

    tmp_path, final_path = prepare_current_snapshot_temp(sid, cleaned_df)
    try:
        return db.log_cleaning_action_atomic(
            project_id=project_id,
            action_name=action_name,
            action_config=action_config,
            rows_before=rows_before,
            rows_after=rows_after,
            cells_changed=cells_changed,
            dq_after=dq_after,
            manifest=manifest,
            expected_version=expected_version,
            before_commit=lambda _hid: promote_current_snapshot_temp(tmp_path, final_path),
        )
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logging.exception("Failed to remove abandoned snapshot temp file")
        raise


@app.route("/preview_clean", methods=["POST"])
def preview_clean():
    """Preview or commit cleaning options.

    Plain preview requests do not mutate authoritative state. Requests marked
    ``persist_apply`` must return server-authored history/manifest evidence.
    """
    filename = request.form.get("filename")
    if not filename:
        return api_error("VALIDATION_ERROR", "No file specified.", status=400, user_action="Upload or reopen a dataset.")

    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return api_error("SESSION_DATASET_MISMATCH", "Session expired. Please re-open your project or re-upload the file.", status=403, user_action="Open Projects and resume this dataset, or upload it again.", retryable=True)
        return jsonify({
            "error": "Session expired — please re-open your project or re-upload the file.",
            "session_expired": True
        }), 403

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return api_error("SOURCE_FILE_MISSING", "The original dataset could not be found.", status=404, user_action="Restore the source file or upload the dataset again.", retryable=False)
        return jsonify({
            "error": "File not found on server. The original upload may have been moved.",
            "session_expired": True
        }), 404

    try:
        form_snapshot  = _FormProxy(request.form.to_dict(flat=False))
        scoped_preview = form_snapshot.get("preview_scope") in ("feature", "section", "manual")
        df   = get_cached_df(sid, file_path)
        opts = parse_clean_options(
            form_snapshot,
            default_enabled=not scoped_preview,
            default_missing_strategy="ignore" if scoped_preview else "fill",
        )
        persist_apply = form_snapshot.get("persist_apply") in ("1", "true", "True")
        expected_version = (
            _require_session_version(form_snapshot)
            if persist_apply else _submitted_dataset_version(form_snapshot)
        )
        operation_token = form_snapshot.get("operation_token")
        try:
            client_applied_at_ms = int(form_snapshot.get("client_applied_at_ms") or 0) or None
        except (TypeError, ValueError):
            client_applied_at_ms = None
        if persist_apply and operation_token:
            cache_key = (sid, operation_token)
            with _operation_cache_lock:
                cached_payload = _operation_cache.get(cache_key)
            if cached_payload is not None:
                return jsonify(make_json_safe(cached_payload))
            project_id = flask_session.get("project_id")
            existing_actions = db.get_actions_by_operation_token(int(project_id), operation_token) if project_id else []
            if existing_actions:
                current_df = load_current_snapshot(sid)
                if current_df is None:
                    current_df = df.copy(deep=True)
                current_report = {
                    "original_shape": list(df.shape), "cleaned_shape": list(current_df.shape),
                    "rows_after": int(current_df.shape[0]), "operations": [],
                    "filled_cells": 0, "duplicates_removed": 0, "outliers_removed": 0,
                    "empty_columns_removed": 0,
                }
                duplicate_payload = build_clean_payload(df, current_df, current_report, sid=sid, file_path=file_path)
                duplicate_payload.update({
                    "dataset_version": int(flask_session.get("dataset_version", 0) or 0),
                    "operation_token": operation_token,
                    "idempotent_replay": True,
                    "history_entries": make_json_safe(existing_actions),
                    "history_entry": make_json_safe(existing_actions[0]),
                })
                return jsonify(make_json_safe(duplicate_payload))
        # Preview and Apply both start from the same committed snapshot.
        # Preview remains non-mutating; Apply additionally persists its diff.
        previous_state = load_current_snapshot(sid)
        action_base_df = previous_state if previous_state is not None else df
        requested_drops = opts.get("drop_selected_columns") or []
        unresolved_drops = [column for column in requested_drops if column not in action_base_df.columns]
        if unresolved_drops:
            return api_error(
                "SELECTED_COLUMNS_NOT_FOUND",
                "The selected columns no longer exist in the current dataset.",
                status=409,
                user_action="Refresh the dataset and select the columns again.",
                retryable=True,
                details={"unresolved_columns": unresolved_drops},
            )
        selected_missing = {
            **(opts.get("numeric_col_strategies") or {}),
            **(opts.get("categorical_col_strategies") or {}),
        }
        unresolved_missing = [column for column in selected_missing if column not in action_base_df.columns]
        if unresolved_missing:
            return api_error(
                "MISSING_COLUMNS_NOT_FOUND",
                "One or more missing-value columns no longer exist.",
                status=409,
                user_action="Refresh Missing Values and choose the strategies again.",
                retryable=True,
                details={"unresolved_columns": unresolved_missing},
            )
        missing_before = {
            column: int(action_base_df[column].isna().sum())
            for column, strategy in selected_missing.items() if strategy != "skip"
        }
        cleaned_df, report = clean_dataframe(action_base_df, options=opts)
        cleaned_df, report = apply_custom_code_blocks(cleaned_df, report, form_snapshot)
        unresolved_repairs = []
        for column, before_count in missing_before.items():
            if before_count <= 0:
                continue
            strategy = selected_missing[column]
            after_count = int(cleaned_df[column].isna().sum()) if column in cleaned_df.columns else 0
            rows_removed = len(cleaned_df) < len(action_base_df) and strategy == "drop"
            if after_count >= before_count and not rows_removed:
                unresolved_repairs.append({
                    "column": column,
                    "strategy": strategy,
                    "missing_count": before_count,
                })
        if unresolved_repairs:
            report.setdefault("notices", []).extend([
                f"{item['column']}: {item['strategy']} could not fill the current missing cells; choose another strategy or type a value."
                for item in unresolved_repairs
            ])

        # --- Persistent, independent undo/redo -------------------------------
        # Only real commits (persist_apply=1, sent by the frontend exactly
        # when persistApply is true — i.e. an actual Apply, not a live preview
        # while the user is still adjusting a toggle) touch the on-disk
        # snapshot. Plain previews behave exactly as before: no snapshot
        # read/write, no manifest, same performance.
        manifest = None
        if persist_apply:
            if previous_state is None:
                # No snapshot yet. Two cases:
                #  (a) this is genuinely the first-ever committed action for
                #      this session — `df` (the raw file) IS the correct
                #      "before" state, so a manifest can be computed normally.
                #  (b) this session has prior actions logged before this
                #      feature existed (legacy) — `df` is NOT the correct
                #      "before" state for THIS action (it would make the
                #      manifest look like it includes every prior action's
                #      changes too). We can't safely build a manifest for
                #      *this* action in that case, but we CAN start the
                #      snapshot chain now so every action from here on has one.
                has_prior_history = False
                try:
                    project_id = flask_session.get("project_id")
                    if project_id:
                        has_prior_history = len(db.get_cleaning_history(project_id)) > 0
                except Exception:
                    pass
                if not has_prior_history:
                    try:
                        manifest = _operation_manifest(action_base_df, cleaned_df, report.get("operations"))
                        if manifest is None:
                            manifest = snapshot_diff.compute_full_diff(df, cleaned_df)
                    except Exception:
                        logging.exception("compute_full_diff failed on first action — continuing without a manifest")
            else:
                try:
                    manifest = _operation_manifest(previous_state, cleaned_df, report.get("operations"))
                    if manifest is None:
                        manifest = snapshot_diff.compute_full_diff(previous_state, cleaned_df)
                except Exception:
                    logging.exception("compute_full_diff failed — continuing without a manifest")
                try:
                    _patch_cell_update_old_values(report.get("operations"), previous_state)
                except Exception:
                    logging.exception("Failed to patch cell-update 'old' values — Change Log display may show the raw value instead")
        report["manifest"] = manifest if (manifest and not snapshot_diff.manifest_is_empty(manifest)) else None
        if report["manifest"] is not None:
            report["manifest"]["row_evidence"] = _full_row_evidence(
                action_base_df, cleaned_df, report["manifest"]
            )

        can_patch_editor = persist_apply and report.get("manifest") is not None
        needs_row_index_map = bool(
            persist_apply
            and (not can_patch_editor or (report.get("manifest") or {}).get("added_rows"))
        )
        # On large datasets the explainable quality score is the slowest part
        # of Apply and does not affect the cleaned data, undo/reapply evidence,
        # or exports. Keep it exact for normal/small data, and defer it for
        # large data so Apply buttons stay responsive.
        exact_quality_now = persist_apply and int(cleaned_df.shape[0] * cleaned_df.shape[1]) <= 500_000
        payload = build_clean_payload(
            df, cleaned_df, report, sid=sid, file_path=file_path,
            include_editor_data=(persist_apply and not can_patch_editor),
            include_row_index_map=needs_row_index_map,
            manifest=report.get("manifest"),
            include_quality=exact_quality_now,
        )
        if persist_apply:
            operations = _enrich_operations_with_manifest(report.get("operations", []), report.get("manifest"))
            report["operations"] = operations
            payload["report"]["operations"] = operations
            operation_notices = list(dict.fromkeys(
                list(report.get("notices") or []) + _operation_notices(operations)
            ))
            if operation_notices:
                payload["notices"] = operation_notices
            action_name = str(form_snapshot.get("action_name") or "Apply cleaning action")[:128]
            action_scope = str(form_snapshot.get("action_scope") or form_snapshot.get("preview_scope") or "")
            manifest_for_history = report.get("manifest")
            action_config = {
                "label": action_name,
                "operations": operations,
                "activePanelId": action_scope,
                "server_authoritative": True,
                "preview": _manifest_action_preview(action_base_df, cleaned_df, manifest_for_history),
            }
            if manifest_for_history:
                payload["patch"] = build_manifest_patch(manifest_for_history, "forward")
            if client_applied_at_ms:
                action_config["applied_at_ms"] = client_applied_at_ms
            rows_before = int(action_base_df.shape[0])
            rows_after = int(cleaned_df.shape[0])
            cells_changed = _operation_cells_changed(operations, manifest_for_history)
            if not manifest_for_history and _df_fingerprint(action_base_df) == _df_fingerprint(cleaned_df):
                payload["no_op"] = True
                payload["no_op_message"] = _no_op_message(operations, action_name)
                payload["result_message"] = payload["no_op_message"]
                payload["dataset_version"] = expected_version
                if operation_token:
                    payload["operation_token"] = operation_token
                    with _operation_cache_lock:
                        if len(_operation_cache) > 200:
                            _operation_cache.pop(next(iter(_operation_cache)))
                        _operation_cache[(sid, operation_token)] = make_json_safe(payload)
                return jsonify(make_json_safe(payload))
            try:
                project_id = flask_session.get("project_id")
                history_actions = _sequential_history_actions(
                    action_base_df, cleaned_df, operations, manifest_for_history,
                    action_name, action_scope, payload.get("summary", {}).get("quality_after"),
                    client_applied_at_ms,
                )
                for history_action in history_actions:
                    history_action["action_config"]["operation_token"] = operation_token
                inserted = []
                if project_id and len(history_actions) > 1:
                    tmp_path, final_path = prepare_current_snapshot_temp(sid, cleaned_df)
                    try:
                        inserted = db.log_cleaning_actions_atomic(
                            int(project_id), history_actions, expected_version,
                            before_commit=lambda _ids: promote_current_snapshot_temp(tmp_path, final_path),
                        )
                    except Exception:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        raise
                    committed_version = int(inserted[-1]["version_after"])
                else:
                    first = history_actions[0]
                    hid = _persist_apply_history_and_snapshot(
                        sid=sid, cleaned_df=cleaned_df,
                        action_name=first["action_name"], action_config=first["action_config"],
                        rows_before=first["rows_before"], rows_after=first["rows_after"],
                        cells_changed=first["cells_changed"], dq_after=first["dq_after"],
                        manifest=first["manifest"], expected_version=expected_version,
                    )
                    committed_version = expected_version + 1
                    if hid:
                        inserted = [{"id": hid, "version_before": expected_version, "version_after": committed_version}]
                flask_session["dataset_version"] = committed_version
            except db.StaleDatasetVersionError:
                raise
            except Exception as exc:
                logging.exception("Committed apply failed while saving history/snapshot")
                message, user_action = _operation_failure_explanation(exc, "save")
                return api_error(
                    "ACTION_PERSISTENCE_FAILED",
                    message,
                    status=500,
                    user_action=user_action,
                    retryable=True,
                )
            history_entries = []
            for stored, item in zip(inserted, history_actions):
                history_entries.append({
                    "id": stored["id"], "action_name": item["action_name"],
                    "action_config": item["action_config"],
                    "rows_before": item["rows_before"], "rows_after": item["rows_after"],
                    "cells_changed": item["cells_changed"], "status": "applied",
                    "has_manifest": True,
                    "version_before": stored["version_before"],
                    "version_after": stored["version_after"],
                })
            payload["history_entries"] = make_json_safe(history_entries)
            if history_entries:
                payload["history_entry"] = make_json_safe(history_entries[0])
            payload["result_message"] = _apply_result_message(history_actions, operation_notices)
            payload["dataset_version"] = committed_version
            if payload.get("report", {}).get("manifest") is not None:
                payload["report"]["has_manifest"] = True
                payload["report"].pop("manifest", None)
            try:
                if project_id:
                    active_metrics = _aggregate_active_history_metrics(
                        db.get_cleaning_history(int(project_id))
                    )
                    for key in ("filled_cells", "duplicates_removed", "outliers_removed", "empty_columns_removed"):
                        payload["report"][key] = active_metrics[key]
                        payload["summary"][key] = active_metrics[key]
                    payload["report"]["operations"] = active_metrics["operations"]
                else:
                    payload["summary"]["duplicates_removed"] = int(payload["report"].get("duplicates_removed") or 0)
            except Exception:
                logging.exception("Failed to refresh cumulative metrics after apply")
        else:
            payload["dataset_version"] = int(flask_session.get("dataset_version", 0) or 0)
        if persist_apply and operation_token:
            payload["operation_token"] = operation_token
            with _operation_cache_lock:
                if len(_operation_cache) > 200:
                    _operation_cache.pop(next(iter(_operation_cache)))
                _operation_cache[(sid, operation_token)] = make_json_safe(payload)
        return jsonify(make_json_safe(payload))
    except db.StaleDatasetVersionError as e:
        return api_error(
            "STALE_DATASET_VERSION", "The dataset changed after these controls were opened.",
            status=409, user_action="Refresh the current project state and try again.", retryable=True,
            details={"submitted_version": e.expected, "current_version": e.current},
        )
    except ValueError as e:
        return api_error(
            "OPERATION_VALIDATION_FAILED", str(e), status=422,
            user_action="Review the selected rows, columns, and methods, then try again.", retryable=False,
        )
    except Exception as e:
        logging.exception("preview_clean failed")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error("INTERNAL_ERROR", message, status=500, user_action=user_action, retryable=True)


# --- Preview custom code -----------------------------------------------------

@app.route("/preview_custom_code", methods=["POST"])
def preview_custom_code():
    if not ENABLE_CUSTOM_CODE:
        return api_error(
            "CUSTOM_CODE_DISABLED", "Custom code is disabled outside trusted local mode.",
            status=403, user_action="Run the desktop/local version to use Custom Pandas Code.", retryable=False,
        )
    # Always enabled — local tool, no sandbox restriction
    filename = request.form.get("filename")
    if not filename:
        return api_error("VALIDATION_ERROR", "No dataset was specified.", status=400, user_action="Upload or resume a project.")

    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return api_error("SESSION_DATASET_MISMATCH", "The active dataset does not match this request.", status=403, user_action="Resume the project and retry.", retryable=True)

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return api_error("SOURCE_FILE_MISSING", "The original dataset could not be found.", status=404, user_action="Restore the source file or upload it again.")

    code = request.form.get("code", "").strip()
    if not code:
        return api_error("CUSTOM_CODE_EMPTY", "The code cell is empty.", status=400, user_action="Enter Pandas code before running the cell.")

    try:
        form_snapshot = _FormProxy(request.form.to_dict(flat=False))
        source_df = get_cached_df(sid, file_path)
        cleaned_df = load_current_snapshot(sid)
        if cleaned_df is None:
            cleaned_df = source_df.copy(deep=True)
        from custom_code_runner import run_custom_code
        result    = run_custom_code(cleaned_df, code)
        operation = result["operation"]
        return jsonify({
            "success":      True,
            "operation":    make_json_safe(operation),
            "output_blocks":make_json_safe(operation.get("output_blocks", [])),
            "rows_before":  operation["rows_before"],
            "rows_after":   operation["rows_after"],
            "changed_cells":operation["changed_cells"],
        })
    except Exception as e:
        logging.exception("preview_custom_code failed")
        if isinstance(e, (SyntaxError, NameError, KeyError, TypeError, ValueError)):
            detail = str(e).strip()[:300] or type(e).__name__
            message = f"Custom code could not run: {detail}"
            user_action = "Correct the indicated code or DataFrame reference, then run the cell again."
        else:
            message, user_action = _operation_failure_explanation(e, "process")
        return api_error(
            "CUSTOM_CODE_PREVIEW_FAILED", message, status=422,
            user_action=user_action, retryable=False,
        )


# --- Editor data (full dataset for canvas editor) ----------------------------

@app.route("/get_editor_data", methods=["POST"])
def get_editor_data():
    """Return the authoritative current snapshot for the Dataset Editor."""
    filename = request.form.get("filename")
    if not filename:
        return api_error("VALIDATION_ERROR", "No file specified.", status=400, user_action="Upload or resume a project.")

    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return api_error("SESSION_DATASET_MISMATCH", "The active dataset does not match this request.", status=403, user_action="Resume the project and retry.", retryable=True)

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return api_error("SOURCE_FILE_MISSING", "The original dataset could not be found.", status=404, user_action="Restore the source file or upload it again.")

    try:
        dataset_version = int(flask_session.get("dataset_version", 0) or 0)
        project_id = flask_session.get("project_id")
        if project_id:
            project = db.get_project(int(project_id))
            if project:
                dataset_version = int(project.get("state_version") or dataset_version)
                flask_session["dataset_version"] = dataset_version
            cdf = load_current_snapshot(sid)
            if cdf is None:
                cdf, _history = replay_project_actions(int(project_id))
                save_current_snapshot(sid, cdf)
        else:
            cdf = load_current_snapshot(sid)
            if cdf is None:
                cdf = get_cached_df(sid, file_path).copy(deep=True)
        return jsonify(make_json_safe({
            "success":       True,
            "columns":       list(cdf.columns),
            "data":          df_to_rows_safe(cdf),
            "row_count":     len(cdf),
            "row_index_map": {
                str(i): int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
                for i, idx in enumerate(cdf.index)
            },
            "dataset_version": dataset_version,
        }))
    except Exception as e:
        logging.exception("get_editor_data failed")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error("EDITOR_LOAD_FAILED", f"The Dataset Editor could not load: {message}", status=500, user_action=user_action, retryable=True)


@app.route("/api/editor-page", methods=["GET"])
def editor_page():
    filename = request.args.get("filename") or flask_session.get("upload_filename")
    sid = flask_session.get("upload_sid")
    if not filename or not sid:
        return api_error("SESSION_DATASET_MISSING", "No active dataset is available.", status=404, user_action="Upload or resume a project.", retryable=False)
    try:
        page = max(1, int(request.args.get("page", "1")))
        per_page = min(500, max(1, int(request.args.get("per_page", "100"))))
    except ValueError:
        return api_error("VALIDATION_ERROR", "Invalid page parameters.", status=400, user_action="Use numeric page and per_page values.")
    project_id = flask_session.get("project_id")
    if project_id:
        try:
            df = load_current_snapshot(sid)
            if df is None:
                df, _history = replay_project_actions(int(project_id))
                save_current_snapshot(sid, df)
            project = db.get_project(int(project_id))
            if project:
                flask_session["dataset_version"] = int(project.get("state_version") or flask_session.get("dataset_version", 0) or 0)
        except Exception:
            logging.exception("editor_page replay failed")
            return api_error("ORDERED_REPLAY_FAILED", "The current project state could not be rebuilt for the editor.", status=409, user_action="Open Projects and resume again.", retryable=True)
    else:
        df = load_current_snapshot(sid)
    if df is None:
        file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
        if not os.path.exists(file_path):
            return api_error("SOURCE_FILE_MISSING", "The original dataset could not be found.", status=404, user_action="Restore the source file or upload the dataset again.")
        try:
            df = get_cached_df(sid, file_path)
        except Exception:
            logging.exception("editor_page load failed")
            return api_error("FILE_PARSE_FAILED", "Could not read the dataset page.", status=500, user_action="Retry or re-upload the dataset.", retryable=True)
    total_rows = int(len(df))
    start = (page - 1) * per_page
    end = min(start + per_page, total_rows)
    if start >= total_rows and total_rows:
        return api_error("PAGE_OUT_OF_RANGE", "Requested page is outside the dataset.", status=416, user_action="Choose a lower page number.")
    page_df = df.iloc[start:end]
    return jsonify(make_json_safe({
        "success": True,
        "page": page,
        "per_page": per_page,
        "rows_total": total_rows,
        "rows_displayed": int(len(page_df)),
        "columns": list(df.columns),
        "row_numbers": [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in page_df.index],
        "rows": df_to_rows_safe(page_df),
        "is_sample": end < total_rows,
        "dataset_version": int(flask_session.get("dataset_version", 0) or 0),
        "row_ids": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in page_df.index],
    }))


# --- Download cleaned file ---------------------------------------------------

@app.route("/download/cleaned/<path:filename>")
def download_cleaned(filename):
    safe     = secure_filename(filename)
    sid      = flask_session.get("cleaned_sid")
    expected = flask_session.get("cleaned_filename")
    if not sid or expected != safe:
        flash("Unauthorized or session expired. Please re-upload.")
        return redirect(url_for("index"))
    folder = os.path.join(UPLOAD_FOLDER, sid, "cleaned")
    full   = os.path.join(folder, safe)
    if not os.path.exists(full):
        flash("File not found.")
        return redirect(url_for("index"))
    response = send_from_directory(folder, safe, as_attachment=True)
    return response


# --- Download cleaning report ------------------------------------------------

@app.route("/download/report/<path:filename>")
def download_report(filename):
    safe     = secure_filename(filename)
    sid      = flask_session.get("cleaned_sid")
    expected = flask_session.get("report_filename")
    if not sid or expected != safe:
        flash("Unauthorized or session expired. Please re-upload.")
        return redirect(url_for("index"))
    folder = os.path.join(UPLOAD_FOLDER, sid, "reports")
    full   = os.path.join(folder, safe)
    if not os.path.exists(full):
        flash("File not found.")
        return redirect(url_for("index"))
    response = send_from_directory(folder, safe, as_attachment=True)
    return response


def _project_owned_path(project: dict, path: str | None) -> bool:
    if not project or not path:
        return False
    try:
        root = os.path.abspath(os.path.join(UPLOAD_FOLDER, str(project.get("upload_sid") or "")))
        target = os.path.abspath(path)
        return os.path.commonpath([root, target]) == root
    except Exception:
        return False


@app.route("/projects/<int:project_id>/download/cleaned")
def download_project_cleaned(project_id: int):
    try:
        project = db.get_project(project_id)
    except Exception:
        logging.exception("download_project_cleaned lookup failed")
        flash("Could not look up that project.")
        return redirect(url_for("projects"))
    if not project:
        flash("Project not found.")
        return redirect(url_for("projects"))
    path = project.get("file_path")
    if not _project_owned_path(project, path):
        flash("That cleaned file is not associated with this project.")
        return redirect(url_for("project_detail", project_id=project_id))
    if not os.path.exists(path):
        flash("That cleaned file is missing. The project needs repair before it can be downloaded.")
        return redirect(url_for("project_detail", project_id=project_id))
    return send_from_directory(os.path.dirname(path), os.path.basename(path), as_attachment=True)


@app.route("/projects/<int:project_id>/versions/<int:version_id>/download")
def download_project_version(project_id: int, version_id: int):
    try:
        project = db.get_project(project_id)
        version = db.get_dataset_version(project_id, version_id)
    except Exception:
        logging.exception("download_project_version lookup failed")
        flash("Could not look up that dataset version.")
        return redirect(url_for("project_detail", project_id=project_id))
    if not project:
        flash("Project not found.")
        return redirect(url_for("projects"))
    if not version:
        flash("Dataset version not found for this project.")
        return redirect(url_for("project_detail", project_id=project_id))
    path = version.get("file_path")
    if not _project_owned_path(project, path):
        flash("That dataset version is not associated with this project.")
        return redirect(url_for("project_detail", project_id=project_id))
    if not path or not os.path.exists(path):
        flash("That dataset version file is missing. The project needs repair before it can be downloaded.")
        return redirect(url_for("project_detail", project_id=project_id))
    return send_from_directory(os.path.dirname(path), os.path.basename(path), as_attachment=True)


# --- Preview outliers --------------------------------------------------------

@app.route("/preview_outliers", methods=["POST"])
def preview_outliers():
    filename = request.form.get("filename")
    if not filename:
        return api_error("VALIDATION_ERROR", "No dataset was specified.", status=400, user_action="Upload or resume a project.")

    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return api_error("SESSION_DATASET_MISMATCH", "The active dataset does not match this request.", status=403, user_action="Resume the project and retry.", retryable=True)

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return api_error("SOURCE_FILE_MISSING", "The original dataset could not be found.", status=404, user_action="Restore the source file or upload it again.")

    try:
        expected_version = _require_session_version(request.form)
        df = load_current_snapshot(sid)
        if df is None:
            df = get_cached_df(sid, file_path).copy(deep=True)
        multiplier        = float(request.form.get("multiplier", 1.5))
        selected_cols_str = request.form.get("outlier_columns", "")
        selected_cols = json.loads(selected_cols_str) if selected_cols_str else None
        if selected_cols is not None and not isinstance(selected_cols, list):
            raise ValueError("Outlier columns must be a list.")
        missing_columns = [column for column in (selected_cols or []) if column not in df.columns]
        if missing_columns:
            return api_error(
                "OUTLIER_COLUMNS_STALE", "One or more selected outlier columns no longer exist.",
                status=409, user_action="Refresh outlier candidates and select again.", retryable=True,
                details={"unresolved_columns": missing_columns},
            )
        outliers = detect_outliers(df, multiplier=multiplier, columns=selected_cols)
        return jsonify({"success": True, "outliers": outliers, "dataset_version": expected_version})
    except db.StaleDatasetVersionError as e:
        return api_error(
            "STALE_DATASET_VERSION", "The dataset changed while outliers were being selected.",
            status=409, user_action="Refresh outlier candidates and select again.", retryable=True,
            details={"submitted_version": e.expected, "current_version": e.current},
        )
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return api_error(
            "OUTLIER_SELECTION_INVALID", str(e), status=422,
            user_action="Review the multiplier and selected numeric columns, then try again.", retryable=False,
        )
    except Exception as e:
        logging.exception("preview_outliers failed")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error(
            "OUTLIER_PREVIEW_FAILED", f"Outlier detection could not be completed: {message}",
            status=500, user_action=user_action, retryable=True,
        )


# --- Project workspace routes ------------------------------------------------

@app.route("/projects")
def projects():
    # Dashboard: all projects, newest first
    try:
        project_list = db.list_projects(limit=100)
    except Exception as e:
        logging.exception("list_projects failed")
        flash(f"Could not load projects: {e}")
        project_list = []
    return render_template("projects.html", projects=project_list)


def _prune_missing_manifests(sid: str, history: list) -> list:
    """Resolves each history row's real Undo/Reapply availability.

    has_manifest is already accurate for anything logged after manifests
    moved into their own DB table (see get_cleaning_history's LEFT JOIN) —
    nothing to do for those. This only needs to handle LEGACY rows that
    predate that change, whose manifest (if it exists at all) still only
    lives as a file referenced by action_config.manifest_file. For those,
    check the file actually exists on disk and set has_manifest accordingly
    — logging exactly what's going on so a future occurrence of this is
    immediately visible in the server log instead of requiring more
    back-and-forth to diagnose.
    """
    for entry in history:
        if entry.get("has_manifest"):
            continue  # already resolved via the DB join — nothing to check
        config = entry.get("action_config")
        if not isinstance(config, dict):
            continue
        manifest_file = config.get("manifest_file")
        if manifest_file:
            manifest_path = os.path.join(_manifests_dir(sid), manifest_file)
            exists = os.path.exists(manifest_path)
            logging.info(
                f"[_prune_missing_manifests] history_id={entry.get('id')} "
                f"legacy manifest_file={manifest_file!r} path={manifest_path} exists={exists}"
            )
            if exists:
                entry["has_manifest"] = True
            else:
                logging.warning(
                    f"Legacy manifest file missing on disk for history_id={entry.get('id')} "
                    f"(sid={sid}) — Undo/Reapply for this action will show as read-only "
                    f"instead of a misleading enabled-but-broken button."
                )
        elif config.get("manifest"):
            entry["has_manifest"] = True
            logging.info(f"[_prune_missing_manifests] history_id={entry.get('id')} has legacy inline manifest — OK")
        else:
            logging.info(f"[_prune_missing_manifests] history_id={entry.get('id')} has no manifest at all")
    return history


@app.route("/projects/<int:project_id>")
def project_detail(project_id: int):
    try:
        project  = db.get_project(project_id)
        history  = db.get_cleaning_history(project_id)
        versions = db.get_dataset_versions(project_id)
    except Exception as e:
        logging.exception("get_project failed")
        flash(f"Could not load project: {e}")
        return redirect(url_for("projects"))

    if not project:
        flash("Project not found.")
        return redirect(url_for("projects"))

    sid = project["upload_sid"]
    history = _prune_missing_manifests(sid, history)

    # Restore ALL session keys so every route (preview, download, upload) works after resume
    # Verify the original file still exists on disk — warn if missing
    original_path, _repaired_from = _ensure_project_source_available(project)
    file_missing = not (original_path and os.path.exists(original_path))

    return render_template("project_detail.html", project=project,
                           history=history, versions=versions,
                           grouped_history=build_grouped_cleaning_history(history),
                           file_missing=file_missing,
                           resume_url=url_for("index"))


# --- Log action (called by JS after each Apply button) -----------------------

@app.route("/log_action", methods=["POST"])
def log_action():
    return api_error(
        "LEGACY_LOG_ACTION_DISABLED",
        "Client-created cleaning history is disabled.",
        status=410,
        user_action="Retry the operation so it can be committed through the authoritative server action path.",
        retryable=False,
    )
    """
    # JS posts here after every section Apply to record the action in DB.
    # This is the bridge that makes the history log real-time.
    project_id = flask_session.get("project_id")
    if not project_id:
        return jsonify({"ok": False, "reason": "no project in session"}), 200  # soft fail

    data = request.get_json(silent=True) or {}
    action_name   = str(data.get("action_name", "section_apply"))[:128]
    action_config = make_json_safe(data.get("action_config", {}))
    rows_before   = int(data.get("rows_before", 0))
    rows_after    = int(data.get("rows_after", 0))
    cells_changed = int(data.get("cells_changed", 0))
    dq_after_raw  = data.get("dq_after")
    try:
        dq_after = float(dq_after_raw) if dq_after_raw is not None else None
    except (TypeError, ValueError):
        dq_after = None

    # Manifests now live in their own MySQL table (see db.save_manifest),
    # keyed by history_id — not the filesystem. Two separate real-world
    # failures already happened with file-based storage in this app (the
    # raw upload living in a volatile temp folder, and manifest files going
    # missing after moving/rebuilding the app) — a DB table doesn't have
    # "the file went missing" as a failure mode at all. This is why the
    # manifest is saved AFTER db.log_cleaning_action below, once we have the
    # real history_id to key it by, rather than up front.
    manifest = action_config.pop("manifest", None) if isinstance(action_config, dict) else None
    if isinstance(action_config, dict):
        action_config.pop("manifest_file", None)  # deprecated — DB is authoritative now

    try:
        hid = db.log_cleaning_action(
            project_id=project_id,
            action_name=action_name,
            action_config=action_config,
            rows_before=rows_before,
            rows_after=rows_after,
            cells_changed=cells_changed,
            dq_after=dq_after,
        )
        if manifest:
            try:
                db.save_manifest(hid, manifest)
            except Exception:
                logging.exception(
                    f"Failed to save manifest to DB for history_id={hid} "
                    "— Undo/Reapply for this action won't be available"
                )
        # Mark project as active (in-progress) when section work is happening
        db.update_project_status(project_id, "active")
        logging.info(f"Logged action '{action_name}' for project {project_id}")
        return jsonify({"ok": True, "id": hid})
    except db.StaleDatasetVersionError as e:
        return api_error(
            "STALE_DATASET_VERSION",
            "The dataset changed after these controls were opened.",
            status=409,
            user_action="Refresh the current project state and try again.",
            retryable=True,
            details={"submitted_version": e.expected, "current_version": e.current},
        )
    except ValueError as e:
        return api_error(
            "OPERATION_VALIDATION_FAILED", str(e), status=422,
            user_action="Review the selected rows, columns, and methods, then try again.", retryable=False,
        )
    except Exception as e:
        logging.warning(f"log_action DB write failed: {e}")
        return jsonify({"ok": False, "reason": str(e)}), 200  # non-fatal
    """


@app.route("/log_action/<int:history_id>/status", methods=["POST"])
def update_action_status(history_id: int):
    # Called when the user clicks Undo/Reapply on a Change Log card — persists
    # the real status so build_grouped_cleaning_history() and Resume reflect it
    # instead of always showing "Applied".
    project_id = flask_session.get("project_id")
    if not project_id:
        return jsonify({"ok": False, "reason": "no project in session"}), 200

    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in ("applied", "undone"):
        return jsonify({"ok": False, "reason": "invalid status"}), 400

    try:
        db.set_cleaning_action_status(history_id, status)
        return jsonify({"ok": True})
    except Exception as e:
        logging.exception("update_action_status DB write failed")
        message, user_action = _operation_failure_explanation(e, "save")
        return api_error(
            "ACTION_STATUS_SAVE_FAILED",
            f"The Change Log status could not be saved: {message}",
            status=500,
            user_action=user_action,
            retryable=True,
        )


@app.route("/apply_manifest", methods=["POST"])
def apply_manifest_route():
    """Undo or reapply one action by ordered replay, then persist status/snapshot."""
    # Called by Undo/Reapply on a Change Log card. Unlike a full-snapshot
    # restore, this replays only the ONE action's own recorded cell/row/
    # column changes against whatever the dataset currently is — so an
    # unrelated later action's changes are left completely alone, and this
    # works identically whether the action was created in this session or
    # resumed from the DB after a restart.
    project_id = flask_session.get("project_id")
    sid        = flask_session.get("upload_sid")
    if not project_id or not sid:
        return api_error("PROJECT_SESSION_MISSING", "No active project is available.", status=409, user_action="Open Projects and resume a project.", retryable=True)

    data = request.get_json(silent=True) or {}
    try:
        expected_version = _require_session_version(data)
    except db.StaleDatasetVersionError as e:
        return api_error(
            "STALE_DATASET_VERSION", "The dataset changed before this action ran.",
            status=409, user_action="Refresh the project and try again.", retryable=True,
            details={"submitted_version": e.expected, "current_version": e.current},
        )
    history_id = data.get("history_id")
    direction  = data.get("direction")  # 'reverse' (Undo) or 'forward' (Reapply)
    if not history_id or direction not in ("forward", "reverse"):
        return api_error("UNDO_REAPPLY_INVALID", "The Undo/Reapply request is invalid.", status=400, user_action="Refresh the Change Log.", retryable=False)

    try:
        row = db.get_cleaning_action(int(history_id), project_id)
    except Exception as e:
        logging.warning(f"apply_manifest: history lookup failed: {e}")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error("ACTION_LOOKUP_FAILED", f"The Change Log action could not be loaded: {message}", status=500, user_action=user_action, retryable=True)

    if not row:
        return api_error("ACTION_NOT_FOUND", "The action does not belong to this project.", status=404, user_action="Refresh the Change Log.", retryable=False)

    history_id_int = int(history_id)
    new_status = "undone" if direction == "reverse" else "applied"
    if row.get("status") == new_status:
        return api_error("ACTION_STATUS_UNCHANGED", "The action already has that status.", status=409, user_action="Refresh the project state.", retryable=True)

    target_status = {history_id_int: new_status}
    try:
        new_df, _history = replay_project_actions(project_id, target_status=target_status)
    except Exception as e:
        logging.exception("apply_manifest: ordered replay failed")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error("ORDERED_REPLAY_FAILED", f"The project could not be rebuilt safely: {message}", status=409, user_action=user_action, retryable=False)

    filename  = flask_session.get("upload_filename")
    file_path = os.path.join(UPLOAD_FOLDER, sid, filename) if filename else None
    try:
        raw_df = get_cached_df(sid, file_path) if (file_path and os.path.exists(file_path)) else new_df
    except Exception:
        raw_df = new_df

    tmp_path = None
    try:
        tmp_path, final_path = prepare_current_snapshot_temp(sid, new_df)
        new_version = db.set_cleaning_action_status_atomic(
            history_id_int,
            int(project_id),
            new_status,
            expected_version,
            before_commit=lambda: promote_current_snapshot_temp(tmp_path, final_path),
        )
        flask_session["dataset_version"] = new_version
    except db.StaleDatasetVersionError as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return api_error(
            "STALE_DATASET_VERSION", "The dataset changed before this action could be saved.",
            status=409, user_action="Refresh the project and try again.", retryable=True,
            details={"submitted_version": e.expected, "current_version": e.current},
        )
    except Exception as e:
        logging.exception("apply_manifest: failed to persist replayed state/status")
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logging.exception("apply_manifest: failed to remove temp snapshot")
        message, user_action = _operation_failure_explanation(e, "save")
        return api_error("UNDO_REAPPLY_PERSISTENCE_FAILED", f"Undo/Reapply was not saved: {message}", status=500, user_action=user_action, retryable=True)

    active_metrics = _aggregate_active_history_metrics(_history, target_status=target_status)
    report = {
        "original_shape":  list(raw_df.shape),
        "cleaned_shape":   list(new_df.shape),
        "rows_after":      int(new_df.shape[0]),
        "cleaned_columns": list(new_df.columns),
        "operations":      active_metrics["operations"],
        "filled_cells":    active_metrics["filled_cells"],
        "duplicates_removed": active_metrics["duplicates_removed"],
        "outliers_removed": active_metrics["outliers_removed"],
        "empty_columns_removed": active_metrics["empty_columns_removed"],
    }
    payload = build_clean_payload(raw_df, new_df, report, sid=sid, file_path=file_path)
    payload["ok"] = True
    payload["success"] = True
    payload["conflicts"] = []
    payload["new_status"] = new_status
    payload["dataset_version"] = new_version
    return jsonify(make_json_safe(payload))


@app.route("/api/history/<int:history_id>/evidence", methods=["GET"])
def history_evidence_page(history_id: int):
    """Page complete row evidence persisted with one project action."""
    project_id = flask_session.get("project_id")
    if not project_id or not db.get_cleaning_action(history_id, int(project_id)):
        return api_error(
            "ACTION_NOT_FOUND", "This Change Log action is not part of the current project.",
            status=404, user_action="Refresh the project history.", retryable=False,
        )
    manifest = db.get_manifest(history_id) or {}
    evidence = manifest.get("row_evidence")
    if not isinstance(evidence, list):
        return jsonify({
            "success": True, "history_id": history_id, "limited_evidence": True,
            "message": "Complete row evidence was not stored for this legacy action.",
            "rows": [], "total": 0, "page": 1, "per_page": 25,
        })
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(1, request.args.get("per_page", 25, type=int)))
    start = (page - 1) * per_page
    return jsonify(make_json_safe({
        "success": True,
        "history_id": history_id,
        "dataset_version": int(flask_session.get("dataset_version", 0) or 0),
        "page": page,
        "per_page": per_page,
        "total": len(evidence),
        "rows": evidence[start:start + per_page],
        "limited_evidence": False,
    }))

    config = row.get("action_config") if isinstance(row.get("action_config"), dict) else {}
    manifest = db.get_manifest(history_id_int) or {}
    if not manifest and isinstance(config, dict):
        # Legacy fallback for actions logged before manifests moved to the
        # DB — try the old file/inline storage, and if found, migrate it
        # into the DB immediately so this fallback is only ever needed once
        # more for this action.
        if config.get("manifest_file"):
            manifest = load_manifest_from_file(sid, config["manifest_file"])
        elif config.get("manifest"):
            manifest = config.get("manifest")
        if manifest:
            try:
                db.save_manifest(int(history_id), manifest)
                logging.info(f"Migrated legacy manifest to DB for history_id={history_id}")
            except Exception:
                logging.exception(f"Failed to migrate legacy manifest to DB for history_id={history_id}")
    if not manifest:
        # Legacy action (predates this feature) or a bundled action that
        # genuinely made no cell-level change — frontend falls back to its
        # existing in-memory snapBefore/snapAfter behaviour for this case.
        return jsonify({"ok": False, "reason": "no_manifest"}), 409
    current_df = new_df
    if current_df is None:
        return jsonify({"ok": False, "reason": "no_snapshot"}), 200

    try:
        result = snapshot_diff.apply_manifest(current_df, manifest, direction)
    except Exception as e:
        logging.exception("apply_manifest: snapshot_diff.apply_manifest failed")
        return jsonify({"ok": False, "reason": f"apply_failed: {e}"}), 200

    new_df = result["df"]
    try:
        save_current_snapshot(sid, new_df)
    except Exception as e:
        logging.exception("apply_manifest: failed to persist updated snapshot")
        return jsonify({"ok": False, "reason": f"save_failed: {e}"}), 200

    new_status = "undone" if direction == "reverse" else "applied"
    try:
        db.set_cleaning_action_status(int(history_id), new_status)
    except Exception as e:
        # The dataset change already succeeded and was saved — a failure to
        # persist the status flag shouldn't roll that back or error out the
        # whole request; the card just might show a stale status until the
        # next successful status sync.
        logging.warning(f"apply_manifest: status update failed: {e}")

    filename  = flask_session.get("upload_filename")
    file_path = os.path.join(UPLOAD_FOLDER, sid, filename) if filename else None
    try:
        raw_df = get_cached_df(sid, file_path) if (file_path and os.path.exists(file_path)) else new_df
    except Exception:
        raw_df = new_df

    conflicts = result.get("conflicts", [])

    if conflicts:
        # A manifest that didn't apply cleanly (a row/column it expected is
        # already gone, or already present) means our patch-only assumptions
        # can't be trusted for the affected part of the dataset. Rather than
        # ship a partial patch that might silently drift from what's actually
        # on disk, fall back to the same full, authoritative rebuild a real
        # Apply uses — rare, and correctness beats speed here.
        report = {
            "original_shape":  list(raw_df.shape),
            "cleaned_shape":   list(new_df.shape),
            "rows_after":      int(new_df.shape[0]),
            "cleaned_columns": list(new_df.columns),
            "operations":      [],
            "filled_cells":    0,
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "empty_columns_removed": 0,
        }
        payload = build_clean_payload(raw_df, new_df, report, sid=sid, file_path=file_path)
        payload["ok"] = True
        payload["conflicts"] = conflicts
        payload["new_status"] = new_status
        return jsonify(make_json_safe(payload))

    # --- Fast path: everything below is O(this manifest's own size) or a
    # cheap O(rows)-only pass — never O(rows x cols) over the full dataset. ---

    patch = build_manifest_patch(manifest, direction)

    dq_before = get_cached_dq(sid, file_path, raw_df) if (file_path and os.path.exists(file_path)) else data_quality_score(raw_df)
    # Genuinely O(this manifest's size), not O(dataset size) — see
    # Full canonical recompute. Exact for ANY combination of independently toggled
    # actions (not just a strict linear undo/redo stack): the engine
    # re-derives every component (missing%, duplicates, placeholders,
    # numeric issues, row-placeholder%, outliers, constant columns) from
    # running state that's updated by this exact manifest's delta, the same
    # way the dataframe itself is — it never assumes a fixed action order.
    # Falls back to the full, known-correct rescan if the incremental path
    # raises for any reason (see get_dq_after_incremental).
    new_df_fp = _df_fingerprint(new_df)
    dq_after = get_cached_dq_after(sid, new_df, fp=new_df_fp)

    # NOTE: analyze_dataframe's output (missing_count, missing_by_column,
    # dtypes, duplicate_count, ...) depends on cell VALUES, not just the
    # column set — a value-only manifest (the overwhelming majority of
    # undo/reapply clicks) still needs a fresh analysis, because the whole
    # point of Undo is that values changed. Caching this by column tuple
    # alone previously served stale missing-value counts after exactly that
    # kind of undo. Keying on the same content fingerprint as the DQ cache
    # makes it correct — it will simply miss on nearly every real
    # value-changing click, which is the price of it being right.
    analysis = get_cached_analysis(sid, new_df, fp=new_df_fp)

    def _label_to_row_number(label_str: str):
        # Row labels are persisted as strings (see save_dirty_rows); an
        # integer-valued label is treated as a position (1-indexed for
        # display), same convention used everywhere else in this file
        # (row_numbers_for, compute_changed_rows).
        try:
            return int(label_str) + 1
        except (TypeError, ValueError):
            return label_str

    removed_row_numbers, added_row_numbers = compute_added_removed_rows(raw_df, new_df)
    dirty_rows = update_dirty_rows_for_manifest(sid, raw_df, new_df, manifest)
    changed_row_numbers = sorted(
        {_label_to_row_number(r) for r in dirty_rows} | set(added_row_numbers),
        key=lambda v: (0, v) if isinstance(v, int) else (1, str(v))
    )

    # Only the rows THIS click actually touched need their row data resent —
    # everything else on screen is already correct and untouched.
    touched_labels = {c["row"] for c in patch["changed_cells"]} | {r["row"] for r in patch["added_rows"]}
    touched_index = [
        _restore_index_dtype_for_lookup(lbl, new_df.index)
        for lbl in touched_labels
    ]
    touched_index = [i for i in touched_index if i in new_df.index]
    changed_preview = df_to_rows_safe(new_df.loc[touched_index]) if touched_index else []

    preview_rows = min(len(new_df), 100)
    dt_cols = [c for c in new_df.columns if pd.api.types.is_datetime64_any_dtype(new_df[c])]
    preview_df = new_df.head(preview_rows).copy() if dt_cols else new_df.head(preview_rows)
    if dt_cols:
        for c in dt_cols:
            preview_df[c] = preview_df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

    row_index_map = {
        str(i): int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
        for i, idx in enumerate(new_df.index)
    }

    missing_by_column_remaining = []
    for col, count in new_df.isna().sum().items():
        count = int(count)
        if count > 0:
            missing_by_column_remaining.append({
                "column": col, "missing": count,
                "percent": round((count / len(new_df)) * 100, 2) if len(new_df) else 0,
            })

    payload = {
        "success": True,
        "ok": True,
        "new_status": new_status,
        "conflicts": conflicts,
        "patch": patch,
        "changed_preview": changed_preview,
        "preview": make_json_safe(preview_df.to_dict(orient="records")),
        "preview_row_numbers": [
            int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in preview_df.index
        ],
        "editor_columns": list(new_df.columns),
        "row_index_map": row_index_map,
        "analysis": analysis,
        "report": {
            "operations":          [],
            "cleaned_columns":     list(new_df.columns),
            "preview_row_numbers": [
                int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in preview_df.index
            ],
            "changed_row_numbers": changed_row_numbers,
            "removed_row_numbers": removed_row_numbers,
            "added_row_numbers":   added_row_numbers,
            "total_changed_rows":  len(changed_row_numbers),
            "has_changes":         len(changed_row_numbers) > 0 or len(removed_row_numbers) > 0,
            "data_quality_before": dq_before,
            "data_quality_after":  dq_after,
            "dq_improvement":      round(dq_after - dq_before, 2),
        },
        "summary": {
            "rows_before": int(raw_df.shape[0]),
            "rows_after":  int(new_df.shape[0]),
            "cols_after":  int(new_df.shape[1]),
            "quality_before": dq_before,
            "quality_after":  dq_after,
            "quality_delta":  round(dq_after - dq_before, 2),
            "missing_remaining":      int(new_df.isna().sum().sum()),
            "missing_rows_remaining": int(new_df.isna().any(axis=1).sum()),
            "missing_by_column_remaining": missing_by_column_remaining,
            "changed_rows": len(changed_row_numbers),
            "removed_rows": len(removed_row_numbers),
        },
    }
    return jsonify(make_json_safe(payload))


@app.route("/persist_client_state", methods=["POST"])
def persist_client_state():
    """Reject legacy browser-snapshot persistence.

    Client-rendered data may help display a preview, but it is never an
    authoritative dataset version.
    """
    return api_error(
        "CLIENT_STATE_PERSISTENCE_DISABLED",
        "Browser table data cannot be used as an authoritative dataset version.",
        status=410,
        user_action="Use a replayable action with server-side history, or resume the project from its saved source.",
        retryable=False,
    )
    # Fallback persistence path for Undo/Reapply when no server-side manifest
    # is available (no_manifest / no_snapshot / apply_failed / save_failed
    # from apply_manifest_route above). In that case the frontend still does
    # a best-effort restore using its own locally-captured preview state
    # (_restoreSnap) — but that only ever updated the browser's own UI. The
    # backend's persisted snapshot never changed, so the dataset silently
    # stayed at whatever it was before the click: the Undo/Reapply status tag
    # flips (that's a pure client-side flag, unconditional), but the actual
    # data — what Download/Resume/the next Apply all see — did not.
    #
    # This takes the client's already-correct editor_data/editor_columns/
    # row_index_map (exactly what's currently showing in the Dataset Editor)
    # and persists it as the current snapshot directly, so that fallback
    # path is no longer purely cosmetic.
    #
    # This is a best-effort safety net, not a replacement for the manifest
    # system: dtypes aren't perfectly preserved on this path (e.g. a datetime
    # column round-trips as whatever string it was already displayed as,
    # same as anywhere else the UI shows dates) — a real manifest-backed
    # Undo/Reapply is always preferred and used first when one exists.
    sid = flask_session.get("upload_sid")
    if not sid:
        return jsonify({"ok": False, "reason": "no_session"}), 200

    data = request.get_json(silent=True) or {}
    columns = data.get("editor_columns") or []
    rows = data.get("editor_data") or []
    row_index_map = data.get("row_index_map") or {}
    if not columns:
        return jsonify({"ok": False, "reason": "no_data"}), 200

    try:
        df = pd.DataFrame(rows, columns=columns)
        if row_index_map:
            df.index = [row_index_map.get(str(i), i) for i in range(len(df))]
        save_current_snapshot(sid, df)
        return jsonify({"ok": True})
    except Exception as e:
        logging.exception("persist_client_state failed")
        return jsonify({"ok": False, "reason": f"persist_failed: {e}"}), 200


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def delete_project_route(project_id: int):
    # Deletes the DB record AND the physical session folder (uploads,
    # cleaned files, manifests, the current-state snapshot) on disk.
    # Each upload_sid belongs to exactly one project — a new upload always
    # clears/replaces the previous SID — so it's safe to wipe the whole
    # folder here. delete_session_folder() itself is a no-op if the folder
    # is already gone (e.g. deleted manually, or an old project record with
    # no upload_sid).
    try:
        project = db.get_project(project_id)
        if project:
            delete_session_folder(project.get("upload_sid"), "project deleted via UI")
        db.delete_project(project_id)
        flash("Project and its files deleted.")
    except Exception as e:
        logging.exception(f"delete_project_route failed for project_id={project_id}")
        flash(f"Could not delete project: {e}")
    return redirect(url_for("projects"))


# Cleanup is strictly manual — nothing here runs on a schedule or timer.
# DB persistence means uploaded/cleaned files survive indefinitely until the
# user explicitly deletes a project (which now also removes its files, see
# delete_project_route above). No background process ever removes anything.


# --- Project: Resume (reload latest cleaned version into workspace) ----------

@app.route("/projects/<int:project_id>/resume")
def resume_project(project_id: int):
    """Load the latest dataset version back into the cleaning workspace."""
    try:
        project  = db.get_project(project_id)
        versions = db.get_dataset_versions(project_id)
    except Exception as e:
        flash(f"Could not load project: {e}")
        return redirect(url_for("projects"))

    if not project:
        flash("Project not found.")
        return redirect(url_for("projects"))

    try:
        history = db.get_cleaning_history(project_id)
        history = _prune_missing_manifests(project["upload_sid"], history)
    except Exception:
        history = []

    # Resume must rebuild the working state from the original upload when
    # saved actions exist. Loading the latest cleaned file and replaying actions
    # double-applies cleaning and makes the UI feel like a fresh upload.
    original_path = os.path.join(UPLOAD_FOLDER, project["upload_sid"], project["original_file"])
    resume_path, repaired_from = _ensure_project_source_available(project)
    if not resume_path:
        flash("The immutable source file is missing. Restore the original upload before resuming this project.")
        return redirect(url_for("project_detail", project_id=project_id))
    if repaired_from:
        flash("The original upload was restored into this app install, so the project can resume normally.")

    if not resume_path or not os.path.exists(resume_path):
        flash("Source file not found on disk — the dataset may have been removed. Please upload again.")
        return redirect(url_for("project_detail", project_id=project_id))

    # CRITICAL FIX: Reuse the project's own upload_sid so all subsequent route
    # calls (preview_clean, preview_outliers, etc.) resolve files correctly.
    # Do NOT create a new session_id — that orphans the files under the old sid.
    project_sid = project["upload_sid"]

    # Clear any previous session that belongs to a DIFFERENT project
    previous_sid = flask_session.get("upload_sid")
    if previous_sid and previous_sid != project_sid:
        # Don't delete — other tab may still need it; just clear session keys
        for key in ("upload_sid", "upload_filename", "cleaned_sid", "cleaned_filename", "report_filename", "project_id"):
            flask_session.pop(key, None)

    # Ensure the session folder exists (may be missing if server was wiped)
    session_dir = os.path.join(UPLOAD_FOLDER, project_sid)
    os.makedirs(session_dir, exist_ok=True)

    # Resume is valid only while the immutable upload exists. A cleaned export
    # must never become a silent replacement source for ordered replay.
    original_path = os.path.join(UPLOAD_FOLDER, project_sid, project["original_file"])
    resume_filename = secure_filename(os.path.basename(resume_path))

    # If resume_path is the cleaned/version file and original is gone, copy it
    # into the session folder as the working file.
    dest_path = os.path.join(session_dir, resume_filename)
    if not os.path.exists(dest_path):
        try:
            shutil.copy2(resume_path, dest_path)
        except Exception as e:
            flash(f"Could not copy dataset: {e}")
            return redirect(url_for("project_detail", project_id=project_id))

    working_filename = project["original_file"]
    working_path     = original_path
    if not os.path.exists(working_path):
        flash("This project cannot be resumed because its original source file is missing.")
        return redirect(url_for("project_detail", project_id=project_id))

    flask_session["upload_sid"]       = project_sid
    flask_session["upload_filename"]  = working_filename
    flask_session["cleaned_sid"]      = project_sid
    flask_session["cleaned_filename"] = os.path.basename(project.get("file_path") or "")
    flask_session["project_id"]       = project_id
    flask_session["dataset_version"]  = int(project.get("state_version") or 0)

    try:
        df = load_dataset(working_path)
        with _cache_lock:
            _df_cache[project_sid] = (os.path.getmtime(working_path), df)
    except Exception as e:
        flash(f"Failed to read dataset: {e}")
        return redirect(url_for("project_detail", project_id=project_id))

    try:
        analysis = analyze_dataframe(df)
        PREVIEW_ROWS = 100
        analysis["original_preview"]       = df_to_rows_safe(df, PREVIEW_ROWS)
        analysis["original_total_rows"]    = len(df)
        analysis["original_preview_count"] = min(len(df), PREVIEW_ROWS)
        analysis["row_index_map"] = {
            str(i): int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
            for i, idx in enumerate(df.index)
        }
        analysis["editor_columns"]   = list(df.columns)
        analysis["editor_row_count"] = len(df)
        analysis["project_id"]       = project_id
        analysis["resumed_project"]  = {
            "id":   project_id,
            "name": project.get("name", ""),
        }
    except Exception as e:
        flash(f"Failed to analyse dataset: {e}")
        return redirect(url_for("project_detail", project_id=project_id))

    return render_template("index.html",
                           uploaded_filename=working_filename,
                           analysis=analysis,
                           cleaned=None,
                           custom_code_enabled=ENABLE_CUSTOM_CODE,
                           resumed_history=history)


# --- Project: Replay cleaning state on resume --------------------------------

@app.route("/resume_replay", methods=["POST"])
def resume_replay():
    """
    Called by JS on resume. Replays saved appliedControlEntries against the
    current session file, returning full cleaned state for UI restore.
    If no entries, returns the current file state as-is (cleaned version).
    """
    sid      = flask_session.get("upload_sid")
    filename = flask_session.get("upload_filename")
    if not sid or not filename:
        return jsonify({"ok": False, "reason": "no session"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        # Try the cleaned file as fallback
        cleaned_filename = flask_session.get("cleaned_filename", "")
        cleaned_path = os.path.join(UPLOAD_FOLDER, sid, "cleaned", cleaned_filename) if cleaned_filename else None
        if cleaned_path and os.path.exists(cleaned_path):
            file_path = cleaned_path
            filename  = cleaned_filename
        else:
            return jsonify({"ok": False, "reason": "file not found"}), 404

    data    = request.get_json(silent=True) or {}
    entries = data.get("entries", [])

    try:
        df = get_cached_df(sid, file_path)

        # Fast path: if a current-state snapshot exists (this session has at
        # least one action logged under the new manifest-based flow), load it
        # directly instead of replaying the whole pipeline from raw + entries.
        # This is what makes Undo/Reapply buttons meaningful after a restart —
        # the snapshot already reflects exactly which actions are currently
        # applied vs undone, cell-for-cell.
        #
        # Falls back to the pre-existing replay-from-entries behaviour
        # whenever no snapshot exists yet — legacy sessions/projects created
        # before this feature keep working exactly as they did before.
        snapshot_df = load_current_snapshot(sid)
        project_id = flask_session.get("project_id")

        if project_id:
            cleaned_df, _history = replay_project_actions(int(project_id))
            save_current_snapshot(sid, cleaned_df)
            project = db.get_project(int(project_id))
            if project:
                flask_session["dataset_version"] = int(project.get("state_version") or flask_session.get("dataset_version", 0) or 0)
            active_metrics = _aggregate_active_history_metrics(_history)
            report = {
                "original_shape":  list(df.shape),
                "cleaned_shape":   list(cleaned_df.shape),
                "rows_after":      int(cleaned_df.shape[0]),
                "cleaned_columns": list(cleaned_df.columns),
                "operations":      active_metrics["operations"],
                "filled_cells":    active_metrics["filled_cells"],
                "duplicates_removed": active_metrics["duplicates_removed"],
                "outliers_removed": active_metrics["outliers_removed"],
                "empty_columns_removed": active_metrics["empty_columns_removed"],
            }
        elif snapshot_df is not None:
            cleaned_df = snapshot_df
            report = {
                "original_shape": list(df.shape),
                "cleaned_shape": list(cleaned_df.shape),
                "rows_after": int(cleaned_df.shape[0]),
                "cleaned_columns": list(cleaned_df.columns),
                "operations": [],
                "filled_cells": 0,
                "duplicates_removed": 0,
                "outliers_removed": 0,
                "empty_columns_removed": 0,
            }
        elif entries:
            # Build a flat form dict from the saved entries and replay cleaning
            form_dict: dict = {}
            for item in entries:
                if isinstance(item, list) and len(item) == 2:
                    k, v = str(item[0]), str(item[1])
                    if k in form_dict:
                        if not isinstance(form_dict[k], list):
                            form_dict[k] = [form_dict[k]]
                        form_dict[k].append(v)
                    else:
                        form_dict[k] = v
            form_multi  = {k: (v if isinstance(v, list) else [v]) for k, v in form_dict.items()}
            form_proxy  = _FormProxy(form_multi)
            opts        = parse_clean_options(form_proxy, default_enabled=False, default_missing_strategy="ignore")
            cleaned_df, report = clean_dataframe(df, options=opts)
            cleaned_df, report = apply_custom_code_blocks(cleaned_df, report, form_proxy)

            # This is the (slow) legacy replay-from-entries path — it only
            # runs because no snapshot existed yet for this session. Saving
            # one now means every resume AFTER this one hits the fast
            # snapshot path above instead of replaying the whole history
            # again from scratch.
            try:
                save_current_snapshot(sid, cleaned_df)
                logging.info(f"Snapshot saved after legacy replay for sid={sid}")
            except Exception:
                logging.exception(f"Failed to save snapshot after replay for sid={sid} — next resume will replay again")
        else:
            # No entries to replay — return the current file state as-is
            # (this covers old DB records without appliedControlEntries)
            from cleaner import analyze_dataframe as _analyze
            cleaned_df = df.copy()
            report = {
                "original_shape":  list(df.shape),
                "cleaned_shape":   list(df.shape),
                "rows_after":      int(df.shape[0]),
                "cleaned_columns": list(df.columns),
                "operations":      [],
                "filled_cells":    0,
                "duplicates_removed": 0,
                "outliers_removed": 0,
                "empty_columns_removed": 0,
                "preview_row_numbers": list(range(1, min(101, len(df)+1))),
                "preview_display_limit": min(100, len(df)),
                "preview": df_to_rows_safe(df, 100),
            }

        dq_before = get_cached_dq(sid, file_path, df)
        quality_breakdown = data_quality_breakdown(cleaned_df, df)
        dq_after  = float(quality_breakdown["score"])
        report["data_quality_before"]         = dq_before
        report["data_quality_after"]          = dq_after
        report["quality_breakdown"]           = quality_breakdown
        report["cleaning_completeness_score"] = cleaning_completeness_score(report)

        preview_rows = min(len(cleaned_df), 100)
        preview_df   = cleaned_df.head(preview_rows).copy()
        dt_cols = [c for c in cleaned_df.columns if pd.api.types.is_datetime64_any_dtype(cleaned_df[c])]
        for c in dt_cols:
            preview_df[c] = preview_df[c].dt.strftime("%Y-%m-%d %H:%M:%S")

        payload = build_clean_payload(
            df, cleaned_df, report, sid=sid, file_path=file_path,
            include_editor_data=False,
            include_changed_rows=False,
        )
        payload["editor_data"] = payload.get("preview", [])
        payload["editor_data_is_sample"] = True
        payload.update({
            "ok": True,
            "success": True,
            "dataset_version": int(flask_session.get("dataset_version", 0) or 0),
        })
        return jsonify(make_json_safe(payload))
    except Exception as e:
        logging.exception("resume_replay failed")
        message, user_action = _operation_failure_explanation(e, "process")
        return api_error(
            "RESUME_REPLAY_FAILED", f"The project could not be rebuilt from its source and action history: {message}",
            status=409, user_action=user_action,
            retryable=False,
        )



@app.route("/projects/<int:project_id>/update_name", methods=["POST"])
def update_project_name_route(project_id: int):
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:255]
    if not name:
        return jsonify({"ok": False, "reason": "Name cannot be empty"}), 400
    try:
        db.update_project_name(project_id, name)
        return jsonify({"ok": True, "name": name})
    except Exception as e:
        logging.exception("update_project_name failed")
        message, user_action = _operation_failure_explanation(e, "save")
        return api_error(
            "PROJECT_NAME_SAVE_FAILED",
            f"The project name could not be saved: {message}",
            status=500,
            user_action=user_action,
            retryable=True,
        )


# --- Project: update notes ---------------------------------------------------

@app.route("/projects/<int:project_id>/update_notes", methods=["POST"])
def update_project_notes_route(project_id: int):
    data = request.get_json(silent=True) or {}
    notes = str(data.get("notes", "")).strip()
    try:
        db.update_project_notes(project_id, notes)
        return jsonify({"ok": True})
    except Exception as e:
        logging.exception("update_project_notes failed")
        message, user_action = _operation_failure_explanation(e, "save")
        return api_error(
            "PROJECT_NOTES_SAVE_FAILED",
            f"The project notes could not be saved: {message}",
            status=500,
            user_action=user_action,
            retryable=True,
        )

# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        from waitress import serve
        logging.info(f"Starting with waitress on {host}:{port} (8 threads)")
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        debug_mode = os.environ.get("FLASK_ENV", "").lower() == "development"
        app.run(host=host, port=port, debug=debug_mode)
