"""
app.py - Flask app for indataout DATA CLEANING
Features:
- Robust CSV/TSV/Excel/JSON loading with encoding fallbacks (utf-8, latin1/cp1252, chardet if installed)
- Safe file saving (secure_filename)
- JSON responses + normal template rendering
- Maintains existing cleaning pipeline integration (analyze_dataframe, clean_dataframe, save_report)
"""

import os
import json
import logging
import shutil
import secrets
import threading
import time
import re
import math
import tempfile
from io import StringIO
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
)
from werkzeug.utils import secure_filename
import pandas as pd

# Import your cleaning helpers (assumes cleaner.py exists and exports these)
from cleaner import analyze_dataframe, clean_dataframe, save_report, data_quality_score, cleaning_completeness_score, detect_outliers

# Optional encoding detector
try:
    import chardet
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

# -----------------------------------
# Flask configuration
# -----------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join(tempfile.gettempdir(), "indataout_uploads"),
)
UPLOAD_RETENTION_SECONDS = int(os.environ.get("UPLOAD_RETENTION_SECONDS", "7200"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "tsv", "txt"}

# Logging
logging.basicConfig(level=logging.INFO)


# -----------------------------------
# Helpers
# -----------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalized_column_name(col_name):
    """Match cleaner.py column normalization for form fields submitted before cleaning."""
    nc = str(col_name).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    nc = nc.strip().lower().replace(" ", "_")
    return "".join(ch if ord(ch) < 128 else "_" for ch in nc)


def make_json_safe(value):
    """Convert pandas/numpy values and non-finite floats into strict JSON-safe data."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]
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


def delete_original_upload(sid, filename):
    """Delete the uploaded source file after final outputs have been generated."""
    if not sid or not filename:
        return
    path = os.path.join(UPLOAD_FOLDER, sid, secure_filename(filename))
    try:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Deleted original uploaded file: {filename}")
    except Exception as e:
        logging.warning(f"Could not delete original uploaded file {filename}: {e}")


def delete_session_folder(sid, reason="session cleanup"):
    """Remove one upload session folder if it is inside the configured upload root."""
    if not sid:
        return
    session_root = os.path.abspath(os.path.join(UPLOAD_FOLDER, str(sid)))
    upload_root = os.path.abspath(UPLOAD_FOLDER)
    if not session_root.startswith(upload_root + os.sep):
        logging.warning(f"Refused to delete path outside upload root: {session_root}")
        return
    if os.path.isdir(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
        logging.info(f"Deleted upload session folder ({reason}): {sid}")


def schedule_session_cleanup(sid, reason, delay=120.0):
    """Schedule temporary session cleanup without keeping the process alive."""
    timer = threading.Timer(delay, lambda: delete_session_folder(sid, reason))
    timer.daemon = True
    timer.start()


def parse_clean_options(form, default_enabled=True, default_missing_strategy="fill"):
    def _bool_from_form(name, default=None):
        if default is None:
            default = default_enabled
        values = form.getlist(name)
        if not values:
            return default
        val = values[-1]
        return str(val).lower() in ("1", "true", "on", "yes")

    opts = {
        "remove_duplicates": _bool_from_form("remove_duplicates"),
        "drop_empty_columns": _bool_from_form("drop_empty_columns"),
        "drop_constant_columns": _bool_from_form("drop_constant_columns"),
        "trim_whitespace": _bool_from_form("trim_whitespace"),
        "rename_columns": True,
        "convert_dtypes": _bool_from_form("convert_dtypes"),
        "handle_outliers": None,
        "encode_categoricals": "label" if _bool_from_form("encode_categoricals", False) else False,
        "missing_strategy": form.get("missing_strategy", default_missing_strategy),
        "fill_numeric": form.get("fill_numeric", "mean"),
        "fill_categorical": form.get("fill_categorical", "mode"),
    }

    selected_outlier_rows_str = form.get("selected_outlier_rows")
    logging.info(f"Raw selected_outlier_rows from form: {selected_outlier_rows_str}")
    logging.info(f"All form keys: {list(form.keys())}")
    if selected_outlier_rows_str:
        try:
            opts["selected_outlier_rows"] = json.loads(selected_outlier_rows_str)
        except Exception as e:
            logging.warning(f"Could not parse selected outlier rows: {e}")
            opts["selected_outlier_rows"] = []
    else:
        opts["selected_outlier_rows"] = []

    dtype_map = {}
    column_renames = {}
    cell_updates = []
    missing_cell_updates = []
    drop_selected_columns = []
    for key, val in form.items():
        if key.startswith("dtype_map[") and key.endswith("]"):
            col_name = normalized_column_name(key[len("dtype_map["):-1])
            if val and val != "auto":
                dtype_map[col_name] = val
        elif key.startswith("column_rename[") and key.endswith("]") and val:
            original_col = normalized_column_name(key[len("column_rename["):-1])
            new_col = normalized_column_name(val)
            if new_col and new_col != original_col:
                column_renames[original_col] = new_col
        elif key.startswith("cell_update["):
            match = re.match(r"^cell_update\[([^\]]+)\]\[(.+)\]$", key)
            if match:
                cell_updates.append({
                    "row_index": match.group(1),
                    "column": normalized_column_name(match.group(2)),
                    "value": val
                })
        elif key.startswith("missing_cell[") and val:
            match = re.match(r"^missing_cell\[([^\]]+)\]\[(.+)\]$", key)
            if match:
                missing_cell_updates.append({
                    "row_index": match.group(1),
                    "column": normalized_column_name(match.group(2)),
                    "value": val
                })
        elif key.startswith("drop_column[") and key.endswith("]") and val:
            drop_selected_columns.append(normalized_column_name(key[len("drop_column["):-1]))
    if dtype_map:
        opts["dtype_map"] = dtype_map
    if column_renames:
        opts["column_renames"] = column_renames
    if cell_updates:
        opts["cell_updates"] = cell_updates
    if missing_cell_updates:
        opts["missing_cell_updates"] = missing_cell_updates
    if drop_selected_columns:
        opts["drop_selected_columns"] = drop_selected_columns

    return opts


def detect_encoding_with_chardet(filepath: str, n_bytes: int = 200000):
    """Return detected encoding or None if chardet not available or fails."""
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
        logging.debug(f"chardet detection failed: {e}")
        return None


def load_dataset(filepath: str):
    """
    Robustly load CSV / TSV / Excel / JSON.
    Tries utf-8, utf-8-sig, latin1, cp1252, iso-8859-1, optional chardet guess,
    then falls back to decoding with replacement.
    Returns a pandas DataFrame.
    Raises ValueError on unrecoverable parse issues.
    """
    ext = os.path.splitext(filepath)[1].lower()
    tried_encodings = []

    # Excel handling
    if ext in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(filepath)
            return df
        except Exception as e:
            logging.exception("Failed to read Excel file")
            raise ValueError("Failed to read Excel file: " + str(e)) from e

    # JSON handling (respect encoding detection)
    if ext == ".json":
        try:
            df = pd.read_json(filepath)
            return df
        except Exception:
            pass

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    chardet_enc = detect_encoding_with_chardet(filepath)
    if chardet_enc and chardet_enc not in encodings_to_try:
        encodings_to_try.insert(0, chardet_enc)

    sep = "\t" if filepath.lower().endswith(".tsv") else None

    for enc in encodings_to_try:
        if enc in tried_encodings:
            continue
        tried_encodings.append(enc)
        try:
            if ext == ".json":
                df = pd.read_json(filepath, encoding=enc)
            elif sep is None:
                df = pd.read_csv(filepath, encoding=enc, sep=None, engine="python")
            else:
                df = pd.read_csv(filepath, encoding=enc, sep=sep)
            logging.info(f"Loaded {os.path.basename(filepath)} using encoding {enc}")
            return df
        except Exception as e:
            logging.debug(f"read failed for encoding={enc}: {e}")

    try:
        if ext == ".json":
            with open(filepath, "rb") as f:
                raw = f.read()
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_json(StringIO(text))
            logging.info("Loaded JSON by decoding with errors='replace'.")
            return df
        else:
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                engine="python",
                sep=sep,
                on_bad_lines="warn",
            )
            logging.info("Loaded using python engine with utf-8 (fallback).")
            return df
    except Exception as e:
        logging.debug(f"python-engine fallback failed: {e}")

    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        sio = StringIO(text)
        if ext == ".json":
            df = pd.read_json(sio)
        else:
            df = pd.read_csv(sio, sep=sep)
        logging.info("Loaded by decoding with errors='replace'. Some characters may be corrupted.")
        return df
    except Exception as e:
        logging.exception("Final fallback failed to parse file.")
        raise ValueError(
            "Unable to parse the uploaded file as CSV/TSV/Excel/JSON. "
            "Consider re-saving the file as UTF-8 or uploading an Excel (.xlsx) file."
        ) from e


# -----------------------------------
# Routes
# -----------------------------------
@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html", uploaded_filename=None, analysis=None, cleaned=None)
    except Exception:
        return (
            """<!doctype html>
            <html><head><meta charset="utf-8"><title>indataout - DATA CLEANING</title></head>
            <body>
            <h3>indataout - DATA CLEANING</h3>
            <form method="post" action="/upload" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Upload</button>
            </form>
            <p>Upload a CSV, Excel, JSON, TSV, or TXT file to start cleaning.</p>
            </body></html>"""
        )


# -----------------------------------
# Upload route (uses robust loader)
# -----------------------------------
@app.route("/upload", methods=["POST"])
def upload():
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
    # Every upload gets its own UUID folder — no two users ever share a path
    from flask import session as flask_session
    previous_sid = flask_session.get("upload_sid")
    if previous_sid:
        delete_session_folder(previous_sid, "replaced by new upload")
        flask_session.pop("upload_sid", None)
        flask_session.pop("upload_filename", None)
        flask_session.pop("cleaned_sid", None)
        flask_session.pop("cleaned_filename", None)
        flask_session.pop("report_filename", None)

    session_id = secrets.token_hex(16)
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path = os.path.join(session_dir, filename)
    flask_session["upload_sid"]      = session_id
    flask_session["upload_filename"] = filename
    try:
        file.save(file_path)
    except Exception as e:
        logging.exception("Failed to save uploaded file")
        flash(f"Failed to save uploaded file: {e}")
        return redirect(url_for("index"))

    try:
        df = load_dataset(file_path)
    except Exception as e:
        logging.exception("Failed to load dataset on upload")
        flash(f"Failed to read uploaded file: {e}")
        return redirect(url_for("index"))

    try:
        analysis = analyze_dataframe(df)
    except Exception as e:
        logging.exception("analyze_dataframe failed")
        flash(f"Failed to analyze dataset: {e}")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        uploaded_filename=filename,
        analysis=analysis,
        cleaned=None,
    )


# -----------------------------------
# Clean data route
# -----------------------------------
@app.route("/clean", methods=["POST"])
def clean():
    filename = request.form.get("filename")
    if not filename:
        flash("No file specified.")
        return redirect(url_for("index"))

    from flask import session as flask_session
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
        df = load_dataset(file_path)
    except Exception as e:
        logging.exception("Failed to read dataset for cleaning")
        flash(f"Failed to read dataset: {e}")
        return redirect(url_for("index"))
    dq_before = data_quality_score(df)


    try:
        opts = parse_clean_options(
            request.form,
            default_enabled=False,
            default_missing_strategy="ignore",
        )
    except Exception as e:
        logging.exception("Invalid advanced option")
        flash(f"Invalid advanced option: {e}")
        return redirect(url_for("index"))

    try:
        cleaned_df, report = clean_dataframe(df, options=opts)
    except Exception as e:
        logging.exception("Error during cleaning")
        flash(f"Error during cleaning: {e}")
        return redirect(url_for("index"))
    # --- Data Quality Score (Before & After) ---
    dq_after = data_quality_score(cleaned_df)
    report["data_quality_before"] = dq_before
    report["data_quality_after"] = dq_after
    report["dq_improvement"] = round(dq_after - dq_before, 2)
    report["cleaning_completeness_score"] = cleaning_completeness_score(report)
    report = make_json_safe(report)

    cleaned_basename = f"cleaned_{filename.rsplit('.', 1)[0]}.csv"
    user_cleaned_dir = os.path.join(UPLOAD_FOLDER, sid, "cleaned")
    os.makedirs(user_cleaned_dir, exist_ok=True)
    cleaned_path = os.path.join(user_cleaned_dir, cleaned_basename)
    flask_session["cleaned_sid"]      = sid
    flask_session["cleaned_filename"] = cleaned_basename
    try:
        # Save with UTF-8 encoding and no index to ensure clean output
        cleaned_df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        logging.exception("Failed to save cleaned file")
        flash(f"Failed to save cleaned file: {e}")
        return redirect(url_for("index"))

    report_basename = f"report_{filename.rsplit('.', 1)[0]}.json"
    user_reports_dir = os.path.join(UPLOAD_FOLDER, sid, "reports")
    os.makedirs(user_reports_dir, exist_ok=True)
    report_path = os.path.join(user_reports_dir, report_basename)
    flask_session["report_filename"] = report_basename
    try:
        save_report(report, report_path)
    except Exception as e:
        logging.exception("Failed to save report")
        flash(f"Failed to save report: {e}")
        return redirect(url_for("index"))

    delete_original_upload(sid, filename)

    return render_template(
        "index.html",
        uploaded_filename=filename,
        analysis=None,
        cleaned={
            "cleaned_filename": cleaned_basename,
            "report_filename": report_basename,
            "report": report,
        },
    )


@app.route("/preview_clean", methods=["POST"])
def preview_clean():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "No file specified."}), 400

    from flask import session as flask_session
    sid = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return jsonify({"error": "Session mismatch. Please re-upload your file."}), 403

    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404

    try:
        df = load_dataset(file_path)
        scoped_preview = request.form.get("preview_scope") in ("feature", "section", "manual")
        opts = parse_clean_options(
            request.form,
            default_enabled=not scoped_preview,
            default_missing_strategy="ignore" if scoped_preview else "fill",
        )
        cleaned_df, report = clean_dataframe(df, options=opts)
        dq_before = data_quality_score(df)
        dq_after = data_quality_score(cleaned_df)
        report["data_quality_before"] = dq_before
        report["data_quality_after"] = dq_after
        report["dq_improvement"] = round(dq_after - dq_before, 2)
        report["cleaning_completeness_score"] = cleaning_completeness_score(report)
        missing_by_column_remaining = []
        missing_counts = cleaned_df.isna().sum()
        for col, count in missing_counts.items():
            count = int(count)
            if count > 0:
                missing_by_column_remaining.append({
                    "column": col,
                    "missing": count,
                    "percent": round((count / len(cleaned_df)) * 100, 2) if len(cleaned_df) else 0,
                })
        payload = {
            "success": True,
            "preview": report.get("preview", []),
            "report": report,
            "summary": {
                "rows_before": int(report["original_shape"][0]),
                "rows_after": int(report["rows_after"]),
                "cols_after": int(report["cleaned_shape"][1]),
                "filled_cells": int(report["filled_cells"]),
                "duplicates_removed": int(report["duplicates_removed"]),
                "outliers_removed": int(report["outliers_removed"]),
                "empty_columns_removed": int(report["empty_columns_removed"]),
                "quality_before": dq_before,
                "quality_after": dq_after,
                "quality_delta": round(dq_after - dq_before, 2),
                "cleaning_completeness": report["cleaning_completeness_score"],
                "missing_remaining": int(cleaned_df.isna().sum().sum()),
                "missing_rows_remaining": int(cleaned_df.isna().any(axis=1).sum()),
                "missing_by_column_remaining": missing_by_column_remaining,
            }
        }
        return jsonify(make_json_safe(payload))
    except Exception as e:
        logging.exception("Live preview failed")
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# Download cleaned file
# -----------------------------------
@app.route("/download/cleaned/<path:filename>")
def download_cleaned(filename):
    from flask import session as flask_session
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
    # Keep both download buttons usable, then let the temporary session expire shortly after.
    response = send_from_directory(folder, safe, as_attachment=True)
    schedule_session_cleanup(sid, "post-download cleanup")
    return response


# -----------------------------------
# Download cleaning report (JSON)
# -----------------------------------
@app.route("/download/report/<path:filename>")
def download_report(filename):
    from flask import session as flask_session
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
    schedule_session_cleanup(sid, "post-report-download cleanup")
    return response


# /files route removed — it exposed all users' filenames publicly.


# -----------------------------------
# Preview outliers route
# -----------------------------------
@app.route('/preview_outliers', methods=['POST'])
def preview_outliers():
    filename = request.form.get('filename')
    if not filename:
        return jsonify({'error': 'No file specified'}), 400

    from flask import session as flask_session
    sid              = flask_session.get("upload_sid")
    session_filename = flask_session.get("upload_filename")
    if not sid or session_filename != filename:
        return jsonify({'error': 'Unauthorized'}), 403
    file_path = os.path.join(UPLOAD_FOLDER, sid, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 400

    try:
        df = load_dataset(file_path)
        if request.form.get("preview_scope"):
            opts = parse_clean_options(
                request.form,
                default_enabled=False,
                default_missing_strategy="ignore",
            )
            opts["selected_outlier_rows"] = []
            df, _ = clean_dataframe(df, options=opts)
        multiplier = float(request.form.get('multiplier', 1.5))
        
        outliers = detect_outliers(df, multiplier=multiplier)
        return jsonify({'success': True, 'outliers': outliers})
    except Exception as e:
        logging.exception('Failed to detect outliers')
        return jsonify({'error': str(e)}), 500


# -----------------------------------
# Background cleanup — deletes abandoned UUID folders older than 2 hours
# Runs every hour. Handles users who upload but never download.
# -----------------------------------
def cleanup_abandoned_uploads():
    now = time.time()
    try:
        for name in os.listdir(UPLOAD_FOLDER):
            if name == "__pycache__":
                continue
            path = os.path.join(UPLOAD_FOLDER, name)
            if os.path.isdir(path):
                age = now - os.path.getmtime(path)
                if age > UPLOAD_RETENTION_SECONDS:
                    shutil.rmtree(path, ignore_errors=True)
                    logging.info(f"Cleanup: removed abandoned upload folder {name}")
    except Exception as e:
        logging.warning(f"Upload cleanup error: {e}")


def _periodic_cleanup():
    while True:
        time.sleep(600)
        cleanup_abandoned_uploads()

cleanup_abandoned_uploads()
threading.Thread(target=_periodic_cleanup, daemon=True).start()


# -----------------------------------
# Main entry
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "").lower() == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

#END
