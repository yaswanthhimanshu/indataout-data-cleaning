"""
app.py - Flask app for indataout DATA CLEANING
Features:
- Robust CSV/TSV/Excel/JSON loading with encoding fallbacks (utf-8, latin1/cp1252, chardet if installed)
- Safe file saving (secure_filename)
- JSON responses + normal template rendering
- Maintains existing cleaning pipeline integration (analyze_dataframe, clean_dataframe, save_report)
"""

import os
import logging
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
from cleaner import analyze_dataframe, clean_dataframe, save_report, data_quality_score

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
app.secret_key = "super-secret-key"  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CLEANED_FOLDER = os.path.join(UPLOAD_FOLDER, "cleaned")
REPORTS_FOLDER = os.path.join(UPLOAD_FOLDER, "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "tsv", "txt"}

# Logging
logging.basicConfig(level=logging.INFO)


# -----------------------------------
# Helpers
# -----------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
            <p>Use /files to list uploads and /download/cleaned/&lt;filename&gt; or /download/report/&lt;filename&gt; to download.</p>
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
    file_path = os.path.join(UPLOAD_FOLDER, filename)
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

    file_path = os.path.join(UPLOAD_FOLDER, filename)
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


    def _bool_from_form(name, default=True):
        val = request.form.get(name)
        if val is None:
            return default
        return str(val).lower() in ("1", "true", "on", "yes")

    try:
        opts = {
            "remove_duplicates": _bool_from_form("remove_duplicates", True),
            "drop_empty_columns": _bool_from_form("drop_empty_columns", True),
            "drop_constant_columns": _bool_from_form("drop_constant_columns", True),
            "trim_whitespace": _bool_from_form("trim_whitespace", True),
            "rename_columns": True,
            "convert_dtypes": _bool_from_form("convert_dtypes", True),
            "handle_outliers": {"method": "iqr", "action": "remove", "multiplier": 1.5}
            if _bool_from_form("handle_outliers", False)
            else None,
            "encode_categoricals": "label" if _bool_from_form("encode_categoricals", False) else False,
            "missing_strategy": request.form.get("missing_strategy", "fill"),
            "fill_numeric": request.form.get("fill_numeric", "mean"),
            "fill_categorical": request.form.get("fill_categorical", "mode"),
            # new options
            "enable_fuzzy": _bool_from_form("enable_fuzzy", False),
            "fuzzy_threshold": int(request.form.get("fuzzy_threshold", 88)),
            "fuzzy_columns": [c.strip() for c in request.form.get("fuzzy_columns", "").split(",") if c.strip()] or None,
            "knn_k": int(request.form.get("knn_k", 5)) if request.form.get("knn_k") else 5,
        }
    except Exception as e:
        logging.exception("Invalid advanced option")
        flash(f"Invalid advanced option: {e}")
        return redirect(url_for("index"))
    # --- Manual datatype mapping (from form fields like dtype_map[column_name]) ---
    dtype_map = {}
    for key, val in request.form.items():
        if key.startswith("dtype_map[") and key.endswith("]"):
            col_name = key[len("dtype_map["):-1]
            if val:
                dtype_map[col_name] = val
    if dtype_map:
        opts["dtype_map"] = dtype_map


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

    cleaned_basename = f"cleaned_{filename.rsplit('.', 1)[0]}.csv"
    cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_basename)
    try:
        cleaned_df.to_csv(cleaned_path, index=False)
    except Exception as e:
        logging.exception("Failed to save cleaned file")
        flash(f"Failed to save cleaned file: {e}")
        return redirect(url_for("index"))

    report_basename = f"report_{filename.rsplit('.', 1)[0]}.json"
    report_path = os.path.join(REPORTS_FOLDER, report_basename)
    try:
        save_report(report, report_path)
    except Exception as e:
        logging.exception("Failed to save report")
        flash(f"Failed to save report: {e}")
        return redirect(url_for("index"))

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


# -----------------------------------
# Download cleaned file
# -----------------------------------
@app.route("/download/cleaned/<path:filename>")
def download_cleaned(filename):
    safe = secure_filename(filename)
    full = os.path.join(CLEANED_FOLDER, safe)
    if not os.path.exists(full):
        flash("File not found.")
        return redirect(url_for("index"))
    return send_from_directory(CLEANED_FOLDER, safe, as_attachment=True)


# -----------------------------------
# Download cleaning report (JSON)
# -----------------------------------
@app.route("/download/report/<path:filename>")
def download_report(filename):
    safe = secure_filename(filename)
    full = os.path.join(REPORTS_FOLDER, safe)
    if not os.path.exists(full):
        flash("File not found.")
        return redirect(url_for("index"))
    return send_from_directory(REPORTS_FOLDER, safe, as_attachment=True)


# -----------------------------------
# List uploaded files (JSON)
# -----------------------------------
@app.route("/files", methods=["GET"])
def list_files():
    try:
        files = []
        for fname in os.listdir(UPLOAD_FOLDER):
            fpath = os.path.join(UPLOAD_FOLDER, fname)
            if os.path.isfile(fpath):
                files.append({"filename": fname, "size_bytes": os.path.getsize(fpath)})
        return jsonify({"success": True, "files": files})
    except Exception as e:
        logging.exception("Failed to list files")
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------------
# Main entry
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "").lower() == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)