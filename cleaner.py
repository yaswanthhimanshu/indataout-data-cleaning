# cleaner.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

# -------------------------
# Helpers
# -------------------------
def _is_datetime_series(s: pd.Series) -> bool:
    """Try to infer datetime series robustly."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_string_dtype(s):
        try:
            pd.to_datetime(s.dropna().sample(min(50, max(1, int(len(s.dropna())*0.1)))), errors='raise')
            return True
        except Exception:
            return False
    return False

def _try_convert_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Attempt to convert to numeric; return (converted_series, changed_flag)."""
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    conv = pd.to_numeric(series, errors='coerce')
    changed = not conv.equals(series) and conv.notna().sum() > 0
    return conv, changed

def _try_convert_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Attempt to convert to datetime; return (converted_series, changed_flag)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False
    try:
        conv = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        changed = conv.notna().sum() > 0 and not conv.equals(series)
        return conv, changed
    except Exception:
        return series, False

def _iqr_outliers_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Return boolean mask of outliers True for outlier rows in a numeric series."""
    if not pd.api.types.is_numeric_dtype(series):
        return pd.Series(False, index=series.index)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr
    return (series < low) | (series > high)

# -------------------------
# Analysis (pre-clean) - what's messy
# -------------------------
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a messiness summary for the UI (before cleaning).
    Keys returned:
      - shape: [rows, cols]
      - missing_count: int
      - missing_percent: float
      - duplicate_count: int
      - empty_columns: list
      - preview: df.head(10) as records (list of dicts)
      - dtypes: mapping column -> dtype (string)
    """
    rows, cols = df.shape
    total_cells = rows * cols if cols > 0 else 1
    missing_count = int(df.isna().sum().sum())
    missing_percent = float(0 if total_cells == 0 else (missing_count / total_cells) * 100)
    duplicate_count = int(df.duplicated().sum())
    empty_columns = [c for c in df.columns if df[c].isna().all()]
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    # 🔹 changed line here
    PREVIEW_ROWS = min(len(df), 1000)
    preview_df = df.head(PREVIEW_ROWS).copy()

    # convert datetimes to iso for preview safety
    for c in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[c]):
            preview_df[c] = preview_df[c].dt.strftime('%Y-%m-%d %H:%M:%S')

    return {
        "shape": [rows, cols],
        "missing_count": missing_count,
        "missing_percent": round(missing_percent, 2),
        "duplicate_count": duplicate_count,
        "empty_columns": empty_columns,
        "dtypes": dtypes,
        "preview": preview_df.to_dict(orient="records")
    }

# -------------------------
# Cleaning Pipeline
# -------------------------
def clean_dataframe(
    df: pd.DataFrame,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataframe according to options and return (cleaned_df, report_dict).

    options keys and defaults:
      - remove_duplicates: True
      - duplicates_subset: None (use all columns) or list of columns
      - missing_strategy: "fill" or "drop"
      - fill_numeric: "mean" | "median" | "zero" | number
      - fill_categorical: "mode" | "constant" | value
      - drop_empty_columns: True
      - rename_columns: True (lowercase & replace spaces with _)
      - trim_whitespace: True
      - convert_dtypes: True (numeric-like -> numeric, dates -> datetime)
      - handle_outliers: None | {"method":"iqr", "action": "remove"|"clip", "multiplier":1.5}
      - encode_categoricals: False | "label"   (one-hot not implemented by default)
      - drop_constant_columns: True
    """
    if options is None:
        options = {}

    # defaults
    opts = {
        "remove_duplicates": options.get("remove_duplicates", True),
        "duplicates_subset": options.get("duplicates_subset", None),
        "missing_strategy": options.get("missing_strategy", "fill"),  # or "drop"
        "fill_numeric": options.get("fill_numeric", "mean"),  # mean|median|zero|number
        "fill_categorical": options.get("fill_categorical", "mode"),  # mode|constant|value
        "drop_empty_columns": options.get("drop_empty_columns", True),
        "rename_columns": options.get("rename_columns", True),
        "trim_whitespace": options.get("trim_whitespace", True),
        "convert_dtypes": options.get("convert_dtypes", True),
        "handle_outliers": options.get("handle_outliers", None),  # dict or None
        "encode_categoricals": options.get("encode_categoricals", False),  # False or "label"
        "drop_constant_columns": options.get("drop_constant_columns", True),
    }

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "original_shape": list(df.shape),
        "operations": []
    }

    working = df.copy(deep=True)

    # 1) Trim whitespace & unify case for strings
    if opts["trim_whitespace"]:
        text_cols = working.select_dtypes(include=["object"]).columns.tolist()
        trimmed = 0
        for c in text_cols:
            before_non_null = working[c].notna().sum()
            working[c] = working[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
            after_non_null = working[c].notna().sum()
            trimmed += 1
        if trimmed:
            report["operations"].append({"action": "trim_whitespace", "columns_processed": text_cols})

    # 2) Rename columns
    renamed_map = {}
    if opts["rename_columns"]:
        new_cols = []
        for c in working.columns:
            nc = c.strip().lower().replace(" ", "_")
            new_cols.append(nc)
            if nc != c:
                renamed_map[c] = nc
        if renamed_map:
            working.columns = new_cols
            report["operations"].append({"action": "rename_columns", "renamed_map": renamed_map, "count": len(renamed_map)})

    # 3) Convert data types (numeric-like -> numeric; strings -> datetime if detected)
    dtype_changes = []
    if opts["convert_dtypes"]:
        for c in list(working.columns):
            ser = working[c]
            conv_num, changed_num = _try_convert_numeric(ser)
            if changed_num:
                working[c] = conv_num
                dtype_changes.append({"column": c, "new_dtype": str(working[c].dtype), "reason": "numeric_conversion"})
                continue
            conv_dt, changed_dt = _try_convert_datetime(ser)
            if changed_dt:
                working[c] = conv_dt
                dtype_changes.append({"column": c, "new_dtype": "datetime64[ns]", "reason": "datetime_parsing"})
        if dtype_changes:
            report["operations"].append({"action": "convert_dtypes", "changes": dtype_changes})

    # 4) Drop empty columns
    if opts["drop_empty_columns"]:
        empty_cols = [c for c in working.columns if working[c].isna().all()]
        if empty_cols:
            working.drop(columns=empty_cols, inplace=True)
            report["operations"].append({"action": "drop_empty_columns", "columns_dropped": empty_cols, "count": len(empty_cols)})

    # 5) Drop constant columns
    if opts["drop_constant_columns"]:
        constant_cols = []
        for c in working.columns:
            if working[c].nunique(dropna=True) <= 1:
                constant_cols.append(c)
        if constant_cols:
            working.drop(columns=constant_cols, inplace=True)
            report["operations"].append({"action": "drop_constant_columns", "columns_dropped": constant_cols, "count": len(constant_cols)})

    # 6) Remove duplicates
    duplicates_removed = 0
    if opts["remove_duplicates"]:
        before = working.shape[0]
        if opts["duplicates_subset"]:
            working = working.drop_duplicates(subset=opts["duplicates_subset"])
        else:
            working = working.drop_duplicates()
        after = working.shape[0]
        duplicates_removed = before - after
        report["operations"].append({"action": "drop_duplicates", "removed_rows": int(duplicates_removed), "subset": opts["duplicates_subset"]})

    # 7) Handle outliers (IQR only for now)
    outliers_removed_total = 0
    if opts["handle_outliers"]:
        method = opts["handle_outliers"].get("method", "iqr")
        action = opts["handle_outliers"].get("action", "remove")
        multiplier = float(opts["handle_outliers"].get("multiplier", 1.5))
        if method == "iqr" and action in ("remove", "clip"):
            numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
            mask_any_outlier = pd.Series(False, index=working.index)
            for c in numeric_cols:
                m = _iqr_outliers_mask(working[c], multiplier=multiplier)
                mask_any_outlier = mask_any_outlier | m
            if action == "remove":
                before = working.shape[0]
                working = working.loc[~mask_any_outlier].copy()
                after = working.shape[0]
                outliers_removed_total = before - after
                report["operations"].append({"action": "outlier_removal", "method": "iqr", "multiplier": multiplier, "removed_rows": int(outliers_removed_total)})
            else:  # clip
                for c in working.select_dtypes(include=[np.number]).columns:
                    m = _iqr_outliers_mask(working[c], multiplier=multiplier)
                    if m.any():
                        q1 = working[c].quantile(0.25)
                        q3 = working[c].quantile(0.75)
                        low = q1 - multiplier * (q3 - q1)
                        high = q3 + multiplier * (q3 - q1)
                        working[c] = working[c].clip(lower=low, upper=high)
                report["operations"].append({"action": "outlier_clip", "method": "iqr", "multiplier": multiplier})

    # 8) Handle missing values
    filled_cells = 0
    dropped_rows = 0
    if opts["missing_strategy"] == "drop":
        before = working.shape[0]
        working = working.dropna()
        after = working.shape[0]
        dropped_rows = before - after
        report["operations"].append({"action": "drop_missing_rows", "rows_dropped": int(dropped_rows)})
    else:  # fill
        numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        for c in numeric_cols:
            strategy = opts["fill_numeric"]
            if strategy == "mean":
                val = working[c].mean()
            elif strategy == "median":
                val = working[c].median()
            elif strategy == "zero":
                val = 0
            else:
                val = strategy
            before_na = working[c].isna().sum()
            if before_na > 0:
                working[c].fillna(val, inplace=True)
                filled_cells += before_na
        obj_cols = working.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in obj_cols:
            strategy = opts["fill_categorical"]
            if strategy == "mode":
                try:
                    val = working[c].mode(dropna=True).iloc[0]
                except Exception:
                    val = ""
            elif strategy == "constant":
                val = ""
            else:
                val = strategy
            before_na = working[c].isna().sum()
            if before_na > 0:
                working[c].fillna(val, inplace=True)
                filled_cells += before_na
        if filled_cells > 0:
            report["operations"].append({"action": "fill_missing", "filled_cells": int(filled_cells)})

    # 9) Encode categoricals
    encoded_columns = []
    if opts["encode_categoricals"] == "label":
        cat_cols = working.select_dtypes(include=["object", "category"]).columns.tolist()
        le = LabelEncoder()
        for c in cat_cols:
            try:
                if working[c].nunique(dropna=True) <= max(1, min(1000, int(working.shape[0] * 0.8))):
                    working[c] = working[c].astype(str).fillna("__nan__")
                    working[c] = le.fit_transform(working[c])
                    encoded_columns.append(c)
            except Exception:
                continue
        if encoded_columns:
            report["operations"].append({"action": "label_encode", "columns_encoded": encoded_columns})

    # Final shape & summary
    report["cleaned_shape"] = list(working.shape)
    report["rows_processed"] = int(report["original_shape"][0])
    report["rows_after"] = int(working.shape[0])
    report["duplicates_removed"] = int(duplicates_removed)
    report["outliers_removed"] = int(outliers_removed_total)
    report["filled_cells"] = int(filled_cells)
    report["columns_renamed"] = len(renamed_map) if renamed_map else 0

    # 🔹 changed line here
    PREVIEW_ROWS = min(len(working), 1000)
    preview = working.head(PREVIEW_ROWS).copy()

    for c in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[c]):
            preview[c] = preview[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    report["preview"] = preview.to_dict(orient="records")

    return working, report

# -------------------------
# Utility: write report to json
# -------------------------
def save_report(report: Dict[str, Any], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

# -------------------------
# Example usage (not executed on import)
# -------------------------
if __name__ == "__main__":
    # small local sanity check if run directly
    sample = pd.DataFrame({
        "Enrollment ID": [1, 2, 3, 3, None],
        "Student ID": [" 193", "111", "132", "132", "999"],
        "Course ID": ["198", "63", "187", "187", "187"],
        "Semester": ["Summer 2024", "Spring 2024", "Summer 2024", "Summer 2024", None],
        "Grade": ["A", "D", "F", "F", "B"]
    })
    print("ANALYZE:")
    print(json.dumps(analyze_dataframe(sample), indent=2, default=str))
    cleaned, r = clean_dataframe(sample, options={
        "remove_duplicates": True,
        "missing_strategy": "fill",
        "fill_numeric": "mean",
        "fill_categorical": "mode",
        "handle_outliers": None,
        "encode_categoricals": "label"
    })
    print("\nREPORT:")
    print(json.dumps(r, indent=2, default=str))
