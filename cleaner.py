import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

# Added imports for advanced cleaning features
from sklearn.impute import KNNImputer
try:
    from rapidfuzz import fuzz
    _FUZZ_FUNC = lambda a, b: fuzz.ratio(a, b)
except Exception:
    try:
        from thefuzz import fuzz
        _FUZZ_FUNC = lambda a, b: fuzz.ratio(a, b)
    except Exception:
        _FUZZ_FUNC = None


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


# Helper: fuzzy text clustering (used for grouping similar string values)
def _build_text_clusters(unique_vals, scorer, threshold=85):
    """Group similar strings using fuzzy similarity."""
    if scorer is None:
        return {}, {}
    remaining = set(unique_vals)
    clusters = {}
    mapping = {}
    for val in sorted(unique_vals, key=lambda x: (-len(str(x)), str(x))):
        if val not in remaining:
            continue
        rep = val
        clusters[rep] = [rep]
        remaining.remove(rep)
        for other in list(remaining):
            try:
                score = scorer(str(rep), str(other))
            except Exception:
                score = 0
            if score >= threshold:
                clusters[rep].append(other)
                mapping[other] = rep
                remaining.remove(other)
        mapping[rep] = rep
    return clusters, mapping


def fuzzy_cluster_column(df, col, threshold=85):
    # Detect and group similar text values in object columns (fix typos, abbreviations)
    # Uses fuzzy string similarity to merge values that are nearly identical
    # Example: "US", "U.S.", "United States" → "United States"
    if _FUZZ_FUNC is None or col not in df.columns:
        return df, None
    ser = df[col].dropna().astype(str)
    if ser.nunique() <= 1:
        return df, None
    clusters, mapping = _build_text_clusters(list(ser.unique()), _FUZZ_FUNC, threshold)
    if not mapping:
        return df, None
    freq = ser.value_counts().to_dict()
    final_map = {}
    for rep, members in clusters.items():
        best = max(members, key=lambda v: freq.get(v, 0))
        for m in members:
            final_map[m] = best
    df[col] = df[col].apply(lambda x: final_map.get(str(x), x) if pd.notna(x) else x)
    return df, {"column": col, "merged_groups": len(clusters), "unique_after": df[col].nunique()}


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

    PREVIEW_ROWS = min(len(df), 1000)
    preview_df = df.head(PREVIEW_ROWS).copy()
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
    """
    if options is None:
        options = {}

    # defaults
    opts = {
        "remove_duplicates": options.get("remove_duplicates", True),
        "duplicates_subset": options.get("duplicates_subset", None),
        "missing_strategy": options.get("missing_strategy", "fill"),
        "fill_numeric": options.get("fill_numeric", "mean"),
        "fill_categorical": options.get("fill_categorical", "mode"),
        "drop_empty_columns": options.get("drop_empty_columns", True),
        "rename_columns": options.get("rename_columns", True),
        "trim_whitespace": options.get("trim_whitespace", True),
        "convert_dtypes": options.get("convert_dtypes", True),
        "handle_outliers": options.get("handle_outliers", None),
        "encode_categoricals": options.get("encode_categoricals", False),
        "drop_constant_columns": options.get("drop_constant_columns", True),
        "enable_fuzzy": options.get("enable_fuzzy", False),
        "fuzzy_threshold": options.get("fuzzy_threshold", 88),
        "knn_k": int(options.get("knn_k", 5))
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

    # Apply fuzzy text clustering if enabled (only affects text columns)
    if opts["enable_fuzzy"]:
        text_cols = working.select_dtypes(include=["object", "category"]).columns.tolist()
        fuzzy_ops = []
        for c in text_cols:
            try:
                working, info = fuzzy_cluster_column(working, c, threshold=opts["fuzzy_threshold"])
                if info:
                    fuzzy_ops.append(info)
            except Exception:
                continue
        if fuzzy_ops:
            report["operations"].append({"action": "fuzzy_cluster", "details": fuzzy_ops})

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
            else:
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

    elif opts["missing_strategy"] == "knn":
        # Handle missing values using KNN imputation (model-based filling)
        # Estimates missing numeric values using nearest rows (based on other features)
        # Runs only when user selects missing_strategy = "knn"
        num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=max(1, opts["knn_k"]))
            before_na = working[num_cols].isna().sum().sum()
            try:
                arr = imputer.fit_transform(working[num_cols])
                working[num_cols] = pd.DataFrame(arr, columns=num_cols, index=working.index)
                after_na = working[num_cols].isna().sum().sum()
                filled_cells = before_na - after_na
                report["operations"].append({
                    "action": "knn_impute",
                    "columns": num_cols,
                    "k": opts["knn_k"],
                    "filled_cells": int(filled_cells)
                })
            except Exception as e:
                report["operations"].append({
                    "action": "knn_impute_failed",
                    "reason": str(e)
                })

    else:
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