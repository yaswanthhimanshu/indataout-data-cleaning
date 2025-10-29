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

# -------------------------
# Data Quality Scoring Helper
# -------------------------
def data_quality_score(df: pd.DataFrame) -> float:
    """
    Compute a simple Data Quality Score (0–100).
    Based on missing %, duplicates %, outliers %, and constant column %.
    """
    if df.empty:
        return 0.0

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    # Missing values
    missing_pct = (df.isna().sum().sum() / total_cells) * 100 if total_cells else 0

    # Duplicates
    dup_pct = (df.duplicated().sum() / n_rows) * 100 if n_rows else 0

    # Constant columns
    const_col_pct = (df.nunique(dropna=False) <= 1).sum() / n_cols * 100

    # Outliers (numeric columns only)
    num_df = df.select_dtypes(include=np.number)
    outlier_pct = 0
    if not num_df.empty:
        z = np.abs((num_df - num_df.mean()) / num_df.std(ddof=0))
        outlier_pct = (z > 3).sum().sum() / (num_df.size) * 100

    # Weighted score (you can tune weights)
    dq = 100 - (0.5 * missing_pct + 0.2 * dup_pct + 0.2 * outlier_pct + 0.1 * const_col_pct)
    dq = max(0, min(100, dq))
    return round(dq, 2)



# -------------------------
# Data type conversion helper
# -------------------------
def convert_dtypes(df: pd.DataFrame, dtype_map: Optional[Dict[str, str]] = None):
    """
    dtype_map: optional mapping {col_name: 'auto'|'string'|'numeric'|'datetime'|'category'|'bool'}
    If dtype_map is None, 'auto' behavior is attempted for all columns.
    Returns (df_converted, meta)
    meta: {col: {'from': old_dtype, 'to': chosen, 'coerced': n, 'success': bool}}
    """
    df = df.copy()
    meta: Dict[str, Dict[str, Any]] = {}
    cols = list(df.columns)

    # default: all auto
    if dtype_map is None:
        dtype_map = {c: "auto" for c in cols}

    def _to_numeric_safe(s: pd.Series):
        conv = pd.to_numeric(s, errors="coerce")
        coerced = int(s.notna().sum() - conv.notna().sum()) if s.notna().any() else 0
        return conv, coerced

    def _to_datetime_safe(s: pd.Series):
        conv = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        coerced = int(s.notna().sum() - conv.notna().sum()) if s.notna().any() else 0
        return conv, coerced

    for c in cols:
        choice = dtype_map.get(c, "auto")
        old_dtype = str(df[c].dtype)
        info = {"from": old_dtype, "to": choice, "coerced": 0, "success": True}
        try:
            if choice == "auto":
                # prefer numeric if mostly numeric, else datetime if mostly datetime, else keep string
                conv_num, coerced_num = _to_numeric_safe(df[c])
                frac_num = conv_num.notna().sum() / max(1, len(df[c]))
                if frac_num >= 0.9:
                    df[c] = conv_num
                    info.update({"to": "numeric", "coerced": int(coerced_num)})
                else:
                    conv_dt, coerced_dt = _to_datetime_safe(df[c])
                    frac_dt = conv_dt.notna().sum() / max(1, len(df[c]))
                    if frac_dt >= 0.9:
                        df[c] = conv_dt
                        info.update({"to": "datetime", "coerced": int(coerced_dt)})
                    else:
                        # leave as-string/object
                        df[c] = df[c].astype(object)
                        info.update({"to": "string", "coerced": 0})
            elif choice == "numeric":
                conv, coerced = _to_numeric_safe(df[c])
                df[c] = conv
                info.update({"to": "numeric", "coerced": int(coerced)})
            elif choice == "datetime":
                conv, coerced = _to_datetime_safe(df[c])
                df[c] = conv
                info.update({"to": "datetime", "coerced": int(coerced)})
            elif choice == "string":
                df[c] = df[c].astype(object)
                info.update({"to": "string"})
            elif choice == "category":
                df[c] = df[c].astype("category")
                info.update({"to": "category"})
            elif choice == "bool":
                mapped = df[c].map({
                    'true': True, 'True': True, 'TRUE': True,
                    'false': False, 'False': False, 'FALSE': False,
                    'yes': True, 'no': False, 'y': True, 'n': False, '1': True, '0': False
                }).where(df[c].notna(), df[c])
                df[c] = mapped.astype("boolean")
                info.update({"to": "bool"})
            else:
                info.update({"success": False, "note": "unknown_choice"})
        except Exception as e:
            info.update({"success": False, "note": str(e)})
        meta[c] = info

    return df, meta


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
# Fill Categorical Helper
# -------------------------
def fill_categorical(
    df: pd.DataFrame,
    strategy: str = "mode",
    constant: Optional[str] = None,
    columns: Optional[List[str]] = None,
    knn_neighbors: int = 5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill missing values in categorical/text columns.
    Supported strategies: 'mode', 'constant', 'drop', 'knn'
    """
    df = df.copy()
    meta = {"strategy": strategy, "filled_cells": 0, "details": {}}

    if columns:
        cat_cols = [c for c in columns if c in df.columns]
    else:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        return df, meta

    if strategy == "drop":
        before = len(df)
        df = df.dropna(subset=cat_cols)
        meta["rows_dropped"] = before - len(df)
        return df, meta

    elif strategy == "constant":
        fill_val = constant if constant else "Unknown"
        for c in cat_cols:
            missing_before = df[c].isna().sum()
            df[c] = df[c].fillna(fill_val)
            meta["details"][c] = {"method": "constant", "filled": int(missing_before)}
            meta["filled_cells"] += int(missing_before)
        return df, meta

    elif strategy == "mode":
        for c in cat_cols:
            try:
                mode_val = df[c].mode(dropna=True).iloc[0]
            except Exception:
                mode_val = "Unknown"
            missing_before = df[c].isna().sum()
            df[c] = df[c].fillna(mode_val)
            meta["details"][c] = {"method": "mode", "filled": int(missing_before), "value": str(mode_val)}
            meta["filled_cells"] += int(missing_before)
        return df, meta

    else:
        return df, meta



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
        # --- auto data type conversion (uses options dtype_map if provided, else auto-detect)
    try:
        dtype_map = options.get("dtype_map", None)
        working, dtype_meta = convert_dtypes(working, dtype_map=dtype_map)
        if dtype_meta:
            report.setdefault("operations", []).append({"action": "convert_dtypes", "changes": dtype_meta})
    except Exception as e:
        report.setdefault("operations", []).append({"action": "convert_dtypes_failed", "error": str(e)})


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

    # 4) Drop empty columns (track names + count for the report)
    if opts["drop_empty_columns"]:
        empty_cols = [c for c in working.columns if working[c].isna().all()]
        if empty_cols:
            # drop them from the working dataframe
            working.drop(columns=empty_cols, inplace=True)
            # record operation (existing)
            report["operations"].append({
                "action": "drop_empty_columns",
                "columns_dropped": empty_cols,
                "count": len(empty_cols)
            })
            # also expose friendly top-level report fields used by templates
            report["empty_columns_removed"] = int(len(empty_cols))
            report["empty_columns"] = list(empty_cols)
        else:
            # explicitly set zero/empty list so template logic is reliable
            report["empty_columns_removed"] = 0
            report["empty_columns"] = []

    

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
       # Handle categorical missing values (new helper)
        working, cat_meta = fill_categorical(
            working,
            strategy=opts["fill_categorical"],
            constant=options.get("categorical_constant", None),
            columns=options.get("categorical_columns", None),
            knn_neighbors=opts["knn_k"]
        )
        filled_cells += cat_meta.get("filled_cells", 0)
        report["operations"].append({
            "action": "fill_categorical",
            "strategy": cat_meta["strategy"],
            "filled_cells": cat_meta["filled_cells"],
            "details": cat_meta.get("details", {})
        })


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
    # ensure empty_columns keys exist for templates even if drop_empty_columns was False
    if "empty_columns_removed" not in report:
        report["empty_columns_removed"] = 0
    if "empty_columns" not in report:
        report["empty_columns"] = []


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

#END