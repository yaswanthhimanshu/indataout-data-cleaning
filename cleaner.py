"""DataFrame transformation and quality-scoring engine for inDataOut.

The functions in this module operate on pandas DataFrames and return copies or
new DataFrames plus operation metadata for the Flask layer to persist. They do
not own sessions, database rows, project files, or downloads.

Correctness rules implemented here:
* whitespace trimming touches only string values in mixed-type columns;
* strict numeric helpers avoid silent loss where used, while legacy dtype
  conversion branches report coercions for review;
* placeholders are detected separately from replacement and replacement is
  explicit;
* label encoding preserves missing values unless the caller opts into a
  missing category;
* quality scoring is read-only, heuristic, and not proof that cleaning improved
  the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
import logging
import warnings



# -------------------------
# Helpers
# -------------------------
def _try_convert_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Safely convert a Series only when every non-null value converts."""
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    
    # Only attempt conversion if series has non-null values
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return series, False
    
    conv = pd.to_numeric(non_null_series, errors='coerce')
    if int(conv.isna().sum()) > 0:
        return series, False
    full_conv = pd.to_numeric(series, errors='coerce')
    changed = not full_conv.equals(series) and full_conv.notna().sum() > 0
    return full_conv, changed

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


def detect_outliers(df: pd.DataFrame, multiplier: float = 1.5, max_rows: int = 100, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect IQR-flagged outlier values in numeric columns without mutating data.
    Returns a dict with:
      - outlier_rows: list of {row_index, row_position, column, value, row_data}
      - total_count: total number of outlier values found
      - columns_checked: list of numeric columns checked
    If `columns` is provided, only check those columns (must be numeric).
    """
    df = df.copy()
    
    # Create a mapping from position to original index
    pos_to_orig_idx = {i: idx for i, idx in enumerate(df.index)}
    
    # All numeric columns in the dataframe
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # If caller specified columns, filter to only valid numeric ones
    if columns:
        numeric_cols = [c for c in columns if c in all_numeric_cols]
    else:
        numeric_cols = all_numeric_cols

    # Collect all IQR-flagged values with their deviation score first,
    # then sort by deviation magnitude descending so the most extreme outliers
    # appear at the top of the list (just like .sort_values(ascending=False) in analysis).
    raw_candidates = []
    outlier_pairs_seen = set()

    for col in numeric_cols:
        # Compute IQR bounds for deviation scoring
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low  = q1 - multiplier * iqr
        high = q3 + multiplier * iqr

        mask = _iqr_outliers_mask(df[col], multiplier=multiplier)
        outlier_index_labels = df.index[mask].tolist()

        for idx_label in outlier_index_labels:
            pair_key = (idx_label, col)
            if pair_key in outlier_pairs_seen:
                continue
            outlier_pairs_seen.add(pair_key)

            raw_val = df.loc[idx_label, col]
            if pd.isna(raw_val):
                continue
            val_float = float(raw_val)

            # Deviation score = how many IQR-widths outside the fence
            # (positive = above upper fence, negative = below lower fence)
            if val_float > high:
                deviation = (val_float - high) / max(iqr, 1e-9)
            else:
                deviation = (low - val_float) / max(iqr, 1e-9)

            row_position = int(df.index.get_loc(idx_label))
            row_data = df.loc[idx_label].to_dict()
            row_data_str = {k: str(v) if pd.notna(v) else '' for k, v in row_data.items()}
            if isinstance(idx_label, (int, np.integer)):
                row_index_value = int(idx_label)
                display_row_number = row_index_value + 1
            else:
                row_index_value = str(idx_label)
                display_row_number = row_position + 1

            raw_candidates.append({
                'row_index':    row_index_value,
                'row_position': row_position,
                'row_number':   display_row_number,
                'column':       col,
                'value':        val_float,
                'deviation':    deviation,   # for sorting; not sent to frontend
                'iqr_low':      float(low),
                'iqr_high':     float(high),
                'row_data':     row_data_str,
            })

    # Sort: largest deviation first (most extreme outlier at top)
    raw_candidates.sort(key=lambda x: x['deviation'], reverse=True)

    # Trim to max_rows and strip internal sorting fields
    outlier_rows = []
    for candidate in raw_candidates[:max_rows]:
        outlier_rows.append({
            'row_index':    candidate['row_index'],
            'row_position': candidate['row_position'],
            'row_number':   candidate['row_number'],
            'column':       candidate['column'],
            'value':        candidate['value'],
            'iqr_low':      candidate['iqr_low'],
            'iqr_high':     candidate['iqr_high'],
            'row_data':     candidate['row_data'],
        })

    return {
        'outlier_rows':       outlier_rows,
        'total_count':        len(outlier_pairs_seen),
        'columns_checked':    numeric_cols,
        'all_numeric_columns': all_numeric_cols,
        'max_displayed':      max_rows,
        'index_mapping':      pos_to_orig_idx,
    }

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
    if total_cells == 0:
        return 0.0

    placeholder_tokens = {
        "", "unknown", "not applicable", "n/a", "na", "null", "none",
        "nan", "inf", "infinity", "-", "--", "---", "?", "??", "???",
        "missing", "undefined",
    }

    def _text_series(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("").str.strip()

    def _placeholder_mask(series: pd.Series) -> pd.Series:
        text = _text_series(series).str.lower()
        return series.isna() | text.isin(placeholder_tokens) | text.str.fullmatch(r"#+", na=False)

    def _numeric_candidate_column(col_name: str, series: pd.Series) -> bool:
        name = str(col_name).strip().lower()
        numeric_name_parts = (
            "duration", "score", "rating", "vote", "income", "revenue",
            "amount", "price", "cost", "budget", "gross", "year", "count",
            "number", "qty", "quantity", "total", "percent", "percentage",
        )
        if any(part in name for part in numeric_name_parts) and not name.endswith("_id") and name != "id":
            return True
        non_placeholder = series[~_placeholder_mask(series)]
        if len(non_placeholder) < 5:
            return False
        text = _text_series(non_placeholder)
        digit_ratio = text.str.contains(r"\d", regex=True, na=False).mean()
        alpha_ratio = text.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
        return digit_ratio >= 0.85 and alpha_ratio <= 0.2

    def _coerce_numeric_like(series: pd.Series) -> pd.Series:
        text = _text_series(series).str.lower()
        text = text.str.replace(r"[$€£₹,\s]", "", regex=True)
        thousands_dot = text.str.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?", na=False)
        text = text.where(~thousands_dot, text.str.replace(".", "", regex=False))
        return pd.to_numeric(text, errors="coerce")

    # Missing values
    missing_pct = (df.isna().sum().sum() / total_cells) * 100

    # Duplicates
    dup_pct = (df.duplicated().sum() / n_rows) * 100 if n_rows else 0

    # Constant columns: only count columns that contain at least one real value.
    # Fully empty columns are handled by the missing-values/empty-column checks.
    non_empty_cols = df.columns[~df.isna().all()]
    const_col_pct = (df[non_empty_cols].nunique(dropna=True) <= 1).sum() / n_cols * 100 if n_cols else 0

    placeholder_cells = 0
    numeric_issue_cells = 0
    for col in df.columns:
        placeholder_mask = _placeholder_mask(df[col]) & ~df[col].isna()
        placeholder_cells += int(placeholder_mask.sum())
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if _numeric_candidate_column(col, df[col]):
            coerced = _coerce_numeric_like(df[col])
            invalid_numeric = (
                (~placeholder_mask)
                & _text_series(df[col]).str.contains(r"\d", regex=True, na=False)
                & coerced.isna()
            )
            numeric_issue_cells += int(invalid_numeric.sum())

    placeholder_pct = (placeholder_cells / total_cells) * 100
    numeric_issue_pct = (numeric_issue_cells / total_cells) * 100
    row_placeholder_pct = 0
    if n_rows:
        placeholder_frame = pd.DataFrame({c: _placeholder_mask(df[c]) for c in df.columns})
        row_placeholder_pct = (placeholder_frame.mean(axis=1) >= 0.8).mean() * 100

    # Outliers (numeric columns only)
    num_df = df.select_dtypes(include=np.number)
    outlier_pct = 0
    if not num_df.empty:
        # Use the same IQR rule as preview detection.
        outlier_count = 0
        for col in num_df.columns:
            if len(num_df[col].dropna()) > 2:  # Need at least 3 values to calculate outliers
                Q1 = num_df[col].quantile(0.25)
                Q3 = num_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count += ((num_df[col] < lower_bound) | (num_df[col] > upper_bound)).sum()
        outlier_pct = (outlier_count / num_df.size) * 100 if num_df.size > 0 else 0

    dq = 100 - (
        0.45 * missing_pct +
        0.20 * placeholder_pct +
        0.20 * dup_pct +
        0.15 * numeric_issue_pct +
        0.15 * row_placeholder_pct +
        0.10 * outlier_pct +
        0.05 * const_col_pct
    )
    dq = max(0, min(100, dq))
    return round(dq, 2)


def data_quality_breakdown(df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Explainable heuristic quality metrics, not proof of correctness."""
    n_rows, n_cols = df.shape
    total_cells = max(1, n_rows * n_cols)
    placeholder_tokens = {
        "", "not applicable", "n/a", "na", "null", "nan", "inf", "infinity",
        "--", "---", "??", "???", "missing", "undefined",
    }

    def _placeholder_only(series: pd.Series) -> pd.Series:
        text = series.astype("string").fillna("").str.strip().str.lower()
        return series.notna() & (text.isin(placeholder_tokens) | text.str.fullmatch(r"#+", na=False))

    placeholder_cells = int(sum(_placeholder_only(df[c]).sum() for c in df.columns))
    duplicate_rows = int(df.duplicated().sum()) if n_rows else 0
    missing_cells = int(df.isna().sum().sum())
    constant_columns = int((df.nunique(dropna=True) <= 1).sum()) if n_cols else 0

    numeric_issue_cells = 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        non_placeholder = df[c][~_placeholder_only(df[c]) & df[c].notna()]
        if len(non_placeholder) == 0:
            continue
        text = non_placeholder.astype("string").str.strip()
        if text.str.contains(r"\d", regex=True, na=False).mean() >= 0.85:
            numeric_issue_cells += int(pd.to_numeric(text, errors="coerce").isna().sum())

    outlier_cells = 0
    num_df = df.select_dtypes(include=np.number)
    for c in num_df.columns:
        series = num_df[c].dropna()
        if len(series) <= 2:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_cells += int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

    original_rows = int(original_df.shape[0]) if original_df is not None else n_rows
    original_cols = int(original_df.shape[1]) if original_df is not None else n_cols
    return {
        "label": "Explainable heuristic data-quality summary",
        "warning": "Higher scores do not prove correctness; these checks do not encode business rules, keys, allowed values, integrity, or timeliness.",
        "components": {
            "completeness": {
                "formula": "100 * non_missing_cells / total_cells",
                "missing_cells": missing_cells,
                "total_cells": int(n_rows * n_cols),
                "score": round(100 * (1 - missing_cells / total_cells), 2),
            },
            "duplicate_row_rate": {
                "formula": "duplicate_rows / rows",
                "duplicate_rows": duplicate_rows,
                "rows": n_rows,
                "rate_percent": round(100 * duplicate_rows / n_rows, 2) if n_rows else 0.0,
            },
            "placeholder_rate": {
                "formula": "non-null placeholder-token cells / total_cells",
                "placeholder_cells": placeholder_cells,
                "rate_percent": round(100 * placeholder_cells / total_cells, 2),
            },
            "type_conformance_issues": {
                "formula": "numeric-looking text values that fail safe numeric parsing / total_cells",
                "issue_cells": numeric_issue_cells,
                "rate_percent": round(100 * numeric_issue_cells / total_cells, 2),
            },
            "outlier_candidate_rate": {
                "formula": "IQR candidate cells in numeric columns / numeric cells",
                "candidate_cells": outlier_cells,
                "rate_percent": round(100 * outlier_cells / max(1, int(num_df.size)), 2),
                "heuristic": True,
            },
            "constant_column_observation": {"constant_columns": constant_columns, "columns": n_cols},
            "row_retention": {
                "formula": "current_rows / original_rows",
                "current_rows": n_rows,
                "original_rows": original_rows,
                "retention_percent": round((n_rows / original_rows * 100) if original_rows else 100.0, 2),
            },
            "column_retention": {
                "formula": "current_columns / original_columns",
                "current_columns": n_cols,
                "original_columns": original_cols,
                "retention_percent": round((n_cols / original_cols * 100) if original_cols else 100.0, 2),
            },
        },
    }


# Canonical quality implementation. These later definitions intentionally
# replace the legacy score/breakdown functions above at module load time.
QUALITY_PLACEHOLDER_TOKENS = {
    "", "unknown", "not applicable", "n/a", "na", "null", "none",
    "nan", "inf", "infinity", "-", "--", "---", "?", "??", "???",
    "missing", "undefined",
}

QUALITY_WEIGHTS = {
    "missing_pct": 0.45,
    "placeholder_pct": 0.20,
    "duplicate_pct": 0.20,
    "numeric_issue_pct": 0.15,
    "row_placeholder_pct": 0.15,
    "outlier_pct": 0.10,
    "constant_column_pct": 0.05,
}


def _quality_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _quality_placeholder_mask(series: pd.Series, include_nulls: bool) -> pd.Series:
    text = _quality_text_series(series).str.lower()
    mask = text.isin(QUALITY_PLACEHOLDER_TOKENS) | text.str.fullmatch(r"#+", na=False)
    if include_nulls:
        return mask | series.isna()
    return mask & series.notna()


def _quality_numeric_candidate_column(col_name: str, series: pd.Series) -> bool:
    name = str(col_name).strip().lower()
    numeric_name_parts = (
        "duration", "score", "rating", "vote", "income", "revenue",
        "amount", "price", "cost", "budget", "gross", "year", "count",
        "number", "qty", "quantity", "total", "percent", "percentage",
    )
    if any(part in name for part in numeric_name_parts) and not name.endswith("_id") and name != "id":
        return True
    non_placeholder = series[~_quality_placeholder_mask(series, include_nulls=True)]
    if len(non_placeholder) < 5:
        return False
    text = _quality_text_series(non_placeholder)
    digit_ratio = text.str.contains(r"\d", regex=True, na=False).mean()
    alpha_ratio = text.str.contains(r"[A-Za-z]", regex=True, na=False).mean()
    return digit_ratio >= 0.85 and alpha_ratio <= 0.2


def _quality_coerce_numeric_like(series: pd.Series) -> pd.Series:
    text = _quality_text_series(series).str.lower()
    text = text.str.replace(r"[$€£₹,\s]", "", regex=True)
    thousands_dot = text.str.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?", na=False)
    text = text.where(~thousands_dot, text.str.replace(".", "", regex=False))
    return pd.to_numeric(text, errors="coerce")


def data_quality_metrics(df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Canonical, read-only heuristic quality metrics for one complete DataFrame.

    The score is a weighted summary of explainable components, not a validation
    certificate. Nulls and placeholders are counted separately, row/column
    retention is compared to ``original_df`` when supplied, and deleting data
    can reduce the score even when it removes problematic values.
    """
    n_rows, n_cols = df.shape
    total_cells_raw = int(n_rows * n_cols)
    total_cells = max(1, total_cells_raw)

    missing_cells = int(df.isna().sum().sum())
    placeholder_cells = int(sum(_quality_placeholder_mask(df[c], include_nulls=False).sum() for c in df.columns))
    duplicate_rows = int(df.astype("string").duplicated().sum()) if n_rows else 0
    non_empty_cols = [c for c in df.columns if not df[c].isna().all()]
    constant_columns = int(sum(df[c].nunique(dropna=True) <= 1 for c in non_empty_cols)) if n_cols else 0

    numeric_issue_cells = 0
    for c in df.columns:
        series = df[c]
        placeholder_mask = _quality_placeholder_mask(series, include_nulls=False)
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if _quality_numeric_candidate_column(c, series):
            coerced = _quality_coerce_numeric_like(series)
            invalid_numeric = (
                (~placeholder_mask)
                & series.notna()
                & _quality_text_series(series).str.contains(r"\d", regex=True, na=False)
                & coerced.isna()
            )
            numeric_issue_cells += int(invalid_numeric.sum())

    outlier_cells = 0
    num_df = df.select_dtypes(include=np.number)
    for c in num_df.columns:
        series = num_df[c].dropna()
        if len(series) <= 2:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_cells += int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

    row_placeholder_rows = 0
    if n_rows and n_cols:
        placeholder_frame = pd.DataFrame({c: _quality_placeholder_mask(df[c], include_nulls=True) for c in df.columns})
        row_placeholder_rows = int((placeholder_frame.mean(axis=1) >= 0.8).sum())

    missing_pct = (missing_cells / total_cells) * 100
    placeholder_pct = (placeholder_cells / total_cells) * 100
    duplicate_pct = (duplicate_rows / n_rows) * 100 if n_rows else 0.0
    numeric_issue_pct = (numeric_issue_cells / total_cells) * 100
    row_placeholder_pct = (row_placeholder_rows / n_rows) * 100 if n_rows else 0.0
    outlier_pct = (outlier_cells / max(1, int(num_df.size))) * 100
    constant_column_pct = (constant_columns / n_cols) * 100 if n_cols else 0.0

    raw_percentages = {
        "missing_pct": missing_pct,
        "placeholder_pct": placeholder_pct,
        "duplicate_pct": duplicate_pct,
        "numeric_issue_pct": numeric_issue_pct,
        "row_placeholder_pct": row_placeholder_pct,
        "outlier_pct": outlier_pct,
        "constant_column_pct": constant_column_pct,
    }
    penalty = sum(QUALITY_WEIGHTS[k] * raw_percentages[k] for k in QUALITY_WEIGHTS)
    score = round(max(0, min(100, 100 - penalty)), 2) if total_cells_raw else 0.0

    original_rows = int(original_df.shape[0]) if original_df is not None else n_rows
    original_cols = int(original_df.shape[1]) if original_df is not None else n_cols
    return {
        "score": score,
        "score_formula": "100 - weighted penalty",
        "weights": QUALITY_WEIGHTS.copy(),
        "raw_percentages": {k: round(v, 6) for k, v in raw_percentages.items()},
        "label": "Explainable heuristic data-quality summary",
        "warning": "Higher scores do not prove correctness; these checks do not encode business rules, keys, allowed values, integrity, or timeliness.",
        "components": {
            "completeness": {
                "formula": "100 * non_missing_cells / total_cells",
                "missing_cells": missing_cells,
                "total_cells": total_cells_raw,
                "score": round(100 * (1 - missing_cells / total_cells), 2),
            },
            "duplicate_row_rate": {
                "formula": "duplicate_rows / rows",
                "duplicate_rows": duplicate_rows,
                "rows": n_rows,
                "rate_percent": round(duplicate_pct, 2),
            },
            "placeholder_rate": {
                "formula": "non-null placeholder-token cells / total_cells",
                "tokens": sorted(QUALITY_PLACEHOLDER_TOKENS),
                "placeholder_cells": placeholder_cells,
                "rate_percent": round(placeholder_pct, 2),
            },
            "type_conformance_issues": {
                "formula": "numeric-looking text values that fail safe numeric parsing / total_cells",
                "issue_cells": numeric_issue_cells,
                "rate_percent": round(numeric_issue_pct, 2),
            },
            "outlier_candidate_rate": {
                "formula": "IQR candidate cells in numeric columns / numeric cells",
                "candidate_cells": outlier_cells,
                "rate_percent": round(outlier_pct, 2),
                "heuristic": True,
            },
            "constant_column_observation": {
                "constant_columns": constant_columns,
                "columns": n_cols,
                "rate_percent": round(constant_column_pct, 2),
            },
            "row_placeholder_rate": {
                "formula": "rows with placeholders/nulls in at least 80% of columns / rows",
                "rows": row_placeholder_rows,
                "rate_percent": round(row_placeholder_pct, 2),
            },
            "row_retention": {
                "formula": "current_rows / original_rows",
                "current_rows": n_rows,
                "original_rows": original_rows,
                "retention_percent": round((n_rows / original_rows * 100) if original_rows else 100.0, 2),
            },
            "column_retention": {
                "formula": "current_columns / original_columns",
                "current_columns": n_cols,
                "original_columns": original_cols,
                "retention_percent": round((n_cols / original_cols * 100) if original_cols else 100.0, 2),
            },
        },
    }


def data_quality_score(df: pd.DataFrame) -> float:
    """Return only the canonical heuristic score for display compatibility."""
    return float(data_quality_metrics(df)["score"])


def data_quality_breakdown(df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Return the full explainable quality-score breakdown."""
    return data_quality_metrics(df, original_df=original_df)


def cleaning_completeness_score(report: Dict[str, Any]) -> float:
    """
    Score action execution, not data correctness or data quality.
    A completed run can be 100 even when the dataset still has values to review.
    """
    operations = report.get("operations", [])
    failed = [
        op for op in operations
        if str(op.get("action", "")).endswith("_failed") or op.get("success") is False
    ]
    if failed:
        return round(max(0, 100 - 20 * len(failed)), 2)
    return 100.0



# -------------------------
# Data type conversion helper
# -------------------------
def _best_datetime_parse(s: pd.Series) -> tuple:
    """
    Try several date parsers and keep the one that preserves the most values.

    Real-world exports are frequently *inconsistent* within a single column —
    e.g. some rows "1/5/2022" (day-first), others "5/1/2022" (month-first),
    others "2022-05-01" (ISO) — because they were typed by hand, exported
    from different systems, or pasted together from multiple sources.

    A single call to pd.to_datetime(s, errors="coerce") picks ONE format by
    inspecting a sample of the column and applies it to every row; any row
    that doesn't match that exact format silently becomes NaT. On a genuinely
    mixed-format column this can turn 50-60%+ of a column to missing values
    even though almost every value was a perfectly good, parseable date —
    just not in the one format pandas guessed.

    This tries several parsing strategies and keeps whichever one preserves
    the most originally-non-null values:
      1. format="mixed"              — parses each value independently
      2. format="mixed", dayfirst=True
      3. plain pd.to_datetime()      — single inferred format (old behavior)
      4. plain pd.to_datetime(dayfirst=True)

    format="mixed" requires pandas >= 2.0; falls back to just (3)/(4) on
    older pandas.

    Returns (converted_series, coerced_count, method_used).
    """
    non_null_before = s.notna()
    total_non_null = int(non_null_before.sum())
    if total_non_null == 0:
        return s, 0, "skipped_empty"

    candidates = []  # (label, converted_series)
    # Several of these calls (esp. the plain-inferred ones with no format=)
    # trigger pandas' "Could not infer format... falling back to dateutil"
    # UserWarning on genuinely mixed-format columns — which is exactly the
    # case this function exists to handle, so it fires constantly and spams
    # the console. We deliberately try the format anyway (that's the point:
    # keep whichever parser preserves the most values) and just silence
    # the noisy warning about it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for dayfirst in (False, True):
            try:
                conv = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
                candidates.append((f"mixed(dayfirst={dayfirst})", conv))
            except (TypeError, ValueError):
                # pandas < 2.0 doesn't support format="mixed"; skip this parser.
                pass
        for dayfirst in (False, True):
            conv = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
            candidates.append((f"inferred(dayfirst={dayfirst})", conv))

    def _success_count(conv):
        return int((non_null_before & conv.notna()).sum())

    best_label, best_conv = max(candidates, key=lambda pair: _success_count(pair[1]))
    coerced = total_non_null - _success_count(best_conv)
    return best_conv, int(coerced), best_label


def convert_dtypes(df: pd.DataFrame, dtype_map: Optional[Dict[str, str]] = None):
    """
    Return a converted DataFrame copy plus per-column conversion metadata.

    ``dtype_map`` is an optional mapping such as ``{"col": "datetime"}``.
    When a map is supplied, unmentioned columns are left untouched. Metadata
    records requested dtype, actual dtype, coercion count, rounding count where
    relevant, and failure notes so previews/reports can explain destructive
    conversions instead of presenting them as clean success.
    """
    df = df.copy()
    meta: Dict[str, Dict[str, Any]] = {}
    cols = list(df.columns)

    # Only fall back to "auto for every column" when no dtype_map was given
    # at all. When a dtype_map IS given (even a partial one — e.g. the user
    # only chose a type for one column), any column not mentioned in it
    # must be left completely untouched, not silently auto-converted.
    dtype_map_was_provided = dtype_map is not None
    if dtype_map is None:
        dtype_map = {c: "auto" for c in cols}

    def _to_numeric_safe(s: pd.Series):
        # Only attempt conversion if series has non-null values
        non_null_series = s.dropna()
        if len(non_null_series) == 0:
            return s, 0
        
        # Sample a portion of the data for testing conversion
        sample_size = min(10, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        # Test if conversion works on sample
        test_conversion = pd.to_numeric(sample, errors='coerce')
        
        # If most values in sample converted successfully, convert the whole series
        successful_conversions = test_conversion.notna().sum()
        if successful_conversions / len(sample) >= 0.95:  # At least 95% success rate
            conv = pd.to_numeric(s, errors="coerce")
            coerced = int(s.notna().sum() - conv.notna().sum()) if s.notna().any() else 0
            return conv, coerced
        else:
            # Return original series with 0 coerced count if conversion not advisable
            return s, 0

    def _to_datetime_safe(s: pd.Series):
        # NEVER convert numeric columns to datetime - that's the main bug!
        # Numbers like 142 get converted to 1970-01-01 00:00:00.000000142 (nanoseconds)
        if pd.api.types.is_numeric_dtype(s):
            return s, 0
        
        # Only attempt conversion if series has non-null string values
        non_null_series = s.dropna()
        if len(non_null_series) == 0:
            return s, 0
        
        # Check if values look like date strings (contain date separators)
        sample = non_null_series.head(10).astype(str)
        date_like_count = sum(1 for val in sample if any(sep in str(val) for sep in ['-', '/', ':']))
        if date_like_count < len(sample) * 0.5:  # Less than 50% look like dates
            return s, 0
        
        # Sample a portion of the data for testing conversion
        sample_size = min(10, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        # Test if conversion works on sample (using the same multi-format
        # logic as the real conversion below, so the decision to convert and
        # the actual conversion agree on how well the data parses)
        _, sample_coerced, _ = _best_datetime_parse(sample)
        successful_conversions = len(sample) - sample_coerced
        
        # If most values in sample converted successfully, convert the whole series
        if successful_conversions / len(sample) >= 0.7:  # At least 70% success rate
            conv, coerced, _method = _best_datetime_parse(s)
            return conv, coerced
        else:
            # Return original series with 0 coerced count if conversion not advisable
            return s, 0

    for c in cols:
        if dtype_map_was_provided and c not in dtype_map:
            # User made explicit choices for other columns and left this one
            # alone — leave it completely untouched (no entry in meta at all).
            continue
        choice = dtype_map.get(c, "auto")
        old_dtype = str(df[c].dtype)
        info = {
            "from": old_dtype,
            "requested": choice,
            "to": choice,
            "actual_dtype": old_dtype,
            "coerced": 0,
            "success": True,
        }
        try:
            if choice == "auto":
                # Prioritize: numeric > datetime > string
                # This prevents numeric columns like duration/score from being converted to datetime
                conv_num, coerced_num = _to_numeric_safe(df[c])
                # Check if numeric conversion actually worked (data changed AND is now numeric)
                actually_converted_num = (not conv_num.equals(df[c])) or (pd.api.types.is_numeric_dtype(conv_num) and not pd.api.types.is_numeric_dtype(df[c]))
                
                if actually_converted_num and coerced_num < len(df[c]):
                    df[c] = conv_num
                    info.update({"to": str(df[c].dtype), "coerced": int(coerced_num)})
                else:
                    # Only try datetime if numeric conversion failed
                    conv_dt, coerced_dt = _to_datetime_safe(df[c])
                    actually_converted_dt = (not conv_dt.equals(df[c])) or (pd.api.types.is_datetime64_any_dtype(conv_dt) and not pd.api.types.is_datetime64_any_dtype(df[c]))
                    
                    if actually_converted_dt and coerced_dt < len(df[c]):
                        df[c] = conv_dt
                        info.update({"to": str(df[c].dtype), "coerced": int(coerced_dt)})
                    else:
                        # leave as-string/object
                        df[c] = df[c].astype(object)
                        info.update({"to": str(df[c].dtype), "coerced": 0})
            elif choice == "numeric":
                # Kept for backward compatibility with any already-stored history
                # referencing this value. New selections use explicit "int"/"float".
                conv = pd.to_numeric(df[c], errors="coerce")
                coerced = int(df[c].notna().sum() - conv.notna().sum()) if df[c].notna().any() else 0
                df[c] = conv
                info.update({"to": str(df[c].dtype), "coerced": int(coerced)})
            elif choice == "int":
                conv = pd.to_numeric(df[c], errors="coerce")
                coerced = int(df[c].notna().sum() - conv.notna().sum()) if df[c].notna().any() else 0
                # Round non-whole values first — pandas refuses a direct cast to
                # Int64 if any value has a real decimal part (raises rather than
                # silently truncating). Report how many were rounded so the
                # Change Log states this plainly instead of changing values with
                # no record of it.
                rounded_mask = conv.notna() & (conv != conv.round())
                rounded_count = int(rounded_mask.sum())
                df[c] = conv.round().astype("Int64")
                info.update({"to": str(df[c].dtype), "coerced": int(coerced), "rounded": rounded_count})
            elif choice == "float":
                conv = pd.to_numeric(df[c], errors="coerce").astype("float64")
                coerced = int(df[c].notna().sum() - conv.notna().sum()) if df[c].notna().any() else 0
                df[c] = conv
                info.update({"to": str(df[c].dtype), "coerced": int(coerced)})
            elif choice == "datetime":
                conv, coerced, method_used = _best_datetime_parse(df[c])
                df[c] = conv
                info.update({"to": str(df[c].dtype), "coerced": int(coerced), "parse_method": method_used})
            elif choice == "string":
                df[c] = df[c].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
                info.update({"to": str(df[c].dtype)})
            elif choice == "category":
                df[c] = df[c].astype("category")
                info.update({"to": str(df[c].dtype)})
            elif choice == "bool":
                # Map string/numeric values to boolean
                bool_map = {
                    'true': True, 'True': True, 'TRUE': True, 't': True, 'T': True,
                    'false': False, 'False': False, 'FALSE': False, 'f': False, 'F': False,
                    'yes': True, 'Yes': True, 'YES': True, 'y': True, 'Y': True,
                    'no': False, 'No': False, 'NO': False, 'n': False, 'N': False,
                    '1': True, '0': False, 1: True, 0: False,
                    1.0: True, 0.0: False
                }
                # Map values and preserve NaNs
                original_non_missing = df[c].notna()
                mapped_values = df[c].map(bool_map)
                coerced = int((original_non_missing & mapped_values.isna()).sum())
                # Keep original missing values, then cast to pandas' NULLABLE boolean
                # dtype ("boolean", not numpy's plain "bool") — this is the only
                # boolean representation that can hold missing values at all, and
                # it means the column's real dtype now genuinely IS boolean, matching
                # what the dropdown and Change Log both claim happened.
                df[c] = mapped_values.where(df[c].notna(), np.nan).astype("boolean")
                info.update({"to": str(df[c].dtype), "coerced": coerced})
            else:
                info.update({"success": False, "note": "unknown_choice"})
        except Exception as e:
            info.update({"success": False, "note": str(e)})
        info["actual_dtype"] = str(df[c].dtype)
        meta[c] = info

    return df, meta




# -------------------------
# Fill Categorical Helper
# -------------------------
def fill_categorical(
    df: pd.DataFrame,
    strategy: str = "mode",
    constant: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill or drop missing values in selected categorical/text columns.

    Returns a DataFrame copy and metadata with affected rows/counts. ``drop``
    removes rows only when they are missing in the selected columns.
    """
    df = df.copy()
    meta = {"strategy": strategy, "filled_cells": 0, "details": {}, "skipped": []}

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
            missing_mask = df[c].isna()
            missing_before = missing_mask.sum()
            missing_index = df.index[missing_mask].tolist()
            if isinstance(df[c].dtype, pd.CategoricalDtype) and fill_val not in df[c].cat.categories:
                df[c] = df[c].cat.add_categories([fill_val])
            df[c] = df[c].fillna(fill_val)
            meta["details"][c] = {
                "method": "constant",
                "filled": int(missing_before),
                "value": str(fill_val),
                "rows": [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in missing_index],
                "row_ids": [str(i) for i in missing_index],
            }
            meta["filled_cells"] += int(missing_before)
        return df, meta

    elif strategy == "mode":
        for c in cat_cols:
            missing_mask = df[c].isna()
            missing_before = missing_mask.sum()
            missing_index = df.index[missing_mask].tolist()
            modes = df[c].mode(dropna=True)
            if modes.empty:
                meta["skipped"].append({
                    "column": c,
                    "strategy": "mode",
                    "missing": int(missing_before),
                    "reason": "Mode is unavailable because the column has no non-missing values.",
                    "rows": [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in missing_index],
                    "row_ids": [str(i) for i in missing_index],
                })
                continue
            mode_val = modes.iloc[0]
            df[c] = df[c].fillna(mode_val)
            meta["details"][c] = {
                "method": "mode",
                "filled": int(missing_before),
                "value": str(mode_val),
                "rows": [int(i) + 1 if isinstance(i, (int, np.integer)) else str(i) for i in missing_index],
                "row_ids": [str(i) for i in missing_index],
            }
            meta["filled_cells"] += int(missing_before)
        return df, meta

    else:
        return df, meta



# -------------------------
# Analysis (pre-clean) - what's messy
# -------------------------
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a read-only messiness summary for the UI.
    Keys returned:
      - shape: [rows, cols]
      - missing_count: int
      - missing_percent: float
      - duplicate_count: int
      - empty_columns: list
      - preview: df.head(10) as records (list of dicts)
      - dtypes: mapping column -> dtype (string)
    """
    # Missing-value controls describe actual nulls only. Placeholder detection
    # is separate and must not change the Missing Values counts.
    analysis_df = df.copy(deep=True)

    rows, cols = analysis_df.shape
    total_cells = rows * cols if cols > 0 else 1
    missing_count = int(analysis_df.isna().sum().sum())
    missing_percent = float(0 if total_cells == 0 else (missing_count / total_cells) * 100)
    duplicate_count = int(analysis_df.duplicated().sum())
    empty_columns = [c for c in analysis_df.columns if analysis_df[c].isna().all()]
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    missing_by_column = []
    for c in analysis_df.columns:
        col_missing = int(analysis_df[c].isna().sum())
        if col_missing > 0:
            missing_by_column.append({
                "column": c,
                "missing": col_missing,
                "percent": round((col_missing / rows) * 100, 2) if rows else 0,
            })

    def _display_value(value):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value

    MISSING_EDITOR_ROWS = min(len(df), 2000)
    missing_rows = []
    if missing_count > 0:
        missing_mask = analysis_df.isna().any(axis=1)
        missing_df = analysis_df.loc[missing_mask]
        for position, idx in enumerate(missing_df.index, start=1):
            if len(missing_rows) >= MISSING_EDITOR_ROWS:
                break
            row = missing_df.loc[idx]
            missing_cols = [c for c in analysis_df.columns if pd.isna(row[c])]
            row_position = int(analysis_df.index.get_loc(idx)) if idx in analysis_df.index else position - 1
            missing_rows.append({
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "row_number": row_position + 1,
                "data_row_number": row_position + 1,
                "missing_columns": missing_cols,
                "row_data": {c: _display_value(row[c]) for c in analysis_df.columns},
            })

    unnamed_columns = []
    for c in analysis_df.columns:
        if str(c).strip().lower().startswith("unnamed"):
            non_missing = int(analysis_df[c].notna().sum())
            missing = int(analysis_df[c].isna().sum())
            unnamed_columns.append({
                "column": c,
                "non_missing": non_missing,
                "missing": missing,
                "percent_missing": round((missing / rows) * 100, 2) if rows else 0,
                "safe_to_drop": non_missing == 0,
            })

    constant_columns = [
        c for c in analysis_df.columns
        if analysis_df[c].notna().any() and analysis_df[c].nunique(dropna=True) <= 1
    ]
    whitespace_cells = 0
    for c in analysis_df.select_dtypes(include=["object"]).columns:
        whitespace_cells += int(analysis_df[c].map(
            lambda value: isinstance(value, str) and value != value.strip()
        ).sum())

    PREVIEW_ROWS = min(len(df), 100)
    preview_df = analysis_df.head(PREVIEW_ROWS).copy()
    for c in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[c]):
            preview_df[c] = preview_df[c].dt.strftime('%Y-%m-%d %H:%M:%S')

    return {
        "shape": [rows, cols],
        "missing_count": missing_count,
        "missing_percent": round(missing_percent, 2),
        "duplicate_count": duplicate_count,
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "whitespace_cells": whitespace_cells,
        "dtypes": dtypes,
        "numeric_columns": analysis_df.select_dtypes(include=[np.number]).columns.tolist(),
        "missing_by_column": missing_by_column,
        "missing_rows": missing_rows,
        "missing_rows_display_limit": MISSING_EDITOR_ROWS,
        "missing_rows_total": int(missing_mask.sum()) if missing_count > 0 else 0,
        "unnamed_columns": unnamed_columns,
        "preview_display_limit": PREVIEW_ROWS,
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
    Clean a DataFrame copy according to explicit options.

    The input DataFrame is not modified. The returned report is transformation
    metadata for previews, history manifests, final reports, and quality
    summaries; Flask owns persistence and transaction boundaries.
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
        "rename_columns": options.get("rename_columns", False),
        "trim_whitespace": options.get("trim_whitespace", True),
        "convert_dtypes": options.get("convert_dtypes", True),
        "handle_outliers": options.get("handle_outliers", None),
        "encode_categoricals": options.get("encode_categoricals", False),
        "drop_constant_columns": options.get("drop_constant_columns", True),
        "column_renames": options.get("column_renames", {}),
        "cell_updates": options.get("cell_updates", []),
        "missing_cell_updates": options.get("missing_cell_updates", []),
        "drop_selected_columns": options.get("drop_selected_columns", []),

        "selected_outlier_rows": options.get("selected_outlier_rows", []),
        "selected_outlier_records": options.get("selected_outlier_records", [])
    }

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "original_shape": list(df.shape),
        "operations": []
    }

    working = df.copy(deep=True)

    def _original_row_number(label):
        if isinstance(label, (int, np.integer)):
            return int(label) + 1
        try:
            return int(working.index.get_loc(label)) + 1
        except Exception:
            return str(label)

    def _row_numbers(labels):
        return [_original_row_number(label) for label in list(labels)]

    # 0) Replace placeholder values only when explicitly requested.
    placeholder_values = options.get("placeholder_tokens") or [
        '####', '#####', '######', '#######',  # Excel display errors
        '',  # blank or spaces-only cells
        'N/A', 'n/a', 'NA', 'na', 'N.A.', 'n.a.',  # Common NA representations
        'NULL', 'null', 'Null', 'None', 'none', 'NONE',  # NULL values
        '--', '---', '??', '???',  # Placeholder symbols
        'NaN', 'nan', 'NAN', '#N/A', '#NA', '#VALUE!', '#REF!', '#DIV/0!',  # Excel errors
        'undefined', 'UNDEFINED', 'missing', 'MISSING', 'unknown', 'UNKNOWN',
        '..', '...', '**', '***'  # Other placeholders
    ]
    placeholder_set = set(placeholder_values)
    if options.get("replace_placeholders", False):
        replacements = []
        for col in working.columns:
            if working[col].dtype == 'object':
                stripped = working[col].astype(str).str.strip()
                mask = working[col].notna() & (working[col].isin(placeholder_set) | stripped.isin(placeholder_set))
                for idx in working.index[mask]:
                    replacements.append({
                        "row_index": idx,
                        "row_number": _original_row_number(idx),
                        "column": col,
                        "before": str(working.at[idx, col]),
                        "after": None,
                    })
                working[col] = working[col].where(~mask)
        if replacements:
            report["operations"].append({
                "action": "replace_placeholders",
                "tokens": list(placeholder_values),
                "count": len(replacements),
                "replacements": replacements[:200],
                "displayed_change_count": min(len(replacements), 200),
                "total_change_count": len(replacements),
                "is_sample": len(replacements) > 200,
            })

    # Drop reviewed columns by their exact identifiers before any rename can
    # change those identifiers. Reject the whole request if any selection is
    # stale so a partial drop cannot be reported as a successful Apply.
    requested_columns = list(dict.fromkeys(opts["drop_selected_columns"]))
    unresolved_columns = [column for column in requested_columns if column not in working.columns]
    if unresolved_columns:
        raise ValueError(f"Selected columns not found: {unresolved_columns}")
    if requested_columns:
        working.drop(columns=requested_columns, inplace=True)
        report["operations"].append({
            "action": "drop_selected_columns",
            "columns_dropped": requested_columns,
            "count": len(requested_columns),
        })

    # 1) Rename columns first (before other remaining operations)
    renamed_map = {}
    if opts["rename_columns"]:
        new_cols = []
        for c in working.columns:
            # Clean up encoding issues and normalize column names
            nc = str(c).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            nc = nc.strip().lower().replace(" ", "_")
            # Remove any non-ASCII characters that might have encoding issues
            nc = ''.join(ch if ord(ch) < 128 else '_' for ch in nc)
            new_cols.append(nc)
            if nc != c:
                renamed_map[c] = nc
        if renamed_map:
            if len(set(new_cols)) != len(new_cols):
                collisions = {}
                for old, new in zip(list(working.columns), new_cols):
                    collisions.setdefault(new, []).append(old)
                collisions = {new: olds for new, olds in collisions.items() if len(olds) > 1}
                raise ValueError(f"Column normalization collision: {collisions}")
            working.columns = new_cols
            report["operations"].append({"action": "rename_columns", "renamed_map": renamed_map, "count": len(renamed_map)})

    def _apply_cell_updates(updates, action):
        applied = []
        unresolved = []
        for update in updates:
            col = update.get("column")
            if col not in working.columns:
                unresolved.append({"row_id": update.get("row_index"), "column": col})
                continue
            raw_row = update.get("row_index")
            row_label = raw_row
            try:
                int_row = int(raw_row)
                if int_row in working.index:
                    row_label = int_row
            except (TypeError, ValueError):
                pass
            if row_label not in working.index:
                unresolved.append({"row_id": raw_row, "column": col})
                continue
            # Capture the value BEFORE it's overwritten, so the Change Log's
            # Before/After views have something real to show — previously
            # only the new value was logged, so "Before" was always blank.
            old_raw = working.at[row_label, col]
            try:
                old_logged = "" if pd.isna(old_raw) else str(old_raw)
            except (TypeError, ValueError):
                old_logged = str(old_raw)
            value = update.get("value")
            was_blank = isinstance(value, str) and value.strip() == ""
            if was_blank:
                value = np.nan
            if pd.api.types.is_numeric_dtype(working[col]):
                numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric_value):
                    value = numeric_value
                elif not was_blank:
                    working[col] = working[col].astype("object")
            working.at[row_label, col] = value
            try:
                logged_value = "" if pd.isna(value) else str(value)
            except (TypeError, ValueError):
                logged_value = str(value)
            if old_logged == logged_value and not (pd.isna(old_raw) and not was_blank):
                continue
            applied.append({
                "row_index": row_label,
                "row_number": _original_row_number(row_label),
                "column": col,
                "old": old_logged,
                "value": logged_value,
                "blank": bool(was_blank),
            })
        if unresolved:
            raise ValueError(f"Edited cells no longer exist: {unresolved[:10]}")
        if applied:
            report["operations"].append({
                "action": action,
                "count": len(applied),
                "updates": applied,
            })

    # 2) Apply manually edited cells before automated missing handling
    _apply_cell_updates(opts["cell_updates"], "manual_cell_updates")

    # 3) Apply manually typed missing-cell values before automated missing handling
    _apply_cell_updates(opts["missing_cell_updates"], "manual_missing_cell_updates")

    manual_renames = {}
    for old_col, new_col in opts["column_renames"].items():
        if old_col in working.columns and new_col and old_col != new_col:
            manual_renames[old_col] = new_col
    if manual_renames:
        existing = set(working.columns)
        safe_renames = {}
        proposed_columns = [manual_renames.get(c, c) for c in working.columns]
        if len(set(proposed_columns)) != len(proposed_columns):
            raise ValueError("Column rename collision: resulting column names must be unique.")
        for old_col, new_col in manual_renames.items():
            duplicate_target = new_col in existing and new_col not in manual_renames
            if not duplicate_target:
                safe_renames[old_col] = new_col
        if safe_renames:
            working.rename(columns=safe_renames, inplace=True)
            report["operations"].append({
                "action": "manual_column_renames",
                "count": len(safe_renames),
                "renamed_map": safe_renames,
            })

    # 4) Trim whitespace & unify case for strings
    if opts["trim_whitespace"]:
        text_cols = working.select_dtypes(include=["object"]).columns.tolist()
        trim_details = []
        changed_cols = []
        for c in text_cols:
            before = working[c].copy()
            str_mask = working[c].map(lambda v: isinstance(v, str))
            working.loc[str_mask, c] = working.loc[str_mask, c].map(lambda v: v.strip())
            changed_mask = before.ne(working[c]) & before.notna()
            changed_rows = _row_numbers(working.index[changed_mask])
            changed_row_ids = [str(idx) for idx in working.index[changed_mask]]
            if changed_rows:
                changed_cols.append(c)
                trim_details.append({
                    "column": c,
                    "rows": changed_rows,
                    "row_ids": changed_row_ids,
                    "count": len(changed_rows),
                })
        # Only report this as something that happened if it actually changed
        # something. columns_processed previously listed EVERY text-dtype
        # column regardless of whether it had any whitespace to strip — so
        # an already-clean dataset would still show "Trimmed whitespace in
        # N column(s)" for columns that were never touched.
        if trim_details:
            total_trimmed_cells = sum(int(item.get("count") or 0) for item in trim_details)
            affected_trim_rows = sorted({
                str(row)
                for item in trim_details
                for row in (item.get("rows") or [])
            }, key=lambda value: int(value) if str(value).isdigit() else str(value))
            report["operations"].append({
                "action": "trim_whitespace",
                "columns_processed": changed_cols,
                "details": trim_details,
                "changed_cells": total_trimmed_cells,
                "affected_rows": len(affected_trim_rows),
                "affected_row_numbers": affected_trim_rows,
            })



    # 4) Convert data types (numeric-like -> numeric; strings -> datetime if detected)
    dtype_changes = []
    if opts["convert_dtypes"]:
        for c in list(working.columns):
            ser = working[c]
            # Only convert to numeric, skip aggressive datetime conversion
            # Datetime conversion should be explicit, not automatic
            conv_num, changed_num = _try_convert_numeric(ser)
            if changed_num:
                working[c] = conv_num
                dtype_changes.append({"column": c, "new_dtype": str(working[c].dtype), "reason": "numeric_conversion"})
        if dtype_changes:
            report["operations"].append({"action": "convert_dtypes", "changes": dtype_changes})

    # 5) --- auto data type conversion (uses options dtype_map if provided, else auto-detect)
    try:
        dtype_map = options.get("dtype_map", None)
        if dtype_map:
            working, dtype_meta = convert_dtypes(working, dtype_map=dtype_map)
            if dtype_meta:
                report.setdefault("operations", []).append({"action": "manual_convert_dtypes", "changes": dtype_meta})
    except Exception as e:
        report.setdefault("operations", []).append({"action": "manual_convert_dtypes_failed", "error": str(e)})
    # 7) Drop empty columns (track names + count for the report)
    if opts["drop_empty_columns"]:
        empty_cols = [c for c in working.columns if working[c].isna().all()]
        if empty_cols:
            working.drop(columns=empty_cols, inplace=True)
        # Always log — even when 0 columns removed (mirrors drop_duplicates behaviour)
        report["operations"].append({
            "action": "drop_empty_columns",
            "columns_dropped": empty_cols,
            "count": len(empty_cols)
        })
        report["empty_columns_removed"] = int(len(empty_cols))
        report["empty_columns"] = list(empty_cols)

    

    # 8) Drop constant columns
    if opts["drop_constant_columns"]:
        constant_cols = []
        for c in working.columns:
            non_missing_count = working[c].notna().sum()
            unique_non_missing = working[c].nunique(dropna=True)
            if non_missing_count > 0 and unique_non_missing <= 1:
                constant_cols.append(c)
        if constant_cols:
            working.drop(columns=constant_cols, inplace=True)
        # Always log — even when 0 columns removed (mirrors drop_duplicates behaviour)
        report["operations"].append({
            "action": "drop_constant_columns",
            "columns_dropped": constant_cols,
            "count": len(constant_cols)
        })

    # 9) Remove duplicates
    duplicates_removed = 0
    if opts["remove_duplicates"]:
        before = working.shape[0]
        duplicate_subset = opts["duplicates_subset"]
        comparison_columns = list(duplicate_subset) if duplicate_subset else list(working.columns)

        def _duplicate_key(row):
            key = []
            for col in comparison_columns:
                value = row[col]
                key.append("__MISSING__" if pd.isna(value) else value)
            return tuple(key)

        first_seen: dict[tuple, object] = {}
        duplicate_matches = []
        for label, row in working.iterrows():
            key = _duplicate_key(row)
            retained_label = first_seen.get(key)
            if retained_label is None:
                first_seen[key] = label
                continue
            removed_record_id = row.get("record_id") if "record_id" in working.columns else None
            retained_row = working.loc[retained_label]
            retained_record_id = retained_row.get("record_id") if "record_id" in working.columns else None
            duplicate_matches.append({
                "removed_row_id": int(label) if isinstance(label, (int, np.integer)) else str(label),
                "removed_uploaded_row": _original_row_number(label),
                "removed_record_id": None if pd.isna(removed_record_id) else removed_record_id,
                "retained_row_id": int(retained_label) if isinstance(retained_label, (int, np.integer)) else str(retained_label),
                "retained_uploaded_row": _original_row_number(retained_label),
                "retained_record_id": None if pd.isna(retained_record_id) else retained_record_id,
                "comparison_basis": "exact",
                "normalization_used": False,
                "comparison_columns": comparison_columns,
                "differing_fields": [],
                "reason": "Exact duplicate across the selected comparison columns.",
            })

        duplicated_mask = working.duplicated(subset=duplicate_subset, keep="first")
        duplicate_rows = _row_numbers(working.index[duplicated_mask])
        if duplicate_subset:
            working = working.drop_duplicates(subset=duplicate_subset)
        else:
            working = working.drop_duplicates()
        after = working.shape[0]
        duplicates_removed = before - after
        report["operations"].append({
            "action": "drop_duplicates",
            "removed_rows": int(duplicates_removed),
            "subset": duplicate_subset,
            "rows_dropped": duplicate_rows,
            "duplicate_matches": duplicate_matches,
            "comparison_basis": "exact",
            "normalization_used": False,
            "comparison_columns": comparison_columns,
        })

    # 10) Handle outliers (IQR only for now)
    outliers_removed_total = 0
    
    # Handle manually selected outlier rows BEFORE other operations that might change the dataframe structure
    selected_outlier_rows = []
    if 'selected_outlier_rows' in opts and opts['selected_outlier_rows']:
        selected_outlier_rows = opts['selected_outlier_rows']
        
        logging.info(f"Attempting to remove selected outlier rows: {selected_outlier_rows}")
        logging.info(f"Working dataframe index before outlier removal: {working.index.tolist()[:10]}... (first 10)")

        # selected_outlier_rows contains the original DataFrame index *label* values
        # (row_position from detect_outliers = df.index[mask], i.e. actual index labels).
        # After drop_duplicates the index is a subset of original labels — use label-based lookup,
        # NOT positional indexing (working.index[pos] is positional and breaks after row removal).
        valid_indices = []

        # Reliable path first: selected_outlier_rows are original DataFrame index
        # labels straight from the checkbox (data-row), and those labels stay
        # stable across every request since clean_dataframe always rebuilds
        # `working` fresh from the untouched original file. No content-matching
        # is needed or wanted here — the index label IS the identity of the row.
        unresolved = []
        for p in selected_outlier_rows:
            candidates = [p]
            try:
                candidates.insert(0, int(p))
            except (TypeError, ValueError):
                pass
            idx = next((candidate for candidate in candidates if candidate in working.index), None)
            if idx is None:
                unresolved.append(p)
            elif idx not in valid_indices:
                valid_indices.append(idx)

        # Fuzzy content-matching is only a fallback for records whose index
        # couldn't be resolved directly above (e.g. legacy data with no
        # selected_outlier_rows at all). It intentionally does NOT re-check or
        # override rows already resolved by direct index lookup — searching the
        # whole dataframe for a "close enough" content match risks matching a
        # different row entirely when many rows share similar field values.
        if unresolved:
            raise ValueError(
                "Selected outlier row IDs are no longer present: "
                + ", ".join(str(value) for value in unresolved[:10])
            )

        logging.info(f"Valid index labels to remove: {valid_indices}")

        if valid_indices:
            valid_index_keys = {str(value) for value in valid_indices}
            selected_outlier_records = [
                record for record in (opts.get("selected_outlier_records") or [])
                if str(record.get("row_index")) in valid_index_keys
            ]
            working = working.drop(index=valid_indices)
            outliers_removed_total = len(valid_indices)
            # Always emit the operation so the frontend's appliedVisibleOpsCount
            # stays accurate across every apply (cumulative backend replay).
            # The frontend scope filter decides whether to route it to the
            # Outliers change-log block or silently skip it.
            report["operations"].append({
                "action": "manual_outlier_removal",
                "method": "user_selection",
                "rows_removed": len(valid_indices),
                "selected_rows": selected_outlier_rows,
                "valid_indices_dropped": valid_indices,
                "selected_positions": valid_indices,
                "row_numbers": _row_numbers(valid_indices),
                "selected_records": selected_outlier_records,
                "scope_apply": opts.get("outlier_scope_apply", False),
            })
            logging.info(f"Successfully removed {len(valid_indices)} outlier rows")
        else:
            raise ValueError("No selected outlier row IDs resolved in the current dataset")
    
    # Handle automatic outliers if enabled (for backward compatibility)
    # This is independent of manual outlier handling
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
                removed_ids = [str(idx) for idx in working.index[mask_any_outlier]]
                removed_numbers = _row_numbers(working.index[mask_any_outlier])
                before = working.shape[0]
                working = working.loc[~mask_any_outlier].copy()
                after = working.shape[0]
                automatic_outliers_removed = before - after
                outliers_removed_total += automatic_outliers_removed
                report["operations"].append({
                    "action": "outlier_removal", "method": "iqr", "multiplier": multiplier,
                    "removed_rows": int(automatic_outliers_removed),
                    "valid_indices_dropped": removed_ids, "row_numbers": removed_numbers,
                })
            else:
                clipped = []
                for c in working.select_dtypes(include=[np.number]).columns:
                    m = _iqr_outliers_mask(working[c], multiplier=multiplier)
                    if m.any():
                        q1 = working[c].quantile(0.25)
                        q3 = working[c].quantile(0.75)
                        low = q1 - multiplier * (q3 - q1)
                        high = q3 + multiplier * (q3 - q1)
                        for idx in working.index[m]:
                            old_value = working.at[idx, c]
                            new_value = min(max(old_value, low), high)
                            clipped.append({
                                "row_id": str(idx), "row_number": _original_row_number(idx),
                                "column": c, "before": old_value, "after": new_value,
                                "iqr_low": low, "iqr_high": high,
                            })
                        working[c] = working[c].clip(lower=low, upper=high)
                if clipped:
                    report["operations"].append({
                        "action": "outlier_clip", "method": "iqr", "multiplier": multiplier,
                        "count": len(clipped), "columns": list(dict.fromkeys(item["column"] for item in clipped)),
                        "changes": clipped,
                    })

    # 10) Handle missing values
    filled_cells = 0
    dropped_rows = 0
    if opts["missing_strategy"] == "drop":
        before = working.shape[0]
        missing_row_mask = working.isna().any(axis=1)
        rows_dropped = _row_numbers(working.index[missing_row_mask])
        missing_columns_by_row = [
            {
                "row": _original_row_number(idx),
                "columns": [c for c in working.columns if pd.isna(working.at[idx, c])]
            }
            for idx in working.index[missing_row_mask]
        ]
        working = working.dropna()
        after = working.shape[0]
        dropped_rows = before - after
        report["operations"].append({
            "action": "drop_missing_rows",
            "dropped_row_count": int(dropped_rows),
            "row_numbers": rows_dropped,
            "missing_columns_by_row": missing_columns_by_row,
        })



    elif opts["missing_strategy"] == "fill":
        numeric_filled = 0  # track separately from categorical
        global_numeric_strategy = opts["fill_numeric"]
        # per-column overrides: {col: strategy} — takes priority over global
        numeric_col_strategies    = options.get("numeric_col_strategies", {})
        categorical_col_strategies = options.get("categorical_col_strategies", {})

        if global_numeric_strategy != "skip" or numeric_col_strategies:
            numeric_cols = list(dict.fromkeys(
                working.select_dtypes(include=[np.number]).columns.tolist()
                + list(numeric_col_strategies)
            ))
            numeric_details = []
            skipped_numeric_details = []
            rows_dropped_by_col = 0
            for c in numeric_cols:
                strategy = numeric_col_strategies.get(c, global_numeric_strategy)
                if strategy == "skip":
                    continue
                missing_mask = working[c].isna()
                before_na = int(missing_mask.sum())
                if before_na == 0:
                    continue
                if strategy == "drop":
                    # Capture row numbers BEFORE dropping — once the rows are removed
                    # from `working` we can no longer recover which uploaded rows they were.
                    dropped_row_numbers = _row_numbers(working.index[missing_mask])
                    dropped_row_ids = [str(idx) for idx in working.index[missing_mask]]
                    before_rows = len(working)
                    working = working[~missing_mask].copy()
                    dropped = before_rows - len(working)
                    rows_dropped_by_col += dropped
                    numeric_details.append({
                        "column": c, "filled": 0, "dropped": dropped,
                        "strategy": "drop",
                        "value": f"{dropped} rows dropped",
                        "rows": dropped_row_numbers,
                        "row_ids": dropped_row_ids,
                    })
                    continue
                if strategy == "mean":
                    val = working[c].mean()
                elif strategy == "median":
                    val = working[c].median()
                elif strategy == "zero":
                    val = 0
                else:
                    try:
                        val = float(strategy)
                    except (TypeError, ValueError):
                        val = strategy
                if pd.isna(val):
                    skipped_numeric_details.append({
                        "column": c, "missing": before_na, "strategy": strategy,
                        "reason": f"{strategy} is unavailable (column may be all-null).",
                        "rows": _row_numbers(working.index[missing_mask]),
                    })
                    continue
                working[c] = working[c].fillna(val)
                numeric_filled += before_na
                numeric_details.append({
                    "column": c, "filled": before_na, "strategy": strategy,
                    "value": str(round(float(val), 6) if isinstance(val, float) else val),
                    "rows": _row_numbers(working.index[missing_mask]),
                    "row_ids": [str(idx) for idx in working.index[missing_mask]],
                })
        filled_cells += numeric_filled
        _rows_dropped_total = rows_dropped_by_col if 'rows_dropped_by_col' in dir() else 0
        report["operations"].append({
            "action": "fill_numeric",
            "strategy": global_numeric_strategy,
            "column_strategies": numeric_col_strategies,
            "filled_cells": numeric_filled,
            # Aggregate count of rows removed via a per-column "drop" strategy — kept
            # separate from filled_cells so the UI can tell "cells filled" from
            # "rows dropped" apart instead of collapsing both into one number.
            "rows_dropped": _rows_dropped_total,
            "details": numeric_details if 'numeric_details' in dir() else [],
            "skipped": skipped_numeric_details if 'skipped_numeric_details' in dir() else [],
            "reason": None if numeric_filled > 0 or _rows_dropped_total > 0 or (skipped_numeric_details if 'skipped_numeric_details' in dir() else []) else "No numeric missing cells were available for this option.",
        })

        # Handle categorical missing values (global + per-column overrides)
        global_cat_strategy = opts["fill_categorical"]
        if global_cat_strategy != "skip" or categorical_col_strategies:
            # Collect columns that need specific strategies
            cat_cols_all = list(dict.fromkeys(
                working.select_dtypes(include=["object", "category", "bool", "boolean", "datetime"]).columns.tolist()
                + list(categorical_col_strategies)
            ))

            # Group columns by strategy: per-column overrides or fallback to global
            strategy_groups: Dict[str, list] = {}
            for c in cat_cols_all:
                col_strategy = categorical_col_strategies.get(c, global_cat_strategy)
                if col_strategy == "skip":
                    continue
                strategy_groups.setdefault(col_strategy, []).append(c)

            cat_total_filled = 0
            cat_details_merged: Dict[str, Any] = {}
            cat_skipped_merged: List[Dict[str, Any]] = []
            cat_strategy_used = global_cat_strategy

            for strat, cols in strategy_groups.items():
                if strat == "drop":
                    for c in cols:
                        missing_mask = working[c].isna()
                        row_ids = [str(idx) for idx in working.index[missing_mask]]
                        row_numbers = _row_numbers(working.index[missing_mask])
                        working = working.loc[~missing_mask].copy()
                        cat_details_merged[c] = {
                            "method": "drop", "strategy": "drop",
                            "dropped_rows": len(row_ids),
                            "row_ids": row_ids, "rows": row_numbers,
                        }
                    continue
                working, cat_meta = fill_categorical(
                    working,
                    strategy=strat,
                    constant=options.get("categorical_constant", None),
                    columns=cols,
                )
                cat_total_filled += cat_meta.get("filled_cells", 0)
                cat_details_merged.update(cat_meta.get("details", {}))
                cat_skipped_merged.extend(cat_meta.get("skipped", []))
                cat_strategy_used = strat

            filled_cells += cat_total_filled
            report["operations"].append({
                "action": "fill_categorical",
                "strategy": cat_strategy_used,
                "column_strategies": categorical_col_strategies,
                "filled_cells": cat_total_filled,
                "details": cat_details_merged,
                "skipped": cat_skipped_merged,
                "reason": None if cat_total_filled > 0 or cat_details_merged else "No categorical missing cells could be filled with the selected option.",
            })
    else:
        report["operations"].append({"action": "skip_missing_values", "reason": "not selected for this preview"})


    # 12) Encode categoricals
    encoded_columns = []
    if opts["encode_categoricals"] == "label":
        cat_cols = working.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            try:
                if working[c].nunique(dropna=True) <= max(1, min(1000, int(working.shape[0] * 0.8))):
                    missing_mask = working[c].isna()
                    categories = sorted([str(v) for v in working.loc[~missing_mask, c].unique()])
                    mapping = {cat: i for i, cat in enumerate(categories)}
                    working.loc[~missing_mask, c] = working.loc[~missing_mask, c].astype(str).map(mapping)
                    encoded_columns.append({"column": c, "mapping": mapping})
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
    # filled_cells so far only tallies fill_numeric/fill_categorical.
    # Manually typed missing-cell values (manual_missing_cell_updates) fill
    # real missing cells too and were already being itemized correctly in
    # the step-by-step log, but were excluded from this summary number.
    manual_missing_filled = sum(
        op.get("count", 0) for op in report["operations"]
        if op.get("action") == "manual_missing_cell_updates"
    )
    report["filled_cells"] = int(filled_cells) + int(manual_missing_filled)
    report["columns_renamed"] = len(renamed_map) if renamed_map else 0
    report["cleaned_columns"] = list(working.columns)
    # ensure empty_columns keys exist for templates even if drop_empty_columns was False
    if "empty_columns_removed" not in report:
        report["empty_columns_removed"] = 0
    if "empty_columns" not in report:
        report["empty_columns"] = []

    # Final logging to indicate if any outlier removal happened
    # Use opts.get() to check the actual options dictionary
    if not opts.get("selected_outlier_rows") and not opts.get("handle_outliers"):
        logging.info("No outlier removal was requested")
    elif opts.get("selected_outlier_rows"):
        logging.info(f"Processed manually selected outlier rows: {opts['selected_outlier_rows']}")
    elif opts.get("handle_outliers"):
        logging.info("Automatic outlier handling was performed")


    PREVIEW_ROWS = min(len(working), 100)
    preview = working.head(PREVIEW_ROWS).copy()
    for c in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[c]):
            preview[c] = preview[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    report["preview_row_numbers"] = _row_numbers(preview.index)
    report["preview_display_limit"] = PREVIEW_ROWS
    report["preview"] = preview.to_dict(orient="records")

    return working, report


# -------------------------
# Utility: write report to json
# -------------------------
def save_report(report: Dict[str, Any], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, allow_nan=False)

#END
