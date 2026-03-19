import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
import logging



# -------------------------
# Helpers
# -------------------------
def _is_datetime_series(s: pd.Series) -> bool:
    """Try to infer datetime series robustly."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        # Only test a sample to avoid performance issues
        non_null_series = s.dropna()
        if len(non_null_series) == 0:
            return False
        
        sample_size = min(50, max(1, int(len(non_null_series)*0.1)))  # 10% of data or max 50
        sample = non_null_series.sample(n=sample_size, random_state=42)
        try:
            pd.to_datetime(sample.head(10), errors='raise')  # Test first 10 values
            return True
        except Exception:
            return False
    return False

def _try_convert_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Attempt to convert to numeric; return (converted_series, changed_flag)."""
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    
    # Only attempt conversion if series has non-null values
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return series, False
    
    # Sample a portion of the data for testing conversion
    sample_size = min(10, len(non_null_series))
    sample = non_null_series.sample(n=sample_size, random_state=42)
    
    # Test if conversion works on sample
    test_conversion = pd.to_numeric(sample, errors='coerce')
    
    # If most values in sample converted successfully, convert the whole series
    successful_conversions = test_conversion.notna().sum()
    if successful_conversions / len(sample) >= 0.7:  # At least 70% success rate
        conv = pd.to_numeric(series, errors='coerce')
        changed = not conv.equals(series) and conv.notna().sum() > 0
        return conv, changed
    else:
        return series, False

def _try_convert_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Attempt to convert to datetime; return (converted_series, changed_flag)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False
    
    # NEVER convert numeric columns to datetime!
    # Numbers like 142 get converted to 1970-01-01 00:00:00.000000142 (nanoseconds)
    if pd.api.types.is_numeric_dtype(series):
        return series, False
    
    try:
        # Only attempt conversion if series has non-null string values
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return series, False
        
        # Check if values look like date strings (contain date separators)
        sample_str = non_null_series.head(10).astype(str)
        date_like_count = sum(1 for val in sample_str if any(sep in str(val) for sep in ['-', '/', ':']))
        if date_like_count < len(sample_str) * 0.5:  # Less than 50% look like dates
            return series, False
        
        # Sample a portion of the data for testing conversion
        sample_size = min(10, len(non_null_series))
        sample = non_null_series.sample(n=sample_size, random_state=42)
        
        # Test if conversion works on sample
        test_conversion = pd.to_datetime(sample, errors='coerce')
        
        # If most values in sample converted successfully, convert the whole series
        successful_conversions = test_conversion.notna().sum()
        if successful_conversions / len(sample) >= 0.7:  # At least 70% success rate
            conv = pd.to_datetime(series, errors='coerce')
            changed = conv.notna().sum() > 0 and not conv.equals(series)
            return conv, changed
        else:
            return series, False
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


def detect_outliers(df: pd.DataFrame, multiplier: float = 1.5, max_rows: int = 100) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns and return details for user selection.
    Returns a dict with:
      - outlier_rows: list of {row_index, row_position, column, value, row_data}
      - total_count: total number of outlier values found
      - columns_checked: list of numeric columns checked
    """
    df = df.copy()
    
    # Create a mapping from position to original index
    pos_to_orig_idx = {i: idx for i, idx in enumerate(df.index)}
    
    # First, clean numeric strings (same as in clean_dataframe)
    placeholder_values = ['####', '#####', '######', 'N/A', 'n/a', 'NA', 'NULL', 'null', 'None', '-', '--', '?', 'NaN', 'nan']
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace placeholders with NaN
            df[col] = df[col].apply(
                lambda x: np.nan if (isinstance(x, str) and (x.strip() in placeholder_values or x.strip().startswith('###'))) else x
            )

    
    # Now detect outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows = []
    seen_positions = set()  # Track by position instead of index
    
    for col in numeric_cols:
        mask = _iqr_outliers_mask(df[col], multiplier=multiplier)
        outlier_positions = df.index[mask].tolist()
        
        for pos in outlier_positions:
            if pos not in seen_positions and len(outlier_rows) < max_rows:
                # Get the original index for this position
                original_idx = pos_to_orig_idx.get(pos, pos)
                row_data = df.loc[pos].to_dict()
                # Convert values to strings for JSON serialization
                row_data_str = {k: str(v) if pd.notna(v) else '' for k, v in row_data.items()}
                outlier_rows.append({
                    'row_index': int(original_idx),  # Original index
                    'row_position': int(pos),        # Position in original df
                    'column': col,
                    'value': float(df.loc[pos, col]) if pd.notna(df.loc[pos, col]) else None,
                    'row_data': row_data_str
                })
                seen_positions.add(pos)
    
    return {
        'outlier_rows': outlier_rows,
        'total_count': len(seen_positions),
        'columns_checked': numeric_cols,
        'max_displayed': max_rows,
        'index_mapping': pos_to_orig_idx  # Include mapping for debugging
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

    # Missing values
    missing_pct = (df.isna().sum().sum() / total_cells) * 100 if total_cells else 0

    # Duplicates
    dup_pct = (df.duplicated().sum() / n_rows) * 100 if n_rows else 0

    # Constant columns
    const_col_pct = (df.nunique(dropna=False) <= 1).sum() / n_cols * 100 if n_cols else 0

    # Outliers (numeric columns only)
    num_df = df.select_dtypes(include=np.number)
    outlier_pct = 0
    if not num_df.empty:
        # Use IQR method for outlier detection to be more robust
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

    # Weighted score (you can tune weights)
    # Lower percentages result in higher quality scores
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
        
        # Test if conversion works on sample
        test_conversion = pd.to_datetime(sample, errors="coerce")
        
        # If most values in sample converted successfully, convert the whole series
        successful_conversions = test_conversion.notna().sum()
        if successful_conversions / len(sample) >= 0.7:  # At least 70% success rate
            conv = pd.to_datetime(s, errors="coerce")
            coerced = int(s.notna().sum() - conv.notna().sum()) if s.notna().any() else 0
            return conv, coerced
        else:
            # Return original series with 0 coerced count if conversion not advisable
            return s, 0

    for c in cols:
        choice = dtype_map.get(c, "auto")
        old_dtype = str(df[c].dtype)
        info = {"from": old_dtype, "to": choice, "coerced": 0, "success": True}
        try:
            if choice == "auto":
                # Prioritize: numeric > datetime > string
                # This prevents numeric columns like duration/score from being converted to datetime
                conv_num, coerced_num = _to_numeric_safe(df[c])
                # Check if numeric conversion actually worked (data changed AND is now numeric)
                actually_converted_num = (not conv_num.equals(df[c])) or (pd.api.types.is_numeric_dtype(conv_num) and not pd.api.types.is_numeric_dtype(df[c]))
                
                if actually_converted_num and coerced_num < len(df[c]):
                    df[c] = conv_num
                    info.update({"to": "numeric", "coerced": int(coerced_num)})
                else:
                    # Only try datetime if numeric conversion failed
                    conv_dt, coerced_dt = _to_datetime_safe(df[c])
                    actually_converted_dt = (not conv_dt.equals(df[c])) or (pd.api.types.is_datetime64_any_dtype(conv_dt) and not pd.api.types.is_datetime64_any_dtype(df[c]))
                    
                    if actually_converted_dt and coerced_dt < len(df[c]):
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
                mapped_values = df[c].map(bool_map)
                # Keep original NaN values
                df[c] = mapped_values.where(df[c].notna(), np.nan)
                # Convert to object dtype to preserve mixed types if needed
                df[c] = df[c].astype('object')
                info.update({"to": "bool"})
            else:
                info.update({"success": False, "note": "unknown_choice"})
        except Exception as e:
            info.update({"success": False, "note": str(e)})
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
    Fill missing values in categorical/text columns.
    Supported strategies: 'mode', 'constant', 'drop'
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

        "selected_outlier_rows": options.get("selected_outlier_rows", [])
    }

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "original_shape": list(df.shape),
        "operations": []
    }

    working = df.copy(deep=True)

    # 0) Replace common placeholder values with NaN (before any other processing)
    placeholder_values = [
        '####', '#####', '######', '#######',  # Excel display errors
        'N/A', 'n/a', 'NA', 'na', 'N.A.', 'n.a.',  # Common NA representations
        'NULL', 'null', 'Null', 'None', 'none', 'NONE',  # NULL values
        '-', '--', '---', '?', '??', '???',  # Placeholder symbols
        'NaN', 'nan', 'NAN', '#N/A', '#NA', '#VALUE!', '#REF!', '#DIV/0!',  # Excel errors
        'undefined', 'UNDEFINED', 'missing', 'MISSING', 'unknown', 'UNKNOWN',
        '.', '..', '...', '*', '**', '***'  # Other placeholders
    ]
    for col in working.columns:
        if working[col].dtype == 'object':
            # Replace placeholder values with NaN
            working[col] = working[col].apply(
                lambda x: np.nan if (isinstance(x, str) and (x.strip() in placeholder_values or x.strip().startswith('###'))) else x
            )
    report["operations"].append({"action": "replace_placeholders", "values": placeholder_values[:5] + ['...']})



    # 1) Rename columns first (before other operations)
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
            working.columns = new_cols
            report["operations"].append({"action": "rename_columns", "renamed_map": renamed_map, "count": len(renamed_map)})

    # 2) Trim whitespace & unify case for strings
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


    # 6) Drop empty columns (track names + count for the report)
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

    

    # 7) Drop constant columns
    if opts["drop_constant_columns"]:
        constant_cols = []
        for c in working.columns:
            if working[c].nunique(dropna=True) <= 1:
                constant_cols.append(c)
        if constant_cols:
            working.drop(columns=constant_cols, inplace=True)
            report["operations"].append({"action": "drop_constant_columns", "columns_dropped": constant_cols, "count": len(constant_cols)})

    # 8) Remove duplicates
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

    # 9) Handle outliers (IQR only for now)
    outliers_removed_total = 0
    
    # Handle manually selected outlier rows BEFORE other operations that might change the dataframe structure
    selected_outlier_rows = []
    if 'selected_outlier_rows' in opts and opts['selected_outlier_rows']:
        selected_outlier_rows = opts['selected_outlier_rows']
        
        logging.info(f"Attempting to remove selected outlier rows early: {selected_outlier_rows}")
        logging.info(f"Working dataframe index before outlier removal: {working.index.tolist()[:10]}... (first 10)")
        
        # The selected_outlier_rows are now positional indices (0, 1, 2, ...) from the original uploaded file
        # Convert positions to actual dataframe indices for removal
        positions = [int(p) for p in selected_outlier_rows if 0 <= int(p) < len(working)]
        
        logging.info(f"Valid positions to remove: {positions}")
        
        if positions:
            # Get the actual indices from the current working dataframe at those positions
            actual_indices_to_remove = working.index[positions].tolist()
            working = working.drop(actual_indices_to_remove)
            outliers_removed_total = len(positions)
            report["operations"].append({
                "action": "manual_outlier_removal",
                "method": "user_selection",
                "rows_removed": len(positions),
                "selected_rows": selected_outlier_rows,
                "valid_indices_dropped": actual_indices_to_remove,
                "selected_positions": positions
            })
            logging.info(f"Successfully removed {len(positions)} outlier rows early")
        else:
            logging.info("No valid positions found for early outlier removal")
    
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
                before = working.shape[0]
                working = working.loc[~mask_any_outlier].copy()
                after = working.shape[0]
                automatic_outliers_removed = before - after
                outliers_removed_total += automatic_outliers_removed
                report["operations"].append({"action": "outlier_removal", "method": "iqr", "multiplier": multiplier, "removed_rows": int(automatic_outliers_removed)})
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

    # 10) Handle missing values
    filled_cells = 0
    dropped_rows = 0
    if opts["missing_strategy"] == "drop":
        before = working.shape[0]
        working = working.dropna()
        after = working.shape[0]
        dropped_rows = before - after
        report["operations"].append({"action": "drop_missing_rows", "rows_dropped": int(dropped_rows)})



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
        
        # Handle categorical missing values
        working, cat_meta = fill_categorical(
            working,
            strategy=opts["fill_categorical"],
            constant=options.get("categorical_constant", None),
            columns=options.get("categorical_columns", None)
        )
        filled_cells += cat_meta.get("filled_cells", 0)
        report["operations"].append({
            "action": "fill_categorical",
            "strategy": cat_meta["strategy"],
            "filled_cells": cat_meta["filled_cells"],
            "details": cat_meta.get("details", {})
        })


    # 11) Encode categoricals
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

    # Final logging to indicate if any outlier removal happened
    # Use opts.get() to check the actual options dictionary
    if not opts.get("selected_outlier_rows") and not opts.get("handle_outliers"):
        logging.info("No outlier removal was requested")
    elif opts.get("selected_outlier_rows"):
        logging.info(f"Processed manually selected outlier rows: {opts['selected_outlier_rows']}")
    elif opts.get("handle_outliers"):
        logging.info("Automatic outlier handling was performed")


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