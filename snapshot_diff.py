# snapshot_diff.py — generic DataFrame diffing for persistent, independent
# undo/redo.
#
# Core idea: instead of storing "restore the whole dataset to this point in
# time" (which would silently discard any *other* action applied after it),
# each action stores a precise, minimal instruction set describing exactly
# which cells/rows/columns IT changed. Undo/Reapply replay just that
# instruction set against whatever the CURRENT dataset happens to be — so an
# unrelated later action's changes are left completely alone.
#
# A manifest has four independent parts:
#   changed_cells   — {row, col, before, after}            (value changed in place)
#   dropped_rows    — {row, data: {col: value, ...}}       (row existed before, gone after)
#   added_rows      — {row, data: {col: value, ...}}       (row didn't exist before, exists after)
#   dropped_columns — {col, data: {row: value, ...}}       (column existed before, gone after)
#   added_columns   — {col, data: {row: value, ...}}       (column didn't exist before, exists after)
#
# "Forward" = replay before→after (Reapply). "Reverse" = replay after→before
# (Undo). Both are derived from the exact same manifest — nothing is stored
# twice.

"""DataFrame manifest diffing for persisted Before/After evidence.

A manifest describes what one committed action changed: changed cells, added or
dropped rows, and added or dropped columns. Flask stores that manifest beside
the ordered history row, and replay code uses it to reconstruct the current
state from the immutable source plus active actions.

``forward`` means before-to-after (Reapply). ``reverse`` means after-to-before
(Undo). Direct reverse patches are not the authoritative undo model for
overlapping actions; ordered replay in ``app.py`` decides the final state.
These helpers provide exact per-action evidence and low-level patch mechanics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# --- value <-> JSON-safe conversion -------------------------------------------
# Manifests get stored in a JSON column (action_config), so every value in
# them has to round-trip through JSON. NaN/NaT become None; numpy scalars and
# Timestamps become plain Python types.

def _to_json_safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp,)):
        if pd.isna(val):
            return None
        return val.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val


def _row_key(idx: Any) -> str:
    """Row indices become string dict keys for JSON — always converted back
    the same way on read, so int-vs-str never causes a silent mismatch."""
    return str(idx)


# --- computing a manifest -----------------------------------------------------

def compute_full_diff(before_df: pd.DataFrame, after_df: pd.DataFrame,
                       cell_limit: "int | None" = None) -> Dict[str, Any]:
    """Diff two DataFrames sharing the same "logical" row index scheme.

    Returns a JSON-safe manifest covering every changed cell — no cap by
    default, so Undo/Reapply stays exact no matter how many cells an action
    touched. Production stores manifests in the ``manifests`` table rather
    than inside the history JSON column, so history listings stay small while
    Undo/Reapply keeps exact per-action evidence.

    `cell_limit`, if given, caps how many individual changed-cell entries get
    recorded — kept only as an explicit opt-in escape hatch for callers that
    want a bounded manifest for some other reason. dropped/added rows &
    columns are NEVER capped even when cell_limit is set, since skipping
    those would corrupt undo/redo, not just make it incomplete.
    """
    before_index = before_df.index
    after_index  = after_df.index
    common_index = before_index.intersection(after_index)

    before_cols = before_df.columns
    after_cols  = after_df.columns
    common_cols = before_cols.intersection(after_cols)

    manifest: Dict[str, Any] = {
        "changed_cells":   [],
        "dropped_rows":    [],
        "added_rows":      [],
        "dropped_columns": [],
        "added_columns":   [],
        "truncated":       False,
    }

    # --- rows ---
    dropped_row_idx = before_index.difference(after_index)
    added_row_idx   = after_index.difference(before_index)

    for idx in dropped_row_idx:
        row = before_df.loc[idx]
        manifest["dropped_rows"].append({
            "row": _row_key(idx),
            "data": {str(c): _to_json_safe(row[c]) for c in before_cols},
        })
    for idx in added_row_idx:
        row = after_df.loc[idx]
        manifest["added_rows"].append({
            "row": _row_key(idx),
            "data": {str(c): _to_json_safe(row[c]) for c in after_cols},
        })

    # --- columns ---
    dropped_cols = before_cols.difference(after_cols)
    added_cols   = after_cols.difference(before_cols)

    for col in dropped_cols:
        series = before_df[col]
        manifest["dropped_columns"].append({
            "col": str(col),
            "data": {_row_key(idx): _to_json_safe(series.loc[idx]) for idx in before_index},
        })
    for col in added_cols:
        series = after_df[col]
        manifest["added_columns"].append({
            "col": str(col),
            "data": {_row_key(idx): _to_json_safe(series.loc[idx]) for idx in after_index},
        })

    # --- changed cells (common rows x common cols only) ---
    if len(common_index) and len(common_cols):
        b = before_df.loc[common_index, common_cols]
        a = after_df.loc[common_index, common_cols]
        n_added = 0
        for col in common_cols:
            bcol = b[col]
            acol = a[col]
            b_na = bcol.isna()
            a_na = acol.isna()
            both_present = ~b_na & ~a_na
            only_one_na  = b_na ^ a_na
            diff_mask = ((both_present & (bcol != acol)) | only_one_na).astype(bool)
            changed_idx = common_index[diff_mask.to_numpy()]
            for idx in changed_idx:
                if cell_limit is not None and n_added >= cell_limit:
                    manifest["truncated"] = True
                    break
                manifest["changed_cells"].append({
                    "row": _row_key(idx),
                    "col": str(col),
                    "before": _to_json_safe(bcol.loc[idx]),
                    "after":  _to_json_safe(acol.loc[idx]),
                })
                n_added += 1
            if cell_limit is not None and n_added >= cell_limit:
                break

    return manifest


def manifest_is_empty(manifest: Dict[str, Any]) -> bool:
    if not manifest:
        return True
    return not any(manifest.get(k) for k in
                   ("changed_cells", "dropped_rows", "added_rows", "dropped_columns", "added_columns"))


# --- applying a manifest ------------------------------------------------------
#
# Both directions work on WHATEVER the current dataframe is — not necessarily
# before_df or after_df themselves — so a later, unrelated action's changes
# are preserved automatically. Anything the manifest references that no
# longer matches reality (row/column already gone, already present, etc.) is
# skipped rather than forced, and reported back in "conflicts" so the caller
# can surface a warning instead of silently corrupting data.

def _restore_index_dtype(row_key: str, sample_index: pd.Index) -> Any:
    """Row keys are stored as strings; convert back to int if the DataFrame's
    index is integer-based (the overwhelmingly common case here), else leave
    as string."""
    if len(sample_index) and pd.api.types.is_integer_dtype(sample_index.dtype):
        try:
            return int(row_key)
        except (TypeError, ValueError):
            return row_key
    return row_key


def _coerce_to_column_dtype(value, dtype):
    """Manifest values are JSON-safe (a pd.Timestamp becomes a plain date
    string — see _to_json_safe), but apply_manifest() inserts them straight
    back into a live DataFrame whose column may still be real datetime64.
    Inserting a string into a datetime64 column silently upcasts the whole
    column to `object` dtype (a mix of Timestamps and strings), which
    pyarrow/Feather can't serialise later (ArrowTypeError: "Expected bytes,
    got a 'Timestamp' object", surfaced from save_current_snapshot). This
    converts the value back to match the column's actual dtype first, so the
    column stays uniformly typed.
    """
    if value is None:
        return value
    try:
        if pd.api.types.is_datetime64_any_dtype(dtype) and not isinstance(value, pd.Timestamp):
            return pd.to_datetime(value)
    except Exception:
        pass
    return value


def apply_manifest(current_df: pd.DataFrame, manifest: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """direction: 'forward' (replay before->after, i.e. Reapply) or
    'reverse' (replay after->before, i.e. Undo).

    Returns {"df": DataFrame, "conflicts": [str, ...]}. Conflicts are non-fatal
    — the rest of the manifest still gets applied.
    """
    if direction not in ("forward", "reverse"):
        raise ValueError(f"invalid direction: {direction!r}")

    df = current_df.copy(deep=True)
    conflicts: List[str] = []
    sample_index = df.index

    changed_cells   = manifest.get("changed_cells", [])
    dropped_rows    = manifest.get("dropped_rows", [])
    added_rows      = manifest.get("added_rows", [])
    dropped_columns = manifest.get("dropped_columns", [])
    added_columns   = manifest.get("added_columns", [])

    # --- changed cells: forward -> "after" value, reverse -> "before" value ---
    value_key = "after" if direction == "forward" else "before"
    for item in changed_cells:
        col = item["col"]
        row = _restore_index_dtype(item["row"], sample_index)
        if col not in df.columns:
            conflicts.append(f"cell change on missing column '{col}' skipped (column no longer exists)")
            continue
        if row not in df.index:
            conflicts.append(f"cell change on missing row {row} skipped (row no longer exists)")
            continue
        df.at[row, col] = _coerce_to_column_dtype(item[value_key], df[col].dtype)

    # --- rows: forward removes dropped_rows / adds added_rows; reverse is the mirror ---
    rows_to_remove = dropped_rows if direction == "forward" else added_rows
    rows_to_add    = added_rows    if direction == "forward" else dropped_rows

    remove_idx = []
    for item in rows_to_remove:
        row = _restore_index_dtype(item["row"], sample_index)
        if row in df.index:
            remove_idx.append(row)
        else:
            conflicts.append(f"row {row} already absent — skipping removal")
    if remove_idx:
        df = df.drop(index=remove_idx)

    add_frames = []
    for item in rows_to_add:
        row = _restore_index_dtype(item["row"], sample_index if len(sample_index) else df.index)
        if row in df.index:
            conflicts.append(f"row {row} already present — skipping re-add")
            continue
        # Only fill columns that still exist in the current dataframe — a
        # later action may have dropped a column this row also referenced.
        # Values are also coerced back to that column's real dtype (see
        # _coerce_to_column_dtype) — otherwise re-adding a row whose date
        # column was JSON-serialised to a string upcasts the whole column to
        # mixed-type object, which breaks Feather serialisation later.
        data = {
            c: (_coerce_to_column_dtype(v, df[c].dtype) if c in df.columns else v)
            for c, v in item["data"].items()
            if c in df.columns or not len(df.columns)
        }
        add_frames.append(pd.DataFrame([data], index=[row]))
    if add_frames:
        df = pd.concat([df] + add_frames, axis=0)
        try:
            df = df.sort_index()
        except TypeError:
            pass  # mixed index types — leave insertion order as-is

    # --- columns: forward removes dropped_columns / adds added_columns; reverse is the mirror ---
    cols_to_remove = [c["col"] for c in (dropped_columns if direction == "forward" else added_columns)]
    cols_to_add    = dropped_columns if direction == "reverse" else added_columns

    existing_remove = [c for c in cols_to_remove if c in df.columns]
    if existing_remove:
        df = df.drop(columns=existing_remove)
    for c in cols_to_remove:
        if c not in existing_remove:
            conflicts.append(f"column '{c}' already absent — skipping removal")

    for item in cols_to_add:
        col = item["col"]
        if col in df.columns:
            conflicts.append(f"column '{col}' already present — skipping re-add")
            continue
        series_data = {}
        for row_key, val in item["data"].items():
            row = _restore_index_dtype(row_key, df.index)
            series_data[row] = val
        df[col] = pd.Series(series_data)

    return {"df": df, "conflicts": conflicts}
