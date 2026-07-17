# dq_incremental.py — O(manifest size) maintenance of data_quality_score's
# inputs, so Undo/Reapply never has to re-scan the full dataset to get an
# exact quality score, regardless of dataset size.
#
# data_quality_score() (cleaner.py) is a weighted blend of seven full-dataset
# metrics: missing%, placeholder%, duplicate%, numeric-issue%, row-placeholder%,
# outlier%, constant-column%. Every one of them can change from a single
# manifest apply/undo — but none of them need a full rescan to update, IF a
# small amount of running state is maintained alongside the snapshot:
#
#   - total_missing              -> missing_pct
#   - row content-hash -> count  -> dup_pct (duplicate rows)
#   - per-column non-null value counts -> const_col_pct
#   - per-column placeholder/digit/alpha/non-placeholder counts -> placeholder_pct,
#     numeric_issue_pct (and re-derives the numeric-like flag exactly like
#     the ground-truth function does, with a bounded per-column fallback if a
#     column's classification flips)
#   - per-row placeholder counts -> row_placeholder_pct
#   - per-numeric-column Q1/Q3/outlier count -> outlier_pct (bounded: only the
#     touched numeric column is rescanned, not the whole dataset)
#
# This module is intentionally self-contained and independently testable —
# see test_dq_incremental.py, which checks it against cleaner.data_quality_score
# (the ground truth) across randomized edit sequences before it's trusted.

"""Legacy/experimental incremental data-quality statistics.

Production runtime does not import this module. The application now uses the
canonical read-only scoring functions in ``cleaner.py`` for previews, replay,
and final reports so every surface uses the same formula. This file is retained
only as prior exploratory work for manifest-sized score maintenance; comments
above describe that experiment, not the live scoring path.
"""

from __future__ import annotations
import re
from typing import Any, Dict
import pandas as pd

PLACEHOLDER_TOKENS = {
    "", "unknown", "not applicable", "n/a", "na", "null", "none",
    "nan", "inf", "infinity", "-", "--", "---", "?", "??", "???",
    "missing", "undefined",
}
_HASH_RE = re.compile(r"#+$")
_DIGIT_RE = re.compile(r"\d")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_THOUSANDS_DOT_RE = re.compile(r"-?\d{1,3}(?:\.\d{3})+(?:\.\d+)?")
_MONEY_CHARS_RE = re.compile(r"[$€£₹,\s]")
NUMERIC_NAME_PARTS = (
    "duration", "score", "rating", "vote", "income", "revenue",
    "amount", "price", "cost", "budget", "gross", "year", "count",
    "number", "qty", "quantity", "total", "percent", "percentage",
)


def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:  # NaN
        return True
    try:
        import pandas as pd
        if v is pd.NaT:
            return True
    except Exception:
        pass
    return False


def _text_of(v) -> str:
    """Mirrors cleaner._text_series for a single scalar: cast to string,
    strip, empty string for null — exact per-value equivalent."""
    if _is_null(v):
        return ""
    return str(v).strip()


def is_placeholder(v) -> bool:
    """Per-value equivalent of cleaner._placeholder_mask."""
    if _is_null(v):
        return True
    text = _text_of(v).lower()
    return text in PLACEHOLDER_TOKENS or bool(_HASH_RE.fullmatch(text)) if text else True


def has_digit(v) -> bool:
    return bool(_DIGIT_RE.search(_text_of(v)))


def has_alpha(v) -> bool:
    return bool(_ALPHA_RE.search(_text_of(v)))


def coerce_numeric_like(v):
    """Per-value equivalent of cleaner._coerce_numeric_like — returns a float
    or None. Only meaningful for object-dtype numeric-like columns."""
    text = _text_of(v).lower()
    text = _MONEY_CHARS_RE.sub("", text)
    if _THOUSANDS_DOT_RE.fullmatch(text):
        text = text.replace(".", "")
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def column_name_implies_numeric(col_name: str) -> bool:
    name = str(col_name).strip().lower()
    return (any(part in name for part in NUMERIC_NAME_PARTS)
            and not name.endswith("_id") and name != "id")


def _row_hash(values) -> int:
    # Order-sensitive tuple hash of a row's values (None-safe) — used purely
    # as a dict key for duplicate-row counting, not for security.
    return hash(tuple(("__NULL__" if _is_null(v) else v) for v in values))


# --- Stats structure ----------------------------------------------------------
# All of this is JSON-serialisable except the two dict-of-dict counters,
# which use string keys already (see to_json/from_json below).

def init_dq_stats(df) -> Dict[str, Any]:
    """One-time O(n) build — unavoidable exactly once, same as any cache's
    first population. Every subsequent apply/undo/reapply updates this
    incrementally instead of rebuilding it."""
    import pandas as pd
    import numpy as np

    n_rows, n_cols = df.shape
    columns = list(df.columns)

    total_missing = 0
    row_hashes: Dict[int, int] = {}
    col_stats: Dict[str, Dict[str, Any]] = {}
    row_placeholder_counts: Dict[Any, int] = {}

    # Per-row pass (needed for row hashes and row placeholder counts anyway —
    # this whole function is the one-time O(n) cost, same as any cache miss).
    for col in columns:
        is_numeric_dtype = pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])
        col_stats[col] = {
            "missing": 0,
            "placeholder": 0,
            "non_placeholder": 0,
            "digit": 0,
            "alpha": 0,
            "distinct": {},   # str(value) -> count, non-null only (for nunique)
            "numeric_issue": 0,
            "is_numeric_dtype": is_numeric_dtype,
            "name_implies_numeric": column_name_implies_numeric(col),
        }

    for idx, row in zip(df.index, df.itertuples(index=False, name=None)):
        row_hashes[_row_hash(row)] = row_hashes.get(_row_hash(row), 0) + 1
        placeholder_count_this_row = 0
        for col, v in zip(columns, row):
            cs = col_stats[col]
            if _is_null(v):
                cs["missing"] += 1
                total_missing += 1
            else:
                key = str(v)
                cs["distinct"][key] = cs["distinct"].get(key, 0) + 1
            ph = is_placeholder(v)
            if ph:
                cs["placeholder"] += 1
                placeholder_count_this_row += 1
            else:
                cs["non_placeholder"] += 1
                if has_digit(v):
                    cs["digit"] += 1
                if has_alpha(v):
                    cs["alpha"] += 1
        row_placeholder_counts[idx] = placeholder_count_this_row

    rows_over_placeholder_threshold = 0
    for col in columns:
        cs = col_stats[col]
        cs["is_numeric_candidate"] = _recompute_numeric_candidate(cs)
    # numeric_issue needs the numeric-like flag known first, then a real scan.
    for col in columns:
        cs = col_stats[col]
        if cs["is_numeric_dtype"] or not cs["is_numeric_candidate"]:
            continue
        cs["numeric_issue"] = _rescan_numeric_issue_for_column(df[col])

    for idx, cnt in row_placeholder_counts.items():
        if n_cols and (cnt / n_cols) >= 0.8:
            rows_over_placeholder_threshold += 1

    # Numeric outlier bounds (per real-numeric-dtype column only).
    num_stats: Dict[str, Dict[str, float]] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        num_stats[col] = _rescan_outlier_bounds(df[col])

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "total_missing": total_missing,
        "row_hashes": row_hashes,
        "col_stats": col_stats,
        "row_placeholder_counts": row_placeholder_counts,
        "rows_over_placeholder_threshold": rows_over_placeholder_threshold,
        "num_stats": num_stats,
    }


def _recompute_numeric_candidate(cs: Dict[str, Any]) -> bool:
    if cs["is_numeric_dtype"]:
        return False  # ground truth `continue`s past real numeric/datetime dtypes entirely
    if cs["name_implies_numeric"]:
        return True
    if cs["non_placeholder"] < 5:
        return False
    digit_ratio = cs["digit"] / cs["non_placeholder"]
    alpha_ratio = cs["alpha"] / cs["non_placeholder"]
    return digit_ratio >= 0.85 and alpha_ratio <= 0.2


def _rescan_numeric_issue_for_column(series) -> int:
    """Bounded fallback: full rescan of ONE column (O(column height), never
    O(dataset)) — used on init, and on the rare occasion a column's
    numeric-like status flips due to an edit."""
    count = 0
    for v in series:
        if is_placeholder(v):
            continue
        if not has_digit(v):
            continue
        if coerce_numeric_like(v) is None:
            count += 1
    return count


def _rescan_outlier_bounds(series) -> Dict[str, float]:
    """Bounded: full rescan of ONE numeric column (O(column height)) to get
    exact Q1/Q3/outlier count — done only when that specific column is
    touched, never for the whole dataset."""
    clean = series.dropna()
    if len(clean) <= 2:
        return {"q1": None, "q3": None, "outlier_count": 0, "n": int(len(clean))}
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((clean < lower) | (clean > upper)).sum())
    return {"q1": q1, "q3": q3, "outlier_count": outlier_count, "n": int(len(clean))}


# --- Incremental update --------------------------------------------------------

def apply_manifest_to_dq_stats(stats: Dict[str, Any], manifest: Dict[str, Any],
                                direction: str, new_df) -> Dict[str, Any]:
    """Update `stats` in place to reflect applying `manifest` in `direction`
    ('forward' = after values win, 'reverse' = before values win). `new_df`
    is the dataframe AFTER this manifest was applied — used only for bounded,
    bounded-bounded per-column rescans (numeric-like flips, a touched
    numeric column's quantiles), never for a full-dataset scan.

    Every code path here is O(manifest size) or O(one column's height) —
    never O(dataset size).
    """
    value_key_new = "after" if direction == "forward" else "before"
    value_key_old = "before" if direction == "forward" else "after"

    columns = stats["columns"]
    col_stats = stats["col_stats"]

    # --- 1. Group changed_cells by row (a single action can touch several
    # columns of the same row — row-hash and row-placeholder-count updates
    # need the FULL row's old/new values, not per-cell deltas). ---
    rows_touched: Dict[Any, Dict[str, Any]] = {}
    for cell in manifest.get("changed_cells", []):
        row_key = cell["row"]
        rows_touched.setdefault(row_key, {})[cell["col"]] = (cell[value_key_old], cell[value_key_new])

    touched_candidate_columns = set()

    for row_key, col_changes in rows_touched.items():
        row_label = _restore_row_label(row_key, new_df.index)
        if row_label not in new_df.index:
            continue  # this row no longer exists (handled by add/drop below elsewhere)
        current_row = new_df.loc[row_label]
        # Reconstruct the row's OLD full values (swap back just the touched columns)
        old_row_values = {c: current_row[c] for c in columns}
        for c, (old_v, new_v) in col_changes.items():
            old_row_values[c] = old_v

        old_hash = _row_hash(tuple(old_row_values[c] for c in columns))
        new_hash = _row_hash(tuple(current_row[c] for c in columns))
        if old_hash != new_hash:
            _decrement_row_hash(stats, old_hash)
            stats["row_hashes"][new_hash] = stats["row_hashes"].get(new_hash, 0) + 1

        old_ph_count = stats["row_placeholder_counts"].get(row_label, 0)
        ph_delta = 0

        for c, (old_v, new_v) in col_changes.items():
            cs = col_stats[c]
            old_null, new_null = _is_null(old_v), _is_null(new_v)
            if old_null and not new_null:
                cs["missing"] -= 1
                stats["total_missing"] -= 1
            elif new_null and not old_null:
                cs["missing"] += 1
                stats["total_missing"] += 1

            if not old_null:
                _dec_distinct(cs, old_v)
            if not new_null:
                _inc_distinct(cs, new_v)

            old_ph, new_ph = is_placeholder(old_v), is_placeholder(new_v)
            if old_ph and not new_ph:
                cs["placeholder"] -= 1
                cs["non_placeholder"] += 1
                if has_digit(new_v):
                    cs["digit"] += 1
                if has_alpha(new_v):
                    cs["alpha"] += 1
                ph_delta -= 1
            elif new_ph and not old_ph:
                cs["placeholder"] += 1
                cs["non_placeholder"] -= 1
                if has_digit(old_v):
                    cs["digit"] -= 1
                if has_alpha(old_v):
                    cs["alpha"] -= 1
                ph_delta += 1
            elif not old_ph and not new_ph:
                # Value changed but stayed non-placeholder on both sides —
                # digit/alpha membership can still flip (e.g. "abc" -> "123").
                if has_digit(old_v) and not has_digit(new_v):
                    cs["digit"] -= 1
                elif has_digit(new_v) and not has_digit(old_v):
                    cs["digit"] += 1
                if has_alpha(old_v) and not has_alpha(new_v):
                    cs["alpha"] -= 1
                elif has_alpha(new_v) and not has_alpha(old_v):
                    cs["alpha"] += 1

            # Numeric-issue bookkeeping only applies to non-numeric-dtype
            # numeric-like columns, and only once we know that classification
            # (checked/settled after all counter updates, below).
            touched_candidate_columns.add(c)

        new_ph_count = old_ph_count + ph_delta
        stats["row_placeholder_counts"][row_label] = new_ph_count
        n_cols = stats["n_cols"]
        was_over = n_cols and (old_ph_count / n_cols) >= 0.8
        now_over = n_cols and (new_ph_count / n_cols) >= 0.8
        if was_over and not now_over:
            stats["rows_over_placeholder_threshold"] -= 1
        elif now_over and not was_over:
            stats["rows_over_placeholder_threshold"] += 1

    # --- 2. Settle numeric-like status + numeric_issue for touched
    # object-dtype columns. A column's classification is threshold-based.
    # function of running counters that were just updated above — if it
    # flipped, the *only* correct move is a full rescan of THAT ONE column
    # (bounded to its height, never the whole dataset), because numeric_issue
    # under the old status has no cheap relationship to numeric_issue under
    # the new one. ---
    for c in touched_candidate_columns:
        cs = col_stats[c]
        if cs["is_numeric_dtype"]:
            continue
        was_candidate = cs["is_numeric_candidate"]
        now_candidate = _recompute_numeric_candidate(cs)
        cs["is_numeric_candidate"] = now_candidate
        if now_candidate != was_candidate:
            cs["numeric_issue"] = _rescan_numeric_issue_for_column(new_df[c]) if now_candidate else 0
        elif now_candidate:
            # Classification held, but specific changed cells may have
            # flipped their own individual validity. Re-derive just those
            # cells' contribution (O(cells changed in this column), not
            # O(column height)).
            for row_key, col_changes in rows_touched.items():
                if c not in col_changes:
                    continue
                old_v, new_v = col_changes[c]
                old_invalid = (not is_placeholder(old_v)) and has_digit(old_v) and coerce_numeric_like(old_v) is None
                new_invalid = (not is_placeholder(new_v)) and has_digit(new_v) and coerce_numeric_like(new_v) is None
                if old_invalid and not new_invalid:
                    cs["numeric_issue"] -= 1
                elif new_invalid and not old_invalid:
                    cs["numeric_issue"] += 1

    # --- 3. Rows added / dropped (direction-aware, mirroring build_manifest_patch) ---
    rows_to_remove = manifest.get("dropped_rows", []) if direction == "forward" else manifest.get("added_rows", [])
    rows_to_add    = manifest.get("added_rows", [])    if direction == "forward" else manifest.get("dropped_rows", [])

    for item in rows_to_remove:
        row_label = _restore_row_label(item["row"], new_df.index)
        _remove_row_contribution(stats, row_label, item["data"], new_df)
    for item in rows_to_add:
        row_label = _restore_row_label(item["row"], new_df.index)
        _add_row_contribution(stats, row_label, item["data"], new_df)

    stats["n_rows"] = int(new_df.shape[0])

    # --- 4. Columns added / dropped ---
    cols_to_remove = manifest.get("dropped_columns", []) if direction == "forward" else manifest.get("added_columns", [])
    cols_to_add    = manifest.get("added_columns", [])    if direction == "forward" else manifest.get("dropped_columns", [])

    columns_changed = bool(cols_to_remove or cols_to_add)
    for item in cols_to_remove:
        _drop_column_from_stats(stats, item["col"], item["data"], new_df.index)
    for item in cols_to_add:
        _add_column_to_stats(stats, item["col"], item["data"], new_df)

    stats["n_cols"] = len(stats["columns"])

    if columns_changed:
        # Row identity (used for duplicate detection) is a tuple of ALL
        # column values — adding/dropping a column changes every row's
        # identity, so the hash table needs a full, bounded O(rows) rebuild.
        # This never fires on an ordinary cell edit, only on this rarer
        # structural change.
        _rebuild_row_hashes(stats, new_df)
        n_cols = stats["n_cols"]
        stats["rows_over_placeholder_threshold"] = sum(
            1 for cnt in stats["row_placeholder_counts"].values() if n_cols and (cnt / n_cols) >= 0.8
        )

    # Touched numeric-dtype columns: quantiles/outlier bounds need a bounded
    # per-column rescan (exact quantile maintenance under arbitrary edits
    # needs an order-statistics structure — out of scope here; this still
    # bounds the cost to ONE column's height instead of the whole dataset).
    # A row add/drop shifts EVERY numeric column's quantiles, not just the
    # ones a cell-edit touched, so those get refreshed unconditionally too.
    rows_changed = bool(rows_to_remove or rows_to_add)
    touched_numeric_cols = {c for c in touched_candidate_columns if col_stats.get(c, {}).get("is_numeric_dtype")}
    for c in list(stats["num_stats"].keys()):
        if c not in new_df.columns:
            del stats["num_stats"][c]
        elif c in touched_numeric_cols or rows_changed:
            stats["num_stats"][c] = _rescan_outlier_bounds(new_df[c])

    # --- Dtype-flip safety net ---
    # A column's numeric/non-numeric CATEGORY was assumed fixed at init, but
    # pandas can silently flip it — e.g. re-adding a row reconstructed from
    # the JSON-safe manifest (where NaN became Python None) upcasts a
    # float64 column to object dtype on concat (a real, observed case, not
    # theoretical — found via randomized testing). Rather than assume this
    # can't happen, verify every touched-by-structure column every time and
    # do a bounded, one-column full re-init if it did — O(that column's
    # height), never the whole dataset.
    if rows_changed or columns_changed:
        for col in list(stats["columns"]):
            if col not in new_df.columns:
                continue
            cs = stats["col_stats"].get(col)
            if cs is None:
                continue
            actual_is_numeric = bool(pd.api.types.is_numeric_dtype(new_df[col]) or pd.api.types.is_datetime64_any_dtype(new_df[col]))
            if actual_is_numeric != cs["is_numeric_dtype"]:
                _resettle_column_dtype_flip(stats, col, new_df)

    return stats


def _resettle_column_dtype_flip(stats, col, new_df):
    """Full re-init of ONE column's stats (O(column height), bounded — never
    O(dataset)) after its numeric/non-numeric category itself changed. Every
    counter this column contributes (missing, placeholder, digit/alpha,
    distinct values, numeric_issue, num_stats membership) is rebuilt, since
    the fundamental interpretation of its values changed — there's no cheap
    delta from the old category to the new one."""
    series = new_df[col]
    old_cs = stats["col_stats"][col]

    stats["total_missing"] -= old_cs["missing"]
    if col in stats["num_stats"]:
        del stats["num_stats"][col]

    is_numeric_dtype = bool(pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series))
    cs = {
        "missing": 0, "placeholder": 0, "non_placeholder": 0, "digit": 0, "alpha": 0,
        "distinct": {}, "numeric_issue": 0,
        "is_numeric_dtype": is_numeric_dtype,
        "name_implies_numeric": old_cs["name_implies_numeric"],
    }
    for idx, v in series.items():
        if _is_null(v):
            cs["missing"] += 1
        else:
            cs["distinct"][str(v)] = cs["distinct"].get(str(v), 0) + 1
        ph = is_placeholder(v)
        if ph:
            cs["placeholder"] += 1
        else:
            cs["non_placeholder"] += 1
            if has_digit(v):
                cs["digit"] += 1
            if has_alpha(v):
                cs["alpha"] += 1

    stats["total_missing"] += cs["missing"]
    cs["is_numeric_candidate"] = _recompute_numeric_candidate(cs)
    if not is_numeric_dtype and cs["is_numeric_candidate"]:
        cs["numeric_issue"] = _rescan_numeric_issue_for_column(series)
    stats["col_stats"][col] = cs
    if is_numeric_dtype:
        stats["num_stats"][col] = _rescan_outlier_bounds(series)

    # row_placeholder_counts is intentionally left untouched here:
    # is_placeholder() is value-based (NaN and None both count as null
    # identically — see _is_null), not dtype-based, so a pure dtype-category
    # flip (e.g. float64 -> object from a concat quirk) cannot change any
    # individual row's placeholder status even though the column's AGGREGATE
    # numeric-like/outlier bookkeeping above does need a full rebuild.
    # This keeps the safety net itself bounded to O(one column's height),
    # never O(rows x cols).


def _restore_row_label(row_key: str, sample_index):
    import pandas as pd
    if len(sample_index) and pd.api.types.is_integer_dtype(sample_index.dtype):
        try:
            return int(row_key)
        except (TypeError, ValueError):
            return row_key
    return row_key


def _decrement_row_hash(stats, h):
    rh = stats["row_hashes"]
    if h in rh:
        rh[h] -= 1
        if rh[h] <= 0:
            del rh[h]


def _dec_distinct(cs, v):
    key = str(v)
    d = cs["distinct"]
    if key in d:
        d[key] -= 1
        if d[key] <= 0:
            del d[key]


def _inc_distinct(cs, v):
    key = str(v)
    cs["distinct"][key] = cs["distinct"].get(key, 0) + 1


def _remove_row_contribution(stats, row_label, row_data: Dict[str, Any], new_df):
    columns = stats["columns"]
    values = [row_data.get(c) for c in columns]
    h = _row_hash(tuple(values))
    _decrement_row_hash(stats, h)
    ph_count = 0
    for c in columns:
        if c not in stats["col_stats"]:
            continue
        cs = stats["col_stats"][c]
        v = row_data.get(c)
        if _is_null(v):
            cs["missing"] -= 1
            stats["total_missing"] -= 1
        else:
            _dec_distinct(cs, v)
        if is_placeholder(v):
            cs["placeholder"] -= 1
            ph_count += 1
        else:
            cs["non_placeholder"] -= 1
            if has_digit(v):
                cs["digit"] -= 1
            if has_alpha(v):
                cs["alpha"] -= 1
    stats["row_placeholder_counts"].pop(row_label, None)
    n_cols = stats["n_cols"]
    if n_cols and (ph_count / n_cols) >= 0.8:
        stats["rows_over_placeholder_threshold"] -= 1
    _resettle_numeric_candidates(stats, columns, new_df)


def _add_row_contribution(stats, row_label, row_data: Dict[str, Any], new_df):
    columns = stats["columns"]
    values = [row_data.get(c) for c in columns]
    h = _row_hash(tuple(values))
    stats["row_hashes"][h] = stats["row_hashes"].get(h, 0) + 1
    ph_count = 0
    for c in columns:
        if c not in stats["col_stats"]:
            continue
        cs = stats["col_stats"][c]
        v = row_data.get(c)
        if _is_null(v):
            cs["missing"] += 1
            stats["total_missing"] += 1
        else:
            _inc_distinct(cs, v)
        if is_placeholder(v):
            cs["placeholder"] += 1
            ph_count += 1
        else:
            cs["non_placeholder"] += 1
            if has_digit(v):
                cs["digit"] += 1
            if has_alpha(v):
                cs["alpha"] += 1
    stats["row_placeholder_counts"][row_label] = ph_count
    n_cols = stats["n_cols"]
    if n_cols and (ph_count / n_cols) >= 0.8:
        stats["rows_over_placeholder_threshold"] += 1
    _resettle_numeric_candidates(stats, columns, new_df)


def _resettle_numeric_candidates(stats, columns, new_df):
    """After a row add/drop shifts a column's digit/alpha/non-placeholder
    counters, its numeric-like status may cross the threshold. If it
    does, numeric_issue for that column has no cheap relationship to its old
    value, so the only correct move is a bounded rescan of that ONE column
    (O(column height), never the whole dataset) — same rule as the cell-edit
    path."""
    for c in columns:
        cs = stats["col_stats"].get(c)
        if not cs or cs["is_numeric_dtype"]:
            continue
        was = cs["is_numeric_candidate"]
        now = _recompute_numeric_candidate(cs)
        cs["is_numeric_candidate"] = now
        if now != was:
            cs["numeric_issue"] = _rescan_numeric_issue_for_column(new_df[c]) if (now and c in new_df.columns) else 0


def _rebuild_row_hashes(stats, new_df):
    """Full O(n_rows) rebuild of the row-hash table — used only after a
    column add/drop, since that changes every row's identity tuple. This is
    a rare, structural event, not a per-cell one, so bounding it to O(rows)
    (not O(rows x cols), and never re-triggered by ordinary cell edits) is
    the right trade-off."""
    columns = stats["columns"]
    row_hashes: Dict[int, int] = {}
    for _, row in new_df[columns].iterrows():
        h = _row_hash(tuple(row[c] for c in columns))
        row_hashes[h] = row_hashes.get(h, 0) + 1
    stats["row_hashes"] = row_hashes


def _drop_column_from_stats(stats, col, col_data: Dict[str, Any], index_sample):
    """`col_data` is the manifest's own {row_label: value} for this column —
    using it (rather than needing the live pre-drop dataframe) keeps this
    consistent with the rest of the module's 'the manifest is the patch'
    design, and lets total_missing / row_placeholder_counts be corrected
    exactly instead of drifting. `index_sample` is any index (e.g. new_df's)
    used only to know whether row labels are integer- or string-typed."""
    if col in stats["col_stats"]:
        cs = stats["col_stats"][col]
        stats["total_missing"] -= cs["missing"]
        del stats["col_stats"][col]
    if col in stats["columns"]:
        stats["columns"].remove(col)
    if col in stats["num_stats"]:
        del stats["num_stats"][col]

    # Every row loses this column's placeholder contribution (dropping a
    # column changes the 80%-of-columns denominator too — recomputed by the
    # caller after n_cols is updated).
    for row_key, v in col_data.items():
        row_label = _restore_row_label(row_key, index_sample)
        if row_label in stats["row_placeholder_counts"] and is_placeholder(v):
            stats["row_placeholder_counts"][row_label] = max(0, stats["row_placeholder_counts"][row_label] - 1)


def _add_column_to_stats(stats, col, data: Dict[str, Any], new_df):
    import pandas as pd
    stats["columns"].append(col)
    series = new_df[col] if col in new_df.columns else None
    is_numeric_dtype = bool(series is not None and (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)))
    cs = {
        "missing": 0, "placeholder": 0, "non_placeholder": 0, "digit": 0, "alpha": 0,
        "distinct": {}, "numeric_issue": 0,
        "is_numeric_dtype": is_numeric_dtype,
        "name_implies_numeric": column_name_implies_numeric(col),
    }
    for row_label, v in data.items():
        if _is_null(v):
            cs["missing"] += 1
            stats["total_missing"] += 1
        else:
            _inc_distinct(cs, v)
        ph = is_placeholder(v)
        if ph:
            cs["placeholder"] += 1
        else:
            cs["non_placeholder"] += 1
            if has_digit(v):
                cs["digit"] += 1
            if has_alpha(v):
                cs["alpha"] += 1
        restored_label = _restore_row_label(row_label, new_df.index)
        stats["row_placeholder_counts"][restored_label] = stats["row_placeholder_counts"].get(restored_label, 0) + (1 if ph else 0)
    cs["is_numeric_candidate"] = _recompute_numeric_candidate(cs)
    if not is_numeric_dtype and cs["is_numeric_candidate"] and series is not None:
        cs["numeric_issue"] = _rescan_numeric_issue_for_column(series)
    stats["col_stats"][col] = cs
    if is_numeric_dtype and series is not None:
        stats["num_stats"][col] = _rescan_outlier_bounds(series)
    # Recompute rows_over_placeholder_threshold from scratch (O(rows), but
    # only on a column add — a rare structural event, not a per-cell one).
    n_cols = stats["n_cols"] + 1
    stats["rows_over_placeholder_threshold"] = sum(
        1 for cnt in stats["row_placeholder_counts"].values() if n_cols and (cnt / n_cols) >= 0.8
    )


# --- Score derivation -----------------------------------------------------------

def score_from_dq_stats(stats: Dict[str, Any]) -> float:
    n_rows, n_cols = stats["n_rows"], stats["n_cols"]
    total_cells = n_rows * n_cols
    if n_rows == 0 or n_cols == 0 or total_cells == 0:
        return 0.0

    missing_pct = (stats["total_missing"] / total_cells) * 100

    n_unique_rows = len(stats["row_hashes"])
    dup_pct = ((n_rows - n_unique_rows) / n_rows) * 100 if n_rows else 0

    placeholder_cells = sum(cs["placeholder"] for cs in stats["col_stats"].values())
    placeholder_pct = (placeholder_cells / total_cells) * 100

    numeric_issue_cells = sum(cs["numeric_issue"] for cs in stats["col_stats"].values())
    numeric_issue_pct = (numeric_issue_cells / total_cells) * 100

    const_col_count = 0
    for cs in stats["col_stats"].values():
        non_empty = any(True for _ in cs["distinct"]) or cs["missing"] < n_rows
        if not non_empty:
            continue  # fully-empty column excluded, matching ground truth
        n_distinct_nonnull = len(cs["distinct"])
        if n_distinct_nonnull <= 1:
            const_col_count += 1
    const_col_pct = (const_col_count / n_cols) * 100 if n_cols else 0

    row_placeholder_pct = (stats["rows_over_placeholder_threshold"] / n_rows) * 100 if n_rows else 0

    outlier_total = 0
    outlier_denominator = 0
    for c, ns in stats["num_stats"].items():
        outlier_total += ns.get("outlier_count", 0)
        outlier_denominator += ns.get("n", 0)
    # Ground truth divides by num_df.size (n_rows x n_numeric_cols, INCLUDING
    # NaNs), not just non-null count — match that denominator exactly.
    n_numeric_cols = len(stats["num_stats"])
    outlier_pct = (outlier_total / (n_rows * n_numeric_cols) * 100) if (n_rows and n_numeric_cols) else 0

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
