"""
Trusted local custom Pandas code runner for inDataOut.

This module is intentionally for local/trusted use. The public app should keep
INDATAOUT_TRUSTED_LOCAL disabled before network exposure so arbitrary Python
code is not exposed to untrusted users.

The runner reports observed changes between the input DataFrame copy and the
post-execution DataFrame. It deliberately does not try to infer whether a cell
was meant only for display or meant to mutate data; applying code means running
it exactly as written.
"""

from __future__ import annotations

import ast
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _row_number(label: Any, fallback_position: int | None = None) -> Any:
    if isinstance(label, (int, np.integer)):
        return int(label) + 1
    if fallback_position is not None:
        return fallback_position + 1
    return str(label)


def _changed_cell_details(before: pd.DataFrame, after: pd.DataFrame, limit: int = 80) -> List[Dict[str, Any]]:
    """Return capped cell-level evidence for custom-code previews."""
    details: List[Dict[str, Any]] = []
    common_index = before.index.intersection(after.index)
    common_columns = list(before.columns.intersection(after.columns))
    if len(common_index) == 0 or len(common_columns) == 0:
        return details
    # Vectorized diff: only inspect rows that actually changed
    before_sub = before.loc[common_index, common_columns]
    after_sub = after.loc[common_index, common_columns]
    diff_mask = pd.DataFrame(False, index=common_index, columns=common_columns)
    for col in common_columns:
        bcol = before_sub[col]
        acol = after_sub[col]
        diff_mask[col] = bcol.ne(acol) & ~(bcol.isna() & acol.isna())
    row_has_diff = diff_mask.any(axis=1)
    changed_indices = common_index[row_has_diff]
    for pos, idx in enumerate(changed_indices):
        row_diff = diff_mask.loc[idx]
        for col in row_diff[row_diff].index:
            before_value = before.at[idx, col]
            after_value = after.at[idx, col]
            details.append({
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "row_number": _row_number(idx, pos),
                "column": str(col),
                "before": "" if pd.isna(before_value) else str(before_value),
                "after": "" if pd.isna(after_value) else str(after_value),
            })
            if len(details) >= limit:
                return details
    return details


def _serialise_value(val: Any) -> Dict[str, Any]:
    """Turn any Python value into a JSON-serialisable output block."""
    if isinstance(val, pd.DataFrame):
        cols = list(val.columns)
        rows = val.head(200).copy()
        for c in rows.columns:
            if pd.api.types.is_datetime64_any_dtype(rows[c]):
                rows[c] = rows[c].dt.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "type": "dataframe",
            "columns": cols,
            "rows": rows.to_dict(orient="records"),
            "total_rows": len(val),
            "total_cols": len(cols),
        }
    if isinstance(val, pd.Series):
        s = val.reset_index()
        s.columns = [str(c) for c in s.columns]
        return {
            "type": "series",
            "name": str(val.name) if val.name is not None else "value",
            "columns": list(s.columns),
            "rows": s.head(200).to_dict(orient="records"),
            "total_rows": len(val),
        }
    # scalar / numpy scalar
    try:
        import math
        f = float(val)
        return {"type": "scalar", "value": None if math.isnan(f) or math.isinf(f) else f}
    except (TypeError, ValueError):
        pass
    return {"type": "scalar", "value": str(val)}


def run_custom_code(df: pd.DataFrame, code: str) -> Dict[str, Any]:
    """
    Execute trusted local Pandas code against a dataframe copy.

    Captures:
    - print() output
    - The return value of the last expression (DataFrame, Series, scalar)
    - df mutations / reassignments
    - result = ... assignments

    Returns df (possibly mutated) + operation metadata + output blocks for
    display. Raises validation/runtime errors to the caller; this function does
    not persist state.
    """
    if not code or not code.strip():
        raise ValueError("Enter Pandas code before previewing or applying.")

    before = df.copy(deep=True)
    working = df.copy(deep=True)
    local_ns: Dict[str, Any] = {"df": working, "pd": pd, "np": np, "result": None}
    global_ns: Dict[str, Any] = {"pd": pd, "np": np}

    # Split into all-but-last and last line to detect bare expressions
    lines = [l for l in code.strip().splitlines() if l.strip()]
    last_line = lines[-1] if lines else ""
    preceding = "\n".join(lines[:-1])

    last_is_expr = False
    try:
        tree = ast.parse(last_line.strip())
        last_is_expr = (
            len(tree.body) == 1 and
            isinstance(tree.body[0], ast.Expr)
        )
    except SyntaxError:
        last_is_expr = False

    stdout_buf = io.StringIO()
    expr_result: Optional[Any] = None

    with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
        if last_is_expr:
            if preceding.strip():
                exec(preceding, global_ns, local_ns)
            try:
                expr_result = eval(last_line.strip(), global_ns, local_ns)
                # If it's a DataFrame/Series treat it as display output,
                # but only replace df if it looks like a filter (fewer rows)
                if isinstance(expr_result, pd.DataFrame):
                    local_ns["result"] = expr_result
            except Exception:
                exec(last_line, global_ns, local_ns)
        else:
            exec(code, global_ns, local_ns)

    printed_text = stdout_buf.getvalue()

    # Determine the working df after execution
    explicit_result = local_ns.get("result")
    candidate = explicit_result if isinstance(explicit_result, pd.DataFrame) else local_ns.get("df")
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError("Custom code must leave df as a pandas DataFrame or assign a DataFrame to result.")

    after = candidate.copy(deep=True)

    # Build output blocks for display in the UI
    output_blocks: List[Dict[str, Any]] = []

    if printed_text.strip():
        output_blocks.append({"type": "print", "text": printed_text.rstrip()})

    if expr_result is not None and not isinstance(expr_result, pd.DataFrame):
        # Series, scalar, bool, int, float etc.
        output_blocks.append(_serialise_value(expr_result))
    elif expr_result is not None and isinstance(expr_result, pd.DataFrame):
        output_blocks.append(_serialise_value(expr_result))

    # Diff metadata
    common_index = before.index.intersection(after.index)
    common_columns = before.columns.intersection(after.columns)
    changed_details = _changed_cell_details(before, after)
    changed_columns = sorted({item["column"] for item in changed_details})
    changed_rows = sorted({item["row_number"] for item in changed_details}, key=lambda v: str(v))

    operation = {
        "action": "custom_pandas_code",
        "code": code.strip(),
        "rows_before": int(before.shape[0]),
        "rows_after": int(after.shape[0]),
        "columns_before": int(before.shape[1]),
        "columns_after": int(after.shape[1]),
        "rows_removed": int(len(before.index.difference(after.index))),
        "rows_added": int(len(after.index.difference(before.index))),
        "columns_removed": [str(col) for col in before.columns.difference(after.columns)],
        "columns_added": [str(col) for col in after.columns.difference(before.columns)],
        "changed_cells": len(changed_details),
        "changed_rows": changed_rows[:80],
        "changed_columns": changed_columns,
        "changed_cell_details": changed_details,
        "common_rows_checked": int(len(common_index)),
        "common_columns_checked": int(len(common_columns)),
        "output_blocks": output_blocks,
    }
    return {"df": after, "operation": operation}
