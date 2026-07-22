import os
import sys
import time
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd

import app as app_module


ROWS = int(os.environ.get("INDATAOUT_BENCH_ROWS", "135000"))
SID = "bench_large_dataset"
FILENAME = "bench_large.csv"
BENCH_APPLY = (
    os.environ.get("INDATAOUT_BENCH_APPLY", "0").lower() in ("1", "true", "yes")
    or "--apply" in sys.argv
)


def make_large_frame(rows=ROWS):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "booking_id": [f"B{i:06d}" for i in range(rows)],
        "property_id": rng.integers(10000, 99999, size=rows),
        "booking_date": pd.date_range("2022-01-01", periods=rows, freq="min").strftime("%d-%m-%y"),
        "check_in_date": pd.date_range("2022-02-01", periods=rows, freq="min").strftime("%-m/%-d/%Y"),
        "checkout_date": pd.date_range("2022-02-02", periods=rows, freq="min").strftime("%-m/%-d/%Y"),
        "no_guests": rng.integers(1, 6, size=rows),
        "room_category": rng.choice(["RT1", "RT2", "RT3"], size=rows),
        "booking_platform": rng.choice(["direct online", "others", "logtrip", "tripster"], size=rows),
        "ratings_given": rng.choice([1, 2, 3, 4, 5, np.nan], size=rows, p=[.1, .15, .2, .25, .25, .05]),
        "booking_status": rng.choice(["Checked Out", "Cancelled", "No Show"], size=rows),
        "revenue_generated": rng.normal(9000, 2000, size=rows).round(2),
        "revenue_realized": rng.normal(7500, 1800, size=rows).round(2),
    })
    if rows > 10:
        df.loc[2, "revenue_generated"] = 9_100_000
        df.loc[5, "room_category"] = " RT1 "
        df.loc[8, "ratings_given"] = np.nan
        df.loc[10, "booking_id"] = df.loc[9, "booking_id"]
        df.loc[10, df.columns != "booking_id"] = df.loc[9, df.columns != "booking_id"]
    df["empty_col"] = np.nan
    df["constant_col"] = "same"
    return df


def seed_session(client):
    folder = os.path.join(app_module.UPLOAD_FOLDER, SID)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, FILENAME)
    if not os.path.exists(path):
        make_large_frame().to_csv(path, index=False)
    with client.session_transaction() as sess:
        sess["upload_sid"] = SID
        sess["upload_filename"] = FILENAME
        sess["dataset_version"] = 0
    app_module._df_cache.pop(SID, None)
    app_module._dq_cache.pop(SID, None)
    app_module._dq_after_cache.pop(SID, None)
    app_module._analysis_cache.pop(SID, None)
    app_module.save_current_snapshot(SID, app_module.get_cached_df(SID, path))
    return path


@contextmanager
def no_db_project():
    with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
         mock.patch.object(app_module.db, "get_project", return_value=None):
        yield


def post_action(client, name, data, persist=False):
    payload = {
        "filename": FILENAME,
        "dataset_version": "0",
        "preview_scope": "section",
        **data,
    }
    if persist:
        payload.update({
            "persist_apply": "1",
            "operation_token": f"bench-{name.replace(' ', '-')}-{time.time_ns()}",
            "action_name": name,
        })
    start = time.perf_counter()
    response = client.post("/preview_clean", data=payload)
    elapsed = time.perf_counter() - start
    size_mb = len(response.get_data()) / 1024 / 1024
    ok = response.status_code == 200 and bool(response.get_json(silent=True) or {})
    return name, elapsed, response.status_code, size_mb, ok


def main():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    seed_session(client)
    actions = [
        ("editor one cell", {
            "preview_scope": "manual",
            "action_scope": "datasetEditorSlide",
            "cell_update[0][no_guests]": "6",
        }),
        ("manual missing one cell", {
            "preview_scope": "manual",
            "action_scope": "missingValuesGroup",
            "missing_cell[8][ratings_given]": "4",
        }),
        ("drop selected columns", {
            "action_scope": "dropSelectedColumnsGroup",
            "drop_column[empty_col]": "1",
        }),
        ("drop empty columns", {
            "action_scope": "dataIntegrityGroup",
            "drop_empty_columns": "1",
        }),
        ("drop constant columns", {
            "action_scope": "dataIntegrityGroup",
            "drop_constant_columns": "1",
        }),
        ("trim whitespace", {
            "action_scope": "dataIntegrityGroup",
            "trim_whitespace": "1",
        }),
        ("fill numeric missing", {
            "action_scope": "missingValuesGroup",
            "missing_strategy": "fill",
            "fill_numeric_col[ratings_given]": "median",
            "fill_numeric": "skip",
        }),
        ("data type integer", {
            "action_scope": "dataTypesGroup",
            "dtype_map[ratings_given]": "int",
        }),
    ]
    only = os.environ.get("INDATAOUT_BENCH_ONLY", "").strip().lower()
    if only:
        actions = [item for item in actions if only in item[0].lower()]
    print(f"Rows: {ROWS}", flush=True)
    print(f"Mode: {'apply' if BENCH_APPLY else 'preview'}", flush=True)
    print("action,status,seconds,response_mb,ok", flush=True)
    with no_db_project():
        for item in actions:
            if BENCH_APPLY:
                seed_session(client)
            name, elapsed, status, size_mb, ok = post_action(client, *item, persist=BENCH_APPLY)
            print(f"{name},{status},{elapsed:.3f},{size_mb:.2f},{ok}", flush=True)


if __name__ == "__main__":
    main()
