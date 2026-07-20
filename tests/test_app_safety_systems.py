"""Integration tests for app-level safety systems.

The scenarios here exercise CSRF/error envelopes, bounded editor paging,
idempotent committed Apply requests, authoritative history creation, and
failure behavior around snapshots/history.
"""

import io
import logging
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import app as app_module


class AppSafetySystemsTests(unittest.TestCase):
    """Flask route tests using isolated temporary upload storage."""

    def setUp(self):
        self.old_upload = app_module.UPLOAD_FOLDER
        self.tmp = tempfile.TemporaryDirectory()
        app_module.UPLOAD_FOLDER = self.tmp.name
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.UPLOAD_FOLDER = self.old_upload
        self.tmp.cleanup()

    def _seed_session_file(self, name="sample.csv", content="x,y\n a ,1\nb,2\n"):
        sid = "testsid"
        folder = os.path.join(app_module.UPLOAD_FOLDER, sid)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        with self.client.session_transaction() as sess:
            sess["upload_sid"] = sid
            sess["upload_filename"] = name
        return sid, name, path

    def _seed_project_session_file(self, name="sample.csv", content="x,y\n1,A\n2,B\n2,B\n"):
        sid, filename, path = self._seed_session_file(name=name, content=content)
        with self.client.session_transaction() as sess:
            sess["project_id"] = 123
        return sid, filename, path

    def test_structured_csrf_error_schema(self):
        app_module.app.config["TESTING"] = False
        client = app_module.app.test_client()
        response = client.post("/preview_clean", json={"filename": "x.csv"})
        self.assertEqual(response.status_code, 403)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"]["code"], "CSRF_TOKEN_INVALID")
        self.assertIn("request_id", payload["error"])
        app_module.app.config["TESTING"] = True

    def test_editor_page_returns_bounded_rows(self):
        self._seed_session_file(content="a,b\n1,2\n3,4\n5,6\n")
        response = self.client.get("/api/editor-page?page=2&per_page=2")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["rows_total"], 3)
        self.assertEqual(payload["rows_displayed"], 1)
        self.assertEqual(payload["rows"][0]["a"], 5)

    def test_drop_selected_columns_preserves_exact_form_names(self):
        form = app_module._FormProxy({
            "drop_column[Unnamed: 15]": ["1"],
            "drop_column[Customer Name]": ["1"],
        })
        options = app_module.parse_clean_options(
            form,
            default_enabled=False,
            default_missing_strategy="ignore",
        )
        self.assertEqual(options["drop_selected_columns"], ["Unnamed: 15", "Customer Name"])

    def test_drop_selected_columns_route_uses_exact_authoritative_names(self):
        _sid, filename, _path = self._seed_session_file(
            content="Customer Name,Unnamed: 15,keep\nA,,1\nB,legacy,2\n"
        )
        response = self.client.post("/preview_clean", data={
            "filename": filename,
            "preview_scope": "feature",
            "drop_column[Customer Name]": "1",
            "drop_column[Unnamed: 15]": "1",
        })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["report"]["cleaned_columns"], ["keep"])
        operation = next(op for op in payload["report"]["operations"] if op["action"] == "drop_selected_columns")
        self.assertEqual(operation["columns_dropped"], ["Customer Name", "Unnamed: 15"])

    def test_committed_exact_column_drop_creates_history_and_snapshot(self):
        sid, filename, _path = self._seed_project_session_file(
            content="Customer Name,Unnamed: 15,keep\nA,,1\nB,legacy,2\n"
        )

        def fake_atomic(**kwargs):
            kwargs["before_commit"](91)
            return 91

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=fake_atomic):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "preview_scope": "feature",
                "persist_apply": "1",
                "action_scope": "dataIntegrityGroup",
                "action_name": "Applied feature only: Drop Selected Columns",
                "drop_column[Customer Name]": "1",
                "drop_column[Unnamed: 15]": "1",
            })

        self.assertEqual(response.status_code, 200)
        history = response.get_json()["history_entry"]
        self.assertEqual(history["id"], 91)
        operation = next(op for op in history["action_config"]["operations"] if op["action"] == "drop_selected_columns")
        self.assertEqual(operation["columns_dropped"], ["Customer Name", "Unnamed: 15"])
        self.assertEqual(list(app_module.load_current_snapshot(sid).columns), ["keep"])

    def test_stale_selected_column_returns_conflict_without_mutation(self):
        sid, filename, _path = self._seed_session_file(content="Customer Name,keep\nA,1\n")
        authoritative = pd.DataFrame({"Customer Name": ["A"], "keep": [1]})
        app_module.save_current_snapshot(sid, authoritative)

        response = self.client.post("/preview_clean", data={
            "filename": filename,
            "preview_scope": "feature",
            "persist_apply": "1",
            "drop_column[missing-column]": "1",
        })

        self.assertEqual(response.status_code, 409)
        payload = response.get_json()
        self.assertEqual(payload["error"]["code"], "SELECTED_COLUMNS_NOT_FOUND")
        self.assertEqual(payload["error"]["details"]["unresolved_columns"], ["missing-column"])
        pd.testing.assert_frame_equal(app_module.load_current_snapshot(sid), authoritative)

    def test_drop_selected_columns_lists_nonempty_unnamed_only(self):
        with mock.patch.object(app_module.db, "create_project", return_value=123):
            response = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"keep,Unnamed: 15,Unnamed: 16\n1,legacy,\n2,,\n"), "unnamed.csv")},
                content_type="multipart/form-data",
            )
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(html.count('name="drop_column[Unnamed: 15]"'), 1)
        self.assertNotIn('name="drop_column[Unnamed: 16]"', html)

    def test_duplicate_operation_token_returns_same_authoritative_payload(self):
        _sid, filename, _path = self._seed_session_file()
        data = {
            "filename": filename,
            "persist_apply": "1",
            "operation_token": "same-token",
            "preview_scope": "section",
            "trim_whitespace": "1",
            "remove_duplicates": "0",
            "drop_empty_columns": "0",
            "drop_constant_columns": "0",
            "convert_dtypes": "0",
            "missing_strategy": "ignore",
        }
        first = self.client.post("/preview_clean", data=data).get_json()
        second = self.client.post("/preview_clean", data=data).get_json()
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        self.assertEqual(first["operation_token"], second["operation_token"])
        self.assertEqual(first["summary"], second["summary"])

    def test_diagnostic_log_written_without_cell_contents(self):
        with tempfile.TemporaryDirectory() as log_dir:
            os.environ["INDATAOUT_LOG_DIR"] = log_dir
            app_module.configure_logging()
            self.client.get("/")
            log_path = os.path.join(log_dir, "indataout.log")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as f:
                logs = f.read()
            self.assertIn("request_complete", logs)
            self.assertNotIn(" a ", logs)
            for handler in list(logging.getLogger().handlers):
                logging.getLogger().removeHandler(handler)
                handler.close()
            os.environ.pop("INDATAOUT_LOG_DIR", None)
            app_module.configure_logging()

    def test_duplicate_apply_returns_persisted_history_entry_and_manifest(self):
        _sid, filename, _path = self._seed_project_session_file()
        captured = {}

        def fake_atomic(**kwargs):
            captured.update(kwargs)
            kwargs["before_commit"](77)
            return 77

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=fake_atomic):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "feature",
                "action_scope": "dataIntegrityGroup",
                "action_name": "Applied feature only: Remove Duplicate Rows",
                "remove_duplicates": "1",
                "missing_strategy": "ignore",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn("history_entry", payload)
        self.assertEqual(payload["history_entry"]["id"], 77)
        self.assertTrue(payload["history_entry"]["has_manifest"])
        ops = payload["history_entry"]["action_config"]["operations"]
        dedupe = [op for op in ops if op.get("action") == "drop_duplicates"][0]
        self.assertEqual(dedupe["removed_rows"], 1)
        self.assertEqual(dedupe["dropped_rows_total"], 1)
        self.assertEqual(dedupe["dropped_rows_preview"][0]["data"], {"x": 2, "y": "B"})
        preview = payload["history_entry"]["action_config"]["preview"]
        self.assertEqual(preview["before_shape"], [3, 2])
        self.assertEqual(preview["after_shape"], [2, 2])
        self.assertEqual(preview["total_removed_rows"], 1)
        self.assertEqual(preview["removed_rows"][0]["data"], {"x": 2, "y": "B"})
        self.assertEqual(captured["rows_before"], 3)
        self.assertEqual(captured["rows_after"], 2)
        self.assertTrue(captured["manifest"]["dropped_rows"])
        self.assertIsNotNone(app_module.load_current_snapshot(_sid))

    def test_trim_history_stores_exact_authoritative_cell_preview(self):
        _sid, filename, _path = self._seed_project_session_file(content="name,value\n a ,1\nb,2\n")

        def fake_atomic(**kwargs):
            kwargs["before_commit"](88)
            return 88

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=fake_atomic):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "feature",
                "action_scope": "dataIntegrityGroup",
                "action_name": "Applied feature only: Trim Whitespace",
                "trim_whitespace": "1",
                "missing_strategy": "ignore",
            })

        self.assertEqual(response.status_code, 200)
        preview = response.get_json()["history_entry"]["action_config"]["preview"]
        self.assertEqual(preview["total_changed_cells"], 1)
        self.assertEqual(preview["changed_cells"][0]["column"], "name")
        self.assertEqual(preview["changed_cells"][0]["before"], " a ")
        self.assertEqual(preview["changed_cells"][0]["after"], "a")

    def test_committed_manual_missing_cell_editor_saves_history_and_snapshot(self):
        sid, filename, _path = self._seed_project_session_file(content="name,score\nA,\nB,2\n")

        def fake_atomic(**kwargs):
            kwargs["before_commit"](92)
            return 92

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=fake_atomic):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "manual",
                "action_scope": "missingValuesGroup",
                "action_name": "Applied: typed missing-cell values",
                "missing_strategy": "ignore",
                "convert_dtypes": "0",
                "missing_cell[0][score]": "5",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        history = payload["history_entry"]
        self.assertEqual(history["id"], 92)
        operation = history["action_config"]["operations"][0]
        self.assertEqual(operation["action"], "manual_missing_cell_updates")
        self.assertEqual(operation["count"], 1)
        self.assertEqual(app_module.load_current_snapshot(sid).loc[0, "score"], 5.0)

    def test_committed_manual_dtype_conversion_saves_reversible_dtype_manifest(self):
        sid, filename, _path = self._seed_project_session_file(content="code,score\nA,1\nB,2\n")

        def fake_atomic(**kwargs):
            kwargs["before_commit"](93)
            return 93

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=fake_atomic):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "feature",
                "action_scope": "dataTypesGroup",
                "action_name": "Applied feature only: Data Type Conversion",
                "missing_strategy": "ignore",
                "convert_dtypes": "0",
                "dtype_map[score]": "string",
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        history = payload["history_entry"]
        self.assertEqual(history["id"], 93)
        operation = history["action_config"]["operations"][0]
        self.assertEqual(operation["action"], "manual_convert_dtypes")
        self.assertIn("score", operation["changes"])
        preview = history["action_config"]["preview"]
        self.assertEqual(preview["total_dtype_changes"], 1)
        self.assertEqual(preview["dtype_changes"], [{
            "column": "score",
            "before": "int64",
            "after": "object",
        }])
        snapshot = app_module.load_current_snapshot(sid)
        self.assertEqual(str(snapshot["score"].dtype), "object")
        self.assertEqual(snapshot["score"].tolist(), ["1", "2"])

    def test_grouped_history_splits_operations_into_their_permanent_subfeatures(self):
        history = [
            {
                "id": 10,
                "action_name": "trim and deduplicate",
                "action_config": {
                    "activePanelId": "dataIntegrityGroup",
                    "operations": [
                        {"action": "trim_whitespace", "columns_processed": ["name"]},
                        {"action": "drop_duplicates", "removed_rows": 1},
                    ],
                    "preview": {"changed_cells": [], "removed_rows": [], "added_rows": [], "removed_columns": [], "added_columns": []},
                },
                "rows_before": 3,
                "rows_after": 2,
                "cells_changed": 2,
                "status": "undone",
            },
            {
                "id": 11,
                "action_name": "edit cell",
                "action_config": {
                    "activePanelId": "datasetEditorSlide",
                    "operations": [{"action": "manual_cell_updates", "count": 1}],
                },
                "rows_before": 2,
                "rows_after": 2,
                "cells_changed": 1,
                "status": "applied",
            },
            {
                "id": 12,
                "action_name": "drop selected columns",
                "action_config": {
                    "activePanelId": "dataIntegrityGroup",
                    "operations": [{"action": "drop_selected_columns", "count": 1, "columns_dropped": ["legacy"]}],
                },
                "rows_before": 2,
                "rows_after": 2,
                "cells_changed": 1,
                "status": "applied",
            },
        ]

        groups = app_module.build_grouped_cleaning_history(history)
        self.assertEqual([group["label"] for group in groups], [
            "Data Integrity", "Missing Values", "Data Types", "Outliers",
            "Dataset Cell Edits", "Drop Selected Columns", "Custom Pandas Code",
        ])
        self.assertEqual(sum(group["action_count"] for group in groups), 4)
        integrity = groups[0]
        self.assertEqual(len(integrity["actions"]), 0)
        duplicate_actions = next(item for item in integrity["subfeatures"] if item["key"] == "removeDuplicateRows")["actions"]
        trim_actions = next(item for item in integrity["subfeatures"] if item["key"] == "trimWhitespace")["actions"]
        self.assertEqual(len(duplicate_actions), 1)
        self.assertEqual(len(trim_actions), 1)
        self.assertEqual(duplicate_actions[0]["id"], 10)
        self.assertEqual(trim_actions[0]["id"], 10)
        self.assertEqual(duplicate_actions[0]["status"], "Undone")
        self.assertEqual(trim_actions[0]["status"], "Undone")
        cell_edits = groups[4]
        self.assertEqual(cell_edits["label"], "Dataset Cell Edits")
        self.assertEqual(len(cell_edits["actions"]), 1)
        self.assertEqual(cell_edits["actions"][0]["local_number"], 1)
        drop_selected = groups[5]
        self.assertEqual(drop_selected["label"], "Drop Selected Columns")
        self.assertEqual(len(drop_selected["actions"]), 1)
        self.assertEqual(drop_selected["actions"][0]["id"], 12)

    def test_combined_integrity_history_has_human_readable_operation_evidence(self):
        history = [{
            "id": 42,
            "action_name": "data integrity",
            "action_config": {
                "activePanelId": "dataIntegrityGroup",
                "operations": [
                    {"action": "drop_duplicates", "removed_rows": 3, "rows_dropped": [37, 38, 39]},
                    {"action": "drop_empty_columns", "count": 2, "columns_dropped": ["fully_empty", "Unnamed: 16"]},
                ],
                "preview": {
                    "before_shape": [39, 19],
                    "after_shape": [36, 17],
                    "changed_cells": [],
                    "removed_rows": [
                        {"row_id": 36, "row_number": 37, "data": {"City": "Pune", "record_id": 1004, "fully_empty": None}},
                        {"row_id": 37, "row_number": 38, "data": {"City": "Kochi", "record_id": 1012}},
                        {"row_id": 38, "row_number": 39, "data": {"City": "Kochi", "record_id": 1024}},
                    ],
                    "added_rows": [],
                    "removed_columns": [
                        {"column": "fully_empty", "sample": [{"row_id": 0, "value": None}]},
                        {"column": "Unnamed: 16", "sample": [{"row_id": 0, "value": None}]},
                    ],
                    "added_columns": [],
                },
            },
            "rows_before": 39,
            "rows_after": 36,
            "cells_changed": 5,
            "status": "applied",
        }]

        integrity = app_module.build_grouped_cleaning_history(history)[0]
        duplicates = next(item for item in integrity["subfeatures"] if item["key"] == "removeDuplicateRows")["actions"]
        empty_columns = next(item for item in integrity["subfeatures"] if item["key"] == "dropEmptyColumns")["actions"]

        self.assertEqual(integrity["action_count"], 2)
        self.assertEqual([item["local_number"] for item in duplicates], [1])
        self.assertEqual([item["local_number"] for item in empty_columns], [1])
        self.assertEqual([pair["context"] for pair in duplicates[0]["pairs"]], ["Row 37", "Row 38", "Row 39"])
        self.assertTrue(all(pair["after"] == "Removed" for pair in duplicates[0]["pairs"]))
        self.assertIn("City: Pune", duplicates[0]["pairs"][0]["before"])
        self.assertNotIn("{", duplicates[0]["pairs"][0]["before"])
        self.assertEqual(
            [(pair["context"], pair["before"], pair["after"]) for pair in empty_columns[0]["pairs"]],
            [("fully_empty", "Column existed", "Dropped"), ("Unnamed: 16", "Column existed", "Dropped")],
        )

    def test_resume_replay_returns_authoritative_snapshot_without_browser_ledger(self):
        sid, filename, _path = self._seed_project_session_file(content="name,value\n a ,1\nb,2\n")
        current = pd.DataFrame({"name": ["a", "b"], "value": [1, 2]})
        app_module.save_current_snapshot(sid, current)

        with mock.patch.object(app_module, "replay_project_actions", return_value=(current, [])) as replay:
            response = self.client.post("/resume_replay", json={"entries": [], "filename": filename})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["editor_data"][0]["name"], "a")
        self.assertEqual(payload["preview"][0]["name"], "a")
        replay.assert_called_once_with(123)

    def test_manual_missing_editor_resume_sync_uses_exact_missing_analysis(self):
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "script.js")
        with open(script_path, "r", encoding="utf-8") as handle:
            script = handle.read()

        self.assertIn("function remainingMissingCellSetFromAnalysis(analysis)", script)
        self.assertIn("const cellStillMissing = hasExactRemainingCells", script)
        self.assertIn("input.value = handled ? nextValue : '';", script)
        self.assertIn(
            "window._ido.syncMissingEditorFromPreview(replayData.preview, replayData.preview_row_numbers, "
            "replayData.editor_columns, replayData.summary, replayData.analysis)",
            script,
        )

    def test_change_log_action_time_uses_client_timestamp_on_resume(self):
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "script.js")
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
        with open(script_path, "r", encoding="utf-8") as handle:
            script = handle.read()
        with open(app_path, "r", encoding="utf-8") as handle:
            app_source = handle.read()

        self.assertIn("client_applied_at_ms", script)
        self.assertIn("applied_at_ms", script)
        self.assertIn("function _historyTs(entry, config)", script)
        self.assertIn("const ts         = _historyTs(entry, config);", script)
        self.assertIn("client_applied_at_ms = int", app_source)
        self.assertIn('config["applied_at_ms"] = applied_at_ms', app_source)
        self.assertIn('action_config["applied_at_ms"] = client_applied_at_ms', app_source)

    def test_apply_history_failure_preserves_previous_snapshot(self):
        sid, filename, _path = self._seed_project_session_file()
        original = pd.DataFrame({"x": [1, 2, 2], "y": ["A", "B", "B"]})
        app_module.save_current_snapshot(sid, original)

        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=RuntimeError("db down")):
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "feature",
                "action_scope": "dataIntegrityGroup",
                "action_name": "Applied feature only: Remove Duplicate Rows",
                "remove_duplicates": "1",
                "missing_strategy": "ignore",
            })

        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"]["code"], "ACTION_PERSISTENCE_FAILED")
        restored = app_module.load_current_snapshot(sid)
        self.assertEqual(restored.shape[0], 3)
        self.assertEqual(restored.to_dict(orient="records"), original.to_dict(orient="records"))

    def test_legacy_log_action_endpoint_is_not_authoritative(self):
        with self.client.session_transaction() as sess:
            sess["project_id"] = 123
        with mock.patch.object(app_module.db, "log_cleaning_action") as log_action:
            response = self.client.post("/log_action", json={"action_name": "unsafe"})
        self.assertEqual(response.status_code, 410)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"]["code"], "LEGACY_LOG_ACTION_DISABLED")
        log_action.assert_not_called()

    def test_project_detail_does_not_replace_active_session(self):
        with self.client.session_transaction() as sess:
            sess["upload_sid"] = "active_sid"
            sess["upload_filename"] = "active.csv"
            sess["project_id"] = 10
        project_folder = os.path.join(app_module.UPLOAD_FOLDER, "project_sid")
        os.makedirs(project_folder, exist_ok=True)
        with open(os.path.join(project_folder, "source.csv"), "w", encoding="utf-8") as f:
            f.write("x\n1\n")
        project = {
            "id": 99,
            "name": "Project",
            "notes": "",
            "original_file": "source.csv",
            "upload_sid": "project_sid",
            "file_path": None,
            "row_count": 1,
            "col_count": 1,
            "status": "active",
            "created_at": None,
        }
        with mock.patch.object(app_module.db, "get_project", return_value=project), \
             mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "get_dataset_versions", return_value=[]):
            response = self.client.get("/projects/99")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        for label in ("Data Integrity", "Missing Values", "Data Types", "Outliers", "Dataset Cell Edits", "Drop Selected Columns", "Custom Pandas Code"):
            self.assertIn(label, page)
        self.assertNotIn("Advanced Options", page)
        self.assertIn("Remove Duplicate Rows", page)
        self.assertIn("No actions yet", page)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["upload_sid"], "active_sid")
            self.assertEqual(sess["upload_filename"], "active.csv")
            self.assertEqual(sess["project_id"], 10)

    def test_resume_repairs_source_after_packaged_upload_root_moves(self):
        with tempfile.TemporaryDirectory() as old_root:
            old_project_dir = os.path.join(old_root, "project_sid")
            old_cleaned_dir = os.path.join(old_project_dir, "cleaned")
            os.makedirs(old_cleaned_dir, exist_ok=True)
            old_source = os.path.join(old_project_dir, "source.csv")
            old_cleaned = os.path.join(old_cleaned_dir, "cleaned.csv")
            with open(old_source, "w", encoding="utf-8") as f:
                f.write("x,y\n1,A\n2,B\n")
            with open(old_cleaned, "w", encoding="utf-8") as f:
                f.write("x,y\n1,A\n")
            project = {
                "id": 99,
                "name": "Project",
                "notes": "",
                "original_file": "source.csv",
                "upload_sid": "project_sid",
                "file_path": old_cleaned,
                "row_count": 2,
                "col_count": 2,
                "status": "cleaned",
                "state_version": 0,
                "created_at": None,
            }
            restored = os.path.join(app_module.UPLOAD_FOLDER, "project_sid", "source.csv")
            self.assertFalse(os.path.exists(restored))
            with mock.patch.object(app_module.db, "get_project", return_value=project), \
                 mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
                 mock.patch.object(app_module.db, "get_dataset_versions", return_value=[]):
                response = self.client.get("/projects/99/resume")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(os.path.exists(restored))
            with self.client.session_transaction() as sess:
                self.assertEqual(sess["upload_sid"], "project_sid")
                self.assertEqual(sess["upload_filename"], "source.csv")

    def test_uploading_second_project_preserves_first_project_folder(self):
        with mock.patch.object(app_module.db, "create_project", side_effect=[1, 2]):
            first = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"x\n1\n"), "first.csv")},
                content_type="multipart/form-data",
            )
            self.assertEqual(first.status_code, 200)
            with self.client.session_transaction() as sess:
                first_sid = sess["upload_sid"]
                csrf = sess["_csrf_token"]
            first_path = os.path.join(app_module.UPLOAD_FOLDER, first_sid, "first.csv")
            self.assertTrue(os.path.exists(first_path))
            second = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(b"x\n2\n"), "second.csv"), "csrf_token": csrf},
                content_type="multipart/form-data",
            )
            self.assertEqual(second.status_code, 200)
        self.assertTrue(os.path.exists(first_path))

    def test_project_scoped_current_download_ignores_active_session_file(self):
        project_folder = os.path.join(app_module.UPLOAD_FOLDER, "project_sid", "cleaned")
        os.makedirs(project_folder, exist_ok=True)
        cleaned_path = os.path.join(project_folder, "cleaned_project.csv")
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.write("x\n42\n")
        project = {
            "id": 99,
            "upload_sid": "project_sid",
            "file_path": cleaned_path,
            "original_file": "source.csv",
        }
        with self.client.session_transaction() as sess:
            sess["cleaned_sid"] = "other_sid"
            sess["cleaned_filename"] = "other.csv"
        with mock.patch.object(app_module.db, "get_project", return_value=project):
            response = self.client.get("/projects/99/download/cleaned")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode("utf-8").replace("\r\n", "\n"), "x\n42\n")
        response.close()

    def test_apply_manifest_uses_ordered_replay_once_without_direct_patch(self):
        sid, filename, _path = self._seed_project_session_file(content="x\n1\n2\n")
        with self.client.session_transaction() as sess:
            sess["project_id"] = 123
        tmp_path = os.path.join(app_module.UPLOAD_FOLDER, sid, "tmp_snapshot.parquet")
        final_path = os.path.join(app_module.UPLOAD_FOLDER, sid, "current_snapshot.parquet")
        replayed = pd.DataFrame({"x": [3, 4]})
        def atomic_status(_history_id, _project_id, _status, _version, before_commit=None):
            before_commit()
            return 1

        with mock.patch.object(app_module.db, "get_cleaning_action", return_value={"id": 77, "status": "applied"}), \
             mock.patch.object(app_module, "replay_project_actions", return_value=(replayed, [])) as replay, \
             mock.patch.object(app_module, "prepare_current_snapshot_temp", return_value=(tmp_path, final_path)) as prepare, \
             mock.patch.object(app_module, "promote_current_snapshot_temp") as promote, \
             mock.patch.object(app_module.db, "set_cleaning_action_status_atomic", side_effect=atomic_status) as set_status, \
             mock.patch.object(app_module.snapshot_diff, "apply_manifest") as apply_manifest:
            response = self.client.post("/apply_manifest", json={"history_id": 77, "direction": "reverse"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        replay.assert_called_once_with(123, target_status={77: "undone"})
        prepare.assert_called_once()
        promote.assert_called_once_with(tmp_path, final_path)
        self.assertEqual(set_status.call_args.args[:4], (77, 123, "undone", 0))
        apply_manifest.assert_not_called()

    def test_noop_committed_apply_does_not_create_replay_history_gap(self):
        sid, filename, _path = self._seed_project_session_file(content="x,y\nA,1\nB,2\n")
        current = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        app_module.save_current_snapshot(sid, current)
        with mock.patch.object(app_module.db, "log_cleaning_action_atomic") as log_atomic:
            response = self.client.post("/preview_clean", data={
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "section",
                "action_scope": "dataTypesGroup",
                "action_name": "Applied section only: Data Types",
                "convert_dtypes": "0",
                "missing_strategy": "ignore",
            })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["no_op"])
        self.assertNotIn("history_entry", payload)
        log_atomic.assert_not_called()

    def test_full_row_evidence_contains_all_columns_and_changed_markers(self):
        before = pd.DataFrame({"record_id": [10], "name": [" old "], "score": [4]}, index=[7])
        after = pd.DataFrame({"record_id": [10], "name": ["old"], "score": [4]}, index=[7])
        manifest = app_module.snapshot_diff.compute_full_diff(before, after)
        evidence = app_module._full_row_evidence(before, after, manifest)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["columns"], ["record_id", "name", "score"])
        self.assertEqual(evidence[0]["before"]["name"], " old ")
        self.assertEqual(evidence[0]["after"]["name"], "old")
        self.assertEqual(evidence[0]["changed_columns"], ["name"])
        self.assertEqual(evidence[0]["row_id"], "7")
        self.assertEqual(evidence[0]["original_row_number"], "7")

    def test_combined_operations_split_into_sequential_independent_manifests(self):
        before = pd.DataFrame({"x": [1, 1], "empty": [None, None]}, index=[0, 1])
        after = before.drop(index=[1]).drop(columns=["empty"])
        manifest = app_module.snapshot_diff.compute_full_diff(before, after)
        operations = [
            {"action": "drop_duplicates", "removed_rows": 1, "rows_dropped": [2]},
            {"action": "drop_empty_columns", "count": 1, "columns_dropped": ["empty"]},
        ]
        actions = app_module._sequential_history_actions(
            before, after, operations, manifest, "Integrity", "dataIntegrityGroup", 90.0
        )
        self.assertEqual(len(actions), 2)
        self.assertEqual((actions[0]["rows_before"], actions[0]["rows_after"]), (2, 1))
        self.assertEqual((actions[1]["rows_before"], actions[1]["rows_after"]), (1, 1))
        self.assertNotEqual(actions[0]["action_name"], actions[1]["action_name"])
        self.assertEqual(actions[0]["action_config"]["preview"]["after_shape"], [1, 2])
        self.assertEqual(actions[1]["action_config"]["preview"]["before_shape"], [1, 2])


if __name__ == "__main__":
    unittest.main()
