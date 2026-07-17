"""Unit tests for correctness-critical DataFrame transformations and replay.

These tests protect regressions where cleaning could silently change values,
where manifests could lose authoritative Before/After evidence, or where
Undo/Reapply could corrupt overlapping actions.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import cleaner
import snapshot_diff


class CleaningCoreCorrectnessTests(unittest.TestCase):
    """Core transformation and manifest invariants independent of Flask routes."""

    def test_mixed_type_whitespace_trimming_only_strips_strings(self):
        df = pd.DataFrame({"mixed": [" a ", 5, True, None]})
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "trim_whitespace": True,
                "convert_dtypes": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )
        self.assertEqual(cleaned["mixed"].tolist()[0], "a")
        self.assertEqual(cleaned["mixed"].tolist()[1], 5)
        self.assertEqual(cleaned["mixed"].tolist()[2], True)
        self.assertIsNone(cleaned["mixed"].tolist()[3])
        self.assertEqual(report["operations"][0]["action"], "trim_whitespace")

    def test_unsafe_auto_numeric_conversion_leaves_column_unchanged(self):
        df = pd.DataFrame({"amount": ["1", "2", "bad"]})
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "convert_dtypes": True,
                "trim_whitespace": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )
        self.assertEqual(cleaned["amount"].tolist(), ["1", "2", "bad"])
        self.assertFalse(any(op["action"] == "convert_dtypes" for op in report["operations"]))

    def test_placeholder_detection_does_not_replace_without_explicit_operation(self):
        df = pd.DataFrame({"status": ["Unknown", "-", ".", "*", "ok", None]})
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "trim_whitespace": False,
                "convert_dtypes": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )
        self.assertEqual(cleaned["status"].tolist(), ["Unknown", "-", ".", "*", "ok", None])
        self.assertFalse(any(op["action"] == "replace_placeholders" for op in report["operations"]))

    def test_explicit_placeholder_replacement_records_cells(self):
        df = pd.DataFrame({"status": ["N/A", "ok", None]})
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "replace_placeholders": True,
                "placeholder_tokens": ["N/A"],
                "trim_whitespace": False,
                "convert_dtypes": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )
        self.assertTrue(pd.isna(cleaned.loc[0, "status"]))
        op = next(op for op in report["operations"] if op["action"] == "replace_placeholders")
        self.assertEqual(op["count"], 1)
        self.assertEqual(op["replacements"][0]["before"], "N/A")

    def test_column_normalization_collision_rejected(self):
        df = pd.DataFrame({"A B": [1], "A_B": [2]})
        with self.assertRaises(ValueError):
            cleaner.clean_dataframe(
                df,
                {
                    "rename_columns": True,
                    "trim_whitespace": False,
                    "convert_dtypes": False,
                    "remove_duplicates": False,
                    "drop_empty_columns": False,
                    "drop_constant_columns": False,
                    "missing_strategy": "ignore",
                },
            )

    def test_selected_columns_use_exact_identifiers(self):
        df = pd.DataFrame({
            "Customer Name": ["A", "B"],
            "Unnamed: 15": [None, "legacy"],
            "keep": [1, 2],
        })
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "drop_selected_columns": ["Customer Name", "Unnamed: 15"],
                "trim_whitespace": False,
                "convert_dtypes": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )

        self.assertEqual(list(cleaned.columns), ["keep"])
        operation = next(op for op in report["operations"] if op["action"] == "drop_selected_columns")
        self.assertEqual(operation["columns_dropped"], ["Customer Name", "Unnamed: 15"])

    def test_unresolved_selected_column_rejects_entire_operation(self):
        df = pd.DataFrame({"Customer Name": ["A"], "keep": [1]})
        with self.assertRaisesRegex(ValueError, "Selected columns not found"):
            cleaner.clean_dataframe(
                df,
                {
                    "drop_selected_columns": ["Customer Name", "missing-column"],
                    "trim_whitespace": False,
                    "convert_dtypes": False,
                    "remove_duplicates": False,
                    "drop_empty_columns": False,
                    "drop_constant_columns": False,
                    "missing_strategy": "ignore",
                },
            )
        self.assertEqual(list(df.columns), ["Customer Name", "keep"])

    def test_label_encoding_preserves_null_and_saves_mapping(self):
        df = pd.DataFrame({"category": ["b", None, "a"]})
        cleaned, report = cleaner.clean_dataframe(
            df,
            {
                "encode_categoricals": "label",
                "trim_whitespace": False,
                "convert_dtypes": False,
                "remove_duplicates": False,
                "drop_empty_columns": False,
                "drop_constant_columns": False,
                "missing_strategy": "ignore",
            },
        )
        self.assertTrue(pd.isna(cleaned.loc[1, "category"]))
        op = next(op for op in report["operations"] if op["action"] == "label_encode")
        self.assertEqual(op["columns_encoded"][0]["mapping"], {"a": 0, "b": 1})

    def test_ordered_replay_overlap_undo_and_reapply(self):
        raw = pd.DataFrame({"x": [1]})
        after_a = pd.DataFrame({"x": [2]})
        after_b = pd.DataFrame({"x": [3]})
        manifest_a = snapshot_diff.compute_full_diff(raw, after_a)
        manifest_b = snapshot_diff.compute_full_diff(after_a, after_b)

        state = raw.copy()
        state = snapshot_diff.apply_manifest(state, manifest_b, "forward")["df"]
        self.assertEqual(int(state.loc[0, "x"]), 3)

        state = raw.copy()
        state = snapshot_diff.apply_manifest(state, manifest_a, "forward")["df"]
        state = snapshot_diff.apply_manifest(state, manifest_b, "forward")["df"]
        self.assertEqual(int(state.loc[0, "x"]), 3)

    def test_quality_breakdown_does_not_double_count_null_as_placeholder(self):
        df = pd.DataFrame({"a": [None, "N/A", "ok"]})
        breakdown = cleaner.data_quality_breakdown(df)
        self.assertEqual(breakdown["components"]["completeness"]["missing_cells"], 1)
        self.assertEqual(breakdown["components"]["placeholder_rate"]["placeholder_cells"], 1)

    def test_final_csv_and_report_can_agree_on_shape_and_columns(self):
        df = pd.DataFrame({"id": ["001", "002"], "value": [1, 2]})
        report = {"rows_after": len(df), "cleaned_columns": list(df.columns)}
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "cleaned.csv")
            report_path = os.path.join(tmp, "report.json")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            cleaner.save_report(report, report_path)
            reread = pd.read_csv(csv_path, dtype=str)
            with open(report_path, "r", encoding="utf-8") as f:
                reread_report = json.load(f)
        self.assertEqual(list(reread.columns), reread_report["cleaned_columns"])
        self.assertEqual(len(reread), reread_report["rows_after"])

    def test_arbitrary_missing_columns_use_independent_strategies(self):
        frame = pd.DataFrame({
            "measure one": [1.0, None, 3.0],
            "label.kind": ["red", None, "red"],
        })
        cleaned, report = cleaner.clean_dataframe(frame, options={
            "remove_duplicates": False,
            "drop_empty_columns": False,
            "drop_constant_columns": False,
            "trim_whitespace": False,
            "convert_dtypes": False,
            "missing_strategy": "fill",
            "fill_numeric": "skip",
            "fill_categorical": "skip",
            "numeric_col_strategies": {"measure one": "median"},
            "categorical_col_strategies": {"label.kind": "mode"},
        })
        self.assertEqual(cleaned.loc[1, "measure one"], 2.0)
        self.assertEqual(cleaned.loc[1, "label.kind"], "red")
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        operations = {item["action"]: item for item in report["operations"]}
        self.assertEqual(operations["fill_numeric"]["column_strategies"], {"measure one": "median"})
        self.assertEqual(operations["fill_categorical"]["column_strategies"], {"label.kind": "mode"})

    def test_typed_missing_cells_use_exact_row_and_column_identity(self):
        frame = pd.DataFrame({"A value": [None, "keep"], "B.value": [1, None]})
        cleaned, report = cleaner.clean_dataframe(frame, options={
            "remove_duplicates": False, "drop_empty_columns": False,
            "drop_constant_columns": False, "trim_whitespace": False,
            "convert_dtypes": False, "missing_strategy": "ignore",
            "missing_cell_updates": [
                {"row_index": "0", "column": "A value", "value": "typed"},
                {"row_index": "1", "column": "B.value", "value": "9"},
            ],
        })
        self.assertEqual(cleaned.loc[0, "A value"], "typed")
        self.assertEqual(cleaned.loc[1, "B.value"], 9)
        operation = next(item for item in report["operations"] if item["action"] == "manual_missing_cell_updates")
        self.assertEqual(operation["count"], 2)

    def test_stale_outlier_row_id_cannot_remove_another_row(self):
        frame = pd.DataFrame({"value": [1, 100, 1], "group": ["x", "x", "x"]})
        with self.assertRaisesRegex(ValueError, "no longer present"):
            cleaner.clean_dataframe(frame, options={
                "remove_duplicates": False, "drop_empty_columns": False,
                "drop_constant_columns": False, "trim_whitespace": False,
                "convert_dtypes": False, "missing_strategy": "ignore",
                "selected_outlier_rows": [99],
            })
        self.assertEqual(frame["value"].tolist(), [1, 100, 1])


if __name__ == "__main__":
    unittest.main()
