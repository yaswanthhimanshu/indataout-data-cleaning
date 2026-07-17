"""Tests for canonical heuristic data-quality scoring.

The score is intentionally treated as an explainable heuristic, not a proof of
correctness. These tests ensure score and breakdown share one formula and that
row/column loss, hidden coercion, and encoded missingness cannot masquerade as
unconditional improvement.
"""

import io
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import app as app_module
import cleaner
import snapshot_diff


class CanonicalQualityTests(unittest.TestCase):
    """Canonical scoring parity and route integration checks."""

    def setUp(self):
        self.old_upload = app_module.UPLOAD_FOLDER
        self.tmp = tempfile.TemporaryDirectory()
        app_module.UPLOAD_FOLDER = self.tmp.name
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.UPLOAD_FOLDER = self.old_upload
        self.tmp.cleanup()

    def _mixed_df(self):
        return pd.DataFrame({
            "amount": [1, "2", "bad-3", None, "-", True],
            "when": [pd.Timestamp("2024-01-01"), "bad-date", None, "2024-01-03", "?", False],
            "category": ["A", "Unknown", None, "###", "B", "A"],
            "score": [1.0, "2.5", "oops", None, 1000.0, 3],
        })

    def _score_from_percentages(self, metrics):
        return round(max(0, min(100, 100 - sum(
            metrics["weights"][k] * metrics["raw_percentages"][k]
            for k in metrics["weights"]
        ))), 2)

    def _seed_upload(self, csv_bytes=b"name,amount\n A ,1\nB,\nB,\n"):
        self.client.get("/")
        with self.client.session_transaction() as sess:
            csrf = sess["_csrf_token"]
        response = self.client.post(
            "/upload",
            data={"csrf_token": csrf, "file": (io.BytesIO(csv_bytes), "quality.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as sess:
            return csrf, sess["upload_sid"], sess["upload_filename"]

    def test_score_and_breakdown_are_same_canonical_result(self):
        df = self._mixed_df()
        metrics = cleaner.data_quality_metrics(df)
        self.assertEqual(cleaner.data_quality_score(df), metrics["score"])
        self.assertEqual(cleaner.data_quality_breakdown(df)["score"], metrics["score"])
        self.assertEqual(metrics["score"], self._score_from_percentages(metrics))

    def test_nulls_are_not_counted_as_placeholder_strings(self):
        df = pd.DataFrame({"x": [None, "-", "Unknown", "ok"]})
        metrics = cleaner.data_quality_metrics(df)
        self.assertEqual(metrics["components"]["completeness"]["missing_cells"], 1)
        self.assertEqual(metrics["components"]["placeholder_rate"]["placeholder_cells"], 2)

    def test_placeholder_tokens_are_identical_for_score_and_breakdown(self):
        df = pd.DataFrame({"x": [None, "-", "?", "Unknown", "ok"]})
        metrics = cleaner.data_quality_metrics(df)
        breakdown = cleaner.data_quality_breakdown(df)
        self.assertEqual(
            breakdown["components"]["placeholder_rate"]["tokens"],
            sorted(cleaner.QUALITY_PLACEHOLDER_TOKENS),
        )
        self.assertEqual(breakdown["components"]["placeholder_rate"]["placeholder_cells"], 3)
        self.assertEqual(cleaner.data_quality_score(df), metrics["score"])

    def test_mixed_type_scoring_does_not_raise_or_mutate(self):
        df = self._mixed_df()
        before = df.copy(deep=True)
        score = cleaner.data_quality_score(df)
        breakdown = cleaner.data_quality_breakdown(df)
        self.assertIsInstance(score, float)
        self.assertEqual(score, breakdown["score"])
        pd.testing.assert_frame_equal(df, before)

    def test_common_transform_results_match_fresh_full_calculation(self):
        original = pd.DataFrame({
            "a": [" x ", "x", "x", None],
            "amount": ["1", "2", "bad", "4"],
            "constant": ["c", "c", "c", "c"],
        })
        variants = []
        variants.append(original.drop_duplicates())
        variants.append(original.drop(columns=["constant"]))
        renamed = original.rename(columns={"a": "label"})
        variants.append(renamed)
        edited = original.copy()
        edited.loc[2, "amount"] = "3"
        variants.append(edited)
        added = original.copy()
        added["new_col"] = [1, 2, 3, 4]
        variants.append(added)
        removed = original.iloc[:2].copy()
        variants.append(removed)
        encoded = original.copy()
        encoded["a"] = encoded["a"].astype("category").cat.codes
        variants.append(encoded)

        for variant in variants:
            self.assertEqual(cleaner.data_quality_score(variant), cleaner.data_quality_metrics(variant)["score"])

    def test_manifest_apply_results_match_fresh_full_calculation(self):
        before = pd.DataFrame({"x": [" a ", "b", "b"], "amount": [1, 2, 2]})
        after = before.drop_duplicates().copy()
        manifest = snapshot_diff.compute_full_diff(before, after)
        replayed = snapshot_diff.apply_manifest(before, manifest, "forward")["df"]
        self.assertEqual(cleaner.data_quality_score(replayed), cleaner.data_quality_metrics(replayed)["score"])

    def test_preview_and_final_report_use_fresh_canonical_score(self):
        csrf, sid, filename = self._seed_upload()
        with mock.patch.object(app_module.db, "get_cleaning_history", return_value=[]), \
             mock.patch.object(app_module.db, "log_cleaning_action_atomic", side_effect=lambda **kw: (kw["before_commit"](77), 77)[1]):
            preview = self.client.post("/preview_clean", data={
                "csrf_token": csrf,
                "filename": filename,
                "persist_apply": "1",
                "preview_scope": "section",
                "operation_token": "quality-token",
                "trim_whitespace": "1",
                "missing_strategy": "ignore",
            })
        self.assertEqual(preview.status_code, 200)
        preview_json = preview.get_json()
        current = app_module.load_current_snapshot(sid)
        expected = cleaner.data_quality_score(current)
        self.assertEqual(preview_json["summary"]["quality_after"], expected)
        self.assertEqual(preview_json["report"]["quality_breakdown"]["score"], expected)

        final = self.client.post("/clean", data={"csrf_token": csrf, "filename": filename})
        self.assertEqual(final.status_code, 200)
        with self.client.session_transaction() as sess:
            report_name = sess["report_filename"]
        report_path = os.path.join(app_module.UPLOAD_FOLDER, sid, "reports", report_name)
        with open(report_path, "r", encoding="utf-8") as handle:
            import json
            report = json.load(handle)
        self.assertEqual(report["data_quality_after"], cleaner.data_quality_score(current))
        self.assertEqual(report["quality_breakdown"]["score"], report["data_quality_after"])

    def test_noop_action_keeps_exact_score_and_no_false_improvement(self):
        csrf, sid, filename = self._seed_upload(b"x,y\nA,1\nB,2\n")
        original = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        app_module.save_current_snapshot(sid, original)
        before_score = cleaner.data_quality_score(original)
        response = self.client.post("/preview_clean", data={
            "csrf_token": csrf,
            "filename": filename,
            "persist_apply": "1",
            "preview_scope": "section",
            "operation_token": "noop-quality",
            "convert_dtypes": "0",
            "missing_strategy": "ignore",
        })
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["no_op"])
        self.assertEqual(payload["summary"]["quality_after"], before_score)
        self.assertEqual(payload["report"]["quality_breakdown"]["score"], before_score)


if __name__ == "__main__":
    unittest.main()
