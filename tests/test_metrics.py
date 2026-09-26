"""
Unit tests for the anti-spoofing metrics in src/evaluate.py.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import compute_anti_spoof_metrics  # noqa: E402

LIVE, SPOOF = 0, 1


class TestAntiSpoofMetrics(unittest.TestCase):
    def test_perfect_classifier(self):
        y_true = np.array([LIVE, LIVE, SPOOF, SPOOF])
        m = compute_anti_spoof_metrics(y_true, y_true.copy())
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["apcer"], 0.0)
        self.assertEqual(m["bpcer"], 0.0)
        self.assertEqual(m["hter"], 0.0)

    def test_apcer_counts_spoof_accepted_as_live(self):
        # 4 spoofs, 1 wrongly accepted as live -> APCER = 25%
        y_true = np.array([SPOOF, SPOOF, SPOOF, SPOOF, LIVE, LIVE])
        y_pred = np.array([SPOOF, SPOOF, SPOOF, LIVE, LIVE, LIVE])
        m = compute_anti_spoof_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["apcer"], 0.25)
        self.assertEqual(m["bpcer"], 0.0)

    def test_bpcer_counts_live_rejected_as_spoof(self):
        # 4 live, 2 wrongly rejected as spoof -> BPCER = 50%
        y_true = np.array([LIVE, LIVE, LIVE, LIVE, SPOOF, SPOOF])
        y_pred = np.array([LIVE, LIVE, SPOOF, SPOOF, SPOOF, SPOOF])
        m = compute_anti_spoof_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["bpcer"], 0.5)
        self.assertEqual(m["apcer"], 0.0)

    def test_hter_is_mean_of_apcer_and_bpcer(self):
        y_true = np.array([SPOOF, SPOOF, LIVE, LIVE])
        y_pred = np.array([LIVE, SPOOF, SPOOF, LIVE])  # APCER=0.5, BPCER=0.5
        m = compute_anti_spoof_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["hter"], (m["apcer"] + m["bpcer"]) / 2)

    def test_always_live_predictor_has_full_apcer(self):
        y_true = np.array([LIVE, LIVE, SPOOF, SPOOF])
        y_pred = np.zeros(4, dtype=int)
        m = compute_anti_spoof_metrics(y_true, y_pred)
        self.assertEqual(m["apcer"], 1.0)
        self.assertEqual(m["bpcer"], 0.0)

    def test_missing_class_yields_nan_not_crash(self):
        y_true = np.array([LIVE, LIVE])
        y_pred = np.array([LIVE, SPOOF])
        m = compute_anti_spoof_metrics(y_true, y_pred)
        self.assertTrue(np.isnan(m["apcer"]))
        self.assertAlmostEqual(m["bpcer"], 0.5)


if __name__ == "__main__":
    unittest.main()
