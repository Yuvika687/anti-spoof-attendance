"""
Unit tests for src/build_summary.py (TASK K).

Runs against the REAL results/ and outputs/predictions/ directories
(read-only -- build_summary() never writes anything), skipped if the
required source file isn't present yet (e.g. a fresh clone before the
TASK G/H/I pipeline has been run).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
from build_summary import CANONICAL_PROTOCOL, build_summary  # noqa: E402

RESULTS_DIR = os.path.join(config.PROJECT_ROOT, "results")
PREDICTIONS_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "predictions")
REQUIRED_SOURCE = os.path.join(RESULTS_DIR, "v1_vs_v2.json")


@unittest.skipUnless(os.path.isfile(REQUIRED_SOURCE), f"{REQUIRED_SOURCE} not found -- run evaluate_v1_vs_v2.py first")
class TestBuildSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.missing = build_summary(RESULTS_DIR, PREDICTIONS_DIR)

    def test_top_level_sections_present(self):
        for key in ("canonical_protocol", "blazeface_only_protocol", "consistency_notes",
                    "headline", "v1_vs_v2", "backbone_comparison", "breakdowns",
                    "gray_zone", "masked_eval"):
            self.assertIn(key, self.summary)

    def test_headline_uses_canonical_protocol_and_n(self):
        self.assertEqual(self.summary["headline"]["protocol"], CANONICAL_PROTOCOL)
        self.assertEqual(self.summary["headline"]["n"], 3900)

    def test_headline_metrics_are_real_probabilities(self):
        h = self.summary["headline"]
        for key in ("accuracy", "apcer", "bpcer", "hter"):
            self.assertIsNotNone(h[key])
            self.assertGreaterEqual(h[key], 0.0)
            self.assertLessEqual(h[key], 1.0)

    def test_v1_vs_v2_uses_canonical_protocol(self):
        self.assertEqual(self.summary["v1_vs_v2"]["protocol"], CANONICAL_PROTOCOL)
        self.assertEqual(self.summary["v1_vs_v2"]["n"], 3900)
        self.assertIn("v1", self.summary["v1_vs_v2"])
        self.assertIn("v2", self.summary["v1_vs_v2"])

    def test_backbone_comparison_uses_canonical_protocol_and_has_all_three(self):
        bc = self.summary["backbone_comparison"]
        self.assertEqual(bc["protocol"], CANONICAL_PROTOCOL)
        self.assertEqual(bc["n"], 3900)
        for backbone in ("mobilenet_v3_small", "resnet18", "efficientnet_b0"):
            self.assertIn(backbone, bc)
            self.assertIsNotNone(bc[backbone], f"{backbone} metrics missing")

    def test_gray_zone_and_masked_eval_are_not_labeled_canonical(self):
        # These sections intentionally use a DIFFERENT protocol from the
        # headline -- the whole point of TASK K is that they must never be
        # silently equated with it.
        self.assertNotEqual(self.summary["gray_zone"]["protocol"], CANONICAL_PROTOCOL)
        self.assertNotEqual(self.summary["masked_eval"]["protocol"], CANONICAL_PROTOCOL)

    def test_consistency_notes_explain_both_known_discrepancies(self):
        notes_text = " ".join(n["numbers"] for n in self.summary["consistency_notes"])
        self.assertIn("95.94%", notes_text)
        self.assertIn("94.54%", notes_text)
        self.assertIn("99.04%", notes_text)
        self.assertIn("98.69%", notes_text)

    def test_is_json_serializable(self):
        import json
        json.dumps(self.summary)  # must not raise


if __name__ == "__main__":
    unittest.main()
