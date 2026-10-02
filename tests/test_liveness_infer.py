"""
Unit tests for the three-way LIVE/SPOOF/UNCERTAIN decision logic in
src/liveness_infer.py (TASK H). classify_three_way is a pure function
(no model/IO), so these run instantly with no checkpoint needed.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from liveness_infer import classify_three_way  # noqa: E402


class TestClassifyThreeWay(unittest.TestCase):
    def test_below_band_is_live(self):
        self.assertEqual(classify_three_way(0.01, band_low=0.3, band_high=0.7, threshold=0.5), "LIVE")

    def test_above_band_is_spoof(self):
        self.assertEqual(classify_three_way(0.99, band_low=0.3, band_high=0.7, threshold=0.5), "SPOOF")

    def test_inside_band_is_uncertain(self):
        self.assertEqual(classify_three_way(0.5, band_low=0.3, band_high=0.7, threshold=0.5), "UNCERTAIN")

    def test_band_lower_boundary_is_inclusive(self):
        self.assertEqual(classify_three_way(0.3, band_low=0.3, band_high=0.7, threshold=0.5), "UNCERTAIN")

    def test_band_upper_boundary_is_inclusive(self):
        self.assertEqual(classify_three_way(0.7, band_low=0.3, band_high=0.7, threshold=0.5), "UNCERTAIN")

    def test_just_below_lower_boundary_is_decided(self):
        # Just outside the band, below threshold -> LIVE, not UNCERTAIN.
        self.assertEqual(classify_three_way(0.2999, band_low=0.3, band_high=0.7, threshold=0.5), "LIVE")

    def test_just_above_upper_boundary_is_decided(self):
        self.assertEqual(classify_three_way(0.7001, band_low=0.3, band_high=0.7, threshold=0.5), "SPOOF")

    def test_asymmetric_band_like_real_gray_zone_json(self):
        # Mirrors the real v2 band shape (not centered on 0.5).
        self.assertEqual(classify_three_way(0.5, band_low=0.024, band_high=0.976, threshold=0.5), "UNCERTAIN")
        self.assertEqual(classify_three_way(0.01, band_low=0.024, band_high=0.976, threshold=0.5), "LIVE")
        self.assertEqual(classify_three_way(0.99, band_low=0.024, band_high=0.976, threshold=0.5), "SPOOF")

    def test_zero_width_band_never_triggers_uncertain(self):
        # band_low == band_high: only a single exact probability value
        # would ever land inside it -- effectively disables UNCERTAIN for
        # any realistic continuous probability.
        self.assertEqual(classify_three_way(0.5, band_low=0.5, band_high=0.5, threshold=0.5), "UNCERTAIN")
        self.assertEqual(classify_three_way(0.5001, band_low=0.5, band_high=0.5, threshold=0.5), "SPOOF")

    def test_threshold_only_affects_decided_region_not_band(self):
        # A non-default threshold must still only matter OUTSIDE the band.
        self.assertEqual(classify_three_way(0.5, band_low=0.3, band_high=0.7, threshold=0.9), "UNCERTAIN")
        self.assertEqual(classify_three_way(0.85, band_low=0.3, band_high=0.7, threshold=0.9), "LIVE")
        self.assertEqual(classify_three_way(0.95, band_low=0.3, band_high=0.7, threshold=0.9), "SPOOF")


if __name__ == "__main__":
    unittest.main()
