"""
Unit tests for the pure geometry helpers in src/face_utils.py: Detection's
derived properties and crop_face's margin/clipping logic. These don't touch
MediaPipe's model at all, so they run instantly with no download.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from face_utils import Detection, crop_face  # noqa: E402


class TestDetection(unittest.TestCase):
    def test_x2_y2_derived_correctly(self):
        d = Detection(x=10, y=20, width=30, height=40, confidence=0.9)
        self.assertEqual(d.x2, 40)
        self.assertEqual(d.y2, 60)


class TestCropFace(unittest.TestCase):
    def setUp(self):
        # 100x200 (h x w) blank frame with a distinct value at a known patch
        # so we can check the crop actually contains the right region.
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.frame[20:60, 30:80] = 255  # bright square

    def test_crop_matches_bbox_with_no_margin(self):
        d = Detection(x=30, y=20, width=50, height=40, confidence=1.0)
        crop = crop_face(self.frame, d, margin=0.0)
        self.assertEqual(crop.shape, (40, 50, 3))
        self.assertTrue(np.all(crop == 255))

    def test_margin_expands_crop_symmetrically(self):
        d = Detection(x=30, y=20, width=50, height=40, confidence=1.0)
        crop = crop_face(self.frame, d, margin=0.2)
        # 20% of width/height added as padding on each side
        expected_w = 50 + 2 * int(50 * 0.2)
        expected_h = 40 + 2 * int(40 * 0.2)
        self.assertEqual(crop.shape, (expected_h, expected_w, 3))

    def test_crop_clips_to_frame_bounds_near_edge(self):
        # Detection near the top-left corner with a margin that would
        # otherwise push the crop out of bounds.
        d = Detection(x=0, y=0, width=20, height=20, confidence=1.0)
        crop = crop_face(self.frame, d, margin=0.5)
        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.shape[0], 100)
        self.assertLessEqual(crop.shape[1], 200)
        # top-left corner must stay clipped at (0, 0), not go negative
        self.assertEqual(crop.shape, (30, 30, 3))  # 20 + 0.5*20 on the free side only

    def test_crop_clips_to_frame_bounds_near_bottom_right(self):
        d = Detection(x=190, y=90, width=10, height=10, confidence=1.0)
        crop = crop_face(self.frame, d, margin=1.0)
        self.assertIsNotNone(crop)
        # frame is 200 wide, 100 tall; crop must not exceed that
        self.assertLessEqual(crop.shape[0], 100)
        self.assertLessEqual(crop.shape[1], 200)

    def test_zero_size_detection_returns_none(self):
        d = Detection(x=50, y=50, width=0, height=0, confidence=1.0)
        crop = crop_face(self.frame, d, margin=0.0)
        self.assertIsNone(crop)


if __name__ == "__main__":
    unittest.main()
