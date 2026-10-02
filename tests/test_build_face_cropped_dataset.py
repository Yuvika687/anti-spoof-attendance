"""
Unit tests for the BB.txt-fallback logic in src/build_face_cropped_dataset.py
(used when BlazeFace misses a face during the cropped-dataset build -- see
TASK G). Uses real temp image files + real *_BB.txt sidecar files, no
mocking, following the same style as tests/test_face_utils.py.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_face_cropped_dataset import read_official_bbox  # noqa: E402


class TestReadOfficialBBox(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_image(self, name: str, w: int, h: int) -> str:
        path = os.path.join(self.tmpdir, name)
        cv2.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
        return path

    def _write_bb(self, image_path: str, bbox_224: list, score: float = 0.99):
        bb_path = os.path.splitext(image_path)[0] + "_BB.txt"
        with open(bb_path, "w") as f:
            f.write(" ".join(str(v) for v in bbox_224 + [score]))
        return bb_path

    def test_scales_bbox_per_official_formula(self):
        # Official formula: real = bbox_224_value * (real_dim / 224)
        img_path = self._write_image("a.png", w=448, h=448)  # 2x scale factor
        self._write_bb(img_path, [54, 41, 68, 71])

        det = read_official_bbox(img_path)
        self.assertIsNotNone(det)
        self.assertEqual(det.x, int(54 * (448 / 224)))
        self.assertEqual(det.y, int(41 * (448 / 224)))
        self.assertEqual(det.width, int(68 * (448 / 224)))
        self.assertEqual(det.height, int(71 * (448 / 224)))
        self.assertAlmostEqual(det.confidence, 0.99, places=4)

    def test_missing_bb_file_returns_none(self):
        img_path = self._write_image("no_bb.png", w=300, h=300)
        # No _BB.txt written for this image.
        self.assertIsNone(read_official_bbox(img_path))

    def test_malformed_bb_file_returns_none(self):
        img_path = self._write_image("bad.png", w=300, h=300)
        bb_path = os.path.splitext(img_path)[0] + "_BB.txt"
        with open(bb_path, "w") as f:
            f.write("not numbers at all")
        self.assertIsNone(read_official_bbox(img_path))

    def test_zero_size_bbox_returns_none(self):
        img_path = self._write_image("zero.png", w=300, h=300)
        self._write_bb(img_path, [10, 10, 0, 0])
        self.assertIsNone(read_official_bbox(img_path))

    def test_bbox_clipped_to_frame_bounds(self):
        # A bbox whose scaled coordinates would run past the real image
        # edges (can happen since the 224-space box isn't guaranteed to
        # perfectly round-trip) must come back clipped, not out of bounds.
        img_path = self._write_image("edge.png", w=100, h=100)
        self._write_bb(img_path, [200, 200, 100, 100])  # far out of range at this scale

        det = read_official_bbox(img_path)
        self.assertIsNotNone(det)
        self.assertLessEqual(det.x + det.width, 100)
        self.assertLessEqual(det.y + det.height, 100)
        self.assertGreaterEqual(det.x, 0)
        self.assertGreaterEqual(det.y, 0)

    def test_unreadable_image_returns_none(self):
        # _BB.txt exists but the image file itself is not a valid image.
        img_path = os.path.join(self.tmpdir, "corrupt.png")
        with open(img_path, "wb") as f:
            f.write(b"not an image")
        self._write_bb(img_path, [10, 10, 20, 20])
        self.assertIsNone(read_official_bbox(img_path))


if __name__ == "__main__":
    unittest.main()
