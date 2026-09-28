"""
Unit tests for LivenessDataset's label mapping and class counting in
src/dataset.py. A mistake here would silently mislabel live/spoof samples
and corrupt training without any obvious symptom, so it's worth pinning
down explicitly.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import LABEL_LIVE, LABEL_SPOOF  # noqa: E402
from dataset import LivenessDataset  # noqa: E402


def _make_fake_dataset_dir(root, live_count, spoof_count, extra_dirs=None):
    for class_name, count in (("live", live_count), ("spoof", spoof_count)):
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(count):
            Image.new("RGB", (8, 8), color=(i, i, i)).save(
                os.path.join(class_dir, f"img_{i}.jpg")
            )
    for d in extra_dirs or []:
        os.makedirs(os.path.join(root, d), exist_ok=True)


class TestLivenessDataset(unittest.TestCase):
    def test_live_images_get_label_live(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_dataset_dir(root, live_count=3, spoof_count=2)
            ds = LivenessDataset(root)
            live_labels = {label for path, label in ds.samples if "live" in path}
            self.assertEqual(live_labels, {LABEL_LIVE})

    def test_spoof_images_get_label_spoof(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_dataset_dir(root, live_count=3, spoof_count=2)
            ds = LivenessDataset(root)
            spoof_labels = {label for path, label in ds.samples if "spoof" in path}
            self.assertEqual(spoof_labels, {LABEL_SPOOF})

    def test_class_counts_matches_files_on_disk(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_dataset_dir(root, live_count=5, spoof_count=7)
            ds = LivenessDataset(root)
            self.assertEqual(len(ds), 12)
            self.assertEqual(ds.class_counts(), {"live": 5, "spoof": 7})

    def test_non_image_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_dataset_dir(root, live_count=2, spoof_count=2)
            with open(os.path.join(root, "live", "readme.txt"), "w") as f:
                f.write("not an image")
            ds = LivenessDataset(root)
            self.assertEqual(len(ds), 4)

    def test_missing_class_folder_is_tolerated(self):
        # Only a live/ folder present (e.g. a masked-samples-only check dir).
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "live"))
            Image.new("RGB", (8, 8)).save(os.path.join(root, "live", "a.jpg"))
            ds = LivenessDataset(root)
            self.assertEqual(ds.class_counts(), {"live": 1, "spoof": 0})

    def test_empty_root_raises(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                LivenessDataset(root)

    def test_getitem_returns_image_and_correct_label(self):
        with tempfile.TemporaryDirectory() as root:
            _make_fake_dataset_dir(root, live_count=1, spoof_count=1)
            ds = LivenessDataset(root)
            for i in range(len(ds)):
                image, label = ds[i]
                self.assertIn(label, (LABEL_LIVE, LABEL_SPOOF))
                self.assertEqual(image.size, (8, 8))


if __name__ == "__main__":
    unittest.main()
