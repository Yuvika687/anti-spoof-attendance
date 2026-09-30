"""
Unit tests for prepare_celeba_subset.py's official-label correction logic.

This exists because a real audit of the actual downloaded CelebA-Spoof data
found 2,022 images (out of 561,575) filed under a "live" folder while being
officially labeled spoof in the dataset's own metas/*.json ground truth.
scan_identities() must use the official label as authoritative, not the
folder name, when both are available -- these tests pin that behavior down
with a synthetic mock dataset that deliberately includes such a mismatch.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prepare_celeba_subset import load_official_labels, scan_identities  # noqa: E402


def _make_mock_raw_dataset(root, with_official_labels=True, with_mismatch=True):
    """
    Builds a tiny mock raw CelebA-Spoof tree:

        Data/train/id1/live/{a.jpg, b.jpg}
        Data/train/id1/spoof/{c.jpg}
        metas/intra_test/train_label.json  (44-element vectors)

    b.jpg is physically in the "live" folder but, if with_mismatch=True,
    officially labeled spoof (1) -- this is the exact real-world case found
    during the audit.
    """
    live_dir = os.path.join(root, "Data", "train", "id1", "live")
    spoof_dir = os.path.join(root, "Data", "train", "id1", "spoof")
    os.makedirs(live_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)

    for d, fname in [(live_dir, "a.jpg"), (live_dir, "b.jpg"), (spoof_dir, "c.jpg")]:
        with open(os.path.join(d, fname), "wb") as f:
            f.write(b"\xff\xd8\xff")  # minimal fake jpeg bytes, content irrelevant here

    if with_official_labels:
        metas_dir = os.path.join(root, "metas", "intra_test")
        os.makedirs(metas_dir, exist_ok=True)

        def vec(label):
            v = [0] * 44
            v[43] = label
            return v

        b_official_label = 1 if with_mismatch else 0  # spoof if mismatch requested
        labels = {
            "Data/train/id1/live/a.jpg": vec(0),              # agrees: folder=live, official=live
            "Data/train/id1/live/b.jpg": vec(b_official_label),  # the mismatch case
            "Data/train/id1/spoof/c.jpg": vec(1),              # agrees: folder=spoof, official=spoof
        }
        with open(os.path.join(metas_dir, "train_label.json"), "w") as f:
            json.dump(labels, f)
        with open(os.path.join(metas_dir, "test_label.json"), "w") as f:
            json.dump({}, f)


class TestLoadOfficialLabels(unittest.TestCase):
    def test_loads_and_merges_label_files(self):
        with tempfile.TemporaryDirectory() as root:
            _make_mock_raw_dataset(root)
            labels = load_official_labels(root)
            self.assertEqual(labels[os.path.normpath("Data/train/id1/live/a.jpg")], 0)
            self.assertEqual(labels[os.path.normpath("Data/train/id1/live/b.jpg")], 1)
            self.assertEqual(labels[os.path.normpath("Data/train/id1/spoof/c.jpg")], 1)

    def test_missing_label_files_returns_empty_dict(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Data", "train"))
            labels = load_official_labels(root)
            self.assertEqual(labels, {})


class TestScanIdentitiesLabelCorrection(unittest.TestCase):
    def test_official_label_overrides_folder_name_on_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            _make_mock_raw_dataset(root, with_official_labels=True, with_mismatch=True)
            identities = scan_identities(root)

            key = "train_id1"
            live_files = [os.path.basename(p) for p in identities[key]["live"]]
            spoof_files = [os.path.basename(p) for p in identities[key]["spoof"]]

            # b.jpg was physically in live/ but officially labeled spoof ->
            # must end up in the spoof list, not the live list.
            self.assertIn("a.jpg", live_files)
            self.assertNotIn("b.jpg", live_files)
            self.assertIn("b.jpg", spoof_files)
            self.assertIn("c.jpg", spoof_files)

    def test_folder_and_official_label_agreeing_is_unaffected(self):
        with tempfile.TemporaryDirectory() as root:
            _make_mock_raw_dataset(root, with_official_labels=True, with_mismatch=False)
            identities = scan_identities(root)

            key = "train_id1"
            live_files = [os.path.basename(p) for p in identities[key]["live"]]
            spoof_files = [os.path.basename(p) for p in identities[key]["spoof"]]

            # No mismatch this time -> b.jpg stays in live as its folder suggests.
            self.assertIn("a.jpg", live_files)
            self.assertIn("b.jpg", live_files)
            self.assertIn("c.jpg", spoof_files)

    def test_falls_back_to_folder_name_when_no_official_labels_present(self):
        with tempfile.TemporaryDirectory() as root:
            _make_mock_raw_dataset(root, with_official_labels=False)
            identities = scan_identities(root)

            key = "train_id1"
            live_files = [os.path.basename(p) for p in identities[key]["live"]]
            spoof_files = [os.path.basename(p) for p in identities[key]["spoof"]]

            # No ground truth available at all -> must trust folder names as before.
            self.assertIn("a.jpg", live_files)
            self.assertIn("b.jpg", live_files)
            self.assertIn("c.jpg", spoof_files)


if __name__ == "__main__":
    unittest.main()
