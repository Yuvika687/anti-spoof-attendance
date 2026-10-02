"""
Unit tests for src/gradcam.py.

Loads the real trained checkpoint (models/best_liveness.pth) -- Grad-CAM
fundamentally needs a real conv net with real gradients to test against,
a mock model wouldn't exercise anything meaningful. Skipped automatically
if that checkpoint doesn't exist (e.g. a fresh clone before training).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402
from gradcam import GradCAM  # noqa: E402
from model import build_model  # noqa: E402

CHECKPOINT_EXISTS = os.path.isfile(config.BEST_MODEL_PATH)


@unittest.skipUnless(CHECKPOINT_EXISTS, f"No trained checkpoint at {config.BEST_MODEL_PATH}")
class TestGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = config.DEVICE
        checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=cls.device)
        cls.model = build_model(backbone=checkpoint.get("backbone", config.BACKBONE), pretrained=False).to(cls.device)
        cls.model.load_state_dict(checkpoint["model_state_dict"])
        cls.model.eval()
        cls.image_size = checkpoint.get("image_size", config.IMAGE_SIZE)

    def _dummy_input(self):
        # A real conv net, but the input content doesn't need to be a real
        # face for these structural tests (shape/range/hook-lifecycle) --
        # only the qualitative sanity checks (non-degenerate, live vs spoof
        # differ) need real face crops, and those were verified by hand
        # against real test-set images (see report_notes.md).
        torch.manual_seed(0)
        return torch.randn(1, 3, self.image_size, self.image_size, device=self.device)

    def test_heatmap_shape_matches_input(self):
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            heatmap, prob = cam(self._dummy_input())
        self.assertEqual(heatmap.shape, (self.image_size, self.image_size))

    def test_heatmap_value_range(self):
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            heatmap, prob = cam(self._dummy_input())
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_hooks_removed_on_context_exit(self):
        cam = GradCAM(self.model, self.model.feature_extractor.features[12])
        self.assertIsNotNone(cam._fwd_handle)
        self.assertIsNotNone(cam._bwd_handle)
        with cam:
            cam(self._dummy_input())
        self.assertIsNone(cam._fwd_handle)
        self.assertIsNone(cam._bwd_handle)

    def test_hooks_removed_by_explicit_remove_and_idempotent(self):
        cam = GradCAM(self.model, self.model.feature_extractor.features[12])
        cam(self._dummy_input())
        cam.remove()
        self.assertIsNone(cam._fwd_handle)
        self.assertIsNone(cam._bwd_handle)
        cam.remove()  # must not raise when called twice

    def test_model_parameters_unchanged_after_call(self):
        """Grad-CAM computes gradients but must never actually update
        weights -- no optimizer.step() should ever run here."""
        param_before = next(self.model.parameters()).clone()
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            cam(self._dummy_input())
        param_after = next(self.model.parameters()).clone()
        self.assertTrue(torch.equal(param_before, param_after))

    def test_model_eval_mode_restored(self):
        self.model.eval()
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            cam(self._dummy_input())
        self.assertFalse(self.model.training)

    def test_predicted_target_is_not_degenerate_for_confident_predictions(self):
        """Regression test for a real bug found while building this module:
        backpropagating from the raw spoof-logit unconditionally produced an
        all-zero heatmap for confidently-LIVE predictions (Grad-CAM's ReLU
        zeroes an all-negative gradient-weighted map). target='predicted'
        must avoid this by explaining whichever class was actually
        predicted."""
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            heatmap, prob = cam(self._dummy_input(), target="predicted")
        self.assertGreater(heatmap.std(), 1e-6, "heatmap is degenerate (uniform/all-zero)")

    def test_target_must_be_valid(self):
        with GradCAM(self.model, self.model.feature_extractor.features[12]) as cam:
            with self.assertRaises(ValueError):
                cam(self._dummy_input(), target="not_a_real_target")


if __name__ == "__main__":
    unittest.main()
