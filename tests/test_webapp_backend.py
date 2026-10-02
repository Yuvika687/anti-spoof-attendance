"""
Unit tests for the new/fixed webapp backend endpoints (TASK M's
/api/explain, TASK N's data API, and the stale-v1-endpoint fixes).

Imports the real Flask app and real model (same startup path as running
`python app.py`), so these are skipped if the v2 checkpoint or its result
files don't exist yet, rather than mocking anything out -- consistent
with this project's "no mocking, test for real" approach everywhere else.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "webapp", "backend", "app.py")
V2_CHECKPOINT = os.path.join(config.MODELS_DIR, "v2_cropped", "best_liveness.pth")
GRADCAM_GALLERY = os.path.join(config.PROJECT_ROOT, "results", "v2_gradcam", "gallery.json")
MASKED_EXAMPLES_DIR = os.path.join(config.PROJECT_ROOT, "results", "masked", "examples")
SAMPLE_IMAGE = os.path.join(config.PROJECT_ROOT, "data", "celeba_spoof_cropped", "test", "live", "test_10059_001096.png")


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_backend_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    os.path.isfile(V2_CHECKPOINT) and os.path.isfile(SAMPLE_IMAGE),
    "v2 checkpoint or sample test image not found -- run the TASK G pipeline first",
)
class TestWebappBackendEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = _load_app_module()
        cls.client = cls.app_module.app.test_client()

    def _sample_frame(self):
        with open(SAMPLE_IMAGE, "rb") as f:
            return f.read()

    # ---- TASK N: stale-endpoint fixes now point at the active (v2) model ----

    def test_metrics_endpoint_uses_active_v2_model_not_stale_v1(self):
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        # v1's stale raw-frame number was 0.9915...; v2's real deployment-crop
        # number is 0.9869... -- assert it's NOT the old v1 figure.
        self.assertNotAlmostEqual(data["accuracy"], 0.9915384615384616, places=3)
        self.assertGreater(data["accuracy"], 0.9)

    def test_training_curves_plot_served(self):
        resp = self.client.get("/api/plots/training-curves.png")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "image/png")

    def test_confusion_matrix_plot_served(self):
        resp = self.client.get("/api/plots/confusion-matrix.png")
        self.assertEqual(resp.status_code, 200)

    # ---- TASK N: new data endpoints ----

    def test_summary_endpoint(self):
        resp = self.client.get("/api/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for key in ("headline", "v1_vs_v2", "backbone_comparison", "gray_zone", "masked_eval"):
            self.assertIn(key, data)

    @unittest.skipUnless(os.path.isfile(GRADCAM_GALLERY), "v2 gradcam gallery not generated yet")
    def test_gradcam_gallery_endpoint(self):
        resp = self.client.get("/api/gradcam/gallery")
        self.assertEqual(resp.status_code, 200)
        gallery = resp.get_json()
        self.assertGreater(len(gallery), 0)

        first_file = gallery[0]["file"]
        img_resp = self.client.get(f"/api/gradcam/image/{first_file}")
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.mimetype, "image/png")

    def test_gradcam_image_path_traversal_blocked(self):
        resp = self.client.get("/api/gradcam/image/..%2f..%2f..%2fetc%2fpasswd")
        self.assertIn(resp.status_code, (404, 400))

    @unittest.skipUnless(os.path.isdir(MASKED_EXAMPLES_DIR), "masked examples not generated yet")
    def test_masked_examples_endpoint(self):
        resp = self.client.get("/api/masked/examples")
        self.assertEqual(resp.status_code, 200)
        examples = resp.get_json()["examples"]
        self.assertGreater(len(examples), 0)
        for ex in examples:
            self.assertIn(ex["color"], ("white", "light_blue", "black", None))
            self.assertIn(ex["true_label"], ("live", "spoof", None))

        first_file = examples[0]["file"]
        img_resp = self.client.get(f"/api/masked/image/{first_file}")
        self.assertEqual(img_resp.status_code, 200)

    def test_masked_image_path_traversal_blocked(self):
        resp = self.client.get("/api/masked/image/..%2f..%2f..%2fetc%2fpasswd")
        self.assertIn(resp.status_code, (404, 400))

    def test_breakdown_and_gray_zone_plots_served(self):
        for endpoint in ("/api/plots/breakdown-spoof-type.png", "/api/plots/breakdown-illumination.png",
                          "/api/plots/breakdown-environment.png", "/api/plots/gray-zone-curve.png"):
            resp = self.client.get(endpoint)
            self.assertEqual(resp.status_code, 200, f"{endpoint} failed")

    # ---- TASK M: /api/explain ----

    def test_explain_endpoint_returns_real_prediction_and_overlay(self):
        data = {"frame": (self._sample_frame_io(), "frame.png")}
        resp = self.client.post("/api/explain", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        result = resp.get_json()
        self.assertTrue(result["face_detected"])
        self.assertIn(result["label"], ("LIVE", "SPOOF"))
        self.assertTrue(0.0 <= result["prob_spoof"] <= 1.0)
        self.assertTrue(result["overlay_image_base64"].startswith("data:image/jpeg;base64,"))

    def test_explain_does_not_disturb_detect_smoothing_state(self):
        app_module = self.app_module
        app_module._recent_probs.clear()
        app_module._face_was_present = False

        detect_data_1 = {"frame": (self._sample_frame_io(), "frame.png")}
        r1 = self.client.post("/api/detect", data=detect_data_1, content_type="multipart/form-data")
        window_after_first_detect = r1.get_json()["smoothing_window"]

        explain_data = {"frame": (self._sample_frame_io(), "frame.png")}
        self.client.post("/api/explain", data=explain_data, content_type="multipart/form-data")

        detect_data_2 = {"frame": (self._sample_frame_io(), "frame.png")}
        r2 = self.client.post("/api/detect", data=detect_data_2, content_type="multipart/form-data")
        window_after_second_detect = r2.get_json()["smoothing_window"]

        # The explain call must not have appended to or reset the smoothing
        # window -- it should grow by exactly 1 between the two /api/detect
        # calls, exactly as if /api/explain had never been called.
        self.assertEqual(window_after_second_detect, window_after_first_detect + 1)

    def _sample_frame_io(self):
        import io
        return io.BytesIO(self._sample_frame())


if __name__ == "__main__":
    unittest.main()
