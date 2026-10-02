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

    # ---- TASK P: "Camera / Sample images" demo mode ----

    def test_sample_images_list(self):
        resp = self.client.get("/api/sample-images")
        self.assertEqual(resp.status_code, 200)
        images = resp.get_json()["images"]
        self.assertEqual(len(images), 8)
        ids = {img["id"] for img in images}
        self.assertEqual(ids, {"live", "masked_live", "photo", "phone", "pc_replay",
                                "poster", "mask_3d", "known_failure"})
        for img in images:
            self.assertIn("label", img)
            self.assertIn("description", img)
            self.assertNotIn("path", img)  # file paths must never be exposed

    def test_sample_image_detect_runs_real_pipeline_with_timings(self):
        resp = self.client.post("/api/sample-images/live/detect")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["face_detected"])
        self.assertIn(data["label"], ("LIVE", "SPOOF", "UNCERTAIN"))
        for stage in ("frame_ms", "detect_ms", "crop_ms", "classify_ms", "decision_ms"):
            self.assertIn(stage, data["timings"])
            self.assertGreaterEqual(data["timings"][stage], 0.0)
        # A single demo image is never smoothed with unrelated frames.
        self.assertEqual(data["smoothing_window"], 1)

    def test_sample_image_detect_known_failure_is_the_documented_pc_replay_case(self):
        # Regression test for a real double-crop bug found while building
        # this: the sample images must be sourced from the RAW (uncropped)
        # dataset, not data/celeba_spoof_cropped -- feeding an
        # already-cropped image through detect+crop_face a second time
        # silently zoomed in far more than intended and changed the
        # prediction (0.4251 cached -> ~0.075 double-cropped). Pin the
        # known-failure sample's raw_confidence close to its real,
        # independently-verified cached value.
        resp = self.client.post("/api/sample-images/known_failure/detect")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["face_detected"])
        self.assertEqual(data["raw_label"], "LIVE")
        self.assertAlmostEqual(data["raw_confidence"], 0.5749, delta=0.01)

    def test_sample_image_explain_runs_real_gradcam(self):
        resp = self.client.post("/api/sample-images/pc_replay/explain")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["face_detected"])
        self.assertIn(data["label"], ("LIVE", "SPOOF"))
        self.assertTrue(0.0 <= data["prob_spoof"] <= 1.0)
        self.assertTrue(data["overlay_image_base64"].startswith("data:image/jpeg;base64,"))

    def test_sample_image_unknown_id_returns_404(self):
        resp = self.client.post("/api/sample-images/not_a_real_id/detect")
        self.assertEqual(resp.status_code, 404)
        resp2 = self.client.post("/api/sample-images/not_a_real_id/explain")
        self.assertEqual(resp2.status_code, 404)

    def test_sample_image_detect_does_not_disturb_camera_smoothing_state(self):
        # Sample-mode calls must never pollute the live camera tab's
        # ongoing smoothing window (TASK P: "Camera / Sample images" are
        # two independent modes sharing one backend process).
        app_module = self.app_module
        app_module._recent_probs.clear()
        app_module._face_was_present = False

        detect_data_1 = {"frame": (self._sample_frame_io(), "frame.png")}
        r1 = self.client.post("/api/detect", data=detect_data_1, content_type="multipart/form-data")
        window_after_first = r1.get_json()["smoothing_window"]

        self.client.post("/api/sample-images/photo/detect")
        self.client.post("/api/sample-images/pc_replay/detect")

        detect_data_2 = {"frame": (self._sample_frame_io(), "frame.png")}
        r2 = self.client.post("/api/detect", data=detect_data_2, content_type="multipart/form-data")
        window_after_second = r2.get_json()["smoothing_window"]

        self.assertEqual(window_after_second, window_after_first + 1)


if __name__ == "__main__":
    unittest.main()
